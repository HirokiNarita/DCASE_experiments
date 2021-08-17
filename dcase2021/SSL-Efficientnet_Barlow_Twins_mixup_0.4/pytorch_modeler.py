############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime

# general analysis tool-kit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
from collections import defaultdict

# original library
import common as com
import preprocessing as prep
from augment import Augment

############################################################################
# load config
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
log_folder = config['IO_OPTION']['OUTPUT_ROOT']+'/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, 'pytorch_modeler.py')
############################################################################
# Setting seed
############################################################################
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################
# Make Dataloader
############################################################################
def make_dataloader(train_paths, machine_type, mode='training'):
    transform = transforms.Compose([
        prep.extract_crop_melspectrogram(mode=mode),
        prep.ToTensor()
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform)
    valid_source_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_source'], transform=transform)
    valid_target_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_target'], transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=config['param']['shuffle'],
        #num_workers=2,
        #pin_memory=True
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        #num_workers=2,
        #pin_memory=True
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        #num_workers=2,
        #pin_memory=True
        )

    dataloaders_dict = {"train": train_loader, "valid_source": valid_source_loader, "valid_target": valid_target_loader}
    
    return dataloaders_dict

#############################################################################
# training
#############################################################################
def calc_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
    #logger.info("AUC : {}".format(auc))
    #logger.info("pAUC : {}".format(p_auc))
    return auc, p_auc

def make_subseq(X, hop_mode=False):
    
    n_mels = config['param']['mel_bins']
    n_crop_frames = config['param']['n_crop_frames']
    n_hop_frames = config['param']['extract_hop_len']
    total_frames = len(X.shape[3]) - n_crop_frames + 1
    subseq = []
    # generate feature vectors by concatenating multiframes
    for frame_idx in range(total_frames):
        subseq.append(X[:,:,frame_idx:(frame_idx+1)*n_crop_frames])
    subseq = torch.cat(subseq, dim=0)
    # reduce sample
    if hop_mode:
        vectors = subseq[:,:,:: n_hop_frames]
    
    return vectors

def extract_net(net, dataloaders_dict, phases=['train', 'valid_source', 'valid_target']):
    outputs = []
    def hook(module, input, output):
        #print(output.shape)
        output = output.cpu()
        outputs.append(output.mean(dim=(2,3)))
    # M1 ~ M9
    net.effnet.blocks[0][0].register_forward_hook(hook)
    net.effnet.blocks[1][0].register_forward_hook(hook)
    net.effnet.blocks[2][0].register_forward_hook(hook)
    net.effnet.blocks[3][0].register_forward_hook(hook)
    net.effnet.blocks[4][0].register_forward_hook(hook)
    net.effnet.blocks[5][0].register_forward_hook(hook)
    net.effnet.blocks[6][0].register_forward_hook(hook)
    net.effnet.blocks[6][1].act1.register_forward_hook(hook)
    net.effnet.blocks[6][1].register_forward_hook(hook)
    #for i in range(len(net.layer2)):
    #net.layer2[-1].register_forward_hook(hook)
    #for i in range(len(net.layer3)):
    #net.layer3[-1].register_forward_hook(hook)
    #for i in range(len(net.layer4)):
    #net.layer4[-1].register_forward_hook(hook)
    #net.layer1[-1].register_forward_hook(hook)
    #net.layer1[-1].register_forward_hook(hook)
    #net.layer2[-1].register_forward_hook(hook)
    #net.layer3[-1].register_forward_hook(hook)
    #net.layer4[-1].register_forward_hook(hook)
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    
    output_dicts = {}
    
    #for phase in ['train', 'valid_source', 'valid_target']:
    for phase in phases:
        net.eval()
        M_means = []
        labels = []
        wav_names = []
        #n_crop_frames = config['param']['n_crop_frames']
        #extract_hop_len = config['param']['extract_hop_len']
        for sample in tqdm(dataloaders_dict[phase]):
            wav_name = sample['wav_name']
            wav_names = wav_names + wav_name
            
            print(sample['feature'].shape)
            input = sample['feature'].to(device)
            # print(input.shape)
            # plt.imshow(input[0,0,:,:].to('cpu'), aspect='auto')
            # plt.show()
            #for i in range(input.size()[0]):
            #    per_input = input[i,:,:]
            #x = input[:,:,:,:n_crop_frames-1].to(device)
            #x = input.to(device)
            #input = input.to(device)
            label = sample['label'].to('cpu')
            labels.append(label)

            with torch.no_grad():
                _ = net(input)  # (batch_size,input(2D)) 
                outputs = torch.cat(outputs, dim=1).cpu()
                M_means.append(outputs)
                outputs = []
                
        M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        output_dicts[phase] = {'features' : M_means, 'labels' : labels, 'wav_names' : wav_names}
    
    return output_dicts

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_net(net, dataloaders_dict, writer, optimizer, lambd=0.0051):
    aug = Augment()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    criterion = nn.MSELoss()
    output_dicts = {}
    n_epochs = config['param']['num_epochs']
    n_crop_frames = config['param']['n_crop_frames']
    for epoch in range(n_epochs):
        for phase in ['train']:
            net.train()
            labels = []
            wav_names = []
            losses = 0
            for sample in tqdm(dataloaders_dict[phase]):
                wav_name = sample['wav_name']
                wav_names = wav_names + wav_name
                input = sample['feature']   # (batch, ch, mel_bins, n_frames)
                
                N = input.size(0)
                # augmentation
                a = aug.do_augment(input)
                b = aug.do_augment(input)
                # plt.imshow(a[0,0,:,:], aspect='auto')
                # plt.show()
                # plt.imshow(b[0,0,:,:], aspect='auto')
                # plt.show()
                # f(x)
                a = a.to(device)
                a = net(a)
                a = a.to('cpu')
                b = b.to(device)
                b = net(b)
                b = b.to('cpu')
                # norm
                a = (a - a.mean(0)) / a.std(0) # NxD
                b = (b - b.mean(0)) / b.std(0) # NxD
                # cross-correlation matrix
                c = torch.mm(a.T, b) / N # DxD
                # multiply off-diagonal elems of c_diff by lambda
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                loss = on_diag + lambd * off_diag
                
                label = sample['label'].to('cpu')
                labels.append(label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
        # processing per epoch
        logger.info(f'epoch:{epoch+1}/{n_epochs}, tr_loss:{losses:.6f}')
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        writer.add_scalar("tr_loss", losses, epoch+1)
    # end
    output_dicts = {'net' : net}
    
    return output_dicts

