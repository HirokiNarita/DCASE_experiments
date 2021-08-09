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
def make_dataloader(train_paths, machine_type):
    transform = transforms.Compose([
        prep.extract_crop_melspectrogram(),
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

# training function
def extract_net(net, dataloaders_dict):
    outputs = []
    def hook(module, input, output):
        #print(output.shape)
        output = output.cpu()
        outputs.append(output.mean(dim=(2,3)))
    
    for i in range(len(net.blocks)):
        for j in range(len(net.blocks[i])):
            net.blocks[i][j].register_forward_hook(hook)
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
    
    for phase in ['train', 'valid_source', 'valid_target']:
        net.eval()
        M_means = []
        labels = []
        wav_names = []
        for sample in tqdm(dataloaders_dict[phase]):
            wav_name = sample['wav_name']
            wav_names = wav_names + wav_name
            
            input = sample['feature']
            print(input.shape)
            plt.imshow(input[0,0,:,:].to('cpu'), aspect='auto')
            plt.show()
            input = input.to(device)
            label = sample['label'].to('cpu')
            labels.append(label)

            with torch.no_grad():
                _ = net(input)  # (batch_size,input(2D)) 
                outputs = torch.cat(outputs, dim=1).cpu()
                M_means.append(outputs)
                print(outputs.shape)
                outputs = []
                #M_means.append(output_dict['M_means'].to('cpu'))
                
        M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        output_dicts[phase] = {'features' : M_means, 'labels' : labels, 'wav_names' : wav_names}
    
    return output_dicts

def train_net(net, dataloaders_dict, writer, optimizer):

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

                x = input[:,:,:,:n_crop_frames-1].to(device)
                act_y = input[:,0,:,n_crop_frames-1].to(device)
                # print(x.shape)
                # plt.imshow(x[0,0,:,:].to('cpu'), aspect='auto')
                # plt.show()
                
                label = sample['label'].to('cpu')
                labels.append(label)
                pred_y = net(x)
                loss = criterion(pred_y, act_y)
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
