############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime

# general analysis tool-kit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.display import display

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_model import FC_block
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
        prep.extract_waveform(),
        prep.ToTensor()
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform)
    dev_test_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['dev_test'], transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )
    
    dev_test_loader = torch.utils.data.DataLoader(
        dataset=dev_test_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )

    dataloaders_dict = {"train": train_loader, "dev_test": dev_test_loader}
    
    return dataloaders_dict

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition.
    ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        #print(x.shape)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size,
                                                                  self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        #print('distmat', distmat.shape)
        #print(x.shape, self.centers.t().shape)
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #print(distmat.shape)
        dist = distmat * mask.float()
        #print(dist.clamp(min=1e-12, max=1e+12).shape)

        prob = dist.clamp(min=1e-12, max=1e+12)
        loss = prob.mean(dim=1)

        return loss


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
def train_net(net, dataloaders_dict, optimizer, num_epochs, writer, model_out_path, score_out_path, pred_out_path, num_classes):
    outputs = []
    def hook(module, input, output):
        #print(output.shape)
        output = output.cpu()
        outputs.append(output.mean(dim=(2,3)))
    
    for i in range(len(net.resnet.layer1)):
        net.resnet.layer1[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer2)):
        net.resnet.layer2[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer3)):
        net.resnet.layer3[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer4)):
        net.resnet.layer4[i].register_forward_hook(hook)
    center_loss = CenterLoss(num_classes=1, feat_dim=3776)
    out_fc = FC_block(3776, 3776)
    
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    out_fc.to(device)
    output_dicts = {}
    best_criterion = 0
    for epoch in range(num_epochs):
        for phase in ['train', 'dev_test']:
            losses = 0
            if phase == 'train':
                net.train()
            else:
                net.eval()
            preds = []
            labels = []
            wav_names = []
            secs = []
            for sample in tqdm(dataloaders_dict[phase]):
                wav_name = sample['wav_name']
                wav_names = wav_names + wav_name
                
                input = sample['feature']
                input = input.to(device)
                label = sample['label'].to('cpu')
                labels.append(label)
                sec = sample['sec']
                sec = sec.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    _ = net(input)  # (batch_size,input(2D))
                    outputs = torch.cat(outputs, dim=1)
                    fake_sec = torch.zeros_like(sec)
                    outputs = out_fc(outputs.to(device))
                    pred = center_loss(outputs.to(device), fake_sec)
                    outputs = []
                    preds.append(pred.to('cpu'))
                    secs.append(sec.to('cpu'))
                    loss = pred.mean()
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    losses += loss.item()
                # outputs = torch.cat(outputs, dim=1).cpu()
                # M_means.append(outputs)
                # outputs = []
            losses = losses / len(dataloaders_dict[phase])
            if phase == 'train':
                tr_loss = losses
            else:
                val_loss = losses
                preds = torch.cat(preds, dim=0).detach().numpy().copy()
                labels = torch.cat(labels, dim=0).detach().numpy().copy()
                secs = torch.cat(secs, dim=0).detach().numpy().copy()
                # calc score
                all_scores_df = pd.DataFrame()
                all_scores_df = com.calc_DCASE2020_score(all_scores_df, labels, preds, secs, wav_names)
                pred_df = com.get_pred_discribe(labels, preds, secs, wav_names)
                
                val_AUC_hmean = all_scores_df.loc['h_mean']['AUC']
                val_pAUC_hmean = all_scores_df.loc['h_mean']['pAUC']
                epoch_log = (
                            f"epoch:{epoch+1}/{num_epochs},"
                            f" tr_loss:{tr_loss:.6f},"
                            f" val_loss:{val_loss:.6f},"
                            f" val_AUC_hmean:{val_AUC_hmean:.6f},"
                            f" val_pAUC_hmean:{val_pAUC_hmean:.6f},"
                            )
                logger.info(epoch_log)
                # early stopping
                if best_criterion < val_pAUC_hmean:
                    best_score = all_scores_df.copy()
                    best_tr_losses = tr_loss
                    best_criterion = val_pAUC_hmean.copy()
                    best_pred = pred_df.copy()
                    best_epoch = epoch
                    best_model = net
                    best_flag = True
                    # save
                    torch.save(best_model.state_dict(), model_out_path)
                    best_score.to_csv(score_out_path)
                    best_pred.to_csv(pred_out_path)
                    # display score dataframe
                    display(best_score)
                    #logger.info("Save best model")
                    # logger info
            # display best score
    best_log = (
                f"best model,"
                f" epoch:{best_epoch+1}/{num_epochs},"
                f" train_losses:{best_tr_losses:.6f},"
                f" val_pAUC_hmean:{best_criterion:.6f},"
                )
    logger.info(best_log)
    display(best_score)      
    output_dicts = {'best_epoch':best_epoch, 'best_pAUC':best_criterion, 'best_pred':best_pred}

    return output_dicts