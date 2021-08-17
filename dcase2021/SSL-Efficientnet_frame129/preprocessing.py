########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

def random_crop(X):
    n_crop_frames = config['param']['n_crop_frames']
    total_frames = X.shape[1]
    bgn_frame = torch.randint(low=0, high=total_frames - n_crop_frames, size=(1,))[0]
    X = X[:, bgn_frame: bgn_frame+n_crop_frames]
    return X

class extract_crop_melspectrogram(object):
    """
    データロード(波形)
    
    Attributes
    ----------
    sound_data : logmelspectrogram
    """
    def __init__(self, sound_data=None, mode='training'):
        self.sound_data = sound_data
        self.mode = mode
    
    def __call__(self, sample):

        sample_rate=config['param']['sample_rate']
        n_mels = config['param']['mel_bins']
        n_fft = config['param']['window_size']
        hop_length=config['param']['hop_size']
        power = 2.0
        
        #self.resampling = Resample(16000, sample_rate)
        #input = self.resampling(input)
        audio, sample_rate = librosa.load(sample['wav_name'],
                                          sr=config['param']['sample_rate'],
                                          mono=True)
        audio = torch.from_numpy(audio.astype(np.float32)).clone().cuda()
        mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        ).cuda()
        X = mel_spectrogram_transformer(audio)
        X = X.cpu()
        eps = 1e-16
        X = (
            20.0 / power * torch.log10(X + eps)
        )
        if self.mode == 'training':
            X = random_crop(X)
        X = torch.stack([X,X,X], dim=0)
        ############################
        self.sound_data = X
        self.label = np.array(sample['label'])
        self.wav_name = sample['wav_name']
        
        return {'feature': self.sound_data, 'label': self.label, 'wav_name': self.wav_name}
    
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        feature, label, wav_name = sample['feature'], sample['label'], sample['wav_name']
        return {'feature': feature, 'label': torch.from_numpy(label), 'wav_name': wav_name}



class DCASE_task2_Dataset(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None):
        self.transform = transform
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # ファイル名でlabelを判断
        if "normal" in file_path:
            label = 0
        else:
            label = 1
        
        sample = {'wav_name':file_path, 'label':np.array(label)}
        sample = self.transform(sample)
        
        return sample
