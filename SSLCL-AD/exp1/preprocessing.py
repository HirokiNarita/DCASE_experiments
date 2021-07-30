########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import re
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

class extract_waveform(object):
    """
    wavデータロード(波形)
    
    Attributes
    ----------
    sound_data : waveform
    """
    def __init__(self, sound_data=None):
        self.sound_data = sound_data
    
    def __call__(self, sample):
        self.sound_data = com.file_load(sample['wav_name'],
                                        sr=16000,
                                        mono=True)
        self.sound_data = self.sound_data[0]
        self.label = np.array(sample['label'])
        self.wav_name = sample['wav_name']
        self.sec = self.get_section_type(sample['wav_name'])
        return {'feature': self.sound_data, 'label': self.label, 'wav_name': self.wav_name, 'sec': self.sec}
    
    def get_section_type(self, wav_name):
        str_id = re.search(r'id_0\d+', wav_name).group()
        id = int(str_id[4])
        return np.array(id)
    
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        feature, label, wav_name, sec = sample['feature'], sample['label'], sample['wav_name'], sample['sec']
        return {'feature': torch.from_numpy(feature).float(), 'label': torch.from_numpy(label), 'wav_name': wav_name, 'sec':torch.from_numpy(sec).long()}

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
