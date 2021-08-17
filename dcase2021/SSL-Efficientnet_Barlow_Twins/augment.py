import numpy as np
import torch

from torchlibrosa.augmentation import SpecAugmentation

class Augment(object):
    def __init__(self):
        self.spec_augmenter = SpecAugmentation(time_drop_width=8,
                                            time_stripes_num=2,
                                            freq_drop_width=8,
                                            freq_stripes_num=2)
    
    def mixup(self, data, alpha=1, debug=False):
        data = data.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        weights = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        x = np.array([x1[i] * weights[i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
        #y1 = np.array(one_hot_labels).astype(np.float)
        #y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
        #y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
        if debug:
            print('Mixup weights', weights)
        return torch.from_numpy(x).clone()
    
    def do_augment(self, x):
        x = self.mixup(x)
        x = self.spec_augmenter(x)
        
        return x
    