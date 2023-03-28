from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import ToTensor
import general

class Satelite_images(Dataset):
    def __init__(self, path_to_patches, endpoint, transformer = ToTensor()) -> None:
        opt_img = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))
    
        #self.labels = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).reshape((-1,1)).astype(np.int64)
        self.labels = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_LABEL}' + endpoint)).flatten().astype(np.int64)
        self.n_classes = np.unique(self.labels).shape[0]
        self.patches = np.load(path_to_patches)#[:200]
        self.transformer = transformer
        
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch_idx = self.patches[index]
        opt_tensor = self.transformer(self.opt_img[patch_idx])
        #label_tensor = self.transformer(self.labels[patch_idx].astype(np.int64)).squeeze(0)
        label_tensor = torch.tensor(self.labels[patch_idx])
        return (
            opt_tensor,
            label_tensor
        )
    
    def get_close_set_index(self, number):
        n = self.__len__()
        close_set = []
        open_set = []
        for i in range(n):
            patch_idx = self.patches[i]
            label_tensor = torch.tensor(self.labels[patch_idx])
            label_np = label_tensor.numpy()
            label_np = np.unique(label_np)
            if label_np[-1] == number:
                open_set.append(i)
            else:
                close_set.append(i)
        return (
            np.array(close_set),
            np.array(open_set)
        )
    def get_index_tensor(self):
        n = self.__len__()
        index = []
        for i in range(n):
            index.append(i)
        index = torch.tensor(index)
        return index
