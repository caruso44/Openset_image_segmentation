from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import ToTensor
import general
import general

class Satelite_images(Dataset):
    def __init__(self, path_to_patches, transformer = ToTensor()) -> None:
        opt_img = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))

        #self.labels = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).reshape((-1,1)).astype(np.int64)
        self.labels = np.load(os.path.join(general.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).flatten().astype(np.int64)
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