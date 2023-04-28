import torch
from dataloader import Satelite_images
import numpy as np
import torch.nn as nn
from neural import UNET
import torch.optim as optmim
from utils import create_dataloader, train_fn
from general import(
    DEVICE,
    LEARNING_RATE,
    PATCHES_PATH,
    PATCHES_VAL_PATH
)



def main():
    model = UNET(in_channel=4,out_channel=7).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=7)
    optmizier = optmim.Adam(model.parameters(),lr= LEARNING_RATE, weight_decay = 5e-6) 
    endpoint = "_train.npy"
    dl, dl_val = create_dataloader(PATCHES_PATH, endpoint, PATCHES_VAL_PATH)                
    model = train_fn(optmizier, model, loss_fn, dl, dl_val)
    torch.save(model, 'open_set_model.pth')


if __name__ == "__main__":
    main()