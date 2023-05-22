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
    PATCHES_VAL_PATH,
    MODEL_PATH
)



def main():
    model = UNET(in_channel=4,out_channel=7).to(DEVICE)
    optmizier = optmim.Adam(model.parameters(),lr= LEARNING_RATE, weight_decay = 5e-6) 
    endpoint = "_train.npy"
    dl, weights = create_dataloader(PATCHES_PATH, endpoint)   
    weights = weights.to(DEVICE)
    dl_val = Satelite_images(PATCHES_VAL_PATH, "_train.npy")  
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=7, reduction= 'mean')
    model = train_fn(optmizier, model, loss_fn, dl, dl_val)
    torch.save(model, 'open_set_model.pth')


if __name__ == "__main__":
    main()