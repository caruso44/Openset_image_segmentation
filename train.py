import torch
from dataloader import Satelite_images
import numpy as np
import torch.nn as nn
from neural import UNET
import torch.optim as optmim
from utils import create_dataloader, train_fn, fit, choose_class
from general import(
    DEVICE,
    LEARNING_RATE,
    PATCHES_PATH,
)
from tqdm import tqdm


def main():
    model = UNET(in_channel=4,out_channel=8).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optmizier = optmim.Adam(model.parameters(),lr= LEARNING_RATE) 
    endpoint = "_train.npy"
    dl = create_dataloader(PATCHES_PATH, endpoint)                
    model = train_fn(optmizier, model, loss_fn, dl)
    torch.save(model, 'model.pth')


if __name__ == "__main__":
    main()