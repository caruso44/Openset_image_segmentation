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
)



def main():
    model = UNET(in_channel=4,out_channel=8).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optmizier = optmim.Adam(model.parameters(),lr= LEARNING_RATE) 
    dl = create_dataloader(PATCHES_PATH)
    model = train_fn(optmizier, model, loss_fn, dl)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()