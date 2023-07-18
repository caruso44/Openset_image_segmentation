from train import main
from utils.general import DEVICE
from utils.neural import UNET
import numpy as np


if __name__ == "__main__":
    model = UNET(in_channel=4,out_channel=7, feat=True, features=[14,28,56,112]).to(DEVICE)
    #main(True, model) 