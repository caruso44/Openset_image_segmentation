from train import main
from utils.general import DEVICE
from  utils.neural import UNET

if __name__ == "__main__":
    model = UNET(in_channel=4,out_channel=7, feat=True).to(DEVICE)
    main(False, model)