import torch
from utils.general import(
    PATCHES_VAL_PATH,
    DEVICE,
)
from tqdm import tqdm
import numpy as np
from utils.dataloader import Satelite_images
import torch.nn.functional as F
import scipy.stats as stats
from utils.evt import weibull_tailfitting
from utils.openmax import openmax
import pickle
from utils.neural import UNET


if __name__ == "__main__":
    ########### get index map ###########

    path = 'C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    
    model = UNET(4,7)
    model = model.to(DEVICE)
    tensor = torch.ones((16,4,64,64))
    tensor = tensor.to(DEVICE)
    out = model(tensor)
    print(out.size())
    
