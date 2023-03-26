import torch
from utils import create_dataloader, train_fn, fit, choose_class
from general import(
    VAL_PATH,
    PATCHES_PATH
)
from tqdm import tqdm
import pickle
import numpy as np

def main():
    dl = create_dataloader(VAL_PATH)                
    model = torch.load('model.pth')
    weibull = fit(model, dl)
    for image, label in dl:
        for i in range(64):
            for j in range(64):
                output = choose_class(weibull, dl[i][j][:,0,0],[1,1,1,1,1,1,1,1],8)


if __name__ == "__main__":
    '''
    path = 'C:/Users/jpcar/OneDrive/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    '''

        
