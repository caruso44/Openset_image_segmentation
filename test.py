import torch
from utils import create_dataloader, train_fn, fit, choose_class
from general import(
    VAL_PATH,
    PATCHES_PATH,
    DEVICE,
    BATCH_SIZE
)
from tqdm import tqdm
import pickle
import numpy as np
from dataloader import Satelite_images

def main():
    dl = Satelite_images(PATCHES_PATH, "_train.npy")
    close_idx, open_idx = dl.get_close_set_index(7)           
    model = torch.load("open_set_model.pth")
    weibull = fit(model, dl, close_idx)
    print(weibull.wbFits[:, 0], weibull.wbFits[:, 1])
    correct = 0
    total = 0
    with tqdm(total=len(open_idx)) as pbar:
        for idx in open_idx:
            for i in range(64):
                for j in range(64):
                    output = choose_class(weibull, dl[idx][0][:,i,j],[1,1,1,1,1,1,1,1],7)
                    if output == dl[idx][1][i,j]:
                        correct += 1
                    total += 1
            pbar.update(1)
    print(correct/total)
                


if __name__ == "__main__":
    #main()
    dl = Satelite_images(PATCHES_PATH, "_test.npy")
    dl = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= dl.get_index_tensor())
    for image, mask in dl:
        image = image.to(DEVICE)
        model = torch.load("open_set_model.pth")
        model = model.to(DEVICE)
        output = model(image)
        output = output.to(DEVICE)
        output = torch.argmax(output, dim=1)
        output = output.to("cpu")
        print((output == mask).sum())
        break
    ########### get index map ###########
    '''
    path = 'C:/Users/jpcar/OneDrive/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    '''

        
