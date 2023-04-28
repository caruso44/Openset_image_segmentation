import torch
from utils import create_dataloader, train_fn, fit, choose_class, create_test_patches
from general import(
    PATCHES_TEST_PATH,
    PATCHES_PATH,
    PATCHES_VAL_PATH,
    DEVICE,
    MODEL_PATH
)
from tqdm import tqdm
import pickle
import numpy as np
from dataloader import Satelite_images
import torch.nn.functional as F

def main():
    dl = Satelite_images(PATCHES_TEST_PATH, "_test.npy")      
    model = torch.load("open_set_model.pth")
    total = 0
    correct = 0
    k = 0
    for image, label in dl:
        print(k)
        image = image.unsqueeze(0)
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        predictions = F.softmax(output, dim = 1)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim = 0)
        for i in range(64):
            for j in range(64):
                if label[i][j] != 7:
                    total += 1
                    if predictions[i][j] == label[i][j]:
                        correct += 1
        k += 1
        if k == 200:
            break
    print(correct/total)

   
def create_openset_classifier():
    dl = Satelite_images(PATCHES_TEST_PATH, "_test.npy") 
    dl2 = Satelite_images(PATCHES_VAL_PATH, "_train.npy")  
    model = torch.load(MODEL_PATH)
    model = model.to(DEVICE)
    weibull = fit(model, dl2)
    image = dl[32][0]
    image = image.unsqueeze(0).to(DEVICE)
    label = model(image)
    label = F.softmax(label, dim = 0)
    label = label.squeeze(0).to("cpu").detach().numpy()
    pixel = label[:,42,42]
    print(choose_class(weibull, pixel, [0.5,0.5,0.5,0.5,0.5,0.5,0.5], 7))


if __name__ == "__main__":
    create_openset_classifier()
    ########### get index map ###########
    '''
    path = 'D:/Caruso/code/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    '''
    
