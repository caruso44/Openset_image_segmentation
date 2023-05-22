import torch
from utils import create_dataloader, train_fn, create_test_patches, get_confusion_matrix
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from openPCS import fit, get_mean, euclidean_distance, fit_libmr, fit_optimized
import scipy.stats as stats


def main():
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    model = torch.load("open_set_model.pth")
    model = model.to(DEVICE)
    param = np.load("weibull2.npy")
    mean = np.load("mean.npy")
    print(param)
    confusion_matrix = np.zeros((8,8))
    precision = np.zeros(8)
    recall = np.zeros(8)
    f1_score = np.zeros(8)
    idx = 0
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                prediction = F.softmax(output, dim = 1)
                prediction = prediction.squeeze(0).to("cpu")
                output = output.squeeze(0).to("cpu")
                label = label.to("cpu")
                for i in range(64):
                    for j in range(64):
                        pred_label = np.argmax(prediction[:,i,j])
                        distance = euclidean_distance(output[:,i,j].numpy(), mean[pred_label])
                        ans = stats.weibull_min(param[pred_label,0], loc = param[pred_label,1], scale = param[pred_label,2]).cdf(distance)
                        l = label[i,j].numpy()
                        if ans < 0.1:
                            confusion_matrix[l,7] += 1
                        else:
                            confusion_matrix[l,pred_label] += 1
            pbar.update(1)
            if idx == 100:
                break
            idx += 1
    for i in range(8):
        precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])
        print(f"A precisão para a classe {i} é {precision[i]}")  
        print(f"A recall para a classe {i} é {recall[i]}")  
        print(f"O F1 Score para a classe {i} é {f1_score[i]}")
                    
                    
                    
def check_model():
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    model = torch.load("open_set_model.pth")
    model = model.to(DEVICE)
    get_confusion_matrix(model, dl)
    

if __name__ == "__main__":
    mean = np.load("mean.npy")
    fit_libmr(mean)
    ########### get index map ###########
    '''
    path = 'D:/Caruso/code/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    '''
    
