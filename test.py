import torch
from general import(
    PATCHES_VAL_PATH,
    DEVICE,
)
from tqdm import tqdm
import numpy as np
from dataloader import Satelite_images
import torch.nn.functional as F
from utils import euclidean_distance
import scipy.stats as stats
from evt import weibull_tailfitting
from openmax import openmax



def test():
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
                    
                    
                    


if __name__ == "__main__":
    mean = np.load("mean.npy")
    model = torch.load("open_set_model.pth").to(DEVICE)
    distance = np.load("distances.npy")
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    weibull = weibull_tailfitting(mean, distance, 7, 20)
    confusion_matrix = np.zeros((8,8))
    precision = np.zeros(8)
    recall = np.zeros(8)
    f1_score = np.zeros(8)
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            image = image.unsqueeze(0).to(DEVICE)
            output = model(image)
            prediction = F.softmax(output, dim = 1)
            prediction = prediction.squeeze(0).to("cpu")
            output = output.squeeze(0).to("cpu")
            for i in range(64):
                for j in range(64):
                    prob = openmax(weibull, output[:,i,j].detach().numpy(), prediction[:,i,j].detach().numpy())
                    pred = prob.index(max(prob))
                    
                    confusion_matrix[pred, label[i,j]] += 1
            pbar.update(1)

        
        for i in range(8):
            precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
            recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
            f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])
            print(f"A precisão para a classe {i} é {precision[i]}")  
            print(f"A recall para a classe {i} é {recall[i]}")  
            print(f"O F1 Score para a classe {i} é {f1_score[i]}")
    ########### get index map ###########
    '''
    path = 'D:/Caruso/code/OpenMax-main/prepared/map.data'
    with open(path, 'rb') as file:
        data = file.read()
        print(pickle.loads(data))
    '''
    
