from dataloader import Satelite_images
from general import(
    DEVICE,
    EPOCHS
)
from tqdm import tqdm
import torch.nn.functional as F
import torch
from openmax import fit_high
import numpy as np
from tqdm import tqdm


def create_dataloader(path_to_patches, endpoint):
    dl = Satelite_images(path_to_patches, endpoint)
    return dl



def train_fn(optmizier, model, loss_fn, dl):
    for epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                image = image.float().unsqueeze(0).to(DEVICE)

                ##################fowards####################

                predictions = model(image)
                predictions = predictions.squeeze()
                predictions = F.softmax(predictions, dim = 0)
                one_hot_mask = torch.zeros(8, 64 ,64)
                for i in range(64):
                    for j in range(64):
                        one_hot_mask[mask[i][j]][i][j] = 1
                one_hot_mask = one_hot_mask.to(DEVICE)
                loss = loss_fn(predictions,one_hot_mask)
                
                #################bachwards#################

                loss.backward()
                optmizier.step()
                optmizier.zero_grad()
                running_loss += loss.item()
                pbar.update(1)
            print(f'EPOCH {epoch}:\n running loss = {running_loss/len(dl)}')

    return model


def fit(model, dl):
    amount = np.zeros(8)
    mean = np.zeros((8,8))
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                image = image.to(DEVICE)
                output = model(image)
                output = output.squeeze().to("cpu")
                for i in range(64):
                    for j in range(64):
                        mean[label[i][j]] += output[:,i,j].numpy()
                        amount[label[i][j]] += 1
            pbar.update(1)
        for i in range(8):
            mean[i] = mean[i]/amount[i]
    mean = torch.from_numpy(mean)
    weibull = fit_high(mean, 1, 20)
    return weibull


def choose_class(weibull, pixel, th_conficence, n):
    scale = weibull.wbFits[:, 1]
    shape = weibull.wbFits[:, 0]
    predict_confidence = np.zeros(n)
    for i in range(n):
        predict_confidence[i] = (shape[i]/scale[i]) * (np.linalg.norm(pixel/scale[i]) ** (shape[i] - 1)) * np.exp(-np.linalg.norm(pixel/scale[i]) ** (shape[i]))
        print(np.linalg.norm(pixel/scale[i]))
    ans = np.argmax(predict_confidence)
    if(predict_confidence[ans] > th_conficence[ans]):
        return ans + 1
    return 0

