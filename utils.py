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
from general import BATCH_SIZE
from scipy.stats import weibull_min


def create_dataloader(path_to_patches, endpoint):
    dl = Satelite_images(path_to_patches, endpoint)
    close_idx, _ = dl.get_close_set_index(7)
    train_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler=close_idx)
    return train_loader



def train_fn(optmizier, model, loss_fn, dl):
    for epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                image = image.float().to(DEVICE)

                ##################fowards####################

                predictions = model(image)
                predictions = F.softmax(predictions, dim = 1)
                one_hot_mask = torch.zeros(mask.size()[0], 7, 64 ,64)
                for k in range(mask.size()[0]):
                    for i in range(64):
                        for j in range(64):
                            one_hot_mask[k][mask[k][i][j]][i][j] = 1
                one_hot_mask = one_hot_mask.to(DEVICE)
                loss = loss_fn(predictions,one_hot_mask)
                
                #################bachwards#################

                loss.backward()
                optmizier.step()
                optmizier.zero_grad()
                running_loss += loss.item()
                pbar.update(1)
            print(f'\nEPOCH {epoch}:\n running loss = {running_loss/len(dl)}')

    return model


def fit(model, dl, close_idx):
    amount = np.zeros(7)
    mean = np.zeros((7,7))

    with tqdm(total=len(close_idx)) as pbar:
        for idx in close_idx:
            image, label = dl[idx]
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                image = image.to(DEVICE)
                output = model(image)
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze().to("cpu")
                predictions = predictions.squeeze().to("cpu")
                predictions = predictions.numpy()
                for i in range(64):
                    for j in range(64):
                        if label[i][j] < 7 and np.argmax(predictions[:,i,j]) == label[i][j]:
                            mean[label[i][j]] += output[:,i,j].numpy()
                            amount[label[i][j]] += 1
                        
            pbar.update(1)
    for i in range(7):
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
    ans = np.argmax(predict_confidence)
    if(predict_confidence[ans] > th_conficence[ans]):
        return ans
    return 7

