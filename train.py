import torch
from utils.dataloader import Satelite_images
import numpy as np
import torch.nn as nn
from utils.neural import UNET
import torch.optim as optmim
from utils.utils import create_dataloader
from utils.general import(
    DEVICE,
    LEARNING_RATE,
    PATCHES_PATH,
    PATCHES_VAL_PATH,
    EPOCHS
)
from tqdm import tqdm
import torch.nn.functional as F
from models.fcn_densenet121 import FCNDenseNet121

def check_model(model, loss_fn, dl, epoch, feat = False):
    model.eval()
    with tqdm(total=len(dl)) as pbar:
        val_loss = 0
        confusion_matrix = np.zeros((7,7))
        precision = np.zeros(7)
        recall = np.zeros(7)
        f1_score = np.zeros(7)
        for image, mask in dl:
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)
                if feat == False:
                    distribution = model(image)
                else:
                    distribution,_,_ = model(image)
                predictions = F.softmax(distribution, dim = 1)
                loss = loss_fn(predictions,mask)
                

                val_loss += loss.item()
                pbar.update(1)
                
                predictions = torch.argmax(predictions, dim = 1)
                mask = mask.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                for i in range(64):
                    for j in range(64):
                        if mask[i][j] != 7:
                            confusion_matrix[mask[i][j]][predictions[i][j]] += 1

        print(f'\nEPOCH {epoch}:\n validation loss = {val_loss/len(dl)}')
        for i in range(7):
            precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
            recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
            f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])
    
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
        
        print(f"A acuracia é {accuracy}")

        for i in range(7):
            print(f"A precisão para a classe {i} é {precision[i]}")  
            print(f"A recall para a classe {i} é {recall[i]}")  
            print(f"O F1 Score para a classe {i} é {f1_score[i]}")


def train_fn(optmizier, model, loss_fn, dl, dl_val, feat):
    last_loss = 0
    k = 0
    for epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                image = image.float().to(DEVICE)
                mask = mask.to(DEVICE)
                ##################fowards####################
                if feat == False:
                    predictions = model(image)
                else:
                    predictions,_,_ = model(image)
                predictions = F.softmax(predictions, dim = 1)
                loss = loss_fn(predictions,mask)
                
                #################bachwards#################

                loss.backward()
                optmizier.step()
                optmizier.zero_grad()
                running_loss += loss.item()
                pbar.update(1)
                    
        print(f'\nEPOCH {epoch}:\n running loss = {running_loss/len(dl)}')
        if k % 10 == 0:
            check_model(model, loss_fn, dl_val, epoch, feat=feat)
        if last_loss - running_loss < 1e-3 and epoch > 0:
            if optmizier.param_groups[0]['lr'] > 1e-9:
                optmizier.param_groups[0]['lr'] *= 0.5
                print("learning rate changed")
            else:
                return model
        last_loss = running_loss
        k += 1
    check_model(model, loss_fn, dl_val, EPOCHS, feat=feat)
    return model


def main(feat, model):
    optmizier = optmim.Adam(model.parameters(),lr= LEARNING_RATE, weight_decay = 5e-6) 
    endpoint = "_train.npy"
    dl, weights = create_dataloader(PATCHES_PATH, endpoint)   
    weights = weights.to(DEVICE)
    dl_val = Satelite_images(PATCHES_VAL_PATH, "_train.npy")  
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=7, reduction= 'mean')
    model = train_fn(optmizier, model, loss_fn, dl, dl_val, feat)
    torch.save(model, 'open_set_model_UNET.pth')
