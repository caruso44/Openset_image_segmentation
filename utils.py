from dataloader import Satelite_images
from general import(
    DEVICE,
    EPOCHS,
    VALID_SIZE,
    BATCH_SIZE,
    DISCARDED_CLASS,
    PATCH_SIZE,
    PATCH_OVERLAP
)
from tqdm import tqdm
import torch.nn.functional as F
import torch
from openmax import fit_high
import numpy as np
from tqdm import tqdm
from general import BATCH_SIZE
from scipy.stats import weibull_min
import torch.optim as optmim
from skimage.util import view_as_windows

def validate(model, loss_fn, optmizier, dl, epoch):
    with tqdm(total=len(dl)) as pbar:
        val_loss = 0
        correct = 0
        total = 0
        for image, mask in dl:
            with torch.no_grad():
                image = image.float().to(DEVICE)
                mask = mask.to(DEVICE)
                ##################fowards####################

                distribution = model(image)
                predictions = F.softmax(distribution, dim = 1)
                loss = loss_fn(predictions,mask)
                
                #################bachwards#################

                val_loss += loss.item()
                pbar.update(1)
                
                predictions = torch.argmax(predictions, dim = 1)
                mask = mask.to("cpu")
                predictions = predictions.to("cpu")
                for i in range(64):
                    for j in range(64):
                        arr1 = mask[:,i,j].numpy()
                        arr2 = predictions[:,i,j].numpy()
                        n7_arr1 = arr1 != 7
                        correct += np.sum(np.equal(arr2[n7_arr1], arr1[n7_arr1]))
                        total += arr1[n7_arr1].shape[0]


        
        print(f'\nEPOCH {epoch}:\n validation loss = {val_loss/len(dl)}')
        print(f'\nEPOCH {epoch}:\n validation precision = {correct/total}')


def create_dataloader(path_to_patches, PATCHES_VAL_PATH, endpoint):
    dl = Satelite_images(path_to_patches, PATCHES_VAL_PATH, endpoint)
    index = list(range(len(dl)))
    np.random.shuffle(index)
    split = int(np.floor(VALID_SIZE * len(dl)))
    train_index, val_index = index[split:], index[:split]
    train_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= train_index)
    val_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= val_index)
    return train_loader, val_loader

def one_hot(targets):    
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 8, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot

def train_fn(optmizier, model, loss_fn, dl, dl_val):
    last_loss = 0
    for epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                image = image.float().to(DEVICE)
                mask = mask.to(DEVICE)
                ##################fowards####################

                predictions = model(image)
                predictions = F.softmax(predictions, dim = 1)
                one_hot_mask = one_hot(mask)
                one_hot_mask = one_hot_mask.long().to(DEVICE)
                loss = loss_fn(predictions,mask)
                
                #################bachwards#################

                loss.backward()
                optmizier.step()
                optmizier.zero_grad()
                running_loss += loss.item()
                pbar.update(1)
            print(f'\nEPOCH {epoch}:\n running loss = {running_loss/len(dl)}')
            validate(model, loss_fn, optmizier, dl_val, epoch)
        if last_loss - running_loss < 1e-3 and epoch > 0:
            if optmizier.param_groups[0]['lr'] > 1e-9:
                optmizier.param_groups[0]['lr'] *= 0.5
                print("learning rate changed")
            else:
                return model
        last_loss = running_loss

    return model


def fit(model, dl):
    amount = np.zeros(7)
    mean = np.zeros((7,7))
    with tqdm(total=len(dl)) as pbar:
        idx = 0
        for image, label in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                predictions = predictions.numpy()
                for i in range(64):
                    for j in range(64):
                        if label[i][j] < 7 and np.argmax(predictions[:,i,j]) == label[i][j]:
                            mean[label[i][j]] += output[:,i,j].numpy()
                            amount[label[i][j]] += 1
                        
            pbar.update(1)
            if idx == 100:
                break
            idx += 1
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
    print(predict_confidence)
    if(predict_confidence[ans] > th_conficence[ans]):
        return ans
    return 7



def create_test_patches():
    test_label = np.load("D:/Caruso/code/OpenMax-main/prepared/label_test.npy")
    shape = test_label.shape
    idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)
    train_step = int((1-PATCH_OVERLAP)*PATCH_SIZE)
    label_patches = view_as_windows(test_label, (PATCH_SIZE, PATCH_SIZE), train_step).reshape((-1, PATCH_SIZE, PATCH_SIZE))
    idx_patches = view_as_windows(idx_matrix, (PATCH_SIZE, PATCH_SIZE), train_step).reshape((-1, PATCH_SIZE, PATCH_SIZE))
    keep_patches = np.mean(np.logical_and((label_patches != 0), (label_patches != DISCARDED_CLASS)), axis=(1,2)) >= 0.02
    idx_patches = idx_patches[keep_patches]
    np.save('test_patches.npy', idx_patches)