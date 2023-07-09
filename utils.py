from dataloader import Satelite_images
from general import(
    DEVICE,
    IMAGE_SIZE,
    LEN_VECTOR,
    BATCH_SIZE,
    DISCARDED_CLASS,
    PATCH_SIZE,
    PATCH_OVERLAP
)
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from general import BATCH_SIZE
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
                     
                            
def create_dataloader(path_to_patches, endpoint):
    dl = Satelite_images(path_to_patches, endpoint)
    index = list(range(len(dl)))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= index)
    return train_loader, dl.getweight()


def one_hot(targets):    
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) 
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 8, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def add_to_vector(num, vec):
    if len(vec) < LEN_VECTOR:
        vec.append(num)
    else:
        min_idx = vec.index(min(vec))
        if num > vec[min_idx]:
            vec[min_idx] = num
            
    return vec

    

def get_distances(dl, mean, model):
    collection = [[] for _ in range(7)]
    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu").numpy()
                predictions = predictions.squeeze(0).to("cpu").numpy()
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        if np.argmax(predictions[:,i,j]) == mask[i][j] and mask[i,j] < 7:
                            centroid = mean[mask[i,j]]
                            distance = euclidean_distance(output[:,i,j], centroid)
                            collection[mask[i,j]].append(distance)
            pbar.update(1)
    np.save("distances.npy", np.array(collection))
    


def fit():
    param = []
    collection = np.load("distances.npy")
    for label in range(7):
        distances = np.array(collection[label])
        shape, loc, scale = weibull_min.fit(distances)
        param.append([shape, loc, scale])
    param = np.array(param)
    np.save("weibull2.npy", param)

def get_mean(dl, model):
    amount = np.zeros(7)
    mean = np.zeros((7,7))
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                predictions = predictions.numpy()
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        if label[i][j] < 7 and np.argmax(predictions[:,i,j]) == label[i][j]:
                            mean[label[i][j]] += output[:,i,j].numpy()
                            amount[label[i][j]] += 1
            pbar.update(1)
    for i in range(7):
        mean[i] = mean[i]/amount[i]
    np.save("mean.npy", mean)
    return mean







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