import torch
from utils import create_dataloader, train_fn, create_test_patches, get_confusion_matrix
from general import(
    PATCHES_TEST_PATH,
    PATCHES_PATH,
    PATCHES_VAL_PATH,
    DEVICE,
    MODEL_PATH,
    LEN_VECTOR,
    IMAGE_SIZE
)
import pickle
from dataloader import Satelite_images
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import weibull_min
from evt import weibull_tailfitting



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


def get_vector(dl, label):
    collection = []
    model = torch.load("open_set_model.pth")
    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                predictions = F.softmax(output, dim = 1)
                output = output.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                predictions = predictions.numpy()
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        if np.argmax(predictions[:,i,j]) == mask[i][j] and label == mask[i][j]:
                            collection =  add_to_vector(output[:,i,j], collection)
            pbar.update(1)
    return collection



def fit(dl, mean):
    param = []
    for label in range(7):
        collection = get_vector(dl, label)
        distances = []
        for tensor in collection:
            centroid = mean[label]
            array = tensor.numpy()
            distances.append(euclidean_distance(array, centroid))
        shape, loc, scale = weibull_min.fit(distances)
        print([shape, loc, scale])
        param.append([shape, loc, scale])
    param = np.array(param)
    np.save("weibull.npy", param)

    

def get_vector_optimized(dl, mean):
    collection = [[] for _ in range(7)]
    model = torch.load("open_set_model.pth")
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
                            collection[mask[i,j]] = add_to_vector(distance, collection[mask[i,j]])
            pbar.update(1)
    np.save("distances.npy", collection)
    


def fit_optimized():
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



def fit_libmr(mean):
    collection = np.load("distances.npy")
    weibull = weibull_tailfitting(mean, collection, 7)
    print(type(weibull[0]['weibull_model'][0].get_params()))
