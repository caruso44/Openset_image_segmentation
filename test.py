from osgeo import gdal
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from utils.general import DEVICE, MAP_COLOR, COLOR_TO_RGB, NUM_KNOWN_CLASSES, PATCHES_TEST_PATH, PATCHES_VAL_PATH   
import torch.nn.functional as F
import matplotlib.pyplot as plt
from validate import get_weibull_model
from utils.openmax import recalibrate_scores
from tqdm import tqdm
from utils.dataloader import Satelite_images
from validate import print_confusion_matrix, check_model_closed

def read_image_gdal(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image = dataset.ReadAsArray()
    return image

PATH = 'C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/Urban/RGBNIR.tif'

def image_to_tensor(image):
    image = np.transpose(image, (1, 2, 0))  
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    tensor = transform(image)
    return tensor

def plot_image(image, original_image):
    colored_image = np.array([[MAP_COLOR[label] for label in row] for row in image])   
    rgb_image = np.array([[COLOR_TO_RGB[color] for color in row] for row in colored_image])
    original_image = original_image/256
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title('Image 1')

    axes[1].imshow(original_image)
    axes[1].set_title('Image 2')

    plt.tight_layout()
    plt.show()


def build_image():
    image = read_image_gdal(PATH)
    image_tensor = image_to_tensor(image)
    image_numpy = image_tensor.numpy().astype(np.int64)
    model = torch.load("open_set_model_openpca.pth").to(DEVICE)
    model.eval()
    _,lenght,width = image_tensor.size()
    output = torch.zeros((lenght,width))
    weibull_model = get_weibull_model()
    i = 0
    j = 0
    red = 0
    with torch.no_grad():
        with tqdm(total= lenght) as pbar:
            while(i + 64 <= lenght):
                j = 0
                while(j + 64 <= width):
                    tensor =  image_tensor[:,i:i+64,j:j+64]
                    tensor = tensor.float().unsqueeze(0).to(DEVICE)
                    out = model(tensor)
                    out = out[0]
                    out_soft = F.softmax(out, dim = 1)
                    out_soft = out_soft.squeeze(0).to("cpu")
                    out = out.squeeze(0).to("cpu")
                    probs = recalibrate_scores(
                    weibull_model, out, out_soft, NUM_KNOWN_CLASSES, NUM_KNOWN_CLASSES, 'eucos'
                )
                    for k in range(len(probs)):
                        row = int(k/64)
                        col = int(k % 64)
                        if probs[k, pred] < 0.9999999:
                            pred = np.argmax(probs[k])
                        else:
                            pred = 7
                        output[i+row, j+col] = pred
                    j += 64
                pbar.update(64)
                i+= 64
    output = output.numpy()
    output = output.astype(np.int64)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    plot_image(output, image_numpy[:,:,0:3])


def pixelwise_openset(th):
    model = torch.load("open_set_model_openpca.pth")
    model = model.to(DEVICE)
    model.eval()
    confusion_matrix = np.zeros((8, 8))
    dl = Satelite_images(PATCHES_TEST_PATH, "_test.npy") 
    with tqdm(total=len(dl)) as pbar:
        with torch.no_grad():
            for image, label in dl:
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                output = output[0]
                output_soft = F.softmax(output, dim = 1)
                predictions = torch.argmax(output_soft, dim = 1)
                label = label.squeeze(0).to("cpu")
                output_soft = output_soft.squeeze().to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                for i in range(64):
                        for j in range(64):
                            if output_soft[predictions[i][j], i, j] > th:
                                confusion_matrix[label[i][j]][predictions[i][j]] += 1
                            else:
                                confusion_matrix[label[i][j]][7] += 1
                pbar.update(1)
    print_confusion_matrix(confusion_matrix, 8)
    
build_image()


ths = [0.7, 0.9]
for th in  ths:
    pixelwise_openset(th)