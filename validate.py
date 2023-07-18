import numpy as np
import torch
from tqdm import tqdm
from utils.general import(
    DEVICE,
    PATCHES_VAL_PATH,
    NUM_KNOWN_CLASSES,
    PATCHES_TEST_PATH
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataloader import Satelite_images
from utils.utils import get_distances, get_mean
from utils.evt import weibull_tailfitting
from utils.openmax import recalibrate_scores
from utils.openpca import fit_pca_model, fit_quantiles

def print_confusion_matrix(confusion_matrix, size):
    precision = np.zeros(size)
    recall = np.zeros(size)
    f1_score = np.zeros(size)
    for i in range(size):
        precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        f1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

    print(f"A acuracia é {accuracy}")

    for i in range(size):
        print(f"A precisão para a classe {i} é {precision[i]}")  
        print(f"A recall para a classe {i} é {recall[i]}")  
        print(f"O F1 Score para a classe {i} é {f1_score[i]}")  

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.show()
    print(confusion_matrix)

def get_confusion_matrix(model, dl):
    model.eval()
    confusion_matrix = np.zeros((7,7))

    with tqdm(total=len(dl)) as pbar:
        for image, mask in dl:
            with torch.no_grad():
                image = image.float().unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)
                distribution = model(image)
                predictions = F.softmax(distribution, dim = 1)
                pbar.update(1)
                predictions = torch.argmax(predictions, dim = 1)
                mask = mask.squeeze(0).to("cpu")
                predictions = predictions.squeeze(0).to("cpu")
                for i in range(64):
                    for j in range(64):
                        if mask[i][j] != 7:
                            confusion_matrix[mask[i][j]][predictions[i][j]] += 1
    print_confusion_matrix(confusion_matrix, 7)

def check_model_closed():
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    model = torch.load("open_set_model_UNET.pth")
    model = model.to(DEVICE)
    get_confusion_matrix(model, dl)


def get_lists(net):
    val_loader = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    print("Calculando as medias")
    mean =  get_mean(val_loader, net)
    #mean = np.load("mean.npy")
    print("Calculando as distâncias")
    dist = get_distances(val_loader, mean, net)
    

def get_weibull_model():
    print("Iniciando a determinação do modelo weibull")
    dist_list = np.load("distances/distances_eucos.npy", allow_pickle= True)
    mean_list = np.load("means/mean_eucos.npy")
    weibull_model = weibull_tailfitting(mean_list, dist_list, NUM_KNOWN_CLASSES, tailsize=10)
    return weibull_model

def test(net, weibull_model):
    dl = Satelite_images(PATCHES_VAL_PATH, "_train.npy")
    net.eval() 
    confusion_matrix = np.zeros((8, 8))
    k = 0
    with tqdm(total=len(dl)) as pbar:
        for image, label in dl:
            label = label.numpy()
            label = label.reshape(label.shape[0] * label.shape[1])
            image = image.unsqueeze(0).to(DEVICE)
            output = net(image)
            output_soft = F.softmax(output, dim = 1)
            output_soft = output_soft.squeeze(0).to("cpu")
            output = output.squeeze(0).to("cpu")
            probs = recalibrate_scores(
                weibull_model, output, output_soft, NUM_KNOWN_CLASSES, NUM_KNOWN_CLASSES, 'eucos'
            )
            for i in range(len(label)):
                pred = np.argmax(probs[i])
                if pred != 7 and probs[i, pred] < 0.999999999:
                    confusion_matrix[label[i], 7] += 1
                else:
                    confusion_matrix[label[i], pred] += 1
            pbar.update(1)
    print_confusion_matrix(confusion_matrix, 8)

def validate_OpenFCN():
    model = torch.load("open_set_model_UNET.pth")
    model = model.to(DEVICE)
    #get_lists(model) # determinar e salvar a lista de distâncias e médias
    weibull_model = get_weibull_model() # determinar e salvar o modelo de weibull
    print("iniciando o teste")
    test(model, weibull_model) # testar o modelo em conjunto aberto


def validate_OpenPCA():
    model = torch.load("open_set_model_UNET.pth")
    model = model.to(DEVICE)
    t = torch.ones((1,4,64,64)).to(DEVICE)
    x = model(t)
    print(x[0].size(), x[1].size(),x[2].size())