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
import numpy as np
from tqdm import tqdm
from general import BATCH_SIZE
from skimage.util import view_as_windows
import matplotlib.pyplot as plt


def validate(model, loss_fn, dl, epoch):
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
                distribution = model(image)
                predictions = F.softmax(distribution, dim = 1)
                loss = loss_fn(predictions,mask)
                
                #################bachwards#################

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


def get_confusion_matrix(model, dl):
    confusion_matrix = np.zeros((7,7))
    precision = np.zeros(7)
    recall = np.zeros(7)
    f1_score = np.zeros(7)
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
                            
                            
def create_dataloader(path_to_patches, endpoint):
    dl = Satelite_images(path_to_patches, endpoint)
    index = list(range(len(dl)))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=BATCH_SIZE, sampler= index)
    return train_loader, dl.getweight()


def one_hot(targets):    
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 8, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot


def train_fn(optmizier, model, loss_fn, dl, dl_val):
    last_loss = 0
    k = 0
    for epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dl)) as pbar:
            for image, mask in dl:
                image = image.float().to(DEVICE)
                mask = mask.to(DEVICE)
                ##################fowards####################

                predictions = model(image)
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
            validate(model, loss_fn, dl_val, epoch)
        if last_loss - running_loss < 1e-3 and epoch > 0:
            if optmizier.param_groups[0]['lr'] > 1e-9:
                optmizier.param_groups[0]['lr'] *= 0.5
                print("learning rate changed")
            else:
                return model
        last_loss = running_loss
        k += 1

    return model




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