from dataloader import Satelite_images
from general import(
    DEVICE,
    EPOCHS
)
from tqdm import tqdm
import torch.nn.functional as F
import torch

def create_dataloader(path_to_patches):
    dl = Satelite_images(path_to_patches)
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
            print(f'EPOCH {epoch}:\n running loss = {running_loss}')

    return model

