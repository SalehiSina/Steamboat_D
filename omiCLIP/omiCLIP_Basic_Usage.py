#--------------------
##Import
#--------------------

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from PIL import Image
import cv2

import albumentations as trans
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm

import loki.utils
import loki.preprocess

#---------------------
##Parameters
#---------------------

model_dir = "/home/mosa505e/FM-venv/Loki"
model_path = os.path.join(model_dir, 'checkpoint.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_dir = "/data/horse/ws/mosa505e-xenium-thesis/Data/Mouse_Coronal_Patches"

#Basic Transformation
base_transform = trans.Compose([
    trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class Histo_images(Dataset):
    def __init__(self, patch_paths, transform=base_transform):
        self.transform = transform
        self.cell_ids = []
        self.paths = []

        for p in tqdm(patch_paths):
          self.paths.append(p)
          cell_id = p[-14:-4]
          self.cell_ids.append(cell_id)

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, index):
        p = self.paths[index]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.transform(image=img)['image']

        return self.cell_ids[index], img


def Embed(model, Data_loader, device):

    model.eval()  # set eval mode
    all_cell_ids = []
    embedded_vectors = []

    with torch.inference_mode():  # no_grad

        for cell_ids, frames in tqdm(Data_loader, desc='Inference'):

            # Forward pass
            outputs = model(frames.to(device))

            # Move back to CPU for storage
            outputs = outputs['image_features'].detach().cpu()

            # Store results
            all_cell_ids.extend(cell_ids)
            embedded_vectors.append(outputs)

    # Concatenate all embedded vectors → shape: (N, embed_dim)
    embedded_vectors = torch.cat(embedded_vectors, dim=0)

    return all_cell_ids, embedded_vectors

#---------------------
##Main
#---------------------
if __name__ == "__main__":
    print("Device: ", device)
    model, preprocess, tokenizer = loki.utils.load_model(model_path, device)

    img_list = os.listdir(HE_dir)
    patch_paths = [os.path.join(HE_dir, fn) for fn in img_list]
    Data_set = Histo_images(transform=base_transform, patch_paths=patch_paths)
    Data_loader = DataLoader(Data_set, batch_size=256, shuffle=False, drop_last=False)

    cell_ids, E_vectors = Embed(model, Data_loader, device)

    adata = anndata.AnnData(
    X = np.zeros((len(cell_ids), 1))
    )

    # Store embeddings
    adata.obs["cell_id"] = cell_ids
    adata.obsm["X_custom"] = E_vectors.numpy()

    # Save to file
    adata_path = "/data/horse/ws/mosa505e-xenium-thesis/Data/Mouse_Coronal_Embeddings/cells.h5ad"
    os.makedirs(os.path.dirname(adata_path), exist_ok=True)
    adata.write(adata_path)