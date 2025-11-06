#--------------------
##Import
#--------------------

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from PIL import Image
import torch

import loki.utils
import loki.preprocess

#---------------------
##Parameters
#---------------------

model_dir = "/home/mosa505e/FM-venv/Loki"
model_path = os.path.join(model_dir, 'checkpoint.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_dir = "/data/horse/ws/mosa505e-xenium-thesis/Data/H&E_Patch_Examples_1"

#---------------------
##Main
#---------------------
if __name__ == "__main__":
    model, preprocess, tokenizer = loki.utils.load_model(model_path, device)
    model.eval()
    image_path = os.path.join(HE_dir, 'XeniumH&E_patch_area_around_dimfnjlb-1_center_1_1.png')
    image_embeddings = loki.utils.encode_images(model, preprocess, [image_path], device)
    print(image_embeddings.shape)