import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from segmentation.predict import predect


def rapport_surface(mask):
    St = mask.shape[0] * mask.shape[1]
    Sb = mask[mask == 1].shape[0]
    return Sb / St


# Fonction pour générer les rapport de surfaces r_seg
def generate_r_seg(model_path, image_path, save_path, save_csv_path, save=False):
    mask_array = predect(model_path, image_path, save_path, save)
    # db_r_seg=[]
    # for mask_array in list(mask_arrays.values()):
    r_seg = rapport_surface(mask_array)
    # db_r_seg.append(r_seg)
    #    k+=1
    # db_path=list(mask_array.keys())
    # data=pd.DataFrame(db_path, columns=["image"])
    # data["r_seg"]=db_r_seg
    return r_seg
    # data.to_csv(save_csv_path,index=False)


model_path = "unet16_model.pt"
images_path = "Oneplus9Pro_1m2_cropped_compressed"
save_path = ".."
save_csv_path = "data_rapport_surface.csv"
save = False
generate_r_seg(model_path, images_path, save_path, save_csv_path, save=save)
