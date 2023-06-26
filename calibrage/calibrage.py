import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import streamlit as st
import yaml
from ..yolo.constants import CONFIG_FOLDER, RUNS_FOLDER
from hubconf import custom  

@st.cache_resource(show_spinner=False)
def get_yolo_model(flow, exp,model_type, device):
    weight_pathfile = os.path.join(RUNS_FOLDER, flow, "train", exp, "weights", "best." + model_type)
    model = custom(path=weight_pathfile, autoshape=True, _verbose=True, device=device)
    #model = torch.hub.load('ultralytics/yolov5', 'custom', weight_pathfile, device=device)
    return model

@st.cache_data
def calibrate(image, _model_yolo, img_size, object_size, object_area, device):
    read_image = Image.open(image)
    # Inference
    detection = _model_yolo([read_image], size=img_size)
    result = detection.pandas().xyxy[0]
    object = result.iloc[result['confidence'].argmax()]
    class_detect = result.loc[0, "class"]
    confidence = result.loc[0, "confidence"]
    xmin = result.loc[0, "xmin"]
    ymin = result.loc[0, "ymin"]
    xmax = result.loc[0, "xmax"]
    ymax = result.loc[0, "ymax"]
    # segmeent object
    box = [xmin, ymin ,xmax, ymax]
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    image_array = np.array(read_image)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_array)
    input_boxes = torch.tensor(box, device=device)
    transformed_box = predictor.transform.apply_boxes_torch(input_boxes, image_array.shape[:2])
    masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_box,
    multimask_output=False,
    )
    masks = masks.cpu().numpy()
    object_area = masks.sum(axis=(2,3))[0]
    mask = np.squeeze(masks[0,...])
    mask = np.array(mask,np.uint8)
    ret, thresh = cv2.threshold(255 * mask, 127, 255, 0)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    area_i = 0
    for i, cnt in enumerate(cnts):
        area_t = cv2.contourArea(cnt)
        if area_t > area:
            area = area_t
            area_i = i
    rect = cv2.minAreaRect(cnts[area_i])

    box = cv2.boxPoints(rect)
    w,h = rect[1]
    height = max([w, h])
    width = min([w, h])
    pixel_per_cm_h = height/object_size[0]
    pixel_per_cm_w = width/object_size[1]
    pixel_per_cm_area = object_area/object_area
    return pixel_per_cm_h, pixel_per_cm_w, pixel_per_cm_area

def main():
    # Titre
    st.title("Calibrationn app")
    # Charger le fichier config
    yaml_file = "streamlit_conf.yaml"
    with open(yaml_file, errors="ignore") as f:
        data = yaml.safe_load(f)  # dictionary
    config_file = data["config_file"]
    img_size = data["image_size"]
    save_dir = data["save_dir"]
    select_device = st.selectbox(
    'Device ?',
    ('cuda:0', 'cpu'))
    device = torch.device(select_device if torch.cuda.is_available() else "cpu")
    model_yolo=get_yolo_model("acier", "test_evolve_exp", 'pt', device)
    image = st.file_uploader("Import image", type=["png", "jpg", "JPG"], accept_multiple_files=False)
    object_height = st.number_input(
        'Object size (cm) ?',min_value = 1, max_value = 100)
    object_width = st.number_input(
        'Object size (cm) ?',min_value = 1, max_value = 100)
    object_area = st.number_input(
        'Object size (cm) ?',min_value = 1, max_value = 10000)
    object_size = [object_height, object_width]
    pixel_per_cm_h, pixel_per_cm_w, pixel_per_cm_area = calibrate(
        image, model_yolo, img_size, object_size, object_area, device
        )
    st.text('Pixel per cm (height) : '+ str(pixel_per_cm_h) )
    st.text('Pixel per cm (width) : '+ str(pixel_per_cm_w) )
    st.text('Pixel per cm (area) : '+ str(pixel_per_cm_area) )

if __name__ == "__main__":
    main()