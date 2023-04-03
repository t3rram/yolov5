import argparse
import csv
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from openpyxl import workbook  # pip install openpyxl
from openpyxl import load_workbook
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
from torchvision import transforms
import time
from commons.json_tools import get_data_from_json
from constants import CLASS_FOLDER
#from segmentation.predict import predict
from segmentation.predict_onnx import predict_onnx, predict
from yolo.constants import CONFIG_FOLDER, RUNS_FOLDER
from hubconf import custom  


def get_class_names(yaml_file):
    with open(yaml_file, errors="ignore") as f:
        data = yaml.safe_load(f)  # dictionary
    return data["names"]


def get_jpg_files(files):
    jpg_files = [file for file in files if file.lower().endswith(".jpg") or file.lower().endswith(".png")]
    return jpg_files
@st.cache_resource(show_spinner=False)
def get_seg_model(device):
    model = torch.load(f="segmentation/unet16_model.pt", map_location=device )
    return model
def detect_count_objects(_results, images, model, model_type, conf_thres=[], yaml_file="", data_path="", detection_type="weight", k_neighbors=10, field=1, device="cpu"):
    class_names = get_class_names(yaml_file)
    dict_detect = {class_names[class_detect]: {"quantity": 0, "area": 0} for class_detect in range(len(class_names))}
    img_crv = []
    areas_crv = []
    hgts = []
    wdts = []
    crv_knn_weight = 0
    for result, img in zip(_results.pandas().xyxy, images):
        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        img_w, img_h = img.size
        for i in range(len(result)):

            class_detect = result.loc[i, "class"]
            confidence = result.loc[i, "confidence"]
            xmin = result.loc[i, "xmin"]
            ymin = result.loc[i, "ymin"]
            xmax = result.loc[i, "xmax"]
            ymax = result.loc[i, "ymax"]
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            area = float(width) * float(height)

            if confidence >= conf_thres[class_detect]:
                dict_detect[class_names[class_detect]]["quantity"] += 1

                if class_names[class_detect] == "conserve_ronde":
                    if detection_type == "area" or detection_type == "knn":
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        cropped_img = cropped_img.resize((256, 256))
                        # cropped_img.save("img_"+str(i)+".jpg")
                        # i+=1
                        img_crv.append(cropped_img)
                        areas_crv.append(area)
                        if detection_type == "knn":
                            hgts.append(max(height, width))
                            wdts.append(min(width, height))
                        # area_crv=predict(model_path= "segmentation/unet16_model.pt", images=img_crv, areas=areas_crv)
                    else:
                        dict_detect[class_names[class_detect]]["area"] += area

                else:
                    dict_detect[class_names[class_detect]]["area"] += area
    if detection_type == "area" or detection_type == "knn":
        if model_type == 'onnx':
            areas, heights, widths = predict_onnx(model_path="segmentation/unet16_model.onnx", images=img_crv, areas=areas_crv, hgts=hgts, wdts=wdts, device=device)
        else :
            areas, heights, widths = predict(model=model, images=img_crv, areas=areas_crv, hgts=hgts, wdts=wdts, device=device)
        dict_detect["conserve_ronde"]["area"] = np.sum(areas)
        if detection_type == "knn":
            crv_knn_weight = detect_weight_knn(data_path, areas, heights, widths, k_neighbors, field)
    return dict_detect, crv_knn_weight


def detect_weight_knn(data_path, areas, heights, widths, k_neighbors=10, field=1):
    # acp_data = get_data_from_json(acp_data_path)
    # excel_data = pd.read_excel(data_path, sheet_name=0)
    excel_data = pd.read_csv(data_path)
    data = pd.DataFrame(excel_data, columns=["poids", "h", "w", "S_seg_autre"])
    data["rapp_h_w"] = data["h"] / data["w"]
    data["S_seg_autre"] = data["S_seg_autre"] / 1000000
    weights = data.poids.to_numpy()
    X = np.array([data.rapp_h_w.to_numpy(), data.S_seg_autre.to_numpy()]).reshape(-1, 2)
    neigh = KNeighborsRegressor(n_neighbors=int(k_neighbors))
    neigh.fit(X, weights)
    data2predic = list()

    areas_n = (field * areas - data["S_seg_autre"].mean()) / data["S_seg_autre"].std()
    rapp_h_l = heights / widths

    rapp_h_l_n = (rapp_h_l - data["rapp_h_w"].mean()) / data["rapp_h_w"].std()
    data2predic = [[axe1, axe2] for axe1, axe2 in zip(rapp_h_l_n, areas_n)]

    weights_hat = neigh.predict(data2predic)
    total_weight = np.sum(weights_hat)
    return total_weight


def detection_to_weight(count_detect, flow):
    avg_weight_data = get_data_from_json(os.path.join(CLASS_FOLDER, flow, "average_weight.json"))

    weight_detect = {
        avg_weight_data[_class]["name"]: {
            "weight(g)": int(detection["quantity"] * avg_weight_data[_class]["average weight(g)"]),
            "quantity": detection["quantity"],
        }
        for _class, detection in count_detect.items()
    }
    return weight_detect


def detection_to_area(count_detect, flow):
    avg_area_data = get_data_from_json(os.path.join(CLASS_FOLDER, flow, "average_weight.json"))
    area_detect = {
        avg_area_data[_class]["name"]: {
            "weight(norm)": detection["area"] * avg_area_data[_class]["average weight per area(g/cm2)"],
            "area": detection["area"],
        }
        for _class, detection in count_detect.items()
    }
    return area_detect


def compare_detect_to_manual(classes, weight_detect, area_detect, image_size_cm=100, crv_knn_weight=0):
    coeff_norm_to_m = image_size_cm * 10000
    compare_list = [
        {
            "name": name,
            "detect_qty": weight_detect.get(name)["quantity"] if weight_detect.get(name) else None,
            "detect_weight(g)": weight_detect.get(name)["weight(g)"] if weight_detect.get(name) else None,
            "detect_weight(g)_area": int(area_detect.get(name)["weight(norm)"] * coeff_norm_to_m)
            if area_detect.get(name)
            else None,
            "detect_weight(g)_knn": crv_knn_weight if name == "conserve_ronde" else 0,
        }
        for name in classes
    ]
    return compare_list

@st.cache_resource(show_spinner=False)
def get_yolo_model(flow, exp,model_type, device):
    weight_pathfile = os.path.join(RUNS_FOLDER, flow, "train", exp, "weights", "best." + model_type)
    model = custom(path=weight_pathfile, autoshape=True, _verbose=True, device=device)
    #model = torch.hub.load('ultralytics/yolov5', 'custom', weight_pathfile, device=device)
    return model

@st.cache_data
def get_detection(flow, images, _model_yolo, _model_seg, model_type, detection_type, img_size, device, conf_thres, config_file, k_neighbors, field):
    yaml_file = os.path.join(CONFIG_FOLDER, flow, config_file)
    data_path = os.path.join(CLASS_FOLDER, flow, "db.csv")
    classes = get_class_names(yaml_file)
    
    read_images = [Image.open(img) for img in images]
    # tensor_images = torch.stack(read_images).to(device)
    # Inference
    results = _model_yolo(read_images, size=img_size)
    
    count_detect = detect_count_objects(results, read_images, _model_seg, model_type, conf_thres, yaml_file, data_path, detection_type, k_neighbors, field, device)

    area_detect = detection_to_area(count_detect[0], flow)
    weight_detect = detection_to_weight(count_detect[0], flow)
    compare_list = compare_detect_to_manual(
        classes, weight_detect, area_detect, image_size_cm=field, crv_knn_weight=count_detect[1]
    )


    return compare_list


def generate_report(carac_results, save_dir, flow, detection_type):
    flow_template = 5 if flow=="acier" else 6
    emr_classes_file = os.path.join(CLASS_FOLDER, flow, "classes_emr.json")
    with open(emr_classes_file) as f:
            emr_classes = json.load(f)
    classes_file = os.path.join(CLASS_FOLDER, flow, "classes.txt")
    template_file = os.path.join(CLASS_FOLDER, flow, "template.xlsx")
    class2idx = dict()
    with open(classes_file, "r") as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            class2idx[line] = i
    wb = load_workbook(template_file)
    sheets = wb.sheetnames
    worksheet = wb[sheets[0]]
    #poids_tot = st.number_input("Insert total weight (g)")
    #worksheet.cell(row=1, column=2).value = poids_tot
    carac_weights = dict()
    for k, v in emr_classes.items():
        carac_weights[v] = st.number_input("Insert " + k + " weight (g)")
    for i in range(1, len(carac_results)+1):
        cls = str(worksheet.cell(row=i + 4, column=1).value)
        poids = carac_results[i-1]["detect_weight(g)"]
        if detection_type == "knn" and cls == "conserve_ronde":
            poids = carac_results[class2idx[cls]]["detect_weight(g)_knn"]
        worksheet.cell(row=i + 4, column=2).value = poids
    for i in range(1, len(emr_classes.keys())+1):
        worksheet.cell(row=i + 4, column=flow_template).value = carac_weights[i - 1]
    wb.save(os.path.join(save_dir, "report.xlsx"))
    with open(os.path.join(save_dir, "report.xlsx"), "rb") as f:
        st.download_button("Download report", f, file_name="report.xlsx")


def parse_args():
    parser = argparse.ArgumentParser("compare results of manual caracterisation with deep learning caracterisation")
    parser.add_argument(
        "yaml_file",
        type=str,
        default="",
        help="Args",
    )

    args = parser.parse_args()
    return args


def main():
    st.title("Caracterization using YOLO")
    # opt = parse_args()
    yaml_file = "streamlit_conf.yaml"
    with open(yaml_file, errors="ignore") as f:
        data = yaml.safe_load(f)  # dictionary
    config_file = data["config_file"]
    img_size = data["image_size"]
    save_dir = data["save_dir"]
    flow = st.selectbox(
    'Flow ?',
    ('acier', 'alu'))
    select_device = st.selectbox(
    'Device ?',
    ('cuda:0', 'cpu'))
    if flow=="acier":
        exp = st.selectbox(
        'experience ?',
        ('2022_11_02_exp', 'test_evolve_exp'))
        k_neighbors = st.number_input(
        'Number of neighbors for knn ?',min_value = 1, max_value = 60)
    else :
        exp = 'test_evolve_exp'
        k_neighbors=None
    f1_tresh_file = os.path.join(RUNS_FOLDER, flow, "train", exp, "f1_tresh.yaml")
    with open(f1_tresh_file, errors="ignore") as f:
        data = yaml.safe_load(f)
    conf_thres = data["seuils"]
    device = torch.device(select_device if torch.cuda.is_available() else "cpu")
    model_type = 'pt' if select_device == "cuda:0" else 'onnx'
    field = 2 if flow=="acier" else 1
    detection_type = "knn" if flow=="acier" else "weight"
    data_source = st.radio(
    "Data source ? ",
    ('Upload', 'Local'))
    if data_source == 'Upload':
        images = st.file_uploader("Import images", type=["png", "jpg", "JPG"], accept_multiple_files=True)
    else :
        images_path = os.path.join(CLASS_FOLDER, flow, "streamlit_input")
        images = [os.path.join(images_path, img) for img in get_jpg_files(os.listdir(images_path))]
    #get_detection.clear()
    if len(images):
        
        data_load_state = st.text("Detecting...")
        model_yolo=get_yolo_model(flow, exp, model_type, device)
        if model_type=='onnx':
            model_seg=None
        else :
            model_seg=get_seg_model(device)
        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()
        starttime = time.time()
        compare_list = get_detection(
            flow, images, model_yolo, model_seg, model_type, detection_type, img_size, device, conf_thres, config_file, k_neighbors, field
        )
        #ender.record()
        #torch.cuda.synchronize()
        #curr_time = starter.elapsed_time(ender)
        curr_time = time.time() - starttime
        st.text('Detection time : '+ str(curr_time) + ' s' )
        data_load_state.text("Detecting...done!")
        st.header("Caracterization results : ")
        st.table(compare_list)
        generate_report(compare_list, save_dir, flow, detection_type)


def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if __name__ == "__main__":
    if check_password():
        main()
