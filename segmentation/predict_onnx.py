import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as onnxrt
import streamlit as st
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def predict_onnx(model_path, images, areas, hgts, wdts, device):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device==torch.device("cuda:0") else ['CPUExecutionProvider']
    onnx_session= onnxrt.InferenceSession(model_path, providers=providers)
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )

    obj_areas = []
    heights = []
    widths = []
    for s in range(0, len(images), 16):
        x = []
        area_t = []
        hgts_t = []
        wdts_t = []
        for i in range(16):
            if s + i < len(images):
                x.append(images[s + i])
                area_t.append(areas[i + s])
                if len(wdts):
                    hgts_t.append(hgts[i + s])
                    wdts_t.append(wdts[i + s])
        with torch.no_grad():
            images_ = [img_transform(img) for img in x]
            images_ = torch.stack(images_).to(device)
        onnx_inputs= {onnx_session.get_inputs()[0].name: to_numpy(images_)}
        outputs = onnx_session.run(None, onnx_inputs)
        #outputs = model(images_)
        #outputs = outputs.cpu().detach().numpy()
        #st.header(outputs[0])
        masks_array = (outputs[0] > 0).astype(np.uint8)
        # print(masks_array.shape)
        St = [masks_array.shape[2] * masks_array.shape[3] for i in range(masks_array.shape[0])]
        # print(St)
        Sb = [np.sum(np.squeeze(masks_array[i, ...])) for i in range(masks_array.shape[0])]
        # Sb = [np.squeeze(masks_array[i,...])[np.squeeze(masks_array[i,...])==1].shape[0] for i in range(masks_array.shape[0])]
        # print(Sb)
        rapp = np.array([Sbi / Sti for Sbi, Sti in zip(Sb, St)])
        batch_areas = rapp * np.array(area_t)
        obj_areas.extend(batch_areas.tolist())

        if len(wdts):
            for m in range(masks_array.shape[0]):
                mask = np.squeeze(masks_array[m, ...])
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

                w = np.linalg.norm(box[0] - box[1])
                h = np.linalg.norm(box[1] - box[2])
                heights.append(hgts_t[m] * max([w, h]) / masks_array.shape[2])
                widths.append(wdts_t[m] * min([w, h]) / masks_array.shape[3])
                # box = np.int0(box)
        # print(box)
        # cv2.drawContours(thresh,[box],0,128,2)
        # cv2.imwrite(os.path.join(outputs_dir,"mask_"+str(m+s)+".jpg"), thresh)
        # torch.cuda.empty_cache()
        # mask_arrays[file_name]=mask_array
        # plt.imsave("output_seg/mask_"+str(s)+".jpg", np.squeeze(masks_array[0,...]))
        # x[0].save("output_seg/img_"+str(s)+".jpg")
        # plt.imsave("output_seg/img_"+str(s)+".jpg", x[0])
        # print(rapp[0])

    return np.array(obj_areas), np.array(heights), np.array(widths)

def predict(model, images, areas, hgts, wdts, device):
    model.eval()
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )

    obj_areas = []
    heights = []
    widths = []
    for s in range(0, len(images), 16):
        x = []
        area_t = []
        hgts_t = []
        wdts_t = []
        for i in range(16):
            if s + i < len(images):
                x.append(images[s + i])
                area_t.append(areas[i + s])
                if len(wdts):
                    hgts_t.append(hgts[i + s])
                    wdts_t.append(wdts[i + s])
        with torch.no_grad():
            images_ = [img_transform(img) for img in x]
            images_ = torch.stack(images_).to(device)

        outputs = model(images_)
        outputs = outputs.cpu().detach().numpy()

        masks_array = (outputs > 0).astype(np.uint8)
        # print(masks_array.shape)
        St = [masks_array.shape[2] * masks_array.shape[3] for i in range(masks_array.shape[0])]
        # print(St)
        Sb = [np.sum(np.squeeze(masks_array[i, ...])) for i in range(masks_array.shape[0])]
        # Sb = [np.squeeze(masks_array[i,...])[np.squeeze(masks_array[i,...])==1].shape[0] for i in range(masks_array.shape[0])]
        # print(Sb)
        rapp = np.array([Sbi / Sti for Sbi, Sti in zip(Sb, St)])
        batch_areas = rapp * np.array(area_t)
        obj_areas.extend(batch_areas.tolist())

        if len(wdts):
            for m in range(masks_array.shape[0]):
                mask = np.squeeze(masks_array[m, ...])
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

                w = np.linalg.norm(box[0] - box[1])
                h = np.linalg.norm(box[1] - box[2])
                heights.append(hgts_t[m] * max([w, h]) / masks_array.shape[2])
                widths.append(wdts_t[m] * min([w, h]) / masks_array.shape[3])
                # box = np.int0(box)
        # print(box)
        # cv2.drawContours(thresh,[box],0,128,2)
        # cv2.imwrite(os.path.join(outputs_dir,"mask_"+str(m+s)+".jpg"), thresh)
        # torch.cuda.empty_cache()
        # mask_arrays[file_name]=mask_array
        # plt.imsave("output_seg/mask_"+str(s)+".jpg", np.squeeze(masks_array[0,...]))
        # x[0].save("output_seg/img_"+str(s)+".jpg")
        # plt.imsave("output_seg/img_"+str(s)+".jpg", x[0])
        # print(rapp[0])

    return np.array(obj_areas), np.array(heights), np.array(widths)