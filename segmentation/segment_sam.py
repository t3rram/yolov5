import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as onnxrt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image

def segment(img, boxs, device):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    image = np.array(img)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_boxes = torch.tensor([boxs], device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
    )
    masks = masks.cpu().numpy()
    areas = masks.sum(axis=(2,3))
    #areas = np.array([np.sum([masks[i,...] for i in range(masks.shape[0])])])

    heights = []
    widths = []
    if True:
        for m in range(masks.shape[0]):
            mask = np.squeeze(masks[m, ...])
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
            #w = np.linalg.norm(box[0] - box[1])
            #h = np.linalg.norm(box[1] - box[2])
            heights.append(max([w, h]))
            widths.append(min([w, h]))
            # cv2.drawContours(thresh,[box.astype(int)],0,128,2)
            # cv2.imwrite(os.path.join("output_seg","mask__"+str(m)+".jpg"), thresh)
        # torch.cuda.empty_cache()
        # mask_arrays[file_name]=mask_array
        # for i in range(0,masks.shape[0],10):
        #     xmin = boxs[i][0]
        #     ymin = boxs[i][1]
        #     xmax = boxs[i][2]
        #     ymax = boxs[i][3]           
        #     cropped_img = img.crop((xmin, ymin, xmax, ymax))
        #     cropped_img = cropped_img.resize((256, 256))
        #     plt.imsave("output_seg/mask_"+str(i)+".jpg", np.squeeze(masks[i,...]))
        #     cropped_img.save("output_seg/img_"+str(i)+".jpg")
            #plt.imsave("output_seg/img_"+str(i)+".jpg", cropped_img)
        # print(rapp[0])

    return np.squeeze(areas)/(img.size[0]*img.size[0]), np.array(heights)/img.size[0], np.array(widths)/img.size[0]


# if __name__=="__main__":
#     label_file = "sevran_14_03_23_8.txt"
#     image_file = "sevran_14_03_23_8.jpg"
#     img = Image.open(image_file)
#     img_w = img.size[0]
#     img_h = img.size[1]
#     boxs=[]
#     with open(label_file, "r") as f:
#         lines = f.readlines()
#         for line in lines :
#             line_s = line.split()
#             box_yolo = [float(line_s[1]), float(line_s[2]), float(line_s[3]), float(line_s[4])]
#             box = [img_w*(box_yolo[0]- box_yolo[2]/2), img_w*(box_yolo[1]-box_yolo[3]/2),img_w*(box_yolo[0] + box_yolo[2]/2),img_w*(box_yolo[1] + box_yolo[3]/2)]
#             boxs.append(box)
    
#     device=torch.device("cuda:0")
#     a, h, w= segment(img, boxs, device)
#     print(a)
#     print(h*w)
#     print(a-h*w)
"""from segment_anything.utils.transforms import ResizeLongestSide


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()
def batch_segment(images, results, confidence, device):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.to(device=device)
    batched_input = []
    for result, img in zip(results.pandas().xyxy, images):
        boxs = []
        img_w, img_h = img.size
        result_crv = result[(result[["confidence"]] > confidence) & (result[["class"]] == 2)]
        for i in range(len(result_crv)):
            xmin = result_crv.loc[i, "xmin"]
            ymin = result_crv.loc[i, "ymin"]
            xmax = result_crv.loc[i, "xmax"]
            ymax = result_crv.loc[i, "ymax"]
            boxs.append([xmin,ymin,xmax,ymax])
        dict_input = dict()
        dict_input["image"] = prepare_image(np.array(img), resize_transform, sam)
        dict_input["boxes"] = resize_transform.apply_boxes_torch(torch.tensor(boxs, device=device), img.size)
        dict_input['original_size']= img.size
        batched_input.append(dict_input)
    batched_output = sam(batched_input, multimask_output=False)
    torch.cuda.empty_cache()
    areas_all_imgs = np.array([])
    heights_all_imgs = []
    widths_all_imgs = []
    for l, output in enumerate(batched_output):
        masks = output["masks"]
        masks = masks.cpu().numpy()
        areas = masks.sum(axis=(2,3))/(batched_input[l]["original_size"][0]**2)
        areas_all_imgs = np.concatenate((areas_all_imgs,np.squeeze(areas)))
    #areas = np.array([np.sum([masks[i,...] for i in range(masks.shape[0])])])

        heights = []
        widths = []
        if True:
            for m in range(masks.shape[0]):
                mask = np.squeeze(masks[m, ...])
                mask = np.array(mask,np.uint8)
                ret, thresh = cv2.threshold(255 * mask, 127, 255, 0)
                cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                area = 0
                area_i = 0
                for c, cnt in enumerate(cnts):
                    area_t = cv2.contourArea(cnt)
                    if area_t > area:
                        area = area_t
                        area_i = c
                rect = cv2.minAreaRect(cnts[area_i])

                box = cv2.boxPoints(rect)
                w,h = rect[1]
                #w = np.linalg.norm(box[0] - box[1])
                #h = np.linalg.norm(box[1] - box[2])
                heights.append(max([w, h])/batched_input[l]["original_size"][0])
                widths.append(min([w, h])/batched_input[l]["original_size"][0])
        heights_all_imgs.extend(heights)
        widths_all_imgs.extend(widths)
            # cv2.drawContours(thresh,[box.astype(int)],0,128,2)
            # cv2.imwrite(os.path.join("output_seg","mask__"+str(m)+".jpg"), thresh)
        # torch.cuda.empty_cache()
        # mask_arrays[file_name]=mask_array
        # for i in range(0,masks.shape[0],10):
        #     xmin = boxs[i][0]
        #     ymin = boxs[i][1]
        #     xmax = boxs[i][2]
        #     ymax = boxs[i][3]           
        #     cropped_img = img.crop((xmin, ymin, xmax, ymax))
        #     cropped_img = cropped_img.resize((256, 256))
        #     plt.imsave("output_seg/mask_"+str(i)+".jpg", np.squeeze(masks[i,...]))
        #     cropped_img.save("output_seg/img_"+str(i)+".jpg")
            #plt.imsave("output_seg/img_"+str(i)+".jpg", cropped_img)
        # print(rapp[0])    

    return np.squeeze(areas), np.array(heights), np.array(widths)"""
