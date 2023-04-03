import os

images_path = "taco_dataset/images2/"
masks_path = "taco_dataset/masks2/"
images = os.listdir(images_path)
masks = os.listdir(masks_path)
for i in range(len(images)):
    image_file = images[i]
    mask_file = masks[i]
    dst_image = f"taco_dataset/train_val/images/{image_file}"
    src_image = images_path + image_file
    dst_mask = f"taco_dataset/train_val/masks/{mask_file}"
    src_mask = masks_path + mask_file
    os.rename(src_image, dst_image)
    os.rename(src_mask, dst_mask)
