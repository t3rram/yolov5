# Import libraries
import copy
import os
import random
import shutil
from collections import defaultdict
from urllib.request import urlretrieve

import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import ternausnet.models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

cudnn.benchmark = True
# Split files from the dataset into the train and validation sets
dataset_directory = os.path.join("./", "taco_dataset/train_val")
root_directory = os.path.join(dataset_directory)
images_directory = os.path.join(root_directory, "images")
masks_directory = os.path.join(root_directory, "masks")

images_filenames = list(sorted(os.listdir(images_directory)))
correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(images_directory, i)) is not None]

random.seed(42)
random.shuffle(correct_images_filenames)

train_images_filenames = correct_images_filenames[:1300]
val_images_filenames = correct_images_filenames[1300:]

# test_images_filenames = images_filenames[-10:]

# print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))


# Preprocess the mask
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask = mask.max(axis=2)
    mask /= 255
    mask[mask == 0] = 0.0
    mask[mask == 1] = 1.0
    return mask


# Visualize images
def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(masks_directory, image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()


# display_image_grid(test_images_filenames, images_directory, masks_directory)
# Define a pytorch dataset class
class TacoDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
train_dataset = TacoDataset(
    train_images_filenames,
    images_directory,
    masks_directory,
    transform=train_transform,
)

val_transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
val_dataset = TacoDataset(
    val_images_filenames,
    images_directory,
    masks_directory,
    transform=val_transform,
)


def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


random.seed(42)
visualize_augmentations(train_dataset, idx=55)


# Define a class to track accuracy and loss during training and validation
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=1e-6):
    labels = labels.byte()
    outputs = outputs.byte()
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return iou.max()


# Define train function
def train(train_loader, model, criterion, optimizer, epoch, params, info_file):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    num_iters = len(stream)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        iou = iou_pytorch(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}. IOU         {iou}".format(
                epoch=epoch, metric_monitor=metric_monitor, iou=iou
            )
        )
        texte = f"Epoch {epoch} iter {i}/{num_iters} {metric_monitor} IOU {iou}"
        info_file.write(texte)
        info_file.write("\n")


# Define validation function
def validate(val_loader, model, criterion, epoch, params, info_file, val_loss, PATH):
    metric_monitor = MetricMonitor()
    model.eval()

    stream = tqdm(val_loader)
    num_iters = len(stream)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            iou = iou_pytorch(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}.IOU {iou}".format(
                    epoch=epoch, metric_monitor=metric_monitor, iou=iou
                )
            )
            texte = f"Epoch {epoch} iter {i}/{num_iters} {metric_monitor} IOU {iou}"
            info_file.write(texte)
            info_file.write("\n")
            new_val_loss = metric_monitor.metrics["Loss"]["avg"]
            if new_val_loss <= val_loss:
                # print("saving the model.........")
                torch.save(model, PATH)
            val_loss = new_val_loss
    return val_loss


# Define a function to create the unet16 model
def create_model(params):
    model = getattr(ternausnet.models, params["model"])(pretrained=True)
    model = model.to(params["device"])
    return model


# Define a function to train and validate the model
def train_and_validate(
    model, train_dataset, val_dataset, params, info_file, val_loss=0.999, PATH="unet16_best_model_8.pt"
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params, info_file)
        val_loss = validate(val_loader, model, criterion, epoch, params, info_file, val_loss, PATH)

    return model


# Define a function to make prediction via the trained model
def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.5).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))
    return predictions


# Set the training parameters
params = {
    "model": "UNet16",
    "device": "cuda",
    "lr": 0.00008,
    "batch_size": 16,
    "num_workers": 4,
    "epochs": 100,
}

# Train and validate the model
info_file = open("infos_unet16_8.txt", "w")
model = create_model(params)
model = train_and_validate(model, train_dataset, val_dataset, params, info_file)
info_file.close()
