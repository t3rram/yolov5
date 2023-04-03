import os

MODULE_NAME = "yolo"

WEIGHT_FOLDER = os.path.join(MODULE_NAME, "weight")
OUTPUT_FOLDER = os.path.join(MODULE_NAME, "output")
CHECKPOINT_FOLDER = os.path.join(MODULE_NAME, "checkpoint")
DATA_FOLDER = os.path.join(MODULE_NAME, "data")
CONFIG_FOLDER = os.path.join(MODULE_NAME, "config")
OUTPUT_FOLDER = os.path.join(MODULE_NAME, "output")
RUNS_FOLDER = os.path.join(MODULE_NAME, "runs")

EMR_DETECTION_PREWEIGHT_FILE = "yolov5s.pt"
EMR_DETECTION_WEIGHT_FILE = "best_emr.pt"
EMR_DETECTION_CONFIG_FILE = "train_emr.yaml"
