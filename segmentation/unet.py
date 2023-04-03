# Importer les libraries :
import os
import warnings

warnings.filterwarnings("ignore")


# Définir les chemins de données:
DATA_DIR = "../taco_dataset"

x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "train_labels")

x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "val_labels")

x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "test_labels")

class_names = ["background", "object"]
Cclass_rgb_values = [[0, 0, 0], [255, 255, 255]]
