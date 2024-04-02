from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from art.attacks import PoisoningAttack
from art.attacks.poisoning import (
    FeatureCollisionAttack,
    PoisoningAttackCleanLabelBackdoor,
)
from art.estimators.classification import PyTorchClassifier
from captum.influence import TracInCP, TracInCPFast
from PIL import Image
from torch._C import device
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import TensorDataset as TD

from ibda.influence_functions.dynamic import tracin_torch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np

# - Training Data: Poisoned Dataset
#  New TD(% Training Data (0.01) + NUM_POIONS * TARGET_INSTANCES)
#  Replace Base Instances with their poisoned counterparts on training data.

# Matrix Size = TrainData * TestData * Epochs
# - Understand the disk management on the server


# 1. Load all training data: poisonedModel ckpts and poisonedData
# 2. For every ckpt, load the wights into the model and the compute self influence, and train-test.
# 3. store the matrix in numpy format as np.array.
# 4. order of the influence measure remains unchanged

data_name = "mnist"
model_name = "resnet20"
subfolder = "subset_id0_r0.1"

ckpts_paths = Path("results", model_name, data_name, subfolder, "poisoned", "ckpts/")
path_list = list(ckpts_paths.glob("*.pt"))

# Path list should be string, intead of Path object

path_list = [str(path) for path in path_list]


model = model_dispatcher[model_name](num_classes=10)

train_Data_path = Path("data", "dirty", "mnist", "subset_id0_r0.1", "poisoned_train.pt")
test_data_path = Path("data", "clean", "mnist", "subset_id0_r0.1", "test.pt")

train_data = torch.load(train_Data_path)
test_data = torch.load(test_data_path)


layer_names = model.trainable_layer_names()


testTracIn = tracin_torch.TracInInfluenceTorch(
		model_instance=model,
		ckpts_file_paths=path_list,
		batch_size=128,
		fast_cp=True,
		layers=layer_names,
)


#Make train data the first 10 elements of train data

subset_train = train_data[:10]
subset_images= subset_train[0]
subset_labels = subset_train[1]
new_train = TD(subset_images, subset_labels)

subset_test = test_data[:5]
subset_images = subset_test[0]
subset_labels = subset_test[1]
new_test = TD(subset_images, subset_labels)

matrix_inf = testTracIn.compute_train_to_test_influence(new_train, new_test)



matrix_inf_savedir = Path("results", model_name, data_name, subfolder, "poisoned")
matrix_inf_savedir.mkdir(parents=True, exist_ok=True)

save_as_np(matrix_inf, matrix_inf_savedir, "influence_matrix.npy")

#load the matrix


loaded_matrix_inf = np.load(matrix_inf_savedir / "influence_matrix.npy")

print(loaded_matrix_inf)
