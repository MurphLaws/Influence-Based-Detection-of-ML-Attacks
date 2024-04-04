from pathlib import Path

import click
import numpy as np
import torch
from art.attacks import PoisoningAttack
from art.attacks.poisoning import (
    FeatureCollisionAttack,
    PoisoningAttackCleanLabelBackdoor,
)
from art.estimators.classification import PyTorchClassifier
from captum.influence import TracInCP, TracInCPFast
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import TensorDataset as TD

from ibda.influence_functions.dynamic import tracin_torch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_name", required=True, type=click.STRING)
@click.option("--subset_id", required=True, type=click.STRING)
@click.option("--attack_type", required=True, type=click.Choice(["many_to_one", "one_to_one"]))
def get_influence_matrix(
    data_name: str,
    model_name: str,
    subset_id: str,
    attack_type: str,
    
):  
    ckpts_paths = Path(
            "results", model_name, data_name, subset_id, "poisoned",attack_type, "ckpts/"
         )

    path_list = list(ckpts_paths.glob("*.pt"))
    path_list = [str(path) for path in path_list]
    
    train_data_path = Path("data", "dirty", data_name, subset_id, attack_type,"poisoned_train.pt")
    test_data_path = Path("data", "clean", data_name, subset_id, "test.pt")

    train_data = torch.load(train_data_path)
    test_data = torch.load(test_data_path)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])
    
    model_conf_fp = str(Path("configs", "resnet", f"resnet_{data_name}.json"))

    conf_mger = ConfigManager(model_training_conf=model_conf_fp)

    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    layer_names = model.trainable_layer_names()

    matrix_inf_savedir = Path(
        "results", model_name, data_name, subset_id,"poisoned",  attack_type,  "influence_matrices"
    )
    matrix_inf_savedir.mkdir(parents=True, exist_ok=True)

    for file in matrix_inf_savedir.glob("*.npy"):
        file.unlink()

    for ckpt in path_list:
        model = set_model_weights(model, ckpt)
        model.eval()

        # Get the influence matrix
        tracInObject = tracin_torch.TracInInfluenceTorch(
            model_instance=model,
            ckpts_file_paths=[ckpt],
            batch_size=128,
            fast_cp=True,
            layers=layer_names,
        )

        print("Computing Influence Matrix for ckpt: ", ckpt)

        influence_matrix = tracInObject.compute_train_to_test_influence(
            train_data, test_data
        )

        # Get ckpt name
        ckpt_name = ckpt.split("/")[-1].split(".")[0]
        np.save(
            matrix_inf_savedir / f"IM_{model_name}_{data_name}_{subset_id}_{attack_type}_{ckpt_name}.npy", influence_matrix
        )
        print(influence_matrix.shape)


if __name__ == "__main__":
    get_influence_matrix()
