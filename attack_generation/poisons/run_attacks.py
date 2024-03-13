from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from art.attacks import PoisoningAttack
from art.attacks.poisoning import (
    FeatureCollisionAttack,
    PoisoningAttackCleanLabelBackdoor,
)
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import TensorDataset as TD

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np




@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_conf_path", required=True, type=click.Path(exists=True))
@click.option("--train_data_path", required=True, type=click.Path(exists=True))
@click.option("--test_data_path", required=True, type=click.Path(exists=True))
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--model_ckpt_path", type=click.Path(exists=True), default=None)


# ToDo
# OPTIONS TO ADD
# --target_class: int, required=True
# --base_class: int, required=True
# --target_ids: List[int], required=True
# --base_ids: List[int], required=True
# --num_poisons: int, required=True


# RUNNING FIRST A SINGLE ATTACK
def run_attack(
    # classifier: PyTorchClassifier, #Necessary to have the model and the preprocessing
    data_name,
    test_data_path: TD,  # Necessary to have the test data
    train_data_path: TD,
    model_conf_path,
    device=None,
    model_ckpt_path=None,
):

    train_data = torch.load(train_data_path)
    test_data = torch.load(test_data_path)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])

    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()

    conf_mger = ConfigManager(model_training_conf=model_conf_path)
    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(np.min(test_x), np.max(test_x)),
        loss=loss,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=num_classes,
        device_type=device,
    )

    if model_ckpt_path is None:
        model_savedir = Path(f"results/{model_name}/{data_name}/clean/ckpts")
        model_savedir.mkdir(parents=True, exist_ok=True)
        model, info = train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            epochs=conf_mger.model_training.epochs,
            batch_size=conf_mger.model_training.batch_size,
            learning_rate=conf_mger.model_training.learning_rate,
            reg_strength=conf_mger.model_training.regularization_strength,
            seed=conf_mger.model_training.random_seed,
            device=device,
            save_dir=model_savedir,
        )
        save_as_json(info, savedir=model_savedir, fname="info.json")
    else:
        model = set_model_weights(model, model_ckpt_path)

    ##TARGET SELECTION

    x_test, y_test = test_data.tensors
    target_class = 4
    class_descr = list(np.arange(10))
    target_label = np.zeros(len(class_descr))
    target_instances = x_test[y_test == target_class]
    target_instance = target_instances[21]
    feature_layer = classifier.trainable_layers[-1]
    base_class = 2

    ##BASE SELECTION

    base_idxs = np.where(y_test == base_class)
    base_instances = np.copy(x_test[base_idxs])
    base_labels = y_test[base_idxs][:10]

    # attack = FeatureCollisionAttack(classifier,
    #                            target_instance,
    #                            feature_layer,
    #                            max_iter=10,
    #                            similarity_coeff=256,
    #                            watermark=0.3,
    #                            learning_rate=1)
    # poison, poison_labels = attack.poison(base_instances)


if __name__ == "__main__":

    run_attack()
