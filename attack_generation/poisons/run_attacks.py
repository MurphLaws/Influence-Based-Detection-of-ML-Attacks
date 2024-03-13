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
from PIL import Image
from torch.utils.data import TensorDataset as TD

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np

##ADD the argument target_class, can be integer or string


@click.argument("target_class", type=click.IntRange(min=0), required=True)
@click.argument("base_class", type=click.IntRange(min=0), required=True)
@click.argument("target_ids", nargs=-1, type=click.INT, required=True)
@click.argument("base_ids", nargs=-1, type=click.INT, required=True)
@click.argument("num_classes", type=click.IntRange(min=0), required=True)
def poison_generator(
    classifier: PyTorchClassifier,
    data_name: str,
    test_data: TD,
    target_class: int,
    base_class: int,
    target_ids: list,
    base_ids: list,
    seed: int,
    attack: PoisoningAttack,
):

    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()
    predictions = classifier.predict(test_x)
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.sum(pred_labels == test_y) / len(test_y)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    np.random.seed(seed)

    target_class = target_class
    base_class = base_class
    target_ids = target_ids
    base_ids = base_ids

    all_target_instances = test_x[test_y == target_class]
    target_instance = all_target_instances[target_ids]

    if len(target_instance.shape) == 3:
        target_instance = np.expand_dims(target_instance, axis=0)
        target_instance = np.repeat(target_instance, 3, axis=1)

    feature_layer = classifier.layer_names[-1]

    # The variable base_idxs should be all the indexes of the base class in the whole test_x dataset

    base_idxs = np.where(test_y == base_class)[0]

    if len(base_ids) > 1:
        base_instances = np.copy([test_x[base_idxs][base_ids]])[0]
    else:
        base_instances = np.copy([test_x[base_idxs][base_ids][0]])

    attack = FeatureCollisionAttack(
        classifier,
        target_instance,
        feature_layer,
        max_iter=10,
        similarity_coeff=256,
        watermark=0.0001,
        learning_rate=0.001,
        verbose=True,
    )
    poison, poison_labels = attack.poison(base_instances)

    # print the class of the poison

    poison_pred = np.argmax(classifier.predict(poison), axis=1)

    # Save all the poison images in root
    for i in range(len(poison)):
        img = poison[i][0] * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"poison_{i}.png")

    print(f"Poison class: {poison_pred}")


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_conf_path", required=True, type=click.Path(exists=True))
@click.option("--train_data_path", required=True, type=click.Path(exists=True))
@click.option("--test_data_path", required=True, type=click.Path(exists=True))
@click.option("--dir_suffix", default="", type=click.STRING)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--model_ckpt_path", type=click.Path(exists=True), default=None)
@click.option("--seed", type=click.INT, default=None, help="")


# RUNNING FIRST A SINGLE ATTACK
def run_attack(
    # classifier: PyTorchClassifier, #Necessary to have the model and the preprocessing
    data_name,
    test_data_path: TD,  # Necessary to have the test data
    train_data_path: TD,
    model_conf_path,
    seed=None,
    dir_suffix="",
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

    model.eval()
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

    ##STILL NEED TO DEFINE INPUT OF ATTACK SETUP

    poison_generator(
        classifier=classifier,
        data_name=data_name,
        test_data=test_data,
        target_class=3,
        base_class=9,
        target_ids=[4],
        base_ids=[6, 4, 10, 11, 15],  # [4,5],
        seed=seed,
        attack=PoisoningAttack,
    )

    # Print the model layer names:


if __name__ == "__main__":

    run_attack()
