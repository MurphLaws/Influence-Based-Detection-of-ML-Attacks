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
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import TensorDataset as TD

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np


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
    target_ids: int,
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

    target_instance = test_x[target_ids]

    feature_layer = classifier.layer_names[-1]

    if len(base_ids) > 1:
        base_instances = np.copy([test_x[base_ids]])[0]
    else:
        base_instances = np.copy([test_x[base_ids][0]])

    hyperparam_dict = {
        "mnist": {
            "classifier": classifier,
            "target": target_instance,
            "feature_layer": feature_layer,
            "max_iter": 10,
            "similarity_coeff": 5446456456,
            "watermark": None,
            "learning_rate": 0.001,
            "verbose": True,
        },
        "cifar10": {
            "classifier": classifier,
            "target": target_instance,
            "feature_layer": feature_layer,
            "max_iter": 10,
            "similarity_coeff": 256,
            "watermark": 0.3,
            "learning_rate": 1,
            "verbose": True,
        },
        "fmnist": None,
    }

    attack = FeatureCollisionAttack(**hyperparam_dict[data_name])

    poisons, poison_labels = attack.poison(base_instances)
    poison_labels = np.argmax(poison_labels, axis=1)
    poison_pred = np.argmax(classifier.predict(poisons), axis=1)

    return poisons, poison_labels
    # print the class of the poison

    # TODO C2. Poisons preserve their original class. If you see very few fulfilling C2, increase the max_iter in the FeatureCollisionAttack

    # Save all the poison images in root
    # save all the poison images reshaping them to be 28x28


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_conf_fp", required=True, type=click.Path(exists=True))
@click.option("--train_data_fp", required=True, type=click.Path(exists=True))
@click.option("--test_data_fp", required=True, type=click.Path(exists=True))
@click.option("--dir_suffix", default="", type=click.STRING)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--model_ckpt_fp", type=click.Path(exists=True), default=None)
@click.option("--seed", type=click.INT, default=None, help="")
@click.option("--target_class", type=click.INT, default=None, help="")
@click.option("--base_class", type=click.INT, default=None, help="")
@click.option("--num_poisons", type=click.INT, default=None, help="")


# RUNNING FIRST A SINGLE ATTACK
def run_attack(
    # classifier: PyTorchClassifier, #Necessary to have the model and the preprocessing
    data_name,
    test_data_fp: TD,  # Necessary to have the test data
    train_data_fp: TD,
    model_conf_fp,
    target_class,
    base_class,
    num_poisons,
    seed=None,
    dir_suffix="",
    device=None,
    model_ckpt_fp=None,
):

    train_data = torch.load(train_data_fp)
    test_data = torch.load(test_data_fp)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])

    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()

    conf_mger = ConfigManager(model_training_conf=model_conf_fp)
    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    # TODO change the placement of model.eval() up to PytorchClassifier and put them after IF-stmt for fine-funing

    loss = nn.CrossEntropyLoss()

    # Slect all the instances from base class

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(np.min(test_x), np.max(test_x)),
        loss=loss,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=num_classes,
        device_type=device,
    )

    if model_ckpt_fp is None:
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
        model = set_model_weights(model, model_ckpt_fp)

    model.eval()

    def draw_bases_and_targets(test_y, test_x, base_class, target_class, num_poisons):
        correctly_classified = False

        while not correctly_classified:

            np.random.seed()
            print("Drawing base(s) and target(s)...")
            # BASE INSTANCE SELECTION

            all_xtest_predictions = classifier.predict(test_x)

            base_idxs = np.where(test_y == base_class)[0]
            base_class_certainty = all_xtest_predictions[base_idxs][:, base_class]
            base_class_certainty = base_class_certainty / np.sum(base_class_certainty)
            base_ids = np.random.choice(base_idxs, num_poisons, p=base_class_certainty)

            # TARGET INSTANCE SELECTION
            target_idxs = np.where(test_y == target_class)[0]
            target_class_certainty = all_xtest_predictions[target_idxs][:, target_class]
            target_class_certainty = target_class_certainty / np.sum(
                target_class_certainty
            )
            target_ids = np.random.choice(target_idxs, 1, p=target_class_certainty)

            if np.all(test_y[base_ids] == base_class) and np.all(
                test_y[target_ids] == target_class
            ):
                correctly_classified = True
                print("Found base(s) and target(s) with the correct classification.")
                return base_ids, target_ids

            else:
                print("Redrawing base(s) and target(s)...")
                continue

    base_ids, target_ids = draw_bases_and_targets(
        test_y, test_x, base_class, target_class, num_poisons
    )

    # TODO C1. Ensure that the base and target ids are CORRECTLY classified

    target_base_ids = {
        target_ids[0]: base_ids
    }  # key:value of type int:List[int] which is Target ID: List[Base IDs]

    print(target_base_ids)
    # TODO ensure that the target_ids are DIFFERENT than base ids

    poisons, poison_labels = poison_generator(
        classifier=classifier,
        data_name=data_name,
        test_data=test_data,
        target_class=target_class,
        base_class=base_class,
        target_ids=target_ids,
        base_ids=base_ids,
        seed=seed,
        attack=PoisoningAttack,
    )

    # TODO C3. Save only the base and target IDs for which the target IDs' class has been changed

    max_iter = 100
    not_sucess_condition = True
    i = 0
    while not_sucess_condition and i < max_iter:

        # Create poisoned dataset by append the train_set with the poisons and their labels

        images_tensor = torch.tensor(poisons, dtype=torch.float32)
        labels_tensor = torch.tensor(poison_labels, dtype=torch.long)

        dataset = TD(images_tensor, labels_tensor)
        poisoned_dataset = torch.utils.data.ConcatDataset([train_data, dataset])

        assert len(poisoned_dataset) == len(train_data) + len(
            dataset
        ), "The generated poions have not been added correctly to the dataset"

        # odel = train(model, ..., train_set_with_poisons, ...)
        # check if target ID's class has changed (success condition)
        # 1. target_base_ids[target_id] = base_ids
        # 2. Save the poisons in location: check final_savedir in adversarials/run_attacks.py
        # 3. Save the poisons in format: as in adversarials/run_attacks.py, save the poison data in a poisons_of_id<TARGET_ID>.pt. For example poisons_of_id4.pt
        # 4. break the loop
        success_rate = "end"
        i = 200
        # Save this dict as a json after the loop so we dont have to save n different dictionaries
        # Example of json {4:[6, 10, 11, 15]}. When we want to read the actual poisons, we read like that: json.load(poisons_of_id{ID}.pt where ID is a dictionary key)
        print(success_rate)

    # Print the model layer names:


if __name__ == "__main__":

    run_attack()
