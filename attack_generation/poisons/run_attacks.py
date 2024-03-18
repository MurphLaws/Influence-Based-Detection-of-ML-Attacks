from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
            "max_iter": 100,
            "similarity_coeff": 200,
            "watermark": 0.1,
            "learning_rate": 0.1,
            "verbose": True,
        },
        "cifar10": {
            "classifier": classifier,
            "target": target_instance,
            "feature_layer": feature_layer,
            "max_iter": 10,
            "similarity_coeff": 256,
            "watermark": 0.3,
            "learning_rate": 0.01,
            "verbose": True,
        },
        "fmnist": {
            "classifier": classifier,
            "target": target_instance,
            "feature_layer": feature_layer,
            "max_iter": 100,
            "similarity_coeff": 200,
            "watermark": 0.1,
            "learning_rate": 0.1,
            "verbose": True,
        },
    }

    attack = FeatureCollisionAttack(**hyperparam_dict[data_name])

    poisons, poison_labels = attack.poison(base_instances)
    poison_labels = np.argmax(poison_labels, axis=1)
    poison_pred = np.argmax(classifier.predict(poisons), axis=1)

    return poisons, poison_labels


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
    data_name,
    test_data_fp: TD,
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

    if data_name == "mnist" or data_name == "fmnist":
        repeated_training_images = []
        repeated_training_labels = []
        repeated_test_images = []
        repeated_test_labels = []

        for image, label in train_data:
            image_3_channels = image.repeat(3, 1, 1)
            repeated_training_images.append(image_3_channels)
            repeated_training_labels.append(label)

        for image, label in test_data:
            image_3_channels = image.repeat(3, 1, 1)
            repeated_test_images.append(image_3_channels)
            repeated_test_labels.append(label)

        repeated_training_images = torch.stack(repeated_training_images)
        repeated_training_labels = torch.tensor(repeated_training_labels)

        repeated_test_images = torch.stack(repeated_test_images)
        repeated_test_labels = torch.tensor(repeated_test_labels)

        train_data = TD(repeated_training_images, repeated_training_labels)
        test_data = TD(repeated_test_images, repeated_test_labels)

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

    loss = nn.CrossEntropyLoss()

    if model_ckpt_fp is None:
        model_savedir = Path(
            f"results/{model_name}/{data_name}/{dir_suffix}/clean/ckpts"
        )
        model_savedir.mkdir(parents=True, exist_ok=True)
        model, info = train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            epochs=conf_mger.model_training.epochs,
            batch_size=conf_mger.model_training.batch_size,
            learning_rate=0.01,  # conf_mger.model_training.learning_rate,
            reg_strength=conf_mger.model_training.regularization_strength,
            seed=conf_mger.model_training.random_seed,
            device=device,
            save_dir=model_savedir,
        )
        save_as_json(info, savedir=model_savedir, fname="info.json")
    else:
        model = set_model_weights(model, model_ckpt_fp)

    model.eval()

    classifier = PyTorchClassifier(
        model=model.get_model_instance(),
        clip_values=(np.min(test_x), np.max(test_x)),
        loss=loss,
        optimizer=optim.Adam(model.parameters(), lr=0.0001),
        input_shape=input_shape,
        nb_classes=num_classes,
        device_type=device,
    )

    def draw_bases_and_targets(test_y, test_x, base_class, target_class, num_poisons):
        correctly_classified = False

        while not correctly_classified:

            np.random.seed(seed)

            print("Drawing base(s) and target(s)...")
            # BASE INSTANCE SELECTION
            # This is not enforcer, otherwise you would have problems with generating sevaral poisons
            all_xtest_predictions = classifier.predict(test_x)

            base_idxs = np.where(test_y == base_class)[0]
            base_class_certainty = all_xtest_predictions[base_idxs][:, base_class]
            base_class_certainty = F.softmax(
                torch.tensor(base_class_certainty), dim=0
            ).numpy()
            base_class_certainty = base_class_certainty / np.sum(base_class_certainty)
            base_ids = np.random.choice(base_idxs, num_poisons, p=base_class_certainty)

            # TARGET INSTANCE SELECTION
            target_idxs = np.where(test_y == target_class)[0]
            # Assuring correct classification of the target instances
            predicted_as_target = np.argmax(all_xtest_predictions[target_idxs], axis=1)
            correctly_classified = target_idxs[predicted_as_target == target_class]

            correctly_classified_probs = [
                F.softmax(torch.tensor(i), dim=0).numpy()
                for i in all_xtest_predictions[correctly_classified]
            ]

            targets_bases_certainty = [
                i[base_class] for i in correctly_classified_probs
            ]
            targets_bases_certainty = targets_bases_certainty / np.sum(
                targets_bases_certainty
            )

            target_ids = np.random.choice(
                correctly_classified, 1, p=targets_bases_certainty
            )

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

    target_base_ids = {str(target_ids[0]): str(base_ids)}
    print(target_base_ids)

    # /{dir_suffix}

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

    poison_images_tensor = torch.tensor(poisons, dtype=torch.float32)
    poison_labels_tensor = torch.tensor(poison_labels, dtype=torch.long)

    only_poisons_dataset = TD(poison_images_tensor, poison_labels_tensor)

    # Get only 10 percent of the training data

    training_data_subset = torch.utils.data.Subset(
        train_data, range(0, int(len(train_data) * 0.01))
    )

    # Concatenate training data subset with the poisons

    new_dataset = torch.utils.data.ConcatDataset(
        [training_data_subset, only_poisons_dataset],
    )

    poisoned_dataset = torch.utils.data.ConcatDataset(
        [train_data, only_poisons_dataset],
    )

    # Get a portion of train_x and train_y to train the model with the poisons

    assert len(poisoned_dataset) == len(train_data) + len(
        poison_images_tensor
    ), "The generated poions have not been added correctly to the dataset"

    train_x, train_y = train_data.tensors[0].numpy(), train_data.tensors[1].numpy()
    pois_train = np.vstack((train_x, poisons))
    pois_labels = np.hstack((train_y, poison_labels))

    # Saving poison Imgs in root. These could be color or grayscale, so do it in a way that is compatible with both

    poisons = torch.tensor(poisons, dtype=torch.float32)
    poison_imgs_savedir = Path(
        "results", model_name, data_name, dir_suffix, "dirty", "poison_imgs"
    )
    poison_imgs_savedir.mkdir(parents=True, exist_ok=True)

    images = poisons
    images = images.permute(0, 2, 3, 1).numpy()

    images = (images * 255).astype(np.uint8)

    for i, img in enumerate(images):
        img = Image.fromarray(img.astype(np.uint8))
        img.save(poison_imgs_savedir / f"poison_{i}.png")

    succesful_attack = False
    max_iters = 200
    i = 0
    poisoned_model_savedir = Path(
        "results", model_name, data_name, dir_suffix, "dirty", "ckpts"
    )
    poisoned_model_savedir.mkdir(parents=True, exist_ok=True)
    print("Base class: ", base_class)
    print("Target class: ", target_class)

    while not succesful_attack and i < max_iters:
        print("Epoch: ", i)

        model, info = train(
            model=model,
            train_data=new_dataset,
            test_data=test_data,
            epochs=1,
            save_ckpts=True,
            save_dir=poisoned_model_savedir,
            batch_size=conf_mger.model_training.batch_size,
            learning_rate=conf_mger.model_training.learning_rate,
            reg_strength=conf_mger.model_training.regularization_strength,
            seed=conf_mger.model_training.random_seed,
            device=device,
        )

        model.eval()

        target_instance = test_data[target_ids][0][0]

        with torch.no_grad():

            target_pred = model(target_instance.unsqueeze(0))

            current_precictions = F.softmax(target_pred, dim=1)

            predicted_class = torch.argmax(target_pred, dim=1).item()
            print(
                "Predicted class: ",
                current_precictions[0][predicted_class].item(),
                "Prob: ",
                np.argmax(current_precictions[0].numpy()),
            )
            print("Base Class Probability: ", current_precictions[0][base_class].item())
            print(
                "Target Class Probability: ",
                current_precictions[0][target_class].item(),
            )

            target_pred = torch.argmax(target_pred, dim=1).item()

            if target_pred != target_class:
                print("Target instance missclassified")
                succesful_attack = True
                break
            else:
                i += 1
                continue

    attack_dict_savedir = Path("results", model_name, data_name, "dirty", "attacks")
    attack_dict_savedir.mkdir(parents=True, exist_ok=True)

    save_as_json(
        target_base_ids,
        attack_dict_savedir,
        "poisons_of_id" + next(iter(target_base_ids.keys())) + ".json",
    )


if __name__ == "__main__":

    run_attack()
