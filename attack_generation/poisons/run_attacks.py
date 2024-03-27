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
            "max_iter": 10,
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
    return 0, 0


def draw_bases_and_targets(
    classifier, test_y, test_x, num_poisons, seed, number_of_targets=5
):

    correctly_classified = True

    all_predictions = classifier.predict(test_x)

    def generate_classes(input_array):
        output_array = np.random.randint(1, 10, size=len(input_array))
        output_array = np.where(
            output_array == input_array, (output_array + 1) % (10 + 1), output_array
        )
        return output_array

    def random_permutation(original_array):
        permutation = original_array.copy()
        while True:
            np.random.shuffle(permutation)
            if not np.any(permutation == original_array):
                break
        return permutation

    target_classes = list(range(number_of_targets))
    base_classes = random_permutation(target_classes)

    print("Target classes: ", target_classes)
    print("Base classes: ", base_classes)
    ##TARGET SELECTION
    target_indexes = []
    for idx, target_class in enumerate(target_classes):
        target_ids = np.where(test_y == target_class)[0]
        predicted_as_target = np.argmax(all_predictions[target_ids], axis=1)
        correctly_classified = target_ids[predicted_as_target == target_class]
        correctly_classified_probs = [
            F.softmax(torch.tensor(i), dim=0).numpy()
            for i in all_predictions[correctly_classified]
        ]
        targets_bases_certainty = [
            i[base_classes[idx]] for i in correctly_classified_probs
        ]
        targets_bases_certainty = targets_bases_certainty / np.sum(
            targets_bases_certainty
        )
        np.random.seed(seed)
        target_indexes.append(
            [np.random.choice(correctly_classified, 1, p=targets_bases_certainty)[0]]
        )

    ##BASE SELECTION

    def draw_base_instances(base_classes):
        base_indexes = []
        for idx, base_class in enumerate(base_classes):
            base_ids = np.where(test_y == base_class)[0]
            bases_certainty = F.softmax(
                torch.tensor(all_predictions[base_ids]), dim=0
            ).numpy()
            bases_base_certainty = [base[base_class] for base in bases_certainty]
            bases_base_certainty = bases_base_certainty / np.sum(bases_base_certainty)
            base_indexes.append(
                np.random.choice(base_ids, num_poisons, p=bases_base_certainty)
            )
        return np.array(base_indexes)

    # while True:

    final_base_indexes = draw_base_instances(base_classes)
    # print("Drawing bases...",end="\r")
    # if (final_base_indexes.size == np.unique(final_base_indexes).size):
    #    print("Found bases...", end="\r")
    #    break

    return final_base_indexes, target_indexes, base_classes, target_classes


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_conf_fp", required=True, type=click.Path(exists=True))
@click.option("--train_data_fp", required=True, type=click.Path(exists=True))
@click.option("--test_data_fp", required=True, type=click.Path(exists=True))
@click.option("--dir_suffix", default="", type=click.STRING)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--model_ckpt_fp", type=click.Path(exists=True), default=None)
@click.option("--seed", type=click.INT, default=None, help="")
@click.option("--target_number", type=click.INT, default=None, help="")
@click.option("--max_iter", type=click.INT, default=None, help="")
@click.option("--num_poisons", type=click.INT, default=None, help="")

# TEST
# RUNNING FIRST A SINGLE ATTACK
def run_attack(
    data_name,
    test_data_fp: TD,
    train_data_fp: TD,
    model_conf_fp,
    num_poisons,
    target_number,
    max_iter,
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

    base_ids_array = []
    target_ids_array = []

    base_ids, target_ids, base_classes, target_classes = draw_bases_and_targets(
        classifier, test_y, test_x, num_poisons, seed, number_of_targets=target_number
    )

    target_base_ids = {}
    for base, target in zip(base_ids, target_ids):
        target_base_ids[str(target)] = base

    class_and_id = zip(base_ids, base_classes, target_ids, target_classes)

    final_poisons = []
    final_poisons_labels = []

    for quad in class_and_id:
        poisons, poison_labels = poison_generator(
            classifier=classifier,
            data_name=data_name,
            test_data=test_data,
            base_class=quad[1],
            target_class=quad[3],
            base_ids=quad[0],
            target_ids=quad[2],
            seed=seed,
            attack=PoisoningAttack,
        )

        final_poisons.append(poisons[0])
        final_poisons_labels.append(poison_labels)

    final_poisons = np.array(final_poisons)
    final_poisons_labels = np.array(final_poisons_labels)

    poison_images_tensor = torch.tensor(poisons, dtype=torch.float32)
    poison_labels_tensor = torch.tensor(poison_labels, dtype=torch.long)

    only_poisons_dataset = TD(poison_images_tensor, poison_labels_tensor)

    # Get only 10 percent of the training data, avoiding the poison labels and  saving all labels
    selected_bases_ids = np.array([x for xs in base_ids for x in xs])  # Just a flatten

    train_data_length = len(train_data)
    clean_indexes = np.random.choice(
        [i for i in range(train_data_length) if i not in selected_bases_ids],
        size=int(train_data_length * 0.01),
        replace=False,
    )

    # Save the poisoned dataset and the cleanIds + poisonIds

    training_data_subset = torch.utils.data.Subset(train_data, clean_indexes)

    # Concatenate training data subset with the poisons

    poisoned_dataset = torch.utils.data.ConcatDataset(
        [training_data_subset, only_poisons_dataset],
    )

    assert len(poisoned_dataset) == int(len(train_data) * 0.01) + len(
        poison_images_tensor
    ), "The generated poisons have not been added correctly to the dataset"

    train_x, train_y = train_data.tensors[0].numpy(), train_data.tensors[1].numpy()

    def save_images():

        pois_train = np.vstack((train_x, poisons))
        pois_labels = np.hstack((train_y, poison_labels))
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

    poisoned_dataset_savedir = Path(
        "data",
        "dirty",
        data_name,
        dir_suffix,
    )
    poisoned_dataset_savedir.mkdir(parents=True, exist_ok=True)
    torch.save(poisoned_dataset, poisoned_dataset_savedir / "poisoned_dataset.pt")
    np.save(poisoned_dataset_savedir / "clean_ids.npy", clean_indexes)
    np.save(poisoned_dataset_savedir / "poison_ids.npy", selected_bases_ids)

    succesful_attack = False
    max_iters = max_iter
    i = 0

    # Stopping condition deleted, since is very unlikely to get all target instances missclassified
    # when you have a high number of them
    target_ids = [item[0] for item in target_ids]
    while i < max_iters:
        print("Epoch: ", i)

        poisoned_model_savedir = Path(
            f"results/{model_name}/{data_name}/{dir_suffix}/poisoned/ckpts"
        )
        poisoned_model_savedir.mkdir(parents=True, exist_ok=True)

        model, info = train(
            model=model,
            train_data=poisoned_dataset,
            test_data=test_data,
            epochs=1,
            save_ckpts=True,
            save_dir=poisoned_model_savedir,
            ckpt_name="checkpoint-" + str(i + 1) + ".pt",
            batch_size=conf_mger.model_training.batch_size,
            learning_rate=conf_mger.model_training.learning_rate,
            reg_strength=conf_mger.model_training.regularization_strength,
            seed=conf_mger.model_training.random_seed,
            device=device,
        )

        model.eval()

        target_instances = test_data[target_ids][0]

        with torch.no_grad():

            target_preds = model(target_instances)
            current_predictions = np.argmax(
                F.softmax(target_preds, dim=1), axis=1
            ).numpy()

            i += 1

    # ToDo:

    # 1. Poisoned "Multitarget Attack" is currently working, but I need to fix all the unused input parameters
    # 2. The previous stopping condition is not needed anymore, however, I would like to add more information during
    #    the attacjs printings, just so the user executing it can understand whats happening
    # (3). I removed all the savings right now, because running on my current environement (Low memory), requiered avoiding it.
    #    So right now is commented, even though I think thats strictly necessary for executing influence so this is the
    #    priority.

    print(target_classes != current_predictions)
    print("Missclassified instances: ", np.sum(target_classes != current_predictions))
    print("Total instances: ", len(target_classes))
    print(
        "Attack success rate: ",
        np.sum(target_classes != current_predictions) / len(target_classes),
    )

    for i in range(len(target_ids)):
        if target_classes[i] != current_predictions[i]:
            print("----------------------------------------------------------------")
            print("Missclassified instance: ", target_ids[i])
            print("Target class: ", target_classes[i])
            print("Predicted class: ", current_predictions[i])
            print("----------------------------------------------------------------")
            print("\n")

    #    predicted_class = torch.argmax(target_pred, dim=1).item()
    #    print(
    #    "Predicted class: ",
    #    current_precictions[0][predicted_class].item(),
    #    "Prob: ",
    #    np.argmax(current_precictions[0].numpy()),
    # )
    #    print("Base Class Probability: ", current_precictions[0][base_class].item())
    #    print(
    #    "Target Class Probability: ",
    #    current_precictions[0][target_class].item(),
    # )

    #   acc_each_epoch.append(info["test_acc"])
    #
    #    target_pred = torch.argmax(target_pred, dim=1).item()

    #    if target_pred != target_class:
    #        print("Target instance missclassified")
    #        succesful_attack = True
    #        break
    #    else:
    #        i += 1
    #        continue

    # attack_dict_savedir = Path(
    #     "results", model_name, data_name, "dirty", dir_suffix, "attacks"
    # )
    # attack_dict_savedir.mkdir(parents=True, exist_ok=True)

    # if succesful_attack:
    #     target_base_ids["succesful"] = True
    # else:
    #     target_base_ids["succesful"] = False

    # if num_poisons == 1:
    #     target_base_ids["attackType"] = "OneToOne"
    # else:
    #     target_base_ids["attackType"] = "ManyToOne"

    # target_base_ids["acc_epoch"] = acc_each_epoch

    # save_as_json(
    #     target_base_ids,
    #     attack_dict_savedir,
    #     "poisons_of_id" + next(iter(target_base_ids.keys())) + ".json",
    # )


if __name__ == "__main__":

    run_attack()
