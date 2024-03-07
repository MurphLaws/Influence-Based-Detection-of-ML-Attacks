from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from art.attacks import EvasionAttack
from art.attacks.evasion import BoundaryAttack, CarliniL2Method, FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import TensorDataset as TD

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager

from ibda.utils.writers import save_as_np


@click.argument("adv_ratio", type=click.FloatRange(min=0, max=1), required=True)
@click.argument("num_classes", type=click.IntRange(min=0), required=True)
def run_attack(
    classifier: PyTorchClassifier,
    test_data: TD,
    adv_ratio: float,
    seed: int,
    attack: EvasionAttack,
    plot_adv_examples=False,
):
    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()

    predictions = classifier.predict(test_x)
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.sum(pred_labels == test_y) / len(test_y)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    error_col = np.zeros(len(test_y), dtype=int)

    corrected_classified_ids = np.where(pred_labels == test_y)[0]

    np.random.seed(seed)
    subset_ids = np.random.choice(
        corrected_classified_ids,
        size=int(np.floor(adv_ratio * len(test_y))),
        replace=False,
    )

    # Step 6: Generate adversarial test examples
    x_test_adv = attack.generate(x=test_x[subset_ids])
    predictions_adv = classifier.predict(x_test_adv)
    pred_adv_labels = np.argmax(predictions_adv, axis=1)
    accuracy = np.sum(pred_adv_labels == test_y[subset_ids]) / len(pred_adv_labels)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    adv_success_pos = np.where(pred_adv_labels != test_y[subset_ids])[0]
    final_adv_ids = subset_ids[adv_success_pos]

    print(f'Attack was successful on {(len(final_adv_ids) / len(subset_ids))*100}% of the selected samples')

    error_col[final_adv_ids] = 1

    if plot_adv_examples:
        plot_adversarial_examples(x_test_adv[adv_success_pos])

    return x_test_adv[adv_success_pos], error_col


def plot_adversarial_examples(adv_examples):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    i, j, counter = 0, 0, 0
    for i in range(3):
        for j in range(3):
            if len(adv_examples) > counter:
                adv_ex = adv_examples[counter]
                axes[i, j].imshow(torch.tensor(adv_ex).permute(1, 2, 0))
            counter += 1
    plt.tight_layout()
    plt.show()


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--train_data_fp", required=True, type=click.Path(exists=True))
@click.option("--test_data_fp", required=True, type=click.Path(exists=True))
@click.option("--model_conf_fp", required=True, type=click.Path(exists=True))
@click.option("--dir_suffix", required=True, type=click.Path())
@click.option("--model_ckpt_fp", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(['cuda', 'cpu']), default=None)
@click.option("--seed", type=click.INT, default=None, help="")
def run_all_evasion_attacks(
    data_name,
    train_data_fp,
    test_data_fp,
    model_conf_fp,
    dir_suffix,
    model_ckpt_fp=None,
    device=None,
    seed=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    conf_mger = ConfigManager(model_training_conf=model_conf_fp)
    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name

    train_data = torch.load(train_data_fp)
    test_data = torch.load(test_data_fp)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    if model_ckpt_fp is None:
        model_savedir = Path(f"results/{model_name}/{data_name}/clean/ckpts")
        model_savedir.mkdir(parents=True, exist_ok=True)
        model, _ = train(
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
    else:
        model = set_model_weights(model, model_ckpt_fp)

    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()

    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(np.min(test_x), np.max(test_x)),
        loss=loss,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=num_classes,
        device_type=device
    )

    attacks_dict = {
        "fgsm": FastGradientMethod(estimator=classifier, eps=0.1),
        "cw": CarliniL2Method(classifier=classifier, verbose=True, confidence=0.1, batch_size=64),
        "bound_attack": BoundaryAttack(estimator=classifier, targeted=False, max_iter=1000, verbose=True)
    }

    for attack_name, attack_fn in attacks_dict.items():
        print(f"Running {attack_name}")
        final_savedir = Path('data', 'dirty', attack_name, model_name, data_name, dir_suffix)
        final_savedir.mkdir(parents=True, exist_ok=True)
        adv_examples, error_col = run_attack(
            classifier=classifier,
            test_data=test_data,
            adv_ratio=0.1,
            seed=seed,
            attack=attack_fn,
            plot_adv_examples=True,
        )

        fname = 'adv'

        torch.save(adv_examples, Path(final_savedir, fname + '.pt'))
        save_as_np(error_col, savedir=final_savedir, fname=fname + '_ids.npy')


if __name__ == "__main__":

    run_all_evasion_attacks()
