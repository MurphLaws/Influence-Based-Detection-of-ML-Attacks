from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from art.attacks import EvasionAttack
from art.attacks.evasion import BoundaryAttack, CarliniL2Method, FastGradientMethod, SaliencyMapMethod, \
    AutoAttack, Wasserstein, SquareAttack, ElasticNet
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset as TD, TensorDataset

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np

from ..utils import plot_adversarial_examples

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
    subset_ids = np.sort(subset_ids)

    # Step 6: Generate adversarial test examples
    x_test_adv = attack.generate(x=test_x[subset_ids])
    predictions_adv = classifier.predict(x_test_adv)
    pred_adv_labels = np.argmax(predictions_adv, axis=1)
    accuracy = np.sum(pred_adv_labels == test_y[subset_ids]) / len(pred_adv_labels)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    adv_success_pos = np.where(pred_adv_labels != test_y[subset_ids])[0]
    final_adv_ids = subset_ids[adv_success_pos]

    print(
        f"Attack was successful on {(len(final_adv_ids) / len(subset_ids))*100}% of the selected samples"
    )

    error_col[final_adv_ids] = 1

    if plot_adv_examples:
        plot_adversarial_examples(x_test_adv[adv_success_pos])

    return x_test_adv[adv_success_pos], error_col


@click.command()
@click.option("--attack", required=True, type=click.STRING)
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--train_data_fp", required=True, type=click.Path(exists=True))
@click.option("--test_data_fp", required=True, type=click.Path(exists=True))
@click.option("--model_conf_fp", required=True, type=click.Path(exists=True))
@click.option("--subset_id", default="", type=click.STRING)
@click.option("--model_ckpt_fp", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--seed", type=click.INT, default=None, help="")
@click.option("--plot", type=click.BOOL, default=False, help="")
def run_all_evasion_attacks(
    attack,
    data_name,
    train_data_fp,
    test_data_fp,
    model_conf_fp,
    subset_id,
    model_ckpt_fp=None,
    device=None,
    seed=None,
    plot=False,
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
        model_savedir = Path(f"results/{model_name}/{data_name}/{subset_id}/clean/ckpts")
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

    test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()

    loss = nn.CrossEntropyLoss()
    model.eval()

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(np.min(test_x), np.max(test_x)),
        loss=loss,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=num_classes,
        device_type=device,
    )

    attacks_dict = {
        "fgsm": FastGradientMethod(estimator=classifier, eps=0.1),
        "cw": CarliniL2Method(
            classifier=classifier, verbose=True, confidence=0.1, batch_size=64
        ),
        "bound_attack": BoundaryAttack(
            estimator=classifier, targeted=False, max_iter=1000, verbose=True
        ),
        "jsma": SaliencyMapMethod(classifier=classifier),
        "auto_attack": AutoAttack(estimator=classifier),
        "wasserstein": Wasserstein(estimator=classifier),
        "square": SquareAttack(estimator=classifier, nb_restarts=2, max_iter=500, eps=1e-2),
        "elastic": ElasticNet(classifier=classifier, confidence=0.1, batch_size=64)
    }


    if attack not in attacks_dict:
        raise ValueError(f'{attack} not in {attacks_dict.keys()}')
    attack_fn = attacks_dict[attack]
    print(f"Running {attack}")
    final_savedir = Path(
        "data", "dirty", attack, model_name, data_name, subset_id
    )
    final_savedir.mkdir(parents=True, exist_ok=True)
    adv_examples, error_col = run_attack(
        classifier=classifier,
        test_data=test_data,
        adv_ratio=0.1,
        seed=seed,
        attack=attack_fn,
        plot_adv_examples=plot,
    )

    adv_ids = np.where(error_col == 1)[0]

    print('MSE', mse_loss(torch.Tensor(adv_examples), test_data[adv_ids][0]))

    test_dirty_X = test_x.copy()
    test_dirty_X[adv_ids] = adv_examples
    dirty_test_preds = classifier.predict(test_dirty_X)
    dirty_test_labels = np.argmax(dirty_test_preds, axis=1)
    assert accuracy_score(test_y[adv_ids], dirty_test_labels[adv_ids]) == 0
    test_dirty_y = torch.tensor(dirty_test_labels)
    test_dirty_X = torch.tensor(test_dirty_X)

    torch.save(TensorDataset(test_dirty_X, test_dirty_y), Path(final_savedir, "test_dirty_y_pred.pt"))
    torch.save(torch.tensor(error_col), Path(final_savedir, "is_adv.pt"))
    print()

if __name__ == "__main__":

    run_all_evasion_attacks()
