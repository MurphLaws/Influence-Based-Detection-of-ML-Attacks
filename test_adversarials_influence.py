from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import TensorDataset

from ibda.influence_functions.dynamic import TracInInfluenceTorch
from ibda.influence_functions.base_influence import BaseInfluenceEstimator
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import get_last_ckpt, predict, set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_np

attack_choices = ["fgsm", "cw", "bound_attack"]


def compute_influence(
    influence_fn: BaseInfluenceEstimator,
    train_data: TensorDataset = None,
    test_data: TensorDataset = None,
    adv_data: TensorDataset = None,
    savedir: Path = None
):

    test_self_inf, adv_self_inf, train_test_inf, train_test_inf_adv = (
        None,
        None,
        None,
        None,
    )

    if test_data is not None:
        test_self_inf = influence_fn.compute_self_influence(dataset=test_data)
        if savedir is not None:
            save_as_np(test_self_inf, savedir, 'test_self_inf.npy')

    if adv_data is not None:
        adv_self_inf = influence_fn.compute_self_influence(dataset=adv_data)
        if savedir is not None:
            save_as_np(adv_self_inf, savedir, 'adv_self_inf.npy')

    if train_data is not None and test_data is not None:
        train_test_inf = influence_fn.compute_train_to_test_influence(
            train_set=train_data, test_set=test_data
        )
        if savedir is not None:
            save_as_np(train_test_inf, savedir, 'train_test_inf.npy')

    if train_data is not None and adv_data is not None:
        train_test_inf_adv = influence_fn.compute_train_to_test_influence(
            train_set=train_data,
            test_set=adv_data,
        )
        if savedir is not None:
            save_as_np(train_test_inf_adv, savedir, 'train_test_inf_adv.npy')

    return test_self_inf, adv_self_inf, train_test_inf, train_test_inf_adv


def compute_signals():
    pass


@click.command()
@click.option("--attack", type=click.Choice(attack_choices), required=True)
@click.option("--data_name", type=click.STRING, required=True)
@click.option("--model_name", type=click.STRING, required=True)
@click.option("--inf_fn_name", type=click.STRING, required=True)
@click.option("--subset_id", type=click.STRING, required=True)
@click.option("--model_conf", type=click.Path(exists=True), required=True)
@click.option("--inf_fn_conf", type=click.Path(exists=True), required=True)
@click.option("--ckpt_fname", type=click.STRING, default=None)
@click.option("--dirty_data_dir", type=click.STRING, default=None)
@click.option("--train_data_fp", type=click.STRING, default=None)
@click.option("--test_data_fp", type=click.STRING, default=None)
@click.option("--model_ckpt_fp", type=click.STRING, default=None)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=None)
@click.option("--savedir", type=click.STRING, default=None)
def run_pipeline(
    attack,
    data_name,
    model_name,
    inf_fn_name,
    subset_id,
    model_conf,
    inf_fn_conf,
    ckpt_fname = None,
    dirty_data_dir=None,
    train_data_fp=None,
    test_data_fp=None,
    model_ckpt_fp=None,
    device=None,
    savedir=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_ckpt_fp is None:
        default_ckpt_dir = f"results/{model_name}/{data_name}/clean/ckpts/"
        print(f"Assigning automatically the checkpoint dir to {default_ckpt_dir}")
        if ckpt_fname is None:
            model_ckpt_fp = get_last_ckpt(default_ckpt_dir)
            print(f"As --model_ckpt_fp is None, the last ckpt is used: {model_ckpt_fp}")
        else:
            ckpt_fname = ckpt_fname + '.pt' if not ckpt_fname.endswith('.pt') else ckpt_fname
            model_ckpt_fp = Path(default_ckpt_dir, ckpt_fname)

    if dirty_data_dir is None:
        dirty_data_dir = (
            f"data/dirty/{attack}/{model_name}/{data_name}/{subset_id}"
        )
        print(f"Assigning automatically the dirty data dir to {dirty_data_dir}")

    if train_data_fp is None:
        train_data_fp = f"data/clean/{data_name}/{subset_id}/train.pt"
        print(f"Assigning automatically the train data filepath to {train_data_fp}")

    if test_data_fp is None:
        test_data_fp = f"data/clean/{data_name}/{subset_id}/test.pt"
        print(f"Assigning automatically the test data filepath to {test_data_fp}")

    print(f"Running {attack} on {model_ckpt_fp}")

    conf_mger = ConfigManager(model_training_conf=model_conf, inf_func_conf=inf_fn_conf)

    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name
    inf_batch_size = conf_mger.inf_funcs.batch_size
    inf_layers = conf_mger.inf_funcs.influence_layers
    inf_fact_cp = conf_mger.inf_funcs.fast_cp

    # Initialization

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

    model = set_model_weights(model, model_ckpt_fp)

    with open(Path(dirty_data_dir, "adv_ids.npy"), "rb") as f:
        error_col = np.load(f)

    adv_samples = torch.tensor(torch.load(Path(dirty_data_dir, "adv.pt")), device=device)
    adv_corr_labels = torch.tensor(
        test_data.tensors[1].cpu().numpy()[np.where(error_col == 1)[0]], device=device
    )
    adv_pred_labels, adv_preds, _ = predict(
        model, TensorDataset(adv_samples, adv_corr_labels), device=device
    )

    adv_data = TensorDataset(adv_samples, torch.tensor(adv_pred_labels))

    test_pred_labels, _, _ = predict(
        model, test_data, device=device
    )

    test_data = TensorDataset(test_data.tensors[0], torch.tensor(test_pred_labels))

    if inf_layers is None:
        inf_layers = model.trainable_layer_names()
    else:
        inf_layers_tmp = []
        for l_model in model.trainable_layer_names():
            for l_inf in inf_layers:
                if l_inf in l_model:
                    inf_layers_tmp.append(l_model)
        inf_layers = inf_layers_tmp

    tracin_inf_fn = TracInInfluenceTorch(
        model_instance=model,
        ckpts_file_paths=[model_ckpt_fp],
        batch_size=inf_batch_size,
        layers=inf_layers,
        fast_cp=inf_fact_cp,
    )

    if savedir is None:
        savedir = Path(
        "results", model_name, data_name, subset_id, "dirty", attack, inf_fn_name
        )
        print(f'savedir is automatically set to {savedir}')

    save_as_np(adv_pred_labels, savedir, 'adversarial_pred_labels.npy')
    save_as_np(adv_preds, savedir, 'adversarial_preds.npy')

    test_self_inf, adv_self_inf, train_test_inf, train_test_inf_adv = compute_influence(
        influence_fn=tracin_inf_fn,
        train_data=train_data,
        test_data=test_data,
        adv_data=adv_data,
        savedir=savedir
    )


if __name__ == "__main__":

    run_pipeline()


    # adv_ids = np.where(error_col == 1)[0]
    #
    # final_si, final_train_test_mat = self_inf.copy(), train_test_inf.copy()
    #
    # final_si[adv_ids] = self_inf_adv
    # final_train_test_mat[:, adv_ids] = train_test_inf_adv
    # average_precision_score(error_col, final_si)
    print()
