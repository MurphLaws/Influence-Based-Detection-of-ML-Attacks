import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from click.testing import CliRunner
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset

from ibda.influence_functions.dynamic import TracInInfluenceTorch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import get_last_ckpt, predict, set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_np

from ibda.signals import InfluenceErrorSignals

attack_choices = ["square"] #["fgsm", "cw", "bound_attack" "jsma", "auto_attack"]
datasets = ['mnist', 'fmnist', 'cifar10']

compute_sigs = ['SI', 'MNI', 'ACNI', 'ACNI^*', 'CPI']

RESULTS_FOLDER = 'results_server'
DATA_FOLDER = 'server_data'

@click.command()
@click.option("--attack", type=click.Choice(attack_choices), required=True)
@click.option("--data_name", type=click.STRING, required=True)
@click.option("--model_name", type=click.STRING, required=True)
@click.option("--inf_fn_name", type=click.STRING, required=True)
@click.option("--subset_id", type=click.STRING, required=True)
@click.option("--model_conf", type=click.Path(exists=True), required=True)
@click.option("--inf_fn_conf", type=click.Path(exists=True), required=True)
@click.option("--ckpt_fname", type=click.STRING, default=None)
@click.option("--train_data_fp", type=click.STRING, default=None)
@click.option("--clean_test_data_fp", type=click.STRING, default=None)
@click.option("--dirty_test_data_dir", type=click.Path(exists=True), default=None)
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
    train_data_fp=None,
    clean_test_data_fp=None,
    dirty_test_data_dir=None,
    model_ckpt_fp=None,
    device=None,
    savedir=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_ckpt_fp is None:
        default_ckpt_dir = f"{RESULTS_FOLDER}/{model_name}/{data_name}/{subset_id}/clean/ckpts/"
        print(f"Assigning automatically the checkpoint dir to {default_ckpt_dir}")
        if ckpt_fname is None:
            model_ckpt_fp = get_last_ckpt(default_ckpt_dir)
            print(f"As --model_ckpt_fp is None, the last ckpt is used: {model_ckpt_fp}")
        else:
            ckpt_fname = ckpt_fname + '.pt' if not ckpt_fname.endswith('.pt') else ckpt_fname
            model_ckpt_fp = Path(default_ckpt_dir, ckpt_fname)

    if train_data_fp is None:
        train_data_fp = f"{DATA_FOLDER}/clean/{data_name}/{subset_id}/train.pt"
        print(f"Assigning automatically the train data filepath to {train_data_fp}")

    if clean_test_data_fp is None:
        clean_test_data_fp = f"{DATA_FOLDER}/clean/{data_name}/{subset_id}/test.pt"
        print(f"Assigning automatically the train data filepath to {clean_test_data_fp}")

    if dirty_test_data_dir is None:
        dirty_test_data_dir = (
            f"{DATA_FOLDER}/dirty/{attack}/{model_name}/{data_name}/{subset_id}"
        )
        print(f"Assigning automatically the dirty data dir to {dirty_test_data_dir}")

    print(f"Running {attack} on {model_ckpt_fp}")

    conf_mger = ConfigManager(model_training_conf=model_conf, inf_func_conf=inf_fn_conf)

    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name
    inf_batch_size = conf_mger.inf_funcs.batch_size
    inf_layers = conf_mger.inf_funcs.influence_layers
    inf_fact_cp = conf_mger.inf_funcs.fast_cp

    # Initialization

    train_data = torch.load(train_data_fp)
    clean_test_data = torch.load(clean_test_data_fp)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    model = set_model_weights(model, model_ckpt_fp)
    model = model.to(device)

    dirty_test_data_y_pred = torch.load(Path(dirty_test_data_dir, 'test_dirty_y_pred.pt'))
    error_col = torch.load(Path(dirty_test_data_dir, 'is_adv.pt')).detach().numpy()

    adv_ids = np.where(error_col == 1)[0]

    y_train = train_data.tensors[1].detach().numpy()
    y_test_clean = clean_test_data.tensors[1].detach().numpy()
    y_pred_test_dirty = dirty_test_data_y_pred.tensors[1].detach().numpy()

    labels, _, _ = predict(model, clean_test_data, device=device)

    print(f'Dirty Test accuracy {accuracy_score(y_test_clean, y_pred_test_dirty)}')
    print(f'Clean Test accuracy {accuracy_score(y_test_clean, labels)}')
    mse = mse_loss(clean_test_data[adv_ids][0], dirty_test_data_y_pred[adv_ids][0])
    print(a, d, 'mse', mse)


    # labels, _, _ = predict(model, dirty_test_data_y_pred, device=device)
    # assert np.abs(accuracy_score(y_test_clean, y_pred_test_dirty) - accuracy_score(y_test_clean, labels)) < 1e-2
    
    assert accuracy_score(y_test_clean[adv_ids], y_pred_test_dirty[adv_ids]) == 0

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
        RESULTS_FOLDER, model_name, data_name, subset_id, "dirty", attack, inf_fn_name
        )
        print(f'savedir is automatically set to {savedir}')

    self_inf_time = time.time()
    test_self_inf = tracin_inf_fn.compute_self_influence(dataset=dirty_test_data_y_pred)
    self_inf_time = time.time() - self_inf_time

    train_test_inf_time = time.time()
    train_test_inf = tracin_inf_fn.compute_train_to_test_influence(
        train_set=train_data,
        test_set=dirty_test_data_y_pred,
    )
    train_test_inf_time = time.time() - train_test_inf_time

    if savedir is not None:
        save_as_np(test_self_inf, savedir, 'test_self_inf.npy')
        save_as_np(train_test_inf, savedir, 'train_test_inf_adv.npy')

    ies = InfluenceErrorSignals(train_test_inf_mat=train_test_inf, y_train=y_train,
                                y_test=y_pred_test_dirty,
                                 compute_test_influence=True
                                )
    joint_sigs = ies.compute_signals(verbose=False)
    joint_sigs['SI'] = test_self_inf

    joint_sigs.to_csv(Path(savedir, 'signals.csv'), index=False)

    res_dict = {}
    for sig in compute_sigs:
        auc = roc_auc_score(error_col, joint_sigs[sig])
        avep = average_precision_score(error_col, joint_sigs[sig])
        print(sig, avep, auc)
        res_dict[sig] = {'avep': avep, 'auc': auc}
    res_dict['self_inf_time'] = self_inf_time
    res_dict['train_test_inf_time'] = train_test_inf_time
    res_dict['mse'] = float(mse.cpu().numpy())

    with open(Path(savedir, f'{subset_id}.json'), 'w') as f:
        json.dump(res_dict, f, indent=2)


if __name__ == "__main__":

    model_name = 'resnet20'
    inf_fn_name = 'tracin'
    inf_fn_conf = 'configs/resnet/tracin_resnet.json'
    device = 'cpu'

    subset_ids = [f'subset_id{i}_r0.1' for i in range(5)]

    for a in attack_choices:
        for d in datasets:
            for subset_id in subset_ids:
                print(a,d, subset_id)
                model_conf = f'configs/resnet/resnet_{d}.json'
                runner = CliRunner()
                result = runner.invoke(run_pipeline, ['--attack', a,
                                                     '--data_name', d,
                                                     '--model_name', model_name,
                                                     '--inf_fn_name', inf_fn_name,
                                                     '--subset_id', subset_id,
                                                     '--model_conf', model_conf,
                                                     '--inf_fn_conf', inf_fn_conf,
                                                     '--device', device,
                                                      ])
                if result.exception:
                    raise result.exception
                print(result.output)

