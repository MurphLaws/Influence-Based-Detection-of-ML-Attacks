from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import TensorDataset

from ibda.influence_functions.tracin_torch import TracInInfluenceTorch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import predict
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_np


def run_pipeline():
    pass


if __name__ == "__main__":
    attack_name = "bound_attack"
    dirty_data_dir = f"data/dirty/{attack_name}/resnet20/cifar10/subset_id0_r0.1"
    train_data_fp = "data/clean/cifar10/subset_id0_r0.1/train.pt"
    test_data_fp = "data/clean/cifar10/subset_id0_r0.1/test.pt"
    model_ckpt = "results/resnet20/cifar10/clean/ckpts/checkpoint-7.pt"
    model_conf = "configs/resnet/resnet_cifar10.json"
    inf_fn_conf = "configs/resnet/tracin_resnet.json"
    data_name = "cifar10"
    model_name = "resnet20"
    inf_fn_name = "tracin"
    subset_id = "subset_id0_r0.1"
    device = "cpu"

    # Paths and Configs Init

    # model_conf = f"configs/{model_name}/{model_name}_{data_name}.json"
    # inf_fn_conf = f"configs/{model_name}/{inf_fn_name}_{data_name}.json"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_ckpt is None:
        pass  # if model_ckpt is None we load the last checkpoint

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

    with open(Path(dirty_data_dir, "adv_ids.npy"), "rb") as f:
        error_col = np.load(f)

    adv_samples = torch.tensor(torch.load(Path(dirty_data_dir, "adv.pt")))
    adv_corr_labels = torch.tensor(
        test_data.tensors[1].cpu().numpy()[np.where(error_col == 1)[0]]
    )
    adv_pred_labels, _, _ = predict(
        model, TensorDataset(adv_samples, adv_corr_labels), device=device
    )

    adv_data = TensorDataset(adv_samples, torch.tensor(adv_pred_labels))

    # Compute Influence

    tracin_inf_fn = TracInInfluenceTorch(model_instance=model)

    if inf_layers is None:
        inf_layers = model.trainable_layer_names()
    else:
        inf_layers_tmp = []
        for l_model in model.trainable_layer_names():
            for l_inf in inf_layers:
                if l_inf in l_model:
                    inf_layers_tmp.append(l_model)
        inf_layers = inf_layers_tmp

    self_inf = tracin_inf_fn.compute_self_influence(
        dataset=test_data,
        ckpts_file_paths=[model_ckpt],
        batch_size=inf_batch_size,
        layers=inf_layers,
        fast_cp=inf_fact_cp,
    )

    self_inf_adv = tracin_inf_fn.compute_self_influence(
        dataset=adv_data,
        ckpts_file_paths=[model_ckpt],
        batch_size=inf_batch_size,
        layers=inf_layers,
        fast_cp=inf_fact_cp,
    )

    train_test_inf = tracin_inf_fn.compute_train_to_test_influence(
        train_set=train_data,
        test_set=test_data,
        ckpts_file_paths=[model_ckpt],
        batch_size=inf_batch_size,
        layers=inf_layers,
        fast_cp=inf_fact_cp,
    )

    train_test_inf_adv = tracin_inf_fn.compute_train_to_test_influence(
        train_set=train_data,
        test_set=adv_data,
        ckpts_file_paths=[model_ckpt],
        batch_size=inf_batch_size,
        layers=inf_layers,
        fast_cp=inf_fact_cp,
    )

    savedir = Path(
        "results", model_name, data_name, subset_id, "dirty", attack_name, inf_fn_name
    )

    save_as_np(self_inf, savedir, "self_inf.npy")
    save_as_np(self_inf, savedir, "self_inf_adv.npy")
    save_as_np(train_test_inf, savedir, "train_test_inf.npy")
    save_as_np(train_test_inf, savedir, "train_test_inf_adv.npy")

    adv_ids = np.where(error_col == 1)[0]

    final_si, final_train_test_mat = self_inf.copy(), train_test_inf.copy()

    final_si[adv_ids] = self_inf_adv
    final_train_test_mat[:, adv_ids] = train_test_inf_adv
    average_precision_score(error_col, final_si)
    print()
