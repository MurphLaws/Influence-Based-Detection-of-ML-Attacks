# adapted from https://github.com/SJabin/NNIF/blob/main/src/nnif.py
import json
import os
import sys
import time
from pathlib import Path

from pkbar import pkbar

import torch
import numpy as np
import torch.autograd as autograd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from typing import Any, List

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import set_model_weights, predict, get_last_ckpt
from ibda.utils.config_manager import ConfigManager

from torchvision.models.feature_extraction import create_feature_extractor


from pydvl.influence.torch import ArnoldiInfluence
from pydvl.influence.torch.util import NestedTorchCatAggregator, TorchNumpyConverter
from pydvl.influence import SequentialInfluenceCalculator


M = 50

DATA_FOLDER = 'server_data'
RESULTS_FOLDER = 'results_server'

def get_trainable_parameters(
    model: torch.nn.Module,
    influence_layers: List[str] = None,
):
    layers_tmp = influence_layers.copy() if influence_layers is not None else None
    param_influence = []
    for name, param in model.named_parameters():
        if influence_layers is None:
            param_influence.append(param)
        else:
            for tl in influence_layers:
                if name.startswith(tl):
                    param_influence.append(param)
                    if tl in layers_tmp:
                        layers_tmp.remove(tl)
    if layers_tmp is not None and len(layers_tmp) > 0:
        raise ValueError(f"{layers_tmp} not found in model's layers")
    return param_influence


# k-nn ranks and distances
def calc_all_ranks_and_dists(features, knn):
    n_neighbors = knn.n_neighbors

    all_neighbor_ranks = -1 * np.ones(
        (len(features), 1, n_neighbors), dtype=np.int32
    )  # num_output
    all_neighbor_dists = -1 * np.ones(
        (len(features), 1, n_neighbors), dtype=np.float32
    )  # num_output

    all_neighbor_dists[:, 0], all_neighbor_ranks[:, 0] = knn.kneighbors(
        features, return_distance=True
    )
    del features
    return all_neighbor_ranks, all_neighbor_dists


def find_ranks(
    test_idx,
    sorted_influence_indices,
    all_neighbor_indices,
    all_neighbor_dists,
    mean=False,
):
    ni = all_neighbor_indices
    nd = all_neighbor_dists

    ranks = -1 * np.ones(
        (len(sorted_influence_indices)), dtype=np.int32
    )  # num_output = 1
    dists = -1 * np.ones(
        (len(sorted_influence_indices)), dtype=np.float32
    )  # num_output = 1

    for target_idx in range(len(sorted_influence_indices)):
        idx = sorted_influence_indices[target_idx]
        # print(ni[test_idx])
        # print(np.where(ni[test_idx, 0] == idx))
        loc_in_knn = np.where(ni[test_idx, 0] == idx)[0][0]
        # print("loc_in_knn:", idx, loc_in_knn)
        knn_dist = nd[test_idx, 0, loc_in_knn]
        # print("knn_dist:", knn_dist)
        ranks[target_idx] = loc_in_knn
        dists[target_idx] = knn_dist
    if mean:
        ranks_mean = np.mean(ranks, axis=1)
        dists_mean = np.mean(dists, axis=1)
        return ranks_mean, dists_mean

    return ranks, dists

def logistic_detector(X, y, random_seed):
    """ A Logistic Regression classifier, and return its accuracy. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

    # Build detector
    lr = LogisticRegression(max_iter=500000, random_state=random_seed).fit(X_train, y_train)
    # Evaluate detector
    y_pred = lr.predict(X_test)

    # AUC
    auc = roc_auc_score(y_test, y_pred)
    print('AUC: ', auc)

    return lr


def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact (ranked features) and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("positive artifacts: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("negative artifacts: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def emb_reshaping(embedding):
    if len(embedding.shape) == 4:
        embedding = embedding.view(embedding.shape[0], -1, embedding.shape[-1])
        embedding = torch.mean(embedding, axis=1)   # TODO check if this is correct
        print()
    elif len(embedding.shape) == 2:
        pass  # leave as is
    else:
        raise AssertionError('Expecting size of 2 or 4 but got {}'.format(embedding.shape))
    return embedding

def build_features(scores, all_neighbor_ranks, all_neighbor_dists):

    print('Extracting NNIF characteristics for max_indices={}'.format(M))
    ranks = -1 * np.ones((len(all_neighbor_ranks), 4, M))

    for i in range(len(all_neighbor_ranks)):

        sorted_indices = np.argsort(scores[i, :])

        ni = all_neighbor_ranks
        nd = all_neighbor_dists

        helpful_ranks, helpful_dists = find_ranks(i, sorted_indices[-M:][::-1], ni, nd)
        harmful_ranks, harmful_dists = find_ranks(i, sorted_indices[:M], ni, nd)
        helpful_ranks = np.array(helpful_ranks)
        helpful_dists = np.array(helpful_dists)
        harmful_ranks = np.array(harmful_ranks)
        harmful_dists = np.array(harmful_dists)

        ranks[i, 0, :] = helpful_ranks
        ranks[i, 1, :] = helpful_dists
        ranks[i, 2, :] = harmful_ranks
        ranks[i, 3, :] = harmful_dists

    characteristics = ranks.reshape((ranks.shape[0], -1))

    return characteristics

def run_nnif(train_data, val_data, test_data, adv_val_labels, adv_test_labels, model, feature_layer, batch_size=256, device='cpu'):

    model = model.to(device)

    tr_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    print('Calculating Influence')
    if_model = ArnoldiInfluence(
        model,
        torch.nn.CrossEntropyLoss(),
        hessian_regularization=0.0,
        rank_estimate=10,
        tol=1e-6,
    )

    if_model.fit(tr_dl)

    infl_calc = SequentialInfluenceCalculator(if_model)
    # Lazy object providing arrays batch-wise in a sequential manner
    lazy_influences = infl_calc.influences(val_dl, tr_dl)

    # Trigger computation and pull results to memory
    scores = lazy_influences.compute(aggregator=NestedTorchCatAggregator())

    if not isinstance(scores, np.ndarray):
        scores = scores.detach().numpy()

    #         for DkNN on final layer

    embedder = create_feature_extractor(model, [feature_layer])

    print("Loading train embeddings")
    train_embeds = embedder(train_data.tensors[0])[feature_layer]
    train_embeds = emb_reshaping(train_embeds).detach().numpy()

    print("Loading adv val embeddings")
    val_embeds = embedder(val_data.tensors[0])[feature_layer]
    val_embeds = emb_reshaping(val_embeds).detach().numpy()

    print("Loading adv val embeddings")
    test_embeds = embedder(test_data.tensors[0])[feature_layer]
    test_embeds = emb_reshaping(test_embeds).detach().numpy()

    # start KNN observation
    knn = {}
    layer = 1
    knn[layer] = NearestNeighbors(n_neighbors=len(train_embeds), p=2, n_jobs=os.cpu_count()-1, algorithm='brute')
    knn[layer].fit(train_embeds, train_data.tensors[1].numpy())
    del train_embeds

    # print("result_dir:", result_dir)
    print('predicting knn dist/indices for val data')
    all_neighbor_ranks, all_neighbor_dists = calc_all_ranks_and_dists(val_embeds, knn[layer])

    characteristics = build_features(scores, all_neighbor_dists=all_neighbor_dists, all_neighbor_ranks=all_neighbor_ranks)
    labels = adv_val_labels
    ## Build detector

    detector_model = logistic_detector(characteristics, labels, random_seed=0)  # acc

    # auc_test = roc_auc_score(, )

    print('predicting knn dist/indices for test data')
    te_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    lazy_influences = infl_calc.influences(te_dl, tr_dl)

    # Trigger computation and pull results to memory
    test_inf_scores = lazy_influences.compute(aggregator=NestedTorchCatAggregator())
    if not isinstance(test_inf_scores, np.ndarray):
        test_inf_scores = test_inf_scores.detach().numpy()
    all_neighbor_ranks, all_neighbor_dists = calc_all_ranks_and_dists(test_embeds, knn[layer])

    characteristics = build_features(test_inf_scores, all_neighbor_dists=all_neighbor_dists,
                                     all_neighbor_ranks=all_neighbor_ranks)
    test_preds = detector_model.predict(characteristics)

    print(roc_auc_score(adv_test_labels, test_preds))

    metrics = {'auc': roc_auc_score(adv_test_labels, test_preds),
               'avep': average_precision_score(adv_test_labels, test_preds)}

    return detector_model, if_model, metrics

def prepare_adv_data(model, adv_samples, clean_adv_labels):
    assert isinstance(adv_samples, TensorDataset)
    labels, _, _ = predict(model, adv_samples)
    if isinstance(adv_samples, torch.Tensor):
        adv_samples = adv_samples.detach().numpy()
    valid_positions = np.where(labels != clean_adv_labels)[0]
    return TensorDataset(*adv_samples[valid_positions]), valid_positions

def preprocess_data(data):
    assert isinstance(data, TensorDataset)
    X = data.tensors[0]
    channels = X.shape[1]
    X_aug = X.clone()
    if channels < 3:
        X_aug = torch.cat([X, X, X], dim=1)
    return TensorDataset(X_aug, data.tensors[1])


if __name__ == "__main__":

    subsets = set([f'subset_id{i}_r0.1' for i in range(5)])
    device = 'cpu'
    model_name = 'resnet20'

    for a in ['square']:
        for d in ['mnist', 'fmnist', 'cifar10']:
            print(a, d)
            for subset_focus in subsets:

                result_dir = f'{RESULTS_FOLDER}/{model_name}/{d}/{subset_focus}/dirty/{a}/nnif'
                Path(result_dir).mkdir(parents=True, exist_ok=True)

                secondary_subsets = subsets.difference(subset_focus)

                train_data_fp = f"{DATA_FOLDER}/clean/{d}/{subset_focus}/train.pt"
                clean_test_data_fp = f"{DATA_FOLDER}/clean/{d}/{subset_focus}/test.pt"
                dirty_test_data_dir = f"{DATA_FOLDER}/dirty/{a}/resnet20/{d}/{subset_focus}"

                train_data = torch.load(train_data_fp)
                train_data = preprocess_data(train_data)

                test_data = torch.load(clean_test_data_fp)
                test_data = preprocess_data(test_data)

                dirty_test_data = torch.load(Path(dirty_test_data_dir, 'test_dirty_y_pred.pt'))
                dirty_test_err_col = torch.load(Path(dirty_test_data_dir, 'is_adv.pt')).detach().numpy()
                dirty_test_data = preprocess_data(dirty_test_data)
                test_adv_ids = np.where(dirty_test_err_col == 1)[0]

                model_conf_fp = f"configs/resnet/resnet_{d}.json"

                model_ckpt_dir = f"{RESULTS_FOLDER}/resnet20/{d}/{subset_focus}/clean/ckpts"
                model_ckpt_fp = get_last_ckpt(model_ckpt_dir)

                conf_mger = ConfigManager(model_training_conf=model_conf_fp)
                model_seed = conf_mger.model_training.random_seed
                model_name = conf_mger.model_training.name

                num_classes = len(torch.unique(train_data.tensors[1]))
                input_shape = tuple(train_data.tensors[0].shape[1:])

                model = model_dispatcher[model_name](
                    num_classes=num_classes,
                    input_shape=input_shape,
                    seed=model_seed,
                    trainable_layers=conf_mger.model_training.trainable_layers,
                )

                model = set_model_weights(model, model_ckpt_fp)

                # labels, _, _ = predict(model, TensorDataset(*dirty_test_data[test_adv_ids]), device=device)
                # assert accuracy_score(test_data[test_adv_ids][1].detach().numpy(), labels) == 0

                val_X_data = []
                val_y_data = []
                adv_val_labels_arr = []
                for secondary_s in secondary_subsets:
                    clean_val_data_fp = f"{DATA_FOLDER}/clean/{d}/{secondary_s}/test.pt"
                    dirty_val_data_dir = f"{DATA_FOLDER}/dirty/{a}/resnet20/{d}/{secondary_s}"
                    clean_val = torch.load(clean_val_data_fp)
                    clean_val = preprocess_data(clean_val)

                    dirty_val_data = torch.load(Path(dirty_val_data_dir, 'test_dirty_y_pred.pt'))
                    dirty_val_data = preprocess_data(dirty_val_data)
                    dirty_val_err_col = torch.load(Path(dirty_val_data_dir, 'is_adv.pt')).detach().numpy()
                    adv_ids = np.where(dirty_val_err_col == 1)[0]

                    clean_val = TensorDataset(*clean_val[adv_ids])
                    dirty_val_data = TensorDataset(*dirty_val_data[adv_ids])

                    adv_val_data, adv_ids = prepare_adv_data(model, dirty_val_data, clean_val.tensors[1].detach().numpy())

                    val_X_final = torch.cat([clean_val[adv_ids][0], adv_val_data.tensors[0]])
                    val_y_final = torch.cat([clean_val[adv_ids][1], adv_val_data.tensors[1]])

                    adv_val_labels = np.zeros(2 * len(adv_ids), dtype=int)
                    adv_val_labels[int(len(adv_val_labels) / 2):] = 1

                    val_X_data.append(val_X_final)
                    val_y_data.append(val_y_final)
                    adv_val_labels_arr.extend(adv_val_labels.tolist())

                final_val_data = TensorDataset(torch.cat(val_X_data), torch.cat(val_y_data))

                trainable_layers = [tl[tl.index('.') + 1:] for tl in model.trainable_layer_names()]
                model = model.get_model_instance()

                exec_time = time.time()

                _, _, metrics = run_nnif(
                    train_data=train_data,
                    val_data=final_val_data,
                    test_data=dirty_test_data,
                    model=model,
                    adv_val_labels=adv_val_labels_arr,
                    adv_test_labels=dirty_test_err_col,
                    feature_layer=trainable_layers[-2],
                    batch_size=512,
                    device=device,
                )

                exec_time = time.time() - exec_time

                info = {
                    'exec_time': exec_time,
                    'nnif': metrics
                }

                with open(Path(result_dir, f'{subset_focus}.json'), 'w') as f:
                    json.dump(info, f, indent=2)


    # run nnif on test data
    # keep only the adversarial test data y_i != f(x_i)

    # test_Χ = adv_test_data.tensors[0].detach().numpy()
    # test_preds = detector_model.predict(test_Χ)
