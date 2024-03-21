# adapted from https://github.com/SJabin/NNIF/blob/main/src/nnif.py

import os
import sys
import time
from pathlib import Path

from pkbar import pkbar

import torch
import numpy as np
import torch.autograd as autograd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from typing import Any, List

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import set_model_weights, predict
from ibda.utils.config_manager import ConfigManager

from torchvision.models.feature_extraction import create_feature_extractor


M = 50

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


# influence score
def get_ihvp_score(
    train_data: TensorDataset,
    test_data: TensorDataset,
    model: torch.nn.Module,
    batch_size: int,
    loss: Any = torch.nn.CrossEntropyLoss(),
    influence_layers: List[str] = None,
    workers: int = None,
    device: str = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    param_influence = get_trainable_parameters(
        model=model, influence_layers=influence_layers
    )

    train_grads_mat = []

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    kbar = pkbar.Pbar(
        name=f'Calculating Train Gradients',
        target=len(train_dataloader),
    )

    for batch_idx, (X, y) in enumerate(train_dataloader):

        kbar.update(batch_idx)

        model.train()

        X = X.to(device)
        y = y.to(device)

        ######## L_TRAIN GRADIENT ########
        model.zero_grad()
        logits = model(X)
        for i, l in enumerate(logits):
            train_loss = loss(l, y[i])
            train_grads = autograd.grad(train_loss, param_influence, retain_graph=True)
            train_grads_mat.append(train_grads)

    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()

    ######## L_TEST GRADIENT ########

    lissa_repeat = 1

    influences = []

    train_dataloader = DataLoader(
        train_data, shuffle=False, batch_size=batch_size, num_workers=workers
    )

    kbar = pkbar.Pbar(
        name=f'Influence Calculation of {len(test_data)} examples',
        target=len(test_data),
    )

    for test_id, (x_t, y_t) in enumerate(test_data):

        kbar.update(test_id)

        x_t = x_t.to(device)
        y_t = y_t.to(device)

        if len(x_t.shape) == 3:
            x_t = torch.unsqueeze(x_t, dim=0)
            y_t = torch.unsqueeze(y_t, dim=0)

        model.zero_grad()
        logits = model(x_t)
        test_loss = loss(logits, y_t)
        test_grads = autograd.grad(test_loss, param_influence)

        ######## IHVP ########

        inverse_hvp = get_inverse_hvp_lissa(
            v=test_grads,
            model=model,
            param_influence=param_influence,
            train_data=train_data,
            lissa_repeat=lissa_repeat,
            recursion_depth=None,
            device=device,
        )

        print(f'\nComputing Train to Test Influence for {test_id}')

        val = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()

        if np.isnan(val):
            print("it's nan!")
            sys.exit()

        influences.append(val)

    torch.cuda.empty_cache()

    return np.array(influences)


######## LiSSA ########
def gather_flat_grad(grads):
    views = []
    for p in grads:
        p.data[p.data == float("inf")] = 0.0
        p.data = torch.nan_to_num(p.data, nan=0.0)
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv


def get_inverse_hvp_lissa(
    v,
    model,
    device,
    param_influence,
    train_data,
    lissa_repeat,
    recursion_depth=None,
    batch_size=256,
    scale=1e4,
    damping=0,
    loss: Any = torch.nn.CrossEntropyLoss(),
    workers=os.cpu_count() - 1,
):
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=workers
    )

    if recursion_depth is None:
        recursion_depth = len(train_loader)

    ihvp = None
    model.train()

    # kbar = pkbar.Kbar(
    #     target=recursion_depth,
    #     epoch=0,
    #     num_epochs=lissa_repeat,
    #     always_stateful=True,
    # )

    for i in range(lissa_repeat):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)

        # print("shape of test_grads:", cur_estimate)
        for j in range(recursion_depth):
            # kbar.update(j)
            try:
                X, y = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                X, y = next(lissa_data_iterator)

            X = X.to(device)
            y = y.to(device)

            model.zero_grad()
            logits = model(X)
            train_loss = loss(logits, y)
            hvp = hv(train_loss, param_influence, cur_estimate)
            del X, y

            cur_estimate = [
                _a + (1 - damping) * _b - _c / scale
                for _a, _b, _c in zip(v, cur_estimate, hvp)
            ]

        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= lissa_repeat
    return return_ihvp


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

    sorted_indices = np.argsort(scores)

    for i in range(len(all_neighbor_ranks)):

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

def run_nnif(train_data, val_data, test_data, adv_val_labels, model, feature_layer, influence_layers, batch_size=256, device='cpu'):

    scores = get_ihvp_score(
        train_data=train_data,
        test_data=val_data,
        model=model,
        batch_size=batch_size,
        device=device,
        influence_layers=influence_layers,
    )


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

    # characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
    characteristics = build_features(scores, all_neighbor_dists=all_neighbor_dists, all_neighbor_ranks=all_neighbor_ranks)
    labels = adv_val_labels
    ## Build detector

    detector_model = logistic_detector(characteristics, labels, random_seed=0)  # acc

    # auc_test = roc_auc_score(, )

    return detector_model

def prepare_adv_data(model, adv_samples, clean_samples, ids_to_replace, assert_adv_acc=True):
    assert isinstance(clean_samples, TensorDataset)
    final_X = clean_samples.tensors[0].numpy()
    final_y = clean_samples.tensors[1].numpy()
    labels, _, _ = predict(model, TensorDataset(adv_samples, clean_samples[ids_to_replace][1]))
    if assert_adv_acc:
        assert accuracy_score(clean_samples[ids_to_replace][1].numpy(), labels) == 0
    if isinstance(adv_samples, torch.Tensor):
        adv_samples = adv_samples.numpy()
    final_X[ids_to_replace] = adv_samples
    final_y[ids_to_replace] = labels
    return TensorDataset(torch.tensor(final_X), torch.tensor(final_y))

def preprocess_data(data):
    assert isinstance(data, TensorDataset)
    X = data.tensors[0]
    channels = X.shape[1]
    X_aug = X.clone()
    if channels < 3:
        X_aug = torch.cat([X, X, X], dim=1)
    return TensorDataset(X_aug, data.tensors[1])


if __name__ == "__main__":

    train_val_subset = 'subset_id0_r0.1'
    test_subset = 'subset_id1_r0.1'

    train_data_fp = f"data/clean/mnist/{train_val_subset}/train.pt"
    val_data_fp = f"data/clean/mnist/{train_val_subset}/test.pt"
    adv_val_dir = f"data/dirty/fgsm/resnet20/mnist/{train_val_subset}"

    test_data_fp = f"data/clean/mnist/{test_subset}/test.pt"
    adv_test_dir = f"data/dirty/fgsm/resnet20/mnist/{test_subset}"

    model_conf_fp = "configs/resnet/resnet_mnist.json"

    model_ckpt_fp = f"results/resnet20/mnist/{train_val_subset}/clean/ckpts/checkpoint-7.pt"

    device = 'cuda'

    conf_mger = ConfigManager(model_training_conf=model_conf_fp)
    model_seed = conf_mger.model_training.random_seed
    model_name = conf_mger.model_training.name

    train_data = torch.load(train_data_fp)
    train_data = preprocess_data(train_data)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = tuple(train_data.tensors[0].shape[1:])

    model = model_dispatcher[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        seed=model_seed,
        trainable_layers=conf_mger.model_training.trainable_layers,
    )

    model = set_model_weights(model, model_ckpt_fp)

    # Load rest data

    val_data = torch.load(val_data_fp)
    adv_val_X = torch.tensor(torch.load(Path(adv_val_dir, 'adv.pt')))
    with open(Path(adv_val_dir, 'adv_ids.npy'), 'rb') as f:
        adv_val_ids = np.load(f)
        adv_val_ids = np.where(adv_val_ids == 1)[0]

    clean_val_data = TensorDataset(*val_data[adv_val_ids])
    clean_val_data = preprocess_data(clean_val_data)

    val_data = prepare_adv_data(model=model, adv_samples=adv_val_X, clean_samples=val_data,
                                ids_to_replace=adv_val_ids)
    adv_val_data = preprocess_data(TensorDataset(*val_data[adv_val_ids]))

    val_X_final = torch.cat([clean_val_data.tensors[0], adv_val_data.tensors[0]])
    val_y_final = torch.cat([clean_val_data.tensors[1], adv_val_data.tensors[1]])
    final_val_data = TensorDataset(val_X_final, val_y_final)

    adv_val_labels = np.zeros(2 * len(adv_val_data), dtype=int)
    adv_val_labels[int(len(adv_val_labels)/2):] = 1

    test_data = torch.load(test_data_fp)
    adv_test_X = torch.tensor(torch.load(Path(adv_test_dir, 'adv.pt')))
    with open(Path(adv_test_dir, 'adv_ids.npy'), 'rb') as f:
        adv_test_ids = np.load(f)
        adv_test_ids = np.where(adv_test_ids == 1)[0]

    adv_test_data = prepare_adv_data(model=model, adv_samples=adv_test_X, clean_samples=test_data,
                                     ids_to_replace=adv_test_ids, assert_adv_acc=False)
    adv_test_data = preprocess_data(adv_test_data)


    trainable_layers = [tl[tl.index('.')+1:] for tl in model.trainable_layer_names()]
    model = model.get_model_instance()

    detector_model = run_nnif(
        train_data=train_data,
        val_data=final_val_data,
        test_data=adv_test_data,
        model=model,
        adv_val_labels = adv_val_labels,
        feature_layer=trainable_layers[-2],
        influence_layers=trainable_layers,
        batch_size=512,
        device=device,
    )
    print()

    # run nnif on test data
    # keep only the adversarial test data y_i != f(x_i)

    test_Χ = adv_test_data.tensors[0].detach().numpy()
    test_preds = detector_model.predict(test_Χ)
