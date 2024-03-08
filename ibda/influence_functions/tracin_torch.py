from typing import Any, List

import numpy as np
import torch
from captum.influence import TracInCP, TracInCPFast
from torch import nn
from torch.utils.data import TensorDataset as TD

from ..models.base import BaseModel


class TracInInfluenceTorch:
    def __init__(
        self,
        model_instance: BaseModel,
        loss_fn: Any = nn.CrossEntropyLoss(),
    ):
        self.model_instance = model_instance
        self.loss_fn = loss_fn

    def _init_tracin_cp(
        self,
        dataset: TD,
        ckpts_file_paths: List[str],
        batch_size: int,
        layers: List[str],
        fast_cp: bool,
    ):
        if fast_cp:
            tracin_cp = TracInCPFast(
                model=self.model_instance,
                train_dataset=dataset,
                final_fc_layer=layers[-1],
                checkpoints=ckpts_file_paths,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
            )
        else:
            tracin_cp = TracInCP(
                model=self.model_instance,
                train_dataset=dataset,
                checkpoints=ckpts_file_paths,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
                sample_wise_grads_per_batch=True,
                layers=layers,
            )
        return tracin_cp

    def compute_train_to_test_influence(
        self,
        train_set: TD,
        test_set: TD,
        ckpts_file_paths: List[str],
        batch_size: int = 3500,
        layers: List[str] = None,
        fast_cp: bool = False,
    ) -> np.ndarray:
        tracin_cp = self._init_tracin_cp(
            dataset=train_set,
            ckpts_file_paths=ckpts_file_paths,
            batch_size=batch_size,
            layers=layers,
            fast_cp=fast_cp,
        )
        test_examples_features = torch.stack(
            [test_set[i][0] for i in range(len(test_set))]
        )
        test_examples_true_labels = torch.Tensor(
            [test_set[i][1] for i in range(len(test_set))]
        ).long()

        train_to_test_influence = tracin_cp.influence(
            (test_examples_features, test_examples_true_labels), show_progress=True
        )

        return np.array(train_to_test_influence).transpose()

    def compute_self_influence(
        self,
        dataset: TD,
        ckpts_file_paths: List[str],
        batch_size: int = 3500,
        layers: List[str] = None,
        fast_cp: bool = False,
    ) -> np.ndarray:
        print("Computing Self Influence")
        tracin_cp = self._init_tracin_cp(
            dataset=dataset,
            ckpts_file_paths=ckpts_file_paths,
            batch_size=batch_size,
            layers=layers,
            fast_cp=fast_cp,
        )
        self_influence = tracin_cp.self_influence(show_progress=True)
        return self_influence.cpu().numpy()

    @staticmethod
    def checkpoints_load_func(model: nn.Module, path: str):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["learning_rate"]
