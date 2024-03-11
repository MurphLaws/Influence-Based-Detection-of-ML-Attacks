from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import TensorDataset as TD


class BaseInfluenceEstimator(ABC):

    @abstractmethod
    def compute_train_to_test_influence(
        self,
        train_set: TD,
        test_set: TD,
    ) -> np.ndarray:
        raise NotImplementedError(
            "compute_train_to_test_influence method must implemented"
        )

    @abstractmethod
    def compute_self_influence(self, dataset: TD) -> np.ndarray:
        raise NotImplementedError("compute_self_influence method must implemented")
