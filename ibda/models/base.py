from abc import ABC, abstractmethod

from torch import nn


class BaseModel(ABC, nn.Module):

    def __init__(self, model):
        super(BaseModel, self).__init__()
        self.model = model

    @abstractmethod
    def trainable_layer_names(self):
        return None

    @abstractmethod
    def get_model_instance(self):
        raise NotImplementedError("get_model_instance method must be implemented")

    def count_params(self, trainable=True):
        params = self.model.parameters()
        if trainable:
            params = filter(lambda p: p.requires_grad, params)
        total_params = sum(p.numel() for p in params)
        return total_params
