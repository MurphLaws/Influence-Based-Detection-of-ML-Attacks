from typing import List

import torch
from torch import nn

from ..base import BaseModel


class Resnet20Model(BaseModel):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = None,
        seed=None,
        trainable_layers: List[str] = None,
    ):
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        )
        super(Resnet20Model, self).__init__(model)
        self.__trainable_layers = None

        self.num_classes = num_classes
        self.random_state = seed
        self.freeze_layers()
        self.set_classification_layer()
        self.set_trainable_layers(layers=trainable_layers)

    def forward(self, x):
        channels = x.shape[1]
        if channels < 3:
            x = torch.cat([x, x, x], dim=1)
        return self.model(x)

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_trainable_layers(self, layers: List[str]):
        if layers is None:
            return self.__trainable_layers

        for name, param in self.model.named_parameters():
            for layer in layers:
                if layer in name:
                    param.requires_grad = True
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers = layers + self.__trainable_layers

    def set_classification_layer(self):
        num_ftrs = self.model.fc.in_features
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers.append("fc")

    def trainable_layer_names(self):
        layers = []
        for layer, _ in self.named_modules():
            for tl in self.__trainable_layers:
                if layer.endswith(tl):
                    layers.append(layer)
        return layers

    def get_model_instance(self):
        return self.model
