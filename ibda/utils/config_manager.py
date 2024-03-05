import json
import os
from typing import List


class ConfigManager:
    def __init__(self, model_training_conf: str = None, inf_func_conf: str = None):

        model_training, inf_funcs = None, None

        if model_training_conf is not None:
            with open(model_training_conf, "r") as f:
                model_training = json.load(f)
        if inf_func_conf is not None:
            with open(inf_func_conf, "r") as f:
                inf_funcs = json.load(f)

        if model_training is not None:
            self.model_training = ConfigManager._ModelTraining(model_training)
        if inf_funcs is not None:
            self.inf_funcs = ConfigManager._InfluenceFunctions(inf_funcs)

    class _ModelTraining:
        __NAME = "name"
        __RANDOM_SEED = "random_seed"
        __TRAINABLE_LAYERS = "trainable_layers"
        __REG_STRENGTH = "regularization_strength"
        __LEARNING_RATE = "learning_rate"
        __EPOCHS = "epochs"
        __BATCH_SIZE = "batch_size"

        def __init__(self, data: dict):
            self.data = data

        @property
        def name(self) -> str:
            return self.data[ConfigManager._ModelTraining.__NAME]

        @property
        def random_seed(self):
            return self.data.get(ConfigManager._ModelTraining.__RANDOM_SEED, None)

        @property
        def trainable_layers(self):
            return self.data.get(ConfigManager._ModelTraining.__TRAINABLE_LAYERS, [])

        @property
        def learning_rate(self):
            return self.data.get(ConfigManager._ModelTraining.__LEARNING_RATE, 1e-2)

        @property
        def regularization_strength(self):
            return self.data.get(ConfigManager._ModelTraining.__REG_STRENGTH, 0)

        @property
        def epochs(self):
            return self.data.get(ConfigManager._ModelTraining.__EPOCHS, 100)

        @property
        def batch_size(self):
            return self.data.get(ConfigManager._ModelTraining.__BATCH_SIZE, 128)

    class _InfluenceFunctions:
        __BATCH_SIZE = "batch_size"
        __FAST_CP = "fast_cp"
        __INFLUENCE_LAYERS = "influence_layers"

        def __init__(self, data: dict):
            self.data = data

        @property
        def fast_cp(self):
            return self.data.get(ConfigManager._InfluenceFunctions.__FAST_CP, False)

        @property
        def batch_size(self):
            return self.data.get(ConfigManager._InfluenceFunctions.__BATCH_SIZE, 4096)

        @property
        def influence_layers(self):
            return self.data.get(ConfigManager._InfluenceFunctions.__INFLUENCE_LAYERS, None)