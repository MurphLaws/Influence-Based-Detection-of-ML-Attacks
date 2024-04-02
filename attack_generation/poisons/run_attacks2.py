from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from art.attacks import PoisoningAttack
from art.attacks.poisoning import (FeatureCollisionAttack,
                                   PoisoningAttackCleanLabelBackdoor)
from art.estimators.classification import PyTorchClassifier
from PIL import Image
from torch._C import device
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import TensorDataset as TD

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.utils.writers import save_as_json, save_as_np


class Attack:

    def __init__(
        self,
        data_name,
        num_poisons,
        num_targets,
        seed,
        device,
        dir_suffix,
        model_ckpt_fp,
        train_data_fp,
        test_data_fp,
        model_conf_fp,
    ):

        self.num_poisons = num_poisons
        self.num_targets = num_targets
        self.seed = seed
        self.data_name = data_name
        self.device = device
        self.train_data = torch.load(train_data_fp)
        self.test_data = torch.load(test_data_fp)
        self.num_classes = len(torch.unique(self.train_data.tensors[1]))
        self.input_shape = tuple(self.train_data.tensors[0].shape[1:])
        self.train_data_fp = train_data_fp

        if data_name == "mnist" or data_name == "fmnist":
            repeated_training_images = []
            repeated_training_labels = []
            repeated_test_images = []
            repeated_test_labels = []

            for image, label in self.train_data:
                image_3_channels = image.repeat(3, 1, 1)
                repeated_training_images.append(image_3_channels)
                repeated_training_labels.append(label)

            for image, label in self.test_data:
                image_3_channels = image.repeat(3, 1, 1)
                repeated_test_images.append(image_3_channels)
                repeated_test_labels.append(label)

            repeated_training_images = torch.stack(repeated_training_images)
            repeated_training_labels = torch.tensor(repeated_training_labels)

            repeated_test_images = torch.stack(repeated_test_images)
            repeated_test_labels = torch.tensor(repeated_test_labels)

            self.train_data = TD(repeated_training_images, repeated_training_labels)
            self.test_data = TD(repeated_test_images, repeated_test_labels)

        self.conf_mger = ConfigManager(model_training_conf=model_conf_fp)
        self.model_seed = self.conf_mger.model_training.random_seed
        self.model_name = self.conf_mger.model_training.name
        self.model = model_dispatcher[self.model_name](
            num_classes=self.num_classes,
            input_shape=self.input_shape,
            seed=self.model_seed,
            trainable_layers=self.conf_mger.model_training.trainable_layers,
        )
        self.dir_suffix = dir_suffix
        self.model_ckpt_fp = model_ckpt_fp
        self.loss = nn.CrossEntropyLoss()

    def initial_fine_tune(self):
        if self.model_ckpt_fp is None:
            model_savedir = Path(
                f"results/{self.model_name}/{self.data_name}/{self.dir_suffix}/clean/ckpts"
            )
            model_savedir.mkdir(parents=True, exist_ok=True)

            # NOTE: Take a look at the learning rate
            self.model, info = train(
                model=self.model,
                train_data=self.train_data,
                test_data=self.test_data,
                epochs=self.conf_mger.model_training.epochs,
                batch_size=self.conf_mger.model_training.batch_size,
                learning_rate=self.conf_mger.model_training.learning_rate,
                reg_strength=self.conf_mger.model_training.regularization_strength,
                seed=self.conf_mger.model_training.random_seed,
                device=self.device,
                save_dir=model_savedir,
            )
            save_as_json(info, savedir=model_savedir, fname="info.json")
        else:
            self.model = set_model_weights(self.model, self.model_ckpt_fp)
            self.model.eval()

    def set_classifier(self):
        self.classifier = PyTorchClassifier(
            model=self.model.get_model_instance(),
            clip_values=(
                np.min(self.train_data.tensors[0].numpy()),
                np.max(self.train_data.tensors[0].numpy()),
            ),
            loss=self.loss,
            optimizer=optim.Adam(self.model.parameters(), lr=0.0001),
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            device_type=self.device,
        )

    # DONE: This function is working as intended
    def draw_bases_and_targets(self):

        test_x = self.test_data.tensors[0]
        test_y = self.test_data.tensors[1]

        def shuffle_array(arr):
            iter_seed = 0

            def same_on_index(arr1, arr2):
                for i in range(len(arr1)):
                    if arr1[i] == arr2[i]:
                        return True

            shuffled = arr.copy()
            while same_on_index(arr, shuffled):
                np.random.seed(iter_seed)
                np.random.shuffle(shuffled)
                iter_seed += 1
            return shuffled

        # TODO: This works for experiments with more than 10 classes.
        # Maybe gonna need in the future. Ideally, we won't do that in this project
        #        if self.num_targets > self.num_classes:
        #            np.random.seed(self.seed)
        #            self.target_classes = np.random.randint(
        #                0, self.num_classes, self.num_targets
        #            )
        #        else:
        #            self.target_classes = np.arange(self.num_targets)

        self.target_classes = np.arange(self.num_targets)

        if self.num_targets == 1:
            np.random.seed(self.seed)
            self.target_classes = np.random.randint(0, self.num_classes, 1)
            self.base_classes = np.random.randint(0, self.num_classes, 1)
            iter_seed = 0
            while self.base_classes == self.target_classes:
                np.random.seed(iter_seed)
                self.base_classes = np.random.randint(0, self.num_classes, 1)
                iter_seed += 1
        else:
            self.base_classes = shuffle_array(self.target_classes)

        correctly_classified = []
        all_predictions = self.classifier.predict(test_x)  # type: ignore
        for i in range(len(test_y)):
            if np.argmax(all_predictions[i]) == test_y[i]:
                correctly_classified.append(i)

        # NOTE: TARGET INSTANCES
        self.target_indices = []
        for i in self.target_classes:  # type: ignore
            class_index = np.where(test_y == i)[0]
            correct_within_class = [x for x in class_index if x in correctly_classified]
            correct_within_class_predictions = all_predictions[correct_within_class]
            correct_within_class_predictions = [
                F.softmax(torch.tensor(x), dim=0)
                for x in correct_within_class_predictions
            ]
            correct_within_class_predictions = np.array(
                [element[i] for element in correct_within_class_predictions]
            )
            missclassification_probabilty = 1 - correct_within_class_predictions
            missclassification_probabilty = missclassification_probabilty / np.sum(
                missclassification_probabilty
            )
            np.random.seed(self.seed)
            selected_targets = np.random.choice(
                correct_within_class,
                size=1,
                replace=False,
                p=missclassification_probabilty,
            )
            self.target_indices.append(selected_targets)

        # NOTE: BASE INSTANCES
        self.base_indices = []
        for i in self.base_classes:
            class_index = np.where(test_y == i)[0]
            correct_within_class = [x for x in class_index if x in correctly_classified]
            correct_within_class_predictions = all_predictions[correct_within_class]
            correct_within_class_predictions = [
                F.softmax(torch.tensor(x), dim=0)
                for x in correct_within_class_predictions
            ]
            correct_within_class_predictions = np.array(
                [element[i] for element in correct_within_class_predictions]
            )
            correct_classification_probability = (
                correct_within_class_predictions
                / np.sum(correct_within_class_predictions)
            )
            if self.num_poisons > len(correct_within_class):
                raise ValueError(
                    "Number of poisons is greater than the number of available base instances"
                )
            np.random.seed(self.seed)
            selected_bases = np.random.choice(
                correct_within_class,
                size=self.num_poisons,
                replace=False,
                p=correct_classification_probability,
            )
            self.base_indices.append(selected_bases)

        target_and_bases_dicts = []

        for target_id, base_ids, target_class, base_class in zip(
            self.target_indices,
            self.base_indices,
            self.target_classes,
            self.base_classes,
        ):
            target_and_bases_dicts.append(
                {
                    "target_id": target_id.astype(int).tolist(),
                    "target_class": int(target_class),
                    "base_ids": base_ids.astype(int).tolist(),
                    "base_class": int(base_class),
                }
            )

        self.target_bases_dict = target_and_bases_dicts
        target_bases_savedir = (
            f"results/{self.model_name}/{self.data_name}/{self.dir_suffix}/poisoned"
        )
        save_as_json(
            self.target_bases_dict,
            savedir=target_bases_savedir,
            fname="target_bases.json",
            indent=4,
        )

    @staticmethod
    def poison_generator(
        classifier: PyTorchClassifier,
        data_name: str,
        test_data: TD,
        target_ids: int,
        base_ids: list,
        seed: int,
        attack: PoisoningAttack,
    ):
        test_x, test_y = test_data.tensors[0].numpy(), test_data.tensors[1].numpy()
        predictions = classifier.predict(test_x)
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = np.sum(pred_labels == test_y) / len(test_y)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        np.random.seed(seed)

        target_instance = test_x[target_ids]
        feature_layer = classifier.layer_names[-1]  # type: ignore
        if len(base_ids) > 1:
            base_instances = np.copy([test_x[base_ids]])[0]
        else:
            base_instances = np.copy([test_x[base_ids][0]])

        hyperparam_dict = {
            "mnist": {
                "classifier": classifier,
                "target": target_instance,
                "feature_layer": feature_layer,
                "max_iter": 10,
                "similarity_coeff": 200,
                "watermark": 0.1,
                "learning_rate": 0.1,
                "verbose": True,
            },
            "cifar10": {
                "classifier": classifier,
                "target": target_instance,
                "feature_layer": feature_layer,
                "max_iter": 10,
                "similarity_coeff": 256,
                "watermark": 0.3,
                "learning_rate": 0.01,
                "verbose": True,
            },
            "fmnist": {
                "classifier": classifier,
                "target": target_instance,
                "feature_layer": feature_layer,
                "max_iter": 100,
                "similarity_coeff": 200,
                "watermark": 0.1,
                "learning_rate": 0.1,
                "verbose": True,
            },
        }

        attack = FeatureCollisionAttack(**hyperparam_dict[data_name])
        poisons, poison_labels = attack.poison(base_instances)
        poison_labels = np.argmax(poison_labels, axis=1)
        return poisons, poison_labels

    def saving_setup(self):
        final_poisons = []
        final_poison_labels = []
        for item in self.target_bases_dict:
            target_id = item["target_id"]
            base_ids = item["base_ids"]

            poisons, poison_labels = Attack.poison_generator(
                classifier=self.classifier,
                data_name=self.data_name,
                test_data=self.test_data,
                target_ids=target_id,
                base_ids=base_ids,
                seed=self.seed,
                attack=FeatureCollisionAttack,  # type: ignore
            )

            final_poisons.append(poisons[0])
            final_poison_labels.append(poison_labels[0])

        self.final_poisons = np.array(final_poisons)
        self.final_poison_labels = np.array(final_poison_labels)
        self.poison_tensors = torch.tensor(self.final_poisons, dtype=torch.float32)
        

        # DONE: ONLY POISONS DATASET
        self.only_poison_dataset = TD(
            self.poison_tensors,
            torch.tensor(self.final_poison_labels, dtype=torch.long),
        )

        # DONE: TRAINING DATA SUBSET
        selected_bases_ids = np.array(
            [x for xs in self.base_indices for x in xs]
        )  # Just a flatten
        np.random.seed(self.seed)
        clean_indexes = np.random.choice(
            [i for i in range(len(self.train_data)) if i not in selected_bases_ids],
            size=int(len(self.train_data) * 0.01),
            replace=False,
        )
        self.training_data_subset = torch.utils.data.Subset(self.train_data, clean_indexes)  # type: ignore

        # DONE: POISONED DATA + CLEAN TRAINING DATA SUBSET
        self.poisoned_dataset = ConcatDataset(
            [self.training_data_subset, self.only_poison_dataset]
        )
        assert len(self.poisoned_dataset) == len(self.training_data_subset) + len(
            self.only_poison_dataset
        ), "The generated poisons have not been added to the training data subset correctly"
            
        self.save_poisoned_ckpts = True
        
        poisoned_dataset_savedir = Path(
            f"data/dirty/{self.data_name}/{self.dir_suffix}"
        )
        
        
       #Make a new dataset called training_data_with_replaced_poisons which is a copy of self.train_data
        temp_poison_tensors = [x for xs in self.poison_tensors for x in xs]
        self.training_data_with_replaced_poisons = self.train_data.tensors[0].clone()
        for i, base_id in enumerate(selected_bases_ids):
            self.training_data_with_replaced_poisons[base_id] = temp_poison_tensors [i]
        
        self.training_data_with_replaced_poisons = TD(self.training_data_with_replaced_poisons, torch.load(self.train_data_fp).tensors[1])

        torch.save(self.training_data_with_replaced_poisons, poisoned_dataset_savedir / "poisoned_train.pt")       
        np.save(poisoned_dataset_savedir / "used_clean_indexes.npy", clean_indexes)
        
        




    
    def run(self, max_iters):
        
        self.max_iters = max_iters
        i = 0

        target_ids = [item[0] for item in self.target_indices]
        prediction_on_all_epochs = []

        while i < self.max_iters:
            print("Epoch: ", i + 1)
            self.model, _ = train(
                model=self.model,
                train_data=self.poisoned_dataset,
                test_data=self.test_data,
                epochs=1,
                save_dir=Path(
                    f"results/{self.model_name}/{self.data_name}/{self.dir_suffix}/poisoned/ckpts"
                ),
                ckpt_name=f"checkpoint-{i}.pt",
                save_ckpts=self.save_poisoned_ckpts,
                batch_size=self.conf_mger.model_training.batch_size,
                learning_rate=self.conf_mger.model_training.learning_rate,
                reg_strength=self.conf_mger.model_training.regularization_strength,
                seed=self.conf_mger.model_training.random_seed,
                device=self.device,
            )
            self.model.eval()
            target_instances = self.test_data[target_ids]
                
                

            with torch.no_grad():
                target_preds = self.model(target_instances[0])
                current_predictions = np.argmax(F.softmax(target_preds, dim=1), axis=1).numpy()
                prediction_on_all_epochs.append(current_predictions)
            i += 1
        self.all_target_prediction_by_epoch = [[x[i] for x in prediction_on_all_epochs]for i in range(len(prediction_on_all_epochs[0]))]





#TODO: CREATE MAIN USING CLICK
myAttack = Attack(
    data_name="mnist",
    num_poisons=2,
    num_targets=2,
    seed=0,
    device="cpu",
    dir_suffix="subset_id0_r0.1",
    model_ckpt_fp="results/resnet20/mnist/subset_id0_r0.1/clean/ckpts/checkpoint-7.pt",
    train_data_fp="data/clean/mnist/subset_id0_r0.1/train.pt",
    test_data_fp="data/clean/mnist/subset_id0_r0.1/test.pt",
    model_conf_fp="configs/resnet/resnet_mnist.json",
)


myAttack.initial_fine_tune()
myAttack.set_classifier()
myAttack.draw_bases_and_targets()
myAttack.saving_setup()

myAttack.run(max_iters=10)
#print(myAttack.all_target_prediction_by_epoch)
