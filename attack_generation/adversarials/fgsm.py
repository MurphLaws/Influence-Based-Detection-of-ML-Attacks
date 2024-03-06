import os.path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.train_loop import train
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager

if __name__ == "__main__":

    data_name = "mnist"
    model_conf_fp = "configs/resnet/resnet_mnist.json"
    train_data_fp = "data/clean/mnist/subset_id0_r0.1/train.pt"
    test_data_fp = "data/clean/mnist/subset_id0_r0.1/test.pt"
    savedir = "data/dirty/adv/fgsm"
    seed = 0
    train_model = True
    model_ckpt_fp = None

    savedir = Path(savedir + train_data_fp.split(data_name)[1]).parent

    train_data = torch.load(train_data_fp)
    test_data = torch.load(test_data_fp)

    num_classes = len(torch.unique(train_data.tensors[1]))
    input_shape = train_data.tensors[0].shape[1:]

    conf_mger = ConfigManager(model_training_conf=model_conf_fp)

    model = model_dispatcher[conf_mger]()

    # if train_model:
    # 	model, _ = train(
    # 		model=model,
    # 		train_data=,
    # 		epochs=conf_mger.model_training.epochs,
    # 		batch_size=conf_mger.model_training.batch_size,
    # 		learning_rate=conf_mger.model_training.learning_rate,
    #
    # 	)
    # else:
    # 	model = set_model_weights(model, model_ckpt_fp)

    # model = Net()
    #
    # # Step 2a: Define the loss function and the optimizer
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    # # Step 3: Create the ART classifier
    #
    # classifier = PyTorchClassifier(
    # 	model=model,
    # 	clip_values=(min_pixel_value, max_pixel_value),
    # 	loss=criterion,
    # 	optimizer=optimizer,
    # 	input_shape=(1, 28, 28),
    # 	nb_classes=10,
    # )
    #
    # # Step 4: Train the ART classifier
    #
    # classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    #
    # # Step 5: Evaluate the ART classifier on benign test examples
    #
    # predictions = classifier.predict(x_test)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    # print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    #
    # # Step 6: Generate adversarial test examples
    # attack = FastGradientMethod(estimator=classifier, eps=0.2)
    # x_test_adv = attack.generate(x=x_test)
