
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import mse_loss
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import TensorDataset

from ibda.signals import InfluenceErrorSignals
from test_adversarials_influence import attack_choices

from attack_generation.utils import plot_adversarial_examples


def load_influence_matrices(
		data_name: str,
		model_name: str,
		subset_folder: str,
		attack_type: str,):
	
	influence_matrices_pathlist = list(Path(f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/influence_matrices").glob("*.npy"))
	

		
	test_data = torch.load(f"data/clean/{data_name}/{subset_folder}/test.pt")
	train_data = torch.load(f"data/clean/{data_name}/{subset_folder}/train.pt")
	y_train = train_data.tensors[1].numpy()
	y_test = test_data.tensors[1].numpy()
	signals_dict_list = []
	for matrix in influence_matrices_pathlist:

		#Get index of matrix in the list between the last "-" and ".npy"

		index = int(str(matrix).split("-")[-1].split(".npy")[0])
		IM = np.load(matrix)

#		
#        self.__signals = {
#            'CPI': self.cpi,
#            'ACNI': self.acni,
#            'ACNI^*': self.acni_opt,
#            'CPI^*': self.cpi_opt,
#            'ACI': self.aci,
#            'MNI': self.mani,
#            'MPI': self.mpi,
#            'MTI': self.mti,
#            'MNIC': self.mnic,
#        }
#
		influenceHolder = InfluenceErrorSignals(
				train_test_inf_mat=IM,
				y_train = y_train,
				y_test = y_test,
				compute_test_influence=False,
				)

	#Plot a y=x function in range 0 to 7

	plt.plot(range(8),range(8))
	plt.show()
	
			

if __name__ == "__main__":
	load_influence_matrices(
		data_name="mnist",
		model_name="resnet20",
		subset_folder="subset_id0_r0.1",
		attack_type="many_to_one",
		)
	  




