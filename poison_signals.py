
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
from tabulate import tabulate
from ibda.signals import InfluenceErrorSignals
from test_adversarials_influence import attack_choices

from attack_generation.utils import plot_adversarial_examples


class InfluenceHolder:

	def __init__(
		self, 
		data_name: str,
		model_name: str,
		subset_folder: str,
		attack_type: str,
		):

		self.data_name = data_name
		self.model_name = model_name
		self.subset_folder = subset_folder
		self.attack_type = attack_type

		self.influence_matrices_pathlist = list(Path(f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/influence_matrices").glob("*.npy"))

		self.y_train = torch.load(f"data/clean/{data_name}/{subset_folder}/train.pt").tensors[1].numpy()
		self.y_test = torch.load(f"data/clean/{data_name}/{subset_folder}/test.pt").tensors[1].numpy()

		save_dir = Path(f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/signals")
		save_dir.mkdir(parents=True, exist_ok=True)

		matrix_sum = np.zeros((self.y_train.shape[0], self.y_test.shape[0]))
		for matrix in self.influence_matrices_pathlist:

			index = int(str(matrix).split("-")[-1].split(".npy")[0])
			self.IM = np.load(matrix)
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
			self.signalComputations= InfluenceErrorSignals(
					train_test_inf_mat=self.IM,
					y_train = self.y_train,
					y_test = self.y_test,
					compute_test_influence=False,
					)


			signals_datafame = self.signalComputations.compute_signals()
			#Make sure the to_csv path exists
			signals_datafame.to_csv(save_dir / f"signals_{index}.csv")
			matrix_sum += self.IM


		self.accumSignalComputations= InfluenceErrorSignals(
				train_test_inf_mat=matrix_sum,
				y_train = self.y_train,
				y_test = self.y_test,
				compute_test_influence=False,
				)

		signals_datafame = self.accumSignalComputations.compute_signals()
		signals_datafame.to_csv(save_dir / f"signals_accumulated.csv")	
		
			
			
			
			
			

	

if __name__ == "__main__":
	  
	data_name = "cifar10"
	model_name = "resnet20"
	subset_folder = "subset_id0_r0.1"
	
	influence_holder = InfluenceHolder(data_name, model_name, subset_folder, "one_to_one")
	influence_holder = InfluenceHolder(data_name, model_name, subset_folder, "many_to_one")
	
			
	
	



