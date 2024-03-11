from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import TensorDataset

from adversarials_influence import attack_choices


class Holder:
	def __init__(self):
		self.test_self_inf = None
		self.adv_self_inf = None
		self.train_test_inf = None
		self.train_adv_inf = None
		self.adv_preds = None
		self.adv_labels = None
		self.clean_test = None
		self.adv_samples = None
		self.error_col = None
		self.adv_ids = None
		self.clean_ids = None

	def set_test_self_inf(self, value):
		self.test_self_inf = value

	def set_adv_self_inf(self, value):
		self.adv_self_inf = value

	def set_train_test_inf(self, value):
		self.train_test_inf = value

	def set_train_adv_inf(self, value):
		self.train_adv_inf = value

	def set_adv_preds(self, value):
		self.adv_preds = value

	def set_adv_labels(self, value):
		self.adv_labels = value

	def set_clean_test(self, value):
		self.clean_test = value

	def set_adv_samples(self, value):
		self.adv_samples = value

	def set_error_col(self, value):
		self.error_col = value
		self.adv_ids = np.where(value == 1)[0]
		self.clean_ids = np.where(value == 0)[0]


@click.option("--attack", type=click.Choice(choices=attack_choices), required=True)
@click.option("--data_name", type=click.STRING, required=True)
@click.option("--model_name", type=click.STRING, required=True)
@click.option("--inf_fn_name", type=click.STRING, required=True)
@click.option("--subset_id", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, default=None)
def load_data(data_name, model_name, inf_fn_name, subset_id, attack, device):

	holder = Holder()

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	test_data_fp = f"data/clean/{data_name}/{subset_id}/test.pt"

	dirty_data_dir = (
		f"data/dirty/{attack}/{model_name}/{data_name}/{subset_id}"
	)

	with open(Path(dirty_data_dir, "adv_ids.npy"), "rb") as f:
		holder.set_error_col(np.load(f))

	holder.set_clean_test(torch.load(test_data_fp))

	result_dir = Path('results',model_name,data_name,subset_id, 'dirty', attack, inf_fn_name)

	holder.set_adv_samples(torch.tensor(torch.load(Path(dirty_data_dir, "adv.pt")), device=device))

	with open(Path(result_dir, 'adversarial_pred_labels.npy'), 'rb') as f:
		holder.set_adv_labels(np.load(f))
	with open(Path(result_dir, 'adversarial_preds.npy'), 'rb') as f:
		holder.set_adv_preds(np.load(f))
	with open(Path(result_dir, 'adv_self_inf.npy'), 'rb') as f:
		holder.set_adv_self_inf(np.load(f))
	with open(Path(result_dir, 'test_self_inf.npy'), 'rb') as f:
		holder.set_test_self_inf(np.load(f))
	with open(Path(result_dir, 'train_test_inf.npy'), 'rb') as f:
		holder.set_train_test_inf(np.load(f))
	with open(Path(result_dir, 'train_test_inf_adv.npy'), 'rb') as f:
		holder.set_train_adv_inf(np.load(f))

	return holder


if __name__ == '__main__':

	data = ['mnist', 'fmnist', 'cifar10']

	inf_fn = 'tracin'
	model_name = 'resnet20'
	subset_id = 'subset_id0_r0.1'

	for a in attack_choices:
		for d in data:
			holder = load_data(attack=a, data_name=d, model_name=model_name, subset_id=subset_id, inf_fn_name=inf_fn, device='cpu')

			final_si, final_train_test_mat = holder.test_self_inf.copy(), holder.train_test_inf.copy()

			final_si[holder.adv_ids] = holder.adv_self_inf
			final_train_test_mat[:, holder.adv_ids] = holder.train_adv_inf
			avep = average_precision_score(holder.error_col, final_si)

			ratio_str = str((len(holder.adv_ids) / len(holder.clean_test))*100) + '%'

			print(a,d,ratio_str)