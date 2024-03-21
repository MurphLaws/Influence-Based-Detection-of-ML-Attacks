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

class Holder:
	def __init__(self):
		self.test_self_inf = None
		self.adv_self_inf = None
		self.train_test_inf = None
		self.train_adv_inf = None
		self.adv_preds = None
		self.adv_labels = None
		self.preds = None
		self.preds_labels = None
		self.clean_test = None
		self.clean_train = None
		self.y_test = None
		self.y_train = None
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
		self.y_test = value.tensors[1].numpy()

	def set_clean_train(self, value):
		self.clean_train = value
		self.y_train = value.tensors[1].numpy()

	def set_adv_samples(self, value):
		self.adv_samples = value

	def set_error_col(self, value):
		self.error_col = value
		self.adv_ids = np.where(value == 1)[0]
		self.clean_ids = np.where(value == 0)[0]

	def set_preds(self, value):
		self.preds = np.array(value)

	def set_preds_labels(self, value):
		self.preds_labels = np.array(value)

	def replace_adv_pred(self):
		self.preds_labels[self.adv_ids] = self.adv_labels
		self.preds[self.adv_ids] = self.adv_preds

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
	train_data_fp = f"data/clean/{data_name}/{subset_id}/train.pt"

	dirty_data_dir = (
		f"data/dirty/{attack}/{model_name}/{data_name}/{subset_id}"
	)

	with open(Path(dirty_data_dir, "adv_ids.npy"), "rb") as f:
		holder.set_error_col(np.load(f))

	holder.set_clean_test(torch.load(test_data_fp))
	holder.set_clean_train(torch.load(train_data_fp))

	result_dir = Path('results', model_name, data_name, subset_id, 'dirty', attack, inf_fn_name)

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

	model_info_fp = Path('results', model_name, data_name, subset_id, 'clean', 'ckpts', 'info.json')
	with open(model_info_fp) as f:
		info_dict = json.load(f)
		holder.set_preds(info_dict['test_preds'])
		holder.set_preds_labels(np.array(info_dict['test_preds_labels'], dtype=int))

	holder.replace_adv_pred()

	return holder

def plot_attack_info(df, y_label, title, col_order=None, row_order=None, ylim=None, savedir=None, fname=None):
	df_pivot = df.pivot(index='data', columns='attack', values='values').reset_index().set_index('data')
	if col_order is not None:
		df_pivot = df_pivot[col_order]
	if row_order is not None:
		df_pivot = df_pivot.loc[row_order]
	df_pivot.plot.bar(rot=0)
	plt.ylim(ylim)
	plt.ylabel(y_label)
	plt.title(title)
	plt.tight_layout()
	Path(savedir).mkdir(parents=True, exist_ok=True)
	if savedir is not None:
		fname = fname if fname is not None else 'img.png'
		plt.savefig(Path(savedir,fname), dpi=300)
	plt.show()

def plot_sig_interval_accuracy(signal_dict, sig_name, savedir=None):
	signal_info_accs = pd.DataFrame(signal_dict)
	signal_info_accs = signal_info_accs.groupby(by=0).mean()
	signal_info_accs.index.name = None
	signal_info_accs.T.plot(style='.-')
	plt.xticks([])
	plt.ylabel('Test Accuracy')
	plt.xlabel(f'Increasing {sig_name} Influence')
	plt.title(f'Test Accuracy vs {sig_name} Influence on Clean Test Data')
	Path(savedir).mkdir(parents=True, exist_ok=True)
	if savedir is not None:
		fname = sig_name
		plt.savefig(Path(savedir, fname), dpi=300)
	plt.show()


if __name__ == '__main__':

	data = ['mnist', 'fmnist', 'cifar10']

	signal_misclass_acc = 'ACNI'

	inf_fn = 'tracin'
	model_name = 'resnet20'
	subset_id = 'subset_id0_r0.1'

	mse_info = []
	avep_info = []
	sig_acc_info = []

	for d in data:
		for a in attack_choices:
			holder = load_data(attack=a, data_name=d, model_name=model_name, subset_id=subset_id, inf_fn_name=inf_fn, device='cpu')

			final_si, final_train_test_mat = holder.test_self_inf.copy(), holder.train_test_inf.copy()

			final_si[holder.adv_ids] = holder.adv_self_inf
			final_train_test_mat[:, holder.adv_ids] = holder.train_adv_inf

			ratio_str = (len(holder.adv_ids) / len(holder.clean_test))*100

			mse = mse_loss(holder.clean_test[holder.adv_ids][0], holder.adv_samples)

			# plot_adversarial_examples(holder.adv_samples, nrows=2, ncols=2, adv_labels=holder.adv_labels, title=a + ' ' + d)
			#
			# plot_adversarial_examples(holder.clean_test[holder.adv_ids][0],
			#                           nrows=2, ncols=2,
			#                           adv_labels=holder.clean_test[holder.adv_ids][1].numpy(), title=f'Clean Labels {d}')

			# noise = holder.clean_test[holder.adv_ids][0] - holder.adv_samples
			# plot_adversarial_examples(noise,
			#                           nrows=2, ncols=2,
			#                           title=f'Adv. Noise {d}')

			sigs = InfluenceErrorSignals(train_test_inf_mat=final_train_test_mat, y_train=holder.y_train, y_test=holder.y_test, compute_test_influence=True)
			joint_signals = sigs.compute_signals()
			joint_signals['SI'] = final_si

			df = pd.DataFrame()
			df['sig'] = joint_signals[signal_misclass_acc]
			df['y_pred'] = holder.preds_labels
			df['y_obs'] = holder.y_test
			df['is_adv'] = holder.error_col
			df_split = np.array_split(df.drop(index=holder.adv_ids).sort_values(by='sig'), 10)
			accs = []
			for sdf in df_split:
				accs.append(accuracy_score(sdf['y_obs'], sdf['y_pred']))
			sig_acc_info.append([d, *accs])

			avep = average_precision_score(holder.error_col, df['sig'])

			avg_pred_proba = np.mean(holder.adv_preds)

			adv_acc = accuracy_score(df.loc[holder.adv_ids, 'y_obs'], df.loc[holder.adv_ids, 'y_pred'])
			assert adv_acc == 0

			for sig in joint_signals:
				print(sig, average_precision_score(holder.error_col, joint_signals[sig]),
					  average_precision_score(holder.error_col, -joint_signals[sig]))

			mse_info.append([d, a, float(mse)])
			avep_info.append([d,a,avep])

			print('{} {} {:.2f}% adv, avg pred proba {:.2f}, avep {}, mse {:.4f} \n'.format(a, d, ratio_str, avg_pred_proba, np.around(avep, 2), mse))

	mse_df = pd.DataFrame(mse_info, columns=['data', 'attack', 'values'])
	avep_df = pd.DataFrame(avep_info, columns=['data', 'attack', 'values'])

	result_dir = Path('results', model_name, d, subset_id, 'dirty', a, inf_fn)

	plot_sig_interval_accuracy(signal_dict=sig_acc_info, savedir=Path('figures', 'signals_acc_misclass'), sig_name=f'{signal_misclass_acc}.png')

	plot_attack_info(mse_df, y_label='MSE (x, $x_a$)', title='Adversarial Attack Quality', col_order=attack_choices, row_order=data, savedir=Path('figures', 'attacks_quality'), fname='mse.png')
	plot_attack_info(avep_df, y_label='AveP', title='Test Set Self Influence', col_order=attack_choices, row_order=data, ylim=(0,1), savedir=Path('figures', 'detection'), fname='avep.png')
