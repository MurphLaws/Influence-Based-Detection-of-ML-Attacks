import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from test_adversarials_influence import attack_choices

RESULTS_FOLDER = 'results_server'

signals = ['SI', 'ACNI^*']

columns = ['method', 'avep', 'auc', 'exec_time', 'attack', 'dataset', 'model']

def read_inf_sigs(res_dict, a, d, m):
	method_res = []
	for sig in signals:
		if sig == 'SI':
			time = res_dict['self_inf_time']
		else:
			time = res_dict['train_test_inf_time']
		method_res.append([sig, res_dict[sig]['avep'], res_dict[sig]['auc'], time, a, d, m])
	return method_res

def read_nnif(res_dict, a, d, m):
	return ['nnif', res_dict['nnif']['avep'], res_dict['nnif']['auc'], res_dict['exec_time'], a, d, m]

methods_res_reader = {
	'nnif': read_nnif,
	'tracin': read_inf_sigs
}

def plot_comparison(data_res):
	mean_auc = data_res.pivot_table(index='attack', columns='method', values='auc', aggfunc='mean')
	std_auc = data_res.pivot_table(index='attack', columns='method', values='auc', aggfunc='std')
	mean_auc = mean_auc.loc[attack_choices,:]
	std_auc = std_auc.loc[attack_choices, :]
	stderr_auc = std_auc / np.sqrt(5)
	new_col_names = []
	for c in mean_auc.columns:
		if c in signals:
			c = '$' + c + '$ (ours)'
		else:
			c = '$' + c + '$'
		new_col_names.append(c)
	mean_auc.columns = new_col_names
	stderr_auc.columns = new_col_names
	mean_auc.plot.bar(rot=0, yerr=stderr_auc)
	title = data_res['model'][0] + ' on ' + data_res['dataset'][0]
	plt.title(title, fontsize=16)
	plt.ylabel('AUC of attack detection on test set', fontsize=14)
	plt.tight_layout()
	plt.ylim((0,1))
	plt.show()


def plot_exec_time(data_res):
	mean_exec_time = data_res.pivot_table(index='attack', columns='method', values='exec_time', aggfunc='mean')


def read_results(model_name, d):
	print(d)
	total_res = []
	for f in Path(RESULTS_FOLDER).rglob('*.json'):
		if 'clean' in str(f) or '/' + d not in str(f) or model_name not in str(f):
			continue
		print(f)
		attack = [a for a in attack_choices if a in str(f)][0]
		for method, read_fn in methods_res_reader.items():
			if method not in str(f):
				continue
			with open(str(f), 'r') as f:
				res_dict = json.load(f)
			method_res = read_fn(res_dict=res_dict, a=attack, d=d, m=model_name)
			if len(method_res) < len(columns):
				total_res.extend(method_res)
			else:
				total_res.append(method_res)
	return pd.DataFrame(total_res, columns=columns)


if __name__ == '__main__':

	model_name = 'resnet20'

	data_res_df = pd.DataFrame()

	for d in ['mnist', 'fmnist', 'cifar10']:
		data_res = read_results(model_name, d)
		data_res_df = pd.concat([data_res_df, data_res])
		# plot_comparison(data_res)
	print()