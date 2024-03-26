import os

import click

SEEDS = [0, 1, 2, 3, 4]
attacks = ['fgsm', 'cw', 'bound_attack']

@click.command()
@click.option("--data", required=True, type=click.Choice(['mnist', 'cifar10', 'fmnist']))
@click.option("--device", required=True, type=click.STRING)
def run_command(data: str, device: str):
	for seed in SEEDS:
			commands = (
				f"make prepare_data SEED={seed};"
				f"make run_evasion_attack_single  DATA={data} SUBSET_FOLDER=subset_id{seed}_r0.1 SEED={seed} device={device};"
			)
			os.system(commands)


if __name__ == '__main__':
	run_command()
	pass
		# for a in attacks:
		# 	command = f"make adv_influence_single DATA={dataset} ATTACK={a} SUBSET_FOLDER=subset_id{seed}_r0.1"
		# 	os.system(command)
