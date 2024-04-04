
#run_poison_attacks:
#	python -m attack_generation.poisons.run_attacks \
#	--data_name $(DATA_NAME) \
#	--train_data_fp data/clean/$(DATA_NAME)/$(SUBSET_FOLDER)/train.pt \
#	--test_data_fp data/clean/$(DATA_NAME)/$(SUBSET_FOLDER)/test.pt \
#	--model_conf_fp configs/resnet/resnet_$(DATA_NAME).json \
#	--dir_suffix $(SUBSET_FOLDER) \
#	$(if $(CKPT_NUMBER),--model_ckpt_fp "results/resnet20/$(DATA_NAME)/$(SUBSET_FOLDER)/clean/ckpts/checkpoint-$(CKPT_NUMBER).pt") \
#	$(if $(SEED),--seed $(SEED),--seed 0) \
#	$(if $(DEVICE),--device $(DEVICE),--device cpu) \
#	$(if $(NUM_POISONS),--num_poisons $(NUM_POISONS),--num_poisons 1) \
#	$(if $(NUM_TARGETS),--num_targets $(NUM_TARGETS),--num_targets 2) \
#	$(if $(MAX_ITER),--max_iter $(MAX_ITER),--max_iter 3)

import os

import click

SEEDS = [0, 1]


@click.command()
@click.option("--data", required=True, type=click.Choice(['mnist', 'cifar10', 'fmnist']))
@click.option("--device", required=True, type=click.STRING)
def run_command(data: str, device: str):
	for seed in SEEDS:
			commands = (
				#f"make prepare_data SEED={seed};"
				#f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=10 NUM_TARGETS=10 MAX_ITER=30 SEED={seed};"
				#f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=1 NUM_TARGETS=10 MAX_ITER=30 SEED={seed};"
				f"make poison_influence DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} ATTACK_TYPE=many_to_one;"
			os.system(commands)


if __name__ == '__main__':
	run_command()
	pass
