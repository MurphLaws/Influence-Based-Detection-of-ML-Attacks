import os

SEEDS = [1,2,3,4]
attacks = ['fgsm', 'cw', 'bound_attack']

if __name__ == '__main__':

	for seed in SEEDS:
			commands = (
				f"make prepare_data SEED={seed};"
				f"make run_evasion_attacks SUBSET_FOLDER=subset_id{seed}_r0.1 SEED={seed};"
			)
			os.system(commands)
			for a in attacks:
				command = f"make adv_influence ATTACK={a} SUBSET_FOLDER=subset_id{seed}_r0.1"
				os.system(command)
