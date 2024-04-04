
import json
import os
import time

import click

from ibda.utils.writers import save_as_json, save_as_np

SEEDS = [0, 1,2]

@click.command()
@click.option("--data", required=True, type=click.Choice(['mnist', 'cifar10', 'fmnist']))
@click.option("--device", required=True, type=click.STRING)
def run_command(data: str, device: str):

    # Create a list to store execution times for each seed
    exec_times = []

    for seed in SEEDS:
        commands = [
            f"make prepare_data SEED={seed}",
            f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=10 NUM_TARGETS=10 MAX_ITER=30 SEED={seed}",
            f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=1 NUM_TARGETS=10 MAX_ITER=30 SEED={seed}",
            f"make poison_influence DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 MODEL_NAME=resnet20  DEVICE={device} ATTACK_TYPE=many_to_one"
        ]

        # Create a dictionary to store execution times for each command
        seed_exec_times = {"seed": seed, "data": data}

        # Execute each command and measure its execution time
        for idx, command in enumerate(commands):
            start_time = time.time()
            os.system(command)
            end_time = time.time()
            execution_time = end_time - start_time

            # Store execution time with specific key based on command index
            if idx == 0:
                seed_exec_times["data_preparation"] = execution_time
            elif idx == 1:
                seed_exec_times["many_to_one_attacks"] = execution_time
            elif idx == 2:
                seed_exec_times["one_to_one_attacks"] = execution_time
            elif idx == 3:
                seed_exec_times["many_to_one_influence"] = execution_time

        exec_times.append(seed_exec_times)

    # Save the list of dictionaries to a file
	save_as_json(exec_times, "execution_times.json")

    print("Execution times saved to execution_times.json")

if __name__ == '__main__':
    run_command()
