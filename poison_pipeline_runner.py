import os
import time

import click

from ibda.utils.writers import save_as_json

SEEDS = [2,3]


@click.command()
@click.option(
    "--data", required=True, type=click.Choice(["mnist", "cifar10", "fmnist"])
)
@click.option("--device", required=True, type=click.STRING)
def run_command(data: str, device: str):

    # Create a list to store execution times for each seed
    exec_times = []

    for seed in SEEDS:
        commands = [
            f"make prepare_data SEED={seed}",
             f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=10 NUM_TARGETS=1 MAX_ITER=50 SEED={seed} CKPT_NUMBER=14" ,
             #f"make run_poison_attacks DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 DEVICE={device} NUM_POISONS=1 NUM_TARGETS=10 MAX_ITER=3 SEED={seed}",
             #f"make poison_influence DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 MODEL_NAME=resnet20  DEVICE={device} ATTACK_TYPE=many_to_one",
            #f"make poison_influence DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 MODEL_NAME=resnet20  DEVICE={device} ATTACK_TYPE=one_to_one",
            # f"make get_signals DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 MODEL_NAME=resnet20 DEVICE={device} ATTACK_TYPE=many_to_one",
            #f"make get_signals DATA_NAME={data} SUBSET_FOLDER=subset_id{seed}_r0.1 MODEL_NAME=resnet20 DEVICE={device} ATTACK_TYPE=one_to_one",
        ]

        seed_exec_times = {"seed": seed, "data": data}
        for idx, command in enumerate(commands):
            start_time = time.time()
            os.system(command)
            end_time = time.time()
            execution_time = end_time - start_time
            if idx == 0:
                seed_exec_times["data_preparation"] = execution_time
            elif idx == 1:
                seed_exec_times["many_to_one_attacks"] = execution_time
            elif idx == 2:
                seed_exec_times["one_to_one_attacks"] = execution_time
            elif idx == 3:
                seed_exec_times["many_to_one_influence"] = execution_time

        exec_times.append(seed_exec_times)
    save_as_json(exec_times, savedir=".", fname="execution_times.json", indent=4)
    print("Execution times saved to execution_times.json")


if __name__ == "__main__":
    run_command()
