
from pathlib import Path

import click ## ??????????????????????????????????????
import numpy as np
import pandas as pd ## ???????????????????????????????? Why pandas tho? JUST USE NP!!!!!
import torch

from ibda.signals import InfluenceErrorSignals





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

        self.influence_matrices_pathlist = [list(
            Path(
                f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/influence_matrices"
            ).glob("*.npy")
        )[0]]

        self.y_train = (
            torch.load(f"data/clean/{data_name}/{subset_folder}/train.pt")
            .tensors[1]
            .numpy()
        )
        self.y_test = (
            torch.load(f"data/clean/{data_name}/{subset_folder}/test.pt")
            .tensors[1]
            .numpy()
        )

        save_dir = Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/signals"
        )
        save_dir.mkdir(parents=True, exist_ok=True)


        for matrix in self.influence_matrices_pathlist:

            index = int(str(matrix).split("-")[-1].split(".npy")[0])
            self.IM = np.load(matrix)
            self.signalComputations = InfluenceErrorSignals(
                train_test_inf_mat=self.IM,
                y_train=self.y_train,
                y_test=self.y_test,
                compute_test_influence=False,
            )
            signals_dataframe = self.signalComputations.compute_signals(verbose=False)
            print(signals_dataframe)
         


@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_name", required=True, type=click.STRING)
@click.option("--subset_id", required=True, type=click.STRING)
@click.option("--attack_type", required=True, type=click.STRING)
def main(data_name, model_name, subset_id, attack_type):


    influence_holder = InfluenceHolder(data_name, model_name, subset_id, attack_type)



if __name__ == "__main__":
    main()






