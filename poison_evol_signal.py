
from pathlib import Path

import click
import numpy as np
import pandas as pd
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


        self.subset_folder = subset_folder
        self.attack_type = attack_type


        self.influence_matrices_pathlist = list(
        Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/influence_matrices"
        ).glob("IM_*.npy"))
         
        self.self_influence_matrices_pathlist = list(
        Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/influence_matrices"
        ).glob("SI_*.npy"))

        self.signal_csv_pathlist = list(
        Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/signals"
        ).glob("*.csv"))

        self.signal_accum = list(Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/signals").glob("*accumulated.csv"))[0]


        save_dir = Path(
            f"results/{model_name}/{data_name}/{subset_folder}/poisoned/{attack_type}/signals"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.signal_csv_pathlist.remove(self.signal_accum)

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

        #NOTE: Order matters here, so we order the matrices path list

        self.influence_matrices_pathlist = sorted(
            self.influence_matrices_pathlist,
            key=lambda x: int(x.stem.split("-")[-1]),
        )

        self.self_influence_matrices_pathlist = sorted(
            self.self_influence_matrices_pathlist,
            key=lambda x: int(x.stem.split("-")[-1]),
        )
 
        self.signal_csv_pathlist = sorted(
            self.signal_csv_pathlist,
            key=lambda x: int(x.stem.split("_")[-1]),
        )


        matrix_differences = []
        firstMatrix = np.load(self.influence_matrices_pathlist[0])

        for matrix in self.influence_matrices_pathlist:
            loaded_matrix = np.load(matrix)
            difference = loaded_matrix - firstMatrix    
            matrix_differences.append(difference)

        si_vector_differences = []
        firstVector = np.load(self.self_influence_matrices_pathlist[0])

        for vector in self.self_influence_matrices_pathlist:
            loaded_vector = np.load(vector)
            difference = loaded_vector - firstVector
            si_vector_differences.append(difference)

        signal_df_dict = {} 
        for idx, matrix in enumerate(matrix_differences):
            
            
            self.signalComputations = InfluenceErrorSignals(
                train_test_inf_mat=matrix,
                y_train=self.y_train,
                y_test=self.y_test,
                compute_test_influence=False,
            )
            signals_dataframe = self.signalComputations.compute_signals(verbose=False)
            signal_df_dict[idx] = signals_dataframe


        si_vector_dict = {}
        for idx, vector in  enumerate(self.self_influence_matrices_pathlist):
            si_vector_dict[idx] =  np.load(vector)


        for index in signal_df_dict.keys():
            signals_dataframe = signal_df_dict[index]
            si_vector = si_vector_dict[index]
            signals_dataframe["SI"] = si_vector
            signals_dataframe.to_csv(save_dir / f"diff_signals_{index}.csv")



@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_name", required=True, type=click.STRING)
@click.option("--subset_id", required=True, type=click.STRING)
@click.option("--attack_type", required=True, type=click.STRING)
def main(data_name, model_name, subset_id, attack_type):

    influence_holder = InfluenceHolder(data_name, model_name, subset_id, attack_type)





if __name__ == "__main__":
    main()
