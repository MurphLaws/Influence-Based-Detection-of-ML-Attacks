import json
import os
import re
import warnings
from itertools import chain
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, average_precision_score

warnings.filterwarnings("ignore")


##NOTE: ACCUMULATED.CSV
def find_walk_csv(directory, endstring):
    accumulated_csv_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(endstring):
                accumulated_csv_paths.append(os.path.join(root, file))
    return accumulated_csv_paths


current_directory = os.path.dirname(__file__)

accumulated_csv_paths = find_walk_csv(current_directory, ".csv")


##NOTE: LARGEST_EPOCH_SIGNAL CSV
def find_largest_integer(directory):
    max_integer = float("-inf")
    pattern = r"signals_(\d+)\.csv"
    for root, dirs, files in os.walk(directory):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                integer = int(match.group(1))  # Extract the integer part
                max_integer = max(max_integer, integer)
    return max_integer


current_directory = (
    os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
)
largest_integer = find_largest_integer(current_directory)


def demo_df(dataset):
    csv_paths = [
        path
        for path in find_walk_csv(current_directory, "accumulated.csv")
        if dataset in path
    ]
    demo = pd.read_csv(csv_paths[0], index_col=0)
    demo[:] = 0
    return demo


def mean_of_dataframes(csv_paths):
    sum_df = None
    if csv_paths == []:
        return demo_df("mnist")
    else:
        for path in csv_paths:
            df = pd.read_csv(path, index_col=0)
            sum_df = df if sum_df is None else sum_df.add(df, fill_value=0)
        return sum_df.div(len(csv_paths))  # type: ignore


plot_path = Path("plots")
plot_path.mkdir(exist_ok=True)

SEEDS = [2, 3]

ALL_BINARY_POISONS = {}
accum_dict = {}
for dataset in ["/mnist", "/fmnist", "/cifar10"]:
    for attack_type in ["one_to_one", "many_to_one"]:
        dataset_len = len(demo_df(dataset))
        avg_accum = pd.DataFrame()
        for seed in SEEDS:

            dict_path = Path(
                "results/resnet20"
                + dataset
                + "/subset_id"
                + str(seed)
                + "_r0.1"
                + "/poisoned/"
                + attack_type
                + "/"
                + "target_bases.json"
            )

            selected_bases = []

            if dict_path.exists():
                with open(dict_path, "r") as f:
                    target_bases_dict = json.load(f)

                for dictionary in target_bases_dict:
                    selected_bases.append(dictionary["base_ids"])

                selected_bases = list(chain.from_iterable(selected_bases))
            else:
                selected_bases = []

            BINARY_POISONS = np.zeros(dataset_len)

            for index in selected_bases:
                BINARY_POISONS[index] = 1

            ALL_BINARY_POISONS[dataset[1:] + "_" + attack_type + "_" + str(seed)] = (
                BINARY_POISONS
            )
            accumulated_csv_paths = [
                path
                for path in find_walk_csv(current_directory, "accumulated.csv")
                if dataset in path
                and attack_type in path
                and "subset_id" + str(seed) in path
            ]

            if accumulated_csv_paths == []:
                accum_df = demo_df(dataset)
            else:
                accum_df = pd.read_csv(accumulated_csv_paths[0], index_col=0)

            results = []
            for col in accum_df.columns:
                avg_precision = average_precision_score(BINARY_POISONS, accum_df[col])
                results.append({"signal": col, "avg precision": avg_precision})

            results_df = pd.DataFrame(results)
            avg_accum = avg_accum.add(results_df.set_index("signal"), fill_value=0)
        avg_accum = avg_accum.div(len(SEEDS))

        accum_dict[dataset[1:] + "_" + attack_type] = avg_accum.reset_index()


dataframes = list(accum_dict.values())

colors = ["orange", "orange", "blue", "blue", "green", "green"]

labels = list(accum_dict.keys())


x = np.arange(len(dataframes[0]))

num_bars = len(dataframes)
bar_width = 1.0 / (num_bars + 2)  # Space for each bar plus space between bars
plt.figure(figsize=(10, 8))

for i, df in enumerate(dataframes):
    if i % 2 == 1:  # Make every other column striped
        plt.bar(
            x + i * bar_width,
            df["avg precision"],
            width=bar_width,
            color=colors[i],
            alpha=1,
            hatch="\\" * 5,
            edgecolor="black",
            label=labels[i],
        )
    else:
        plt.bar(
            x + i * bar_width,
            df["avg precision"],
            width=bar_width,
            color=colors[i],
            alpha=1,
            edgecolor="black",
            label=labels[i],
        )

# Add labels and legend
plt.xlabel("Signal")
plt.ylabel("Average Precision")
plt.title("Average Precision vs Signal")
plt.xticks(
    x + 0.5, dataframes[0]["signal"]
)  # Assuming 'signal' column is shared across dataframes
plt.ylim(0, 1)  # Limit y-axis from 0 to 1
plt.legend()
plt.grid(True)
plt.savefig(plot_path / "all_daset_avg_precision")


##DONE: SWARMPLOT

for dataset in ["/mnist", "/fmnist", "/cifar10"]:
    for attack_type in ["one_to_one", "many_to_one"]:
        df_to_accum_list = []
        results_accum_df = pd.DataFrame()
        for seed in SEEDS:
            BINARY_POISONS = ALL_BINARY_POISONS[
                dataset[1:] + "_" + attack_type + "_" + str(seed)
            ]
            all_paths = []
            results = []
            for i in range(0, largest_integer + 1):  # type: ignore
                integer_ending_path = find_walk_csv(current_directory, str(i) + ".csv")
                integer_ending_path = [
                    path
                    for path in integer_ending_path
                    if dataset in path
                    and attack_type in path
                    and "subset_id" + str(seed) in path
                ]
                all_paths.append(integer_ending_path)
## FIXME: Plots priting on wrong order. For some reason (bug) im not indexing the dataframe columns correctly
## Thus completely turning into shit the whole plotting system

            for signal_csv in all_paths:
                signal_df = pd.read_csv(signal_csv[0], index_col=0)
                df_id = str(signal_csv).split("_")[-1].split(".")[0]
                for col in signal_df.columns:
                    avg_precision = average_precision_score(
                        BINARY_POISONS, signal_df[col]
                    )
                    results.append(
                        {"ID": df_id, "signal": col, "avg precision": avg_precision}
                    )

            results_df = pd.DataFrame(results)
            df_to_accum_list.append(results_df)

        for df in df_to_accum_list:
            if results_accum_df.empty:
                results_accum_df = df
            else:
                results_accum_df["avg precision"] = (
                    results_accum_df["avg precision"] + df["avg precision"]
                )
        results_accum_df["avg precision"] = results_accum_df["avg precision"] / len(
            SEEDS
        )

        plt.figure(figsize=(10, 7))

        #barplot = sns.barplot(
        #    data=accum_dict[dataset[1:] + "_" + attack_type],
        #    x="signal",
        #    y="avg precision",
        #    color="#2F837F",
        #)
        #barplot.set_ylim(0, 1)

        swarmplot = sns.swarmplot(
            data=results_accum_df,
            x="signal",
            y="avg precision",
            hue="ID",
            palette="viridis",
            size=6,
            alpha=1,
        )
        swarmplot.set_ylim(0, 1)

        swarmplot.legend_.remove()  # type: ignore
        plt.xlabel("Signals")
        plt.ylabel("Avg Precision")
        plt.title(
            dataset[1:].upper()
            + " Poison detection Avg. Precision score, "
            + attack_type
        )

        cmap = mpl.cm.viridis  # type: ignore
        norm = mpl.colors.Normalize(  # type: ignore
            vmin=results_accum_df["ID"].min(), vmax=results_accum_df["ID"].max()
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # dummy variable required for colorbar

        # Create a new axis for the colorbar
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Epoch")

        plt.savefig(plot_path / (dataset[1:] + "_" + attack_type + "_avg_precision"))
