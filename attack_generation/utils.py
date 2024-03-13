from pathlib import Path

import torch
from matplotlib import pyplot as plt


def plot_adversarial_examples(
    adv_examples,
    nrows=3,
    ncols=3,
    adv_labels=None,
    title=None,
    savedir=None,
    fname=None,
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
    i, j, counter = 0, 0, 0
    for i in range(nrows):
        for j in range(ncols):
            if len(adv_examples) > counter:
                adv_ex = adv_examples[counter]
                cmap = "grey" if adv_ex.shape[0] == 1 else "viridis"
                axes[i, j].imshow(torch.tensor(adv_ex).permute(1, 2, 0), cmap=cmap)
                if adv_labels is not None:
                    axes[i, j].set_title("Label: " + str(adv_labels[counter]))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            counter += 1
    if title is not None:
        plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    if savedir is not None:
        fname = "img.png" if fname is None else fname
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(savedir, fname), dpi=300)
    plt.show()
