import torch
from matplotlib import pyplot as plt


def plot_adversarial_examples(adv_examples):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    i, j, counter = 0, 0, 0
    for i in range(3):
        for j in range(3):
            if len(adv_examples) > counter:
                adv_ex = adv_examples[counter]
                axes[i, j].imshow(torch.tensor(adv_ex).permute(1, 2, 0))
            counter += 1
    plt.tight_layout()
    plt.show()
