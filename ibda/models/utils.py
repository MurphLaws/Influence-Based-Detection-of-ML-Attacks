import glob
import os

import click
import numpy as np
import torch
from pkbar import pkbar
from torch.utils.data import DataLoader


def get_last_ckpt(ckpt_dir: str):
    file_list = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    highest_number = -1
    highest_checkpoint = None
    for file_path in file_list:
        if file_path.endswith(".pt") and file_path.count("-") > 0:
            parts = file_path.split("-")
            try:
                checkpoint_number = int(parts[-1].split(".pt")[0])
                if checkpoint_number > highest_number:
                    highest_number = checkpoint_number
                    highest_checkpoint = file_path
            except ValueError:
                pass  # Ignore files with non-numeric label
    return highest_checkpoint


def set_model_weights(model, ckpt_fp):
    state_dict = torch.load(ckpt_fp)
    model.load_state_dict(state_dict["model_state_dict"])
    return model


def predict(model, data, batch_size=256, num_workers=5, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    preds_labels = None
    preds = None
    data_loader = DataLoader(
        data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    with torch.no_grad():
        model.eval()
        t_total = 0
        t_correct = 0

        kbar = pkbar.Kbar(target=len(data_loader) - 1)

        for batch_id, (inputs_t, targets_t) in enumerate(data_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = model(inputs_t)
            outputs_t = torch.softmax(outputs_t, dim=1)
            curr_preds, curr_preds_labels = outputs_t.max(1)
            t_total += targets_t.size(0)
            t_correct += curr_preds_labels.eq(targets_t).sum().item()
            if preds_labels is None:
                preds_labels = curr_preds_labels.cpu().numpy()
                preds = curr_preds.cpu().numpy()
            else:
                preds_labels = np.hstack(
                    (preds_labels, curr_preds_labels.cpu().numpy())
                )
                preds = np.hstack((preds, curr_preds.cpu().numpy()))

            if kbar.target > 0:
                kbar.update(batch_id)

        acc = t_correct / t_total

    return preds_labels, preds, acc
