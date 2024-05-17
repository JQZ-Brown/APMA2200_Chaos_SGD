import sys
import os.path as path
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, hessian

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    data_name = "usps" # usps, ionosphere
    optim_name_list = ["sgd", "cgd-chaotic", "cgd-pos", "cgd-neg"]
    model_color_dict = {
        "sgd": "#ebce2b",
        "cgd-chaotic": "#702c8c",
        "cgd-pos": "#db6917",
        "cgd-neg": "#96cde6",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for optim_name in optim_name_list:
        res_dict = np.load("./res/Pruning/{}-{}-log.npy".format(data_name, optim_name), allow_pickle=True).item()
        loss_list = res_dict["loss_list"]
        acc_list = res_dict["acc_list"]
        avg_loss = moving_average(loss_list, n=50)
        avg_acc = moving_average(acc_list, n=50)
        # ax1.plot(range(len(loss_list)), loss_list, "-", color=model_color_dict[optim_name], lw=2.0, label=optim_name)
        ax1.plot(range(len(avg_loss)), avg_loss, "-", color=model_color_dict[optim_name], lw=2.0, label=optim_name)
        ax2.plot(np.arange(len(acc_list))*100, acc_list, "-", color=model_color_dict[optim_name], lw=2.0, label=optim_name)

    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Testing Loss")
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Testing Accuracy")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig("../../figs/Fig3_pruning.pdf", dpi=600)
    plt.show()

