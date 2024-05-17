'''
Description:
    Re-implementation of the paper "Chaotic Dynamics are Intrinsic to Neural Network Training with SGD".

Author:
    Jiaqi Zhang

Reference:
    1. Herrmann, L., Granz, M., & Landgraf, T. (2022). Chaotic dynamics are intrinsic to neural network training with sgd.
       Advances in Neural Information Processing Systems, 35, 5219-5229.
    2. Github repo: https://github.com/luisherrmann/chaotic_neurips22
'''

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


# ====================================================

def get_train_data(data_path: str, name: str, transform=[]):
    transform_list = [transforms.ToTensor()]
    transform_list += [*transform]
    transform = transforms.Compose(transform_list)
    train_data = datasets.USPS(
        data_path, train=True, download=False, transform=transform
    )
    return train_data


def get_validation_data(data_path: str, name: str, transform=[]):
    transform_list = [transforms.ToTensor()]
    transform_list += [*transform]
    transform_composition = transforms.Compose(transform_list)
    val_data = datasets.USPS(
        data_path, train=False, download=False, transform=transform_composition
    )
    return val_data


def prepare_ds(data_path: str, name: str, batch_size: int, debug=False):
    train_data = get_train_data(data_path, name)
    val_data = get_validation_data(data_path, name)
    batch_size_tmp = len(train_data)
    train_loader = DataLoader(
        train_data, batch_size=batch_size_tmp, shuffle=True #, num_workers=2
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size_tmp, shuffle=True# , num_workers=2
    )
    batch = next(iter(train_loader))
    x, y = batch
    mean, std = torch.mean(x), torch.std(x)
    transform = [transforms.Normalize(mean, std)]
    train_data = get_train_data(data_path, name, transform=transform)
    val_data = get_validation_data(data_path, name, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True  # , num_workers=6
    )
    val_loader = DataLoader(
        val_data, batch_size=len(val_data), shuffle=False  # , num_workers=6
    )
    return train_loader, val_loader


# ====================================================

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = [-1, shape]

    def __repr__(self):
        dimstr = ", ".join([str(x) for x in self.shape])
        return f"Reshape({dimstr})"

    def forward(self, x):
        return x.reshape(self.shape)


# ====================================================
# HESSIAN COMPUTATION
# https://discuss.pytorch.org/t/efficient-computation-of-hessian-with-respect-to-network-weights-using-autograd-grad-and-symmetry-of-hessian-matrix/156222/7


def fcall(params, inputs):
    outputs = functional_call(mlp_model, params, inputs)
    return outputs


def loss_fn(outputs, targets):
    return criterion(outputs, targets)


def compute_loss(params, inputs, targets):
    # outputs = vmap(fcall, in_dims=(None, 0))(params, inputs)  # vectorize over batch
    outputs = fcall(params, inputs)
    return loss_fn(outputs, targets)


def compute_hessian_loss(params, inputs, targets):
    return hessian(compute_loss, argnums=(0))(params, inputs, targets)



if __name__ == '__main__':
    HOME = "./chaotic_neurips22/experiments/"
    DATA_PATH = path.join(HOME, "data")
    data_name = "usps"
    batch_size = 8
    optim_name = "sgd" # sgd, cgd-chaotic, cgd-pos, cgd-neg
    train_loader, val_loader = prepare_ds(
        DATA_PATH, data_name, batch_size, debug=False
    )
    # -----
    criterion = nn.CrossEntropyLoss()
    mlp_model = nn.Sequential(
        Reshape(256),
        nn.Linear(256, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Softmax()
    )
    print(mlp_model)
    params_name = list(dict(mlp_model.named_parameters()).keys())
    # -----
    lr = 0.01
    momentum = 0.0
    epochs = 5
    optimizer = optim.SGD(mlp_model.parameters(), lr=lr, momentum=momentum)
    time0 = time()
    loss_list = []
    acc_list = []
    par_hess_dict = {p:[] for p in params_name}
    par_hess_eig_dict = {p:[] for p in par_hess_dict}
    par_lm_dict = {p:[] for p in par_hess_dict} # Lyapunov matrix
    par_tangent_dict = {p:[] for p in par_hess_dict} # tangent map
    par_tune_tangent_dict = {p:[] for p in par_hess_dict} # tuned tangent map
    acc = np.nan
    for e in range(epochs):
        pbar = tqdm(train_loader, desc="[ Training | Epoch {}/{}]".format(e+1, epochs))
        # for images, labels in train_loader:
        for i_batch, feed_dict in enumerate(pbar):
            images = feed_dict[0]
            labels = feed_dict[1]
            mlp_model.train()
            optimizer.zero_grad()
            output = mlp_model(images)
            loss = criterion(output, labels)
            loss.backward()
            # -----
            # Compute Hessian matrix and corresponding eigen-value(vector).
            if "cgd" in optim_name:
                params = dict(mlp_model.named_parameters())
                hess = compute_hessian_loss(params, images, labels)
                for p in params_name:
                    p_hess = hess[p][p].view(np.prod(params[p].shape).item(), -1)
                    p_lambda, p_v = torch.linalg.eigh(p_hess)
                    # p_lm = 0.5 * torch.log(torch.square(torch.eye(p_hess.shape[0]) - lr * p_hess))
                    # par_hess_dict[p].append(p_hess)
                    # par_hess_eig_dict[p].append((p_lambda, p_v))
                    # par_lm_dict[p].append(p_lm)
                    # if iter == 0:
                    #     par_tangent_dict[p].append(torch.eye(p_hess.shape[0]))
                    # else:
                    #     par_tangent_dict[p].append(
                    #         torch.mul(torch.eye(p_hess.shape[0]) - lr * p_hess, par_tangent_dict[p][-1]))
                    # -----
                    # pruning
                    par_content = params[p]
                    par_hess_w = p_lambda
                    par_hess_V = p_v
                    par_w = (1 - lr * par_hess_w) ** 2  # eigen value of Lyapunov matrix
                    if "chaotic" in optim_name:
                        keep_idx = (par_w <= 1)
                    elif "pos" in optim_name:
                        keep_idx = (par_hess_w < 0)
                    elif "neg" in optim_name:
                        keep_idx = (par_hess_w > 0)
                    # update parameter gradients
                    V_ = par_hess_V[:, keep_idx]
                    V_T = V_.T
                    grads_ = torch.einsum("ij,j->i", V_T, par_content.grad.view(-1))
                    grads_ = torch.einsum("ij,j->i", V_, grads_)
                    par_content.grad = grads_.view(par_content.grad.size())
                    # del V_, V_T, grads_, par_hess_w, par_hess_V
                    # par_hess_eig_dict[p][-1] = par_hess_eig_dict[p][-1][0]
            ######
            optimizer.step()
            # -----
            # Validation
            if i_batch % 100 == 0:
                mlp_model.eval()
                correct_count, all_count = 0, 0
                for images, labels in val_loader:
                    logps = mlp_model(images)
                    pred_label = torch.argmax(logps, dim=1)
                    correct_count += torch.sum(pred_label == labels)
                    all_count += len(labels)
                acc = correct_count / all_count
                acc_list.append(acc.item())
            # -----
            loss_list.append(loss.item())
            # print("[ Epoch {} | Iter {} ] Train loss={:.4f}, Val Acc={:.4f}".format(e, iter, loss, acc))
            pbar.set_postfix({
                "Train loss": "{:.4f}".format(loss),
                "Val Acc": "{:.4f}".format(acc),
            })
    print("=" * 100)
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    # -----
    np.save(
        "./res/Pruning/{}-{}-log.npy".format(data_name, optim_name),
        {
            "loss_list": loss_list,
            "acc_list": acc_list,
            "par_hess_dict": par_hess_dict,
            "par_hess_eig_dict": par_hess_eig_dict,
            "par_lm_dict": par_lm_dict,
            "par_tangent_dict": par_tangent_dict,
        })
    # # -----
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # ax1.plot(np.arange(len(loss_list)), loss_list, lw=1.5)
    # ax1.set_title("Train Loss")
    # ax2.plot(np.arange(len(acc_list)), acc_list, lw=1.5)
    # ax2.set_title("Val Acc")
    # plt.tight_layout()
    # plt.show()