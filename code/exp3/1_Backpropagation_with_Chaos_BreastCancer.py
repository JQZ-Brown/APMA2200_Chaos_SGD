'''
Description:
    Re-implementation of the paper "Back-propagation with Chaos".
    Implement classification tasks on the Breast Cancer Wisconsin dataset.

Author:
    Jiaqi Zhang

Reference:
    1. Fazayeli, F., Wang, L., & Liu, W. (2008, June). Back-propagation with chaos.
       In 2008 International Conference on Neural Networks and Signal Processing (pp. 5-8). IEEE.
    2. Breast Cancer Wisconsin : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
'''
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import zscore

# ========================================

# Load data
data = pd.read_csv("../../data/breast_cancer/wdbc.data", sep=",", header=None, index_col=None).values
X = np.asarray(data[:, 2:12], dtype=float)
Y = data[:, 1]

# Stardardization
X = zscore(X, axis=0)
X = np.nan_to_num(X)
# One-hot encoding
Y_onehot = np.zeros((len(Y), 2))
Y_onehot[np.where(Y == "M")[0], 0] = 1.0
Y_onehot[np.where(Y == "B")[0], 1] = 1.0
Y = Y_onehot

# Split into training:validation:testing=6:2:2
list_idx = np.arange(X.shape[0])
# np.random.shuffle(list_idx)
train_idx, remaining_idx = np.split(list_idx, [int(0.6 * len(list_idx))])
valid_idx, test_idx = np.split(remaining_idx, [int(0.5 * len(remaining_idx))])
train_X, valid_X, test_X = X[train_idx, :], X[valid_idx, :], X[test_idx, :]
train_Y, valid_Y, test_Y = Y[train_idx, :], Y[valid_idx, :], Y[test_idx, :]

# ========================================

class UniformClipper(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.alpha, self.alpha)
            module.weight.data = w

# ========================================
# MLP structure: 10-2-2-2
bp_model = nn.Sequential(
    nn.Linear(10, 2),
    nn.Tanh(),
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 2),
    nn.Tanh(),
) # conventional BP
# All initial weights were randomly chosen from a uniform distribution in the range (-2.0, 2.0)
for par in bp_model.parameters():
    nn.init.uniform_(par, -0.2, 0.2)
bpc_model = copy.deepcopy(bp_model) # BP + chaos

# Model training
train_X = torch.FloatTensor(train_X)
train_Y = torch.FloatTensor(train_Y)
test_X = torch.FloatTensor(test_X)
test_Y = torch.FloatTensor(test_Y)
n_iters = 3000
lr = 5.0
bp_lr = 0.1
beta = 1e-3
alpha = 0.9
clipper = UniformClipper(alpha)
loss_fn = nn.MSELoss()
bp_optimizer = torch.optim.SGD(bp_model.parameters(), lr=bp_lr)
bpc_optimizer = torch.optim.SGD(bpc_model.parameters(), lr=lr)
train_loss_list = []
tests_loss_list = []
for t in range(n_iters):
    bp_model.train()
    bpc_model.train()
    bp_optimizer.zero_grad()
    bpc_optimizer.zero_grad()
    # BP
    bp_out = bp_model(train_X)
    bp_loss = loss_fn(bp_out, train_Y)
    bp_loss.backward()
    bp_optimizer.step()
    # BP + chaos
    bpc_out = bpc_model(train_X)
    bpc_loss = loss_fn(bpc_out, train_Y)
    bpc_loss.backward()
    bpc_optimizer.step()
    bpc_model.apply(clipper)
    train_loss_list.append((bp_loss.item(), bpc_loss.item()))
    # shrink learning rate
    for g in bpc_optimizer.param_groups:
        g['lr'] = (1-beta) * g['lr']
    if t % 100 == 0:
        print("MSE Loss (training): BP={:.3f}, BPC={:.3f}".format(bp_loss, bpc_loss))
    # -----
    bp_model.eval()
    bpc_model.eval()
    bp_out = bp_model(test_X)
    bp_loss = loss_fn(bp_out, test_Y)
    bpc_out = bpc_model(test_X)
    bpc_loss = loss_fn(bpc_out, test_Y)
    tests_loss_list.append((bp_loss.item(), bpc_loss.item()))

# ========================================
# Prediction on testing set
bp_model.eval()
bpc_model.eval()
bp_pred_Y = bp_model(test_X)
bpc_pred_Y = bpc_model(test_X)

bp_mse = loss_fn(bp_pred_Y, test_Y)
bpc_mse = loss_fn(bpc_pred_Y, test_Y)
print("[ Testing MSE ] BP={:.4f} | BPC={:.4f}".format(bp_mse, bpc_mse))

bp_acc = torch.sum(torch.argmax(test_Y, dim=1) == torch.argmax(bp_pred_Y, dim=1)) / test_Y.shape[0]
bpc_acc = torch.sum(torch.argmax(test_Y, dim=1) == torch.argmax(bpc_pred_Y, dim=1)) / test_Y.shape[0]
print("[ Testing Accuracy ] BP={:.4f} | BPC={:.4f}".format(bp_acc, bpc_acc))


plt.figure(figsize=(10, 5.5))
plt.title("Training Loss during Training")
plt.plot(np.arange(n_iters), [x[0] for x in train_loss_list], "-k", lw=1, label="BP")
plt.plot(np.arange(n_iters), [x[1] for x in train_loss_list], "--r", lw=1, label="BP+Chaos")
plt.xlabel("Iters")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5.5))
plt.title("Testing Loss during Training")
plt.plot(np.arange(n_iters), [x[0] for x in tests_loss_list], "-k", lw=1, label="BP")
plt.plot(np.arange(n_iters), [x[1] for x in tests_loss_list], "--r", lw=1, label="BP+Chaos")
plt.xlabel("Iters")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# np.save("./res/BP_Chaos/WBC-acc.npy", {
#     "test_loss": tests_loss_list,
#     "train_loss": train_loss_list,
#     "bp_acc": bp_acc,
#     "bpc_acc": bpc_acc,
# })
