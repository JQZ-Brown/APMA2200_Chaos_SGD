import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import zscore

ion_res = np.load("./res/BP_Chaos/Ionosphere-acc.npy", allow_pickle=True).item()
wbc_res = np.load("./res/BP_Chaos/WBC-acc.npy", allow_pickle=True).item()

ion_test_loss = ion_res["test_loss"]
ion_bp_acc = ion_res["bp_acc"]
ion_bpc_acc = ion_res["bpc_acc"]

wbc_test_loss = wbc_res["test_loss"]
wbc_bp_acc = wbc_res["bp_acc"]
wbc_bpc_acc = wbc_res["bpc_acc"]

n_iters = 3000

fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("Ionosphere")
ax1.plot(np.arange(n_iters), [x[0] for x in ion_test_loss], "--k", lw=1, label="BP")
ax1.plot(np.arange(n_iters), [x[1] for x in ion_test_loss], "-r", lw=1, label="BP+Chaos")
ax1.set_xlabel("Training Iteration")
ax1.set_ylabel("Testing Loss")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("WBC")
ax2.plot(np.arange(n_iters), [x[0] for x in wbc_test_loss], "--k", lw=1, label="BP")
ax2.plot(np.arange(n_iters), [x[1] for x in wbc_test_loss], "-r", lw=1, label="BP+Chaos")
ax2.set_xlabel("Training Iteration")
ax2.set_ylabel("Testing Loss")

ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
plt.tight_layout()
plt.savefig("../../figs/BP_chaos.pdf", dpi=600)
plt.show()


print("[ ION Testing Accuracy ] BP={:.4f} | BPC={:.4f}".format(ion_bp_acc, ion_bpc_acc))
print("[ WBC Testing Accuracy ] BP={:.4f} | BPC={:.4f}".format(wbc_bp_acc, wbc_bpc_acc))