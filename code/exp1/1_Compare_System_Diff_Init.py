import numpy as np
import matplotlib.pyplot as plt

from codes.chaotic_system.lorenz import lorenz_system
from codes.chaotic_system.rossler import rossler_system
from codes.plotting import *

system_dict = {
    "lorenz": lorenz_system,
    "rossler": rossler_system,
}


# ======================================

def init_pars(system_name):
    if system_name == "lorenz":
        sigma = 10.0
        rho = 25.0
        beta = 4.0
        pars = {"sigma": sigma, "rho": rho, "beta": beta}
    elif system_name == "rossler":
        a = 0.2
        b = 0.2
        c = 5.5
        pars = {"a": a, "b": b, "c": c}
    else:
        raise ValueError("Unknown system name {}!".format(system_name))
    return pars



def runMultipleTrials(system_name, timepoints, num_trials, pars):
    # sample init states
    init_state_list = np.random.normal(0, 0.1, (num_trials, 3))
    # run dynamic system
    trial_traj = [system_dict[system_name](init_state_list[i,:], timepoints, **pars) for i in range(num_trials)]
    trial_traj = np.asarray(trial_traj)
    return trial_traj


# ======================================

def plotTrial(trajs):
    #NOTE: for now, only compare the first two trials
    trial1 = trajs[0, :, :]
    trial2 = trajs[1, :, :]
    first_color = "r"
    second_color = "b"
    # -----
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))
    x, y, z = trial1
    ax[0].plot(x, y, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[0].set_title('x-y phase plane')
    ax[1].plot(x, z, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[1].set_title('x-z phase plane')
    ax[2].plot(y, z, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[2].set_title('y-z phase plane')

    x, y, z = trial2
    ax[0].plot(x, y, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[0].set_title('x-y phase plane')
    ax[1].plot(x, z, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[1].set_title('x-z phase plane')
    ax[2].plot(y, z, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[2].set_title('y-z phase plane')
    plt.tight_layout()
    plt.show()

    # -----
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))
    x, y, z = trial1
    ax[0].plot(timepoints, x, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[0].set_title('x-t')
    ax[1].plot(timepoints, y, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[1].set_title('y-t')
    ax[2].plot(timepoints, z, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax[2].set_title('z-t')

    x, y, z = trial2
    ax[0].plot(timepoints, x, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[0].set_title('x-t')
    ax[1].plot(timepoints, y, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[1].set_title('y-t')
    ax[2].plot(timepoints, z, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax[2].set_title('z-t')
    plt.tight_layout()
    plt.show()


def compareTrialStats(trajs):
    trial_mean = np.mean(trajs, axis=0)
    trial_var = np.var(trajs, axis=0)
    # -----
    color = gray_color
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(timepoints, trial_var[0, :], "-", color=color, alpha=1.0, linewidth=1.0)
    ax[0].set_title('x_var-t')
    ax[1].plot(timepoints, trial_var[1, :], "-", color=color, alpha=1.0, linewidth=1.0)
    ax[1].set_title('y_var-t')
    ax[2].plot(timepoints, trial_var[2, :], "-", color=color, alpha=1.0, linewidth=1.0)
    ax[2].set_title('z_var-t')
    plt.tight_layout()
    plt.show()


def plotTrial3D(trajs):
    trial1 = trajs[0, :, :]
    trial2 = trajs[1, :, :]
    first_color = "r"
    second_color = "b"
    x1, y1, z1 = trial1
    x2, y2, z2 = trial2
    # plot the lorenz attractor in three-dimensional phase space
    fig = plt.figure(figsize=(12, 9))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.plot(x1, y1, z1, color=first_color, alpha=0.7, linewidth=0.6)
    ax.plot(x2, y2, z2, color=second_color, alpha=0.7, linewidth=0.6)
    plt.show()


def plotReportFig(trajs):
    trial1 = trajs[0, :, :]
    trial2 = trajs[1, :, :]
    first_color = "r"
    second_color = "b"
    x1, y1, z1 = trial1
    x2, y2, z2 = trial2
    # -----
    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.xaxis.set_pane_color((1, 1, 1, 1))
    ax1.yaxis.set_pane_color((1, 1, 1, 1))
    ax1.zaxis.set_pane_color((1, 1, 1, 1))
    ax1.plot(x1, y1, z1, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax1.plot(x2, y2, z2, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title("x component")
    ax2.plot(timepoints, x1, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax2.plot(timepoints, x2, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax2.set_xlabel("t")
    # ax2.set_ylabel("x")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_title("y component")
    ax3.plot(timepoints, y1, "-", color=first_color, alpha=1.0, linewidth=1.0)
    ax3.plot(timepoints, y2, "--", color=second_color, alpha=1.0, linewidth=1.0)
    ax3.set_xlabel("t")
    # ax3.set_ylabel("y")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title("z component")
    ax4.plot(timepoints, z1, "-", color=first_color, alpha=1.0, linewidth=1.0, label="init 1")
    ax4.plot(timepoints, z2, "--", color=second_color, alpha=1.0, linewidth=1.0, label="init 2")
    ax4.set_xlabel("t")
    # ax4.set_ylabel("z")

    ax4.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig("../../figs/{}-traj.pdf".format(system_name), dpi=600)
    plt.show()


if __name__ == '__main__':
    system_name = "lorenz" # lorenz, rossler
    num_trials = 5
    start_time = 0
    end_time = 100
    num_points = 10240
    timepoints = np.linspace(start_time, end_time, num_points)
    pars = init_pars(system_name)
    # -----
    print("=" * 70)
    print("[System Name] {}".format(system_name))
    print("[# Timepoints] {}".format(len(timepoints)))
    print("[Parameters] {}".format(pars))
    print("[# Trials] {}".format(num_trials))
    print("-" * 70)
    print("Generating trajectories...")
    traj_res = runMultipleTrials(system_name, timepoints, num_trials, pars)
    print("Trajectory shape: {}".format(traj_res.shape))
    # plotTrial(traj_res)
    # plotTrial3D(traj_res)
    # compareTrialStats(traj_res)
    # -----
    plotReportFig(traj_res)