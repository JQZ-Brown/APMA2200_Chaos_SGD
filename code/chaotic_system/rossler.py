'''
Description:
    Rossler system.

Author:
    Jiaqi Zhang
'''
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt



def _rossler(current_state, t, a, b, c):
    x, y, z = current_state
    dx_dt = -y - z
    dy_dt = x + a*y
    dz_dt = b + z*(x-c)
    return dx_dt, dy_dt, dz_dt



def rossler_system(init_state, timepoints, a, b, c):
    # use odeint() to solve a system of ordinary differential equations
    # the arguments are:
    # 1, a function - computes the derivatives
    # 2, a vector of initial system conditions (aka x, y, z positions in space)
    # 3, a sequence of time points to solve for
    # returns an array of x, y, and z value arrays for each time point, with the initial values in the first row
    xyz = odeint(_rossler, init_state, timepoints, args=(a, b, c))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return x, y, z



if __name__ == '__main__':
    # define the initial system state (aka x, y, z positions in space)
    # initial_state = [0.1, 0, 0]
    initial_state = [0.1, 0.1, 0.1]

    # define the system parameters a, b, c
    a = 0.2
    b = 0.2
    c = 5.7

    # define the time points to solve for, evenly spaced between the start and end times
    start_time = 0
    end_time = 100
    num_points = 10240
    timepoints = np.linspace(start_time, end_time, num_points)

    x, y, z = rossler_system(initial_state, timepoints, a, b, c)

    # now plot two-dimensional cuts of the three-dimensional phase space
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))
    # plot the x values vs the y values
    ax[0].plot(x, y, color='r', alpha=0.7, linewidth=1)
    ax[0].set_title('x-y phase plane')
    # plot the x values vs the z values
    ax[1].plot(x, z, color='m', alpha=0.7, linewidth=1)
    ax[1].set_title('x-z phase plane')
    # plot the y values vs the z values
    ax[2].plot(y, z, color='b', alpha=0.7, linewidth=1)
    ax[2].set_title('y-z phase plane')
    plt.tight_layout()
    plt.show()

    # now plot three-dimensional phase space
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))
    # plot the x values vs the y values
    ax[0].plot(timepoints, x, color='r', alpha=0.7, linewidth=0.3)
    ax[0].set_title('x-t')
    # plot the x values vs the z values
    ax[1].plot(timepoints, y, color='m', alpha=0.7, linewidth=0.3)
    ax[1].set_title('y-t')
    # plot the y values vs the z values
    ax[2].plot(timepoints, z, color='b', alpha=0.7, linewidth=0.3)
    ax[2].set_title('z-t')
    plt.tight_layout()
    plt.show()
