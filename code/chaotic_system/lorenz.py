'''
Description:
    Lorenz system.

Author:
    Jiaqi Zhang

Reference:
    https://github.com/gboeing/lorenz-system
'''
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt



def _lorenz(current_state, t, sigma, rho, beta):
    x, y, z = current_state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt



def lorenz_system(init_state, timepoints, sigma, rho, beta):
    # use odeint() to solve a system of ordinary differential equations
    # the arguments are:
    # 1, a function - computes the derivatives
    # 2, a vector of initial system conditions (aka x, y, z positions in space)
    # 3, a sequence of time points to solve for
    # returns an array of x, y, and z value arrays for each time point, with the initial values in the first row
    xyz = odeint(_lorenz, init_state, timepoints, args=(sigma, rho, beta))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return x, y, z



if __name__ == '__main__':
    # define the initial system state (aka x, y, z positions in space)
    # initial_state = [0.1, 0, 0]
    initial_state = [0.1, 0.1, 0.1]

    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho = 28.
    beta = 8. / 3.

    # define the time points to solve for, evenly spaced between the start and end times
    start_time = 0
    end_time = 100
    num_points = 10240
    timepoints = np.linspace(start_time, end_time, num_points)

    x, y, z = lorenz_system(initial_state, timepoints, sigma, rho, beta)

    # now plot two-dimensional cuts of the three-dimensional phase space
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))
    # plot the x values vs the y values
    ax[0].plot(x, y, color='r', alpha=0.7, linewidth=0.3)
    ax[0].set_title('x-y phase plane')
    # plot the x values vs the z values
    ax[1].plot(x, z, color='m', alpha=0.7, linewidth=0.3)
    ax[1].set_title('x-z phase plane')
    # plot the y values vs the z values
    ax[2].plot(y, z, color='b', alpha=0.7, linewidth=0.3)
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
