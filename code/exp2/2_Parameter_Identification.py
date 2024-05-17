from scipy.optimize import minimize
import numpy as np

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


def _minFunc(x, init_state, timepoints, system_func, true_traj):
    pred_traj = system_func(init_state, timepoints, *x)
    pred_traj = np.asarray(pred_traj)
    rmse = np.sqrt(np.linalg.norm(pred_traj - true_traj, ord="fro"))
    return rmse


def parIdentWithInit(system_name, timepoints, pars):
    '''
    Parameter identification through minimizing the MSE. Initial conditions are known.
    '''
    system_func = system_dict[system_name]
    # sample true trajectory
    init_state = np.array([0.1, 0.1, 0.1])
    true_pars = pars
    true_traj = system_func(init_state, timepoints, **pars)
    true_traj = np.asarray(true_traj)
    # identify parameters
    if system_name == "lorenz":
        x0 = [5.0, 20.0, 2.0]
    elif system_name == "rossler":
        x0 = [0.1, 0.1, 5.0]
    else:
        raise ValueError("Unknown system name {}!".format(system_name))
    print("[Initial Guess] {}".format(x0))
    res = minimize(_minFunc, x0, args=(init_state, timepoints, system_func, true_traj), method='Nelder-Mead', tol=1e-6)
    return res

# ======================================

def parIdentWithDisruptInit(system_name, timepoints, pars):
    '''
    Parameter identification through minimizing the MSE. Disrupt initial conditions with small perturbations.
    '''
    n_disrupt = 10
    noise_scale = 0.1 # 0.01
    system_func = system_dict[system_name]
    # sample true trajectory
    init_state = np.array([0.1, 0.1, 0.1])
    dirupt_init_list = np.asarray([init_state+np.random.normal(0.0, noise_scale, init_state.shape) for _ in range(n_disrupt)])
    true_pars = pars
    print("[True Init Condition] {}".format(init_state))
    # -----
    res_list = []
    if system_name == "lorenz":
        x0 = [5.0, 20.0, 2.0]
    elif system_name == "rossler":
        x0 = [0.1, 0.1, 5.0]
    else:
        raise ValueError("Unknown system name {}!".format(system_name))
    true_traj = system_func(init_state, timepoints, **pars)
    true_traj = np.asarray(true_traj)
    print("[Initial Guess] {}".format(x0))
    for i in range(n_disrupt):
        res = minimize(_minFunc, x0, args=(dirupt_init_list[i, :], timepoints, system_func, true_traj), method='Nelder-Mead', tol=1e-6)
        res_list.append((dirupt_init_list[i, :], res))
    return res_list

# ======================================

def _minFuncWithoutInit(x, timepoints, system_func, true_traj):
    pred_init = x[:3]
    pred_pars = x[3:]
    pred_traj = system_func(pred_init, timepoints, *pred_pars)
    pred_traj = np.asarray(pred_traj)
    rmse = np.sqrt(np.linalg.norm(pred_traj - true_traj, ord="fro"))
    return rmse


def parIdentWithoutInit(system_name, timepoints, pars):
    '''
    Parameter identification through minimizing the MSE. Initial conditionas are known.
    '''
    system_func = system_dict[system_name]
    # sample true trajectory
    init_state = np.array([0.1, 0.1, 0.1])
    true_pars = pars
    true_traj = system_func(init_state, timepoints, **pars)
    true_traj = np.asarray(true_traj)
    print("[True Init Condition] {}".format(init_state))
    # identify parameters
    if system_name == "lorenz":
        x0 = [0.0, 0.0, 0.0, 5.0, 20.0, 2.0]
    elif system_name == "rossler":
        x0 = [0.0, 0.0, 0.0, 0.1, 0.1, 5.0]
    else:
        raise ValueError("Unknown system name {}!".format(system_name))
    print("[Initial Guess] {}".format(x0))
    res = minimize(_minFuncWithoutInit, x0, args=(timepoints, system_func, true_traj), method='Nelder-Mead', tol=1e-6)
    return res





if __name__ == '__main__':
    system_name = "rossler" # lorenz, rossler
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
    # # -----
    # print("-" * 70)
    # print("Parameter identification with known init condition...")
    # pred_res = parIdentWithInit(system_name, timepoints, pars)
    # print("[Predicted Parameters] {}".format(pred_res.x))
    # print("-" * 70)
    # print(pred_res)
    # # -----
    # print("-" * 70)
    # print("Parameter identification with disrupted init condition...")
    # pred_res_list = parIdentWithDisruptInit(system_name, timepoints, pars)
    # print("[Predicted Parameters] {}".format(np.asarray([res[1].x for res in pred_res_list])))
    # print("[Predicted Parameters Residual] {}".format([np.linalg.norm(res[1].x-np.asarray(list(pars.values()))) for res in pred_res_list]))
    # print("-" * 70)
    # -----
    print("-" * 70)
    print("Parameter identification without init condition...")
    pred_res = parIdentWithoutInit(system_name, timepoints, pars)
    print("[Predicted Parameters] {}".format(pred_res.x))
    print("-" * 70)
    print(pred_res)

