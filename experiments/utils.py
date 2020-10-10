import sys
from os import path
sys.path.insert(0, "./ISANet/")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD, NCG, LBFGS
from isanet.optimizer.utils import l_norm
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import isanet.metrics as metrics
import numpy as np
import time
from tabulate import tabulate
import pandas as pd

def optimize_monk_f(monk="1", reg = 0, seed=1, optimizer=None, max_iter = 1000, verbose = 0):
    if monk not in ["1","2", "3"]:
        raise Exception("wrong monk function - accepeted only: '1','2', '3'")
    if optimizer is None:
        raise Exception("an optimatizer must be specifed")
    np.random.seed(seed=seed)
    X_train, Y_train = load_monk(monk, "train")
    kernel_initializer = { "1": [0.003, 0.003],
                           "2": [1/np.sqrt(17), 1/np.sqrt(4)],
                           "3": [0.003, 0.003]}
    model = Mlp()
    model.add(4, input= 17, kernel_initializer = kernel_initializer[monk][0], kernel_regularizer = reg)
    model.add(1, kernel_initializer = kernel_initializer[monk][1], kernel_regularizer = reg)

    model.set_optimizer(optimizer)
    start = time.time()
    model.fit(X_train,
            Y_train, 
            epochs=max_iter, 
            verbose=verbose)
    end = time.time()
    return model.history, optimizer.history, (end - start)


def print_result(f, opt,c1, c2, r, m, history, opt_history, time, latex = False):
    table = [[f, opt, c1, c2, r, m, "{:.2e}".format(history["loss_mse_reg"][-1]), "{:.2e}".format(opt_history["norm_g"][-1]), len(history["loss_mse_reg"]), "{:.2f}".format(time)]]
    header = ["f", "Optimizer", "c1", "c2", "restart", "m", "Loss", "‖gk‖", "Conv. Iter.", "Time (s)"]
    print(tabulate(table, headers=header, tablefmt="fancy_grid"))
    if latex is True:
        print("latex table row:")
        print("${}$ & {} & {} & {}  & {}  & {}  & {:.2e}  & {:.2e} & {}  & {:.2f}".format(
               f, opt, c1, c2, r, m, history["loss_mse_reg"][-1], opt_history["norm_g"][-1], len(history["loss_mse_reg"]), time))

def print_ls_result(ls_max_iter, opt_history, latex = False):
    converged = 0
    tot_iteration = 0
    tot = len(opt_history["ls_conv"])
    for i in range(tot):
        tot_iteration += opt_history["ls_it"][i] + opt_history["zoom_it"][i]
        if opt_history["ls_conv"][i] == "y":
            converged += 1

    table = [[ls_max_iter, tot_iteration, "{:.2f}".format(converged/tot), "{:.2f}".format(np.sum(opt_history["ls_time"]))]]
    header = ["Ls Max Iter.", "Ls Iter.", "Ls Hit Rate", "Ls Time (s)"]
    print(tabulate(table, headers=header, tablefmt="fancy_grid"))
    if latex is True:
        print("latex table row:")
        print(" {} & {} & {:.2f}  & {:.2f} ".format(ls_max_iter, tot_iteration, converged/tot, np.sum(opt_history["ls_time"])))

def save_csv(path, f, model_history, opt_history):
    h = {"f": model_history["loss_mse_reg"],
         "time_iter": model_history["epoch_time"]}
    mpd = pd.DataFrame.from_dict(h)
    opd = pd.DataFrame.from_dict(opt_history)
    df = pd.concat([mpd, opd], axis=1)
    df.to_csv(path, index=False)
    return df

def rate(e):
    d = np.abs(e - e[-1])
    p = []
    for i in range(len(d)-1):
        p.append(np.log(d[i+1])/np.log(d[i]))
    return p