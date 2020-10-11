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
    """Allows you to build a target fusion from a monk dataset,
    and then optimize it with a specific "optimizer".
    The function is built as follows:
                    f_monk = MSE_monk + reg*||w||^2
    Parameters
    ----------

    monk : String
        Allow to specify the monk dataset.
        Accepted values:
            "1" : monk1
            "2" : monk2
            "3" : monk3
    
    reg : float 
        lamda value used in the regualrization term
        reg must be in 0 <= reg <= 1

    seed : int
        Allow to specify a starting point
    
    optimizer : isanet.optimizer
        Allow to specify the optimizer used in the process.
        Must be passed!
    
    max_iter : int
        Define the max iteration in the optimization 
        process as stop criteria.

    verbose : int [0, 1, 2, 3]
        Define the verbosity of the process.

            "0" : no output
            "1" : only iteration output
            "2" : iteration and optimizer log at each iteration
            "3" : same as above, plus the line search log 

    Returns
    -------
        model.history, optimizer.history, (end - start)
            It return the model and optimizer history plus the time of the whole process
    """

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
    """Print in a table in the following form::

                ╒═══════╤═════════════╤════════╤══════╤═══════════╤═════╤════════╤══════════╤═══════════════╤════════════╕
                │ f     │ Optimizer   │     c1 │   c2 │   restart │ m   │   Loss │     ‖gk‖ │   Conv. Iter. │   Time (s) │
                ╞═══════╪═════════════╪════════╪══════╪═══════════╪═════╪════════╪══════════╪═══════════════╪════════════╡
                │ Monk3 │ NCG FR      │ 0.0001 │  0.1 │         6 │ -   │ 0.0384 │ 6.51e-06 │           450 │       2.35 │
                ╘═══════╧═════════════╧════════╧══════╧═══════════╧═════╧════════╧══════════╧═══════════════╧════════════╛
        
        Parameters
        ----------
        f : String
            Name of the function

        opt: String
            Name of the optimizier
        
        c1 : float
            C1 value
        
        c2 : float
            C2 value
        
        r : int
            restart value of NCG
        m : int
            m value of LBFGS
        
        history : dict
            log of the model
        
        opt_history : dict
            log of the optimizer
        
        time : float
            time of the whole process
        
        latex : boolean (Default False)
            Allow to specify if a latex versione of the table must be printed
    """ 

    table = [[f, opt, c1, c2, r, m, "{:.2e}".format(history["loss_mse_reg"][-1]), "{:.2e}".format(opt_history["norm_g"][-1]), len(history["loss_mse_reg"]), "{:.2f}".format(time)]]
    header = ["f", "Optimizer", "c1", "c2", "restart", "m", "Loss", "‖gk‖", "Conv. Iter.", "Time (s)"]
    print(tabulate(table, headers=header, tablefmt="fancy_grid"))
    if latex is True:
        print("latex table row:")
        print("${}$ & {} & {} & {}  & {}  & {}  & {:.2e}  & {:.2e} & {}  & {:.2f}".format(
               f, opt, c1, c2, r, m, history["loss_mse_reg"][-1], opt_history["norm_g"][-1], len(history["loss_mse_reg"]), time))

def print_ls_result(ls_max_iter, opt_history, latex = False):
    """Print in a table in the following form::

            ╒════════════════╤════════════╤═══════════════╤═══════════════╕
            │   Ls Max Iter. │   Ls Iter. │   Ls Hit Rate │   Ls Time (s) │
            ╞════════════════╪════════════╪═══════════════╪═══════════════╡
            │            100 │       2490 │             1 │          1.84 │
            ╘════════════════╧════════════╧═══════════════╧═══════════════╛
        
        Parameters
        ----------
        ls_max_iter : int 
            max number of iteration used in the line search
        
        opt_history : dict
            log of the optimizer

        latex : boolean (Default False)
            Allow to specify if a latex versione of the table must be printed

    """
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
    """Allow to save the result in csv in a specif path.
    The model and optimizer history must be provided through
    model_history and opt_history parameters

    Returns
    -------
    df : Dataframe
        dataframe saved
    """
    h = {"f": model_history["loss_mse_reg"],
         "time_iter": model_history["epoch_time"]}
    mpd = pd.DataFrame.from_dict(h)
    opd = pd.DataFrame.from_dict(opt_history)
    df = pd.concat([mpd, opd], axis=1)
    df.to_csv(path, index=False)
    return df

def rate(e):
    """Compute the rate of convergence and return the list with the p values.
    """
    d = np.abs(e - e[-1])
    p = []
    for i in range(len(d)-1):
        p.append(np.log(d[i+1])/np.log(d[i]))
    return p