import sys
from os import path
sys.path.insert(0, "./ISANet/")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD, NCG, LBFGS
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np
import time
from tabulate import tabulate

def get_fitted_model(X_train, Y_train, optimizer, n_seed = 189, verbose = 1):
    np.random.seed(seed=n_seed)
    print("Build the model")
    model = Mlp()
    model.add(4, input= 17, kernel_initializer = 0.003, kernel_regularizer = 0.001)
    model.add(1, kernel_initializer = 0.003, kernel_regularizer = 0.001)

    model.set_optimizer(optimizer)
    start = time.time()
    model.fit(X_train,
            Y_train, 
            epochs=1000, 
            verbose=verbose)
    end = time.time()
    return model, (end - start)


def print_history(f, opt,c1, c2, r, m, history, opt_history, t):
    print("------------------------------------------------------------------------------------------------------------")
    print("     f   | Optimizer |    c1    |   c2   | restart |   m   |    Loss   |    ‖gk‖    | Conv. Iter. | Time (s) ")
    print("   {}     {}       {}      {}        {}        {}     {:.2e}      {:.2e}       {}         {:.2f}".format(
            f, opt, c1, c2, r, m, history["loss_mse"][-1], opt_history["norm_g"][-1], len(history["loss_mse"]), t))
    print("------------------------------------------------------------------------------------------------------------")
    print("latex table row:")
    print("${}$ & {} & {} & {}  & {}  & {}  & {:.2e}  & {:.2e} & {}  & {:.2f}".format(
            f, opt, c1, c2, r, m, history["loss_mse"][-1], opt_history["norm_g"][-1], len(history["loss_mse"]), t))

def ls_stat(ls_max_iter, info):
    converged = 0
    tot_iteration = 0
    tot = len(info["ls_conv"])
    for i in range(tot):
        tot_iteration += info["ls_it"][i] + info["zoom_it"][i]
        if info["ls_conv"][i] == "y":
            converged += 1
    print("---------------------------------------------------------")
    print("    Ls Max Iter.   | Ls Iter. | Ls Hit Rate | Ls Time (s)")
    print("        {}            {}        {:.2f}        {:.2f}".format(ls_max_iter, tot_iteration, converged/tot, np.sum(info["ls_time"])))
    print("----------------------------------------------------------")
    print("latex table row:")
    print(" {} & {} & {:.2f}  & {:.2f} ".format(ls_max_iter, tot_iteration, converged/tot, np.sum(info["ls_time"])))


print("Load Monk DataSet")
X_train, Y_train = load_monk("1", "train")

seed = 206
seed_buoni = [6, 206]
results = []
ln_maxiter = 100
l_eps = 1e-13
norm_g_eps = 1e-13
#############################
#          NCG f1
#############################
restart = 3
optimizer = NCG(beta_method="fr", c1=1e-4, c2=.3, restart=restart, ln_maxiter = ln_maxiter, l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_fr = model.history
print_history(f="Monk1", opt="NCG FR", c1="1e-4", c2=.3, r=restart, m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ln_maxiter, optimizer.history)
print()

#############################
#          NCG pr+
#############################

optimizer = NCG(beta_method="pr+", c1=1e-4, c2=.4, ln_maxiter = ln_maxiter, l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_pr = model.history
print_history(f="Monk1", opt="NCG PR+", c1="1e-4", c2=.4, r="-", m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ln_maxiter, optimizer.history)
print()
#############################
#          NCG Hs+
#############################

optimizer = NCG(beta_method="hs+", c1=1e-4, c2=.6, ln_maxiter = ln_maxiter, l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_hs = model.history 
print_history(f="Monk1", opt="NCG HS+", c1="1e-4", c2=.6, r="-", m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ln_maxiter, optimizer.history)
print()
#############################
#          L-BFGS
#############################
m=3
optimizer = LBFGS(m=m, c1= 1e-4, c2=0.9, ln_maxiter = ln_maxiter, l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_lbfgs = model.history 
print_history(f="Monk1", opt="L-BFGS", c1="1e-4", c2=.6, r="-", m=m, history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ln_maxiter, optimizer.history)

##############################
# plot
##############################

import matplotlib.pyplot as plt

pos_train = (0,0)
figsize = (12, 4)

plt.plot(h_fr["loss_mse"], linestyle='-')
plt.plot(h_pr["loss_mse"], linestyle = '--')
plt.plot(h_hs["loss_mse"], linestyle='-.')
plt.plot(h_lbfgs["loss_mse"], linestyle=':')
plt.title('Monk1 - seed={}'.format(seed))
plt.ylabel("Loss")
plt.xlabel('Iteration')
plt.grid()
plt.yscale('log')
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS+', 'L-BFGS - m={}'.format(m)], loc='lower left', fontsize='large')    
plt.show()
