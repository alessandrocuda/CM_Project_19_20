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
    reg = 1e-4
    model.add(4, input= 17, kernel_initializer = 0.003, kernel_regularizer = reg)
    model.add(1, kernel_initializer = 0.003, kernel_regularizer = reg)

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
            f, opt, c1, c2, r, m, history["loss_mse_reg"][-1], opt_history["norm_g"][-1], len(history["loss_mse_reg"]), t))
    print("------------------------------------------------------------------------------------------------------------")
    print("latex table row:")
    print("${}$ & {} & {} & {}  & {}  & {}  & {:.2e}  & {:.2e} & {}  & {:.2f}".format(
            f, opt, c1, c2, r, m, history["loss_mse_reg"][-1], opt_history["norm_g"][-1], len(history["loss_mse_reg"]), t))

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

def rate(e):
    d = np.abs(e - e[-1])
    p = []
    for i in range(len(d)-1):
        p.append(np.log(d[i+1])/np.log(d[i]))
    return p

print("Load Monk DataSet")
X_train, Y_train = load_monk("3", "train")

seed = 783
ls_maxiter = 100
l_eps = 1e-6
norm_g_eps = 10e-6

#############################
#          NCG f1
#############################
restart = 6
optimizer = NCG(beta_method="fr", c1=1e-4, c2=.1, restart=restart, ln_maxiter = ls_maxiter,  l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_fr = model.history 
p_fr = rate(model.history["loss_mse_reg"]) 
print_history(f="Monk3", opt="NCG FR", c1="1e-4", c2=.1, r=restart, m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ls_maxiter, optimizer.history)
print()
#############################
#          NCG pr+
#############################

optimizer = NCG(beta_method="pr+", c1=1e-4, c2=.1, ln_maxiter = ls_maxiter,  l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_pr = model.history 
p_pr = rate(model.history["loss_mse_reg"]) 
print_history(f="Monk3", opt="NCG PR+", c1="1e-4", c2=.1, r="-", m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ls_maxiter, optimizer.history)
print()
#############################
#          NCG Hs+
#############################

optimizer = NCG(beta_method="hs+", c1=1e-4, c2=.3, ln_maxiter = ls_maxiter,  l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_hs = model.history 
p_hs = rate(model.history["loss_mse_reg"]) 
print_history(f="Monk3", opt="NCG HS+", c1="1e-4", c2=.3, r="-", m="-", history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ls_maxiter, optimizer.history)
print()
#############################
#          L-BFGS
#############################
m = 30
optimizer = LBFGS(m=m, c1= 1e-4, c2=0.9, ln_maxiter = ls_maxiter,  l_eps = l_eps, norm_g_eps = norm_g_eps)
model, t = get_fitted_model(X_train, Y_train, optimizer, seed, 0)
h_lbfgs = model.history 
p_lbfgs = rate(model.history["loss_mse_reg"]) 
print_history(f="Monk3", opt="L-BFGS", c1="1e-4", c2=.9, r="-", m=m, history=model.history, opt_history=optimizer.history, t=t)
print()
ls_stat(ls_maxiter, optimizer.history)

##############################
# plot
##############################

import matplotlib.pyplot as plt

pos_train = (0,0)
figsize = (12, 4)

plt.plot(h_fr["loss_mse_reg"] - h_fr["loss_mse_reg"][-1], linestyle='-')
plt.plot(h_pr["loss_mse_reg"] - h_pr["loss_mse_reg"][-1],linestyle = '--')
plt.plot(h_hs["loss_mse_reg"] - h_hs["loss_mse_reg"][-1], linestyle='-.')
plt.plot(h_lbfgs["loss_mse_reg"] - h_lbfgs["loss_mse_reg"][-1], linestyle=':')
plt.title('Monk3 - seed={}'.format(seed))
plt.ylabel("Loss")
plt.xlabel('Iteration')
plt.grid()
plt.yscale('log')
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS+', 'L-BFGS - m={}'.format(m)], loc='upper right', fontsize='large')    
plt.show()

plt.plot(p_fr, linestyle='-')
plt.plot(p_pr,linestyle = '--')
plt.plot(p_hs, linestyle='-.')
plt.plot(p_lbfgs, linestyle=':')
plt.title('Monk3 - seed={}'.format(seed))
plt.ylabel("p")
plt.xlabel('Iteration')
plt.grid()
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS+', 'L-BFGS - m={}'.format(m)], loc='upper right', fontsize='large')    
plt.show()

