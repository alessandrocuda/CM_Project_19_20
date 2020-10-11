import sys
from os import path
sys.path.insert(0, "../../ISANet/")
sys.path.insert(0, "../../experiments/")
sys.path.insert(0, "./experiments/")

from utils import optimize_monk_f, print_result, print_ls_result, save_csv, rate
from isanet.optimizer import NCG, LBFGS
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


#############################
monk     =  "3"
reg      =  1e-4
seed     =  783 
ng_eps   =  10e-6
l_eps    =  1e-6
max_iter =  1000
verbose  =  0
#############################

print(" - Start Monk{} comparison -".format(monk))
print()

#############################
#          NCG FR
#############################
beta_method = "fr" 
c1          = 1e-4 
c2          = .1 
restart     = 6 
ln_maxiter  = 100
#############################
optimizer = NCG(beta_method=beta_method, c1=c1, c2=c2, restart=restart, 
                ln_maxiter = ln_maxiter, norm_g_eps = ng_eps, l_eps = l_eps)
model_history, opt_history, time = optimize_monk_f(monk=monk, 
                                                   reg = reg, 
                                                   seed=seed, 
                                                   optimizer=optimizer, 
                                                   max_iter = max_iter, 
                                                   verbose = verbose)
print(" - NCG FR -")
print_result(f="Monk"+monk, opt=("NCG "+beta_method).upper(), c1=c1, c2=c2, r=restart, m="-", history=model_history, opt_history=opt_history, time=time)
print_ls_result(ls_max_iter=ln_maxiter, opt_history=opt_history)
f_fr = model_history["loss_mse_reg"]
p_fr = rate(model_history["loss_mse_reg"]) 


#############################
#          NCG PR+
#############################
beta_method = "pr+" 
c1          = 1e-4 
c2          = .1
restart     = None
ln_maxiter  = 100
#############################
optimizer = NCG(beta_method=beta_method, c1=c1, c2=c2, restart=restart, 
                ln_maxiter = ln_maxiter, norm_g_eps = ng_eps, l_eps = l_eps)
model_history, opt_history, time = optimize_monk_f(monk=monk, 
                                                   reg = reg, 
                                                   seed=seed, 
                                                   optimizer=optimizer, 
                                                   max_iter = max_iter, 
                                                   verbose = verbose)
print(" - NCG PR+ -")
print_result(f="Monk"+monk, opt=("NCG "+beta_method).upper(), c1=c1, c2=c2, r=restart, m="-", history=model_history, opt_history=opt_history, time=time)
print_ls_result(ls_max_iter=ln_maxiter, opt_history=opt_history)
f_prp = model_history["loss_mse_reg"]
p_prp = rate(model_history["loss_mse_reg"]) 


#############################
#          NCG HS+
#############################
beta_method = "hs+" 
c1          = 1e-4 
c2          = .3
restart     = None
ln_maxiter  = 100
#############################
optimizer = NCG(beta_method=beta_method, c1=c1, c2=c2, restart=restart, 
                ln_maxiter = ln_maxiter, norm_g_eps = ng_eps, l_eps = l_eps)
model_history, opt_history, time = optimize_monk_f(monk=monk, 
                                                   reg = reg, 
                                                   seed=seed, 
                                                   optimizer=optimizer, 
                                                   max_iter = max_iter, 
                                                   verbose = verbose)
print(" - NCG HS+ -")
print_result(f="Monk"+monk, opt=("NCG "+beta_method).upper(), c1=c1, c2=c2, r=restart, m="-", history=model_history, opt_history=opt_history, time=time)
print_ls_result(ls_max_iter=ln_maxiter, opt_history=opt_history)
f_hsp = model_history["loss_mse_reg"]
p_hsp = rate(model_history["loss_mse_reg"]) 


#############################
#          L-BFGS
#############################
c1          = 1e-4 
c2          = .9 
m           = 30
ln_maxiter  = 100
#############################
optimizer = LBFGS(m = m, c1=c1, c2=c2, ln_maxiter = ln_maxiter, norm_g_eps = ng_eps, l_eps = l_eps)
model_history, opt_history, time = optimize_monk_f(monk=monk, 
                                                   reg = reg, 
                                                   seed=seed, 
                                                   optimizer=optimizer, 
                                                   max_iter = max_iter, 
                                                   verbose = verbose)
print(" - LBFGS - results")
print_result(f="Monk"+monk, opt="L-BFGS", c1=c1, c2=c2, r="-", m=m, history=model_history, opt_history=opt_history, time=time)
print_ls_result(ls_max_iter=ln_maxiter, opt_history=opt_history)
f_lbfgs = model_history["loss_mse_reg"]
p_lbfgs = rate(model_history["loss_mse_reg"]) 

##############################
# plot
##############################
pos_train = (0,0)
figsize = (12, 4)

plt.plot(f_fr - f_fr[-1], linestyle='-')
plt.plot(f_prp - f_prp[-1],linestyle = '--')
plt.plot(f_hsp - f_hsp[-1], linestyle='-.')
plt.plot(f_lbfgs - f_lbfgs[-1], linestyle=':')
plt.title('Monk{} - seed={}'.format(monk, seed))
plt.ylabel("Loss")
plt.xlabel('Iteration')
plt.grid()
plt.yscale('log')
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS+', 'L-BFGS - m={}'.format(m)], loc='upper right', fontsize='large')    
plt.show()

plt.plot(p_fr, linestyle='-')
plt.plot(p_prp,linestyle = '--')
plt.plot(p_hsp, linestyle='-.')
plt.plot(p_lbfgs, linestyle=':')
plt.title('Monk{} - seed={}'.format(monk, seed))
plt.ylabel("p")
plt.xlabel('Iteration')
plt.grid()
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS+', 'L-BFGS - m={}'.format(m)], loc='upper right', fontsize='large')    
plt.show()

