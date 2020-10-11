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

#########################################
# Construct the Monk1 objective function
# and define a w0 with the seed
#########################################

np.random.seed(seed=seed)
print("Load Monk DataSet")
X_train, Y_train = load_monk(monk, "train")
print("Build the model")
model = Mlp()
model.add(4, input= 17, kernel_initializer = 0.003, kernel_regularizer = reg)
model.add(1, kernel_initializer = 0.003, kernel_regularizer = reg)

#############################
#          L-BFGS
#############################
c1          = 1e-4 
c2          = .9 
m           = 30
ln_maxiter  = 100
#############################
optimizer = LBFGS(m = m, c1=c1, c2=c2, ln_maxiter = ln_maxiter, norm_g_eps = ng_eps, l_eps = l_eps)
model.set_optimizer(optimizer)

print("Start the optimization process:")
model.fit(X_train,
          Y_train, 
          epochs=max_iter, 
          verbose=verbose) 
f = model.history["loss_mse_reg"]

##############################
# plot
##############################
pos_train = (0,0)
figsize = (12, 4)

plt.plot(f - f[-1], linestyle='-')
plt.title('Monk{} - seed={} - (f_k - f^*)'.format(monk, seed))
plt.ylabel("Loss")
plt.xlabel('Iteration')
plt.grid()
plt.yscale('log')
plt.legend(['L-BFGS'], loc='upper right', fontsize='large')    
plt.show()

