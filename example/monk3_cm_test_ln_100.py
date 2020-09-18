import sys
from os import path
sys.path.insert(0, "./ISANet/")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD, NCG, LBFGS
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np

def get_fitted_model(X_train, Y_train, optimizer, n_seed = 189, verbose = 1):
    np.random.seed(seed=n_seed)
    print("Build the model")
    model = Mlp()
    model.add(4, input= 17, kernel_initializer = 0.003, kernel_regularizer = 0.001)
    model.add(1, kernel_initializer = 0.003, kernel_regularizer = 0.001)

    model.set_optimizer(optimizer)

    model.fit(X_train,
            Y_train, 
            epochs=1200, 
            #batch_size=31,
            #validation_data = [X_test, Y_test],
            verbose=verbose)

    return model 

print("Load Monk DataSet")
X_train, Y_train = load_monk("3", "train")

seed = 623

#############################
#          NCG f1
#############################
restart = 6
optimizer = NCG(beta_method="fr", c1=1e-4, c2=.1, restart=restart, tol = 1e-12)
model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_fr = model.history 

#############################
#          NCG pr
#############################

optimizer = NCG(beta_method="pr", c1=1e-4, c2=.1, tol = 1e-12)
model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_pr = model.history 

#############################
#          NCG Hs
#############################

optimizer = NCG(beta_method="hs", c1=1e-4, c2=.4, tol = 1e-12)
model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_hs = model.history 

#############################
#          L-BFGS
#############################
m = 3
optimizer = LBFGS(m=m, c1= 1e-4, c2=0.4, tol=1e-20)
model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_lbfgs = model.history 


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
plt.title('Monk3 - seed={}'.format(seed))
plt.ylabel("MSE")
plt.xlabel('Epoch')
plt.grid()
plt.yscale('log')
plt.legend(['NCG - FR - R={}'.format(restart),'NCG - PR+','NCG - HS', 'L-BFGS - m={}'.format(m)], loc='upper right', fontsize='large')    
plt.show()
