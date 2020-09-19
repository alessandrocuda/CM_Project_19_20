import sys
from os import path
sys.path.insert(0, "./ISANet/")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD, NCG, LBFGS
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(seed=777)
def get_fitted_model(X_train, Y_train, optimizer, n_seed = 189, verbose = 1):
    np.random.seed(seed=n_seed)
    print("Build the model")
    model = Mlp()
    model.add(4, input= 17, kernel_initializer = 1/np.sqrt(17), kernel_regularizer = 0.001)
    model.add(1, kernel_initializer = 1/np.sqrt(4), kernel_regularizer = 0.001)

    model.set_optimizer(optimizer)

    model.fit(X_train,
            Y_train, 
            epochs=600, 
            #batch_size=31,
            #validation_data = [X_test, Y_test],
            verbose=verbose)

    return model 


print("Load Monk DataSet")
X_train, Y_train = load_monk("2", "train")

seed = 189



#############################
#          SGD
#############################

# optimizer = SGD(lr = 0.8, momentum = 0.9, nesterov = True)
# model = get_fitted_model(X_train, Y_train, optimizer, 189, 1)
# outputNet = model.predict(X_test)
# printMSE(outputNet, Y_test, type = "test")
# printAcc(outputNet, Y_test, type = "test")
# h0 = model.history 


#############################
#          NCG f1
#############################

optimizer = NCG(beta_method="fr", c1=1e-4, c2=.9, restart=3, tol = 1e-9)

model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_fr = model.history 

#############################
#          NCG pr
#############################

optimizer = NCG(beta_method="pr", c1=1e-4, c2=.9, tol = 1e-9)

model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_pr = model.history 

#############################
#          NCG Hs
#############################

optimizer = NCG(beta_method="hs", c1=1e-4, c2=.9, tol = 1e-9)

model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_hs = model.history 

#############################
#          L-BFGS
#############################

optimizer = LBFGS(m=20, c1= 1e-4, c2=0.9, tol=1e-20)

model = get_fitted_model(X_train, Y_train, optimizer, seed, 1)
h_lbfgs = model.history 

##############################
# plot
##############################

pos_train = (0,0)
figsize = (12, 4)

plt.plot(h_fr["loss_mse"], linestyle='-')
plt.plot(h_pr["loss_mse"], linestyle = '--')
plt.plot(h_hs["loss_mse"], linestyle='-.')
plt.plot(h_lbfgs["loss_mse"], linestyle=':')
plt.title('Monk2 - seed=189')
plt.ylabel("MSE")
plt.xlabel('Epoch')
plt.grid()
plt.yscale('log')
plt.legend(['NCG - fr - r=3','NCG - pr','NCG - hs', 'BFGS'], loc='upper right', fontsize='large')    
plt.show()
