# Computational Mathematics for Learning and Data Analysis  - Project 2019-2020

*TRAINING  A NEURAL NETWORK WITH NONLINEAR CONJUGATE GRADIENT AND LIMITED-MEMORY BFGS METHODS*

> [`ISANet`](https://github.com/alessandrocuda/ISANet) lib has been extended to include the NCG FR/PR/HS and beta+ variants, and the L-BFGS methods.


## Abastract
Neural Networks are highly expressive models that have achieved the state of the art performance in many tasks as pattern recognition, natural language processing, and many others. Usually, stochastic momentum methods coupled with the classical Backpropagation algorithm for the gradient computation is used in training a neural network.
In recent years several methods have been developed to accelerate the learning convergence of first-order methods such as Classic Momentum, also known as Polyak's heavy ball method, or the Nesterov momentum.
This work aims to go beyond the first-order methods and analyse some variants of the nonlinear conjugate gradient (NCG) and a specific case of limited-memory quasi-Newton class called L-BFGS as optimization methods. Them are combined with  the use of a line search that respects the strong Wolfe conditions to accelerate the learning processes of a feedforward neural network.  
## Introduction
TBW

## Usage
This code requires Python 3.5 or later, to download the repository:

`git clone https://github.com/alessandrocuda/CM_Project_19_20`

Then you need to install the basic dependencies to run the project on your system:

```
cd CM_Project_19_20
pip install -r requirements.txt
```

You also need to fetch the ISANet Library from the [`ISANet`](https://github.com/alessandrocuda/ISANet) repository:

```
git submodule init
git submodule update
```

And you are good to go!

## Example
NCG or LBFGS optimizer examples:

```python
from isanet.model import Mlp
from isanet.optimizer import NCG, LBFGS
from isanet.optimizer.utils import l_norm
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import isanet.metrics as metrics
import numpy as np

X_train, Y_train = load_monk("1", "train")

#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
kernel_initializer = 0.003 
kernel_regularizer = 0.001

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer, kernel_regularizer)
model.add(1, kernel_initializer, kernel_regularizer)

# Define your optimizer, debug parameter helps you to execture step by step by pressing a key on the keyboard.
optimizer = NCG(beta_method="hs+", c1=1e-4, c2=0.1, restart=None, ln_maxiter = 100, norm_g_eps = 1e-9, l_eps = 1e-9, debug = True)
# or you can choose the LBFGS optimizer
#optimizer = LBFGS(m = 30, c1=1e-4, c2=0.9, ln_maxiter = 100, norm_g_eps = 1e-9, l_eps = 1e-9, debug = True)

#start the optimisation phase
# no batch with NCG or LBFGS optimizer
model.fit(X_train,
          Y_train, 
          epochs=600,  
          es = es,
          verbose=0) 
```

## References
TBW
 
 <!-- CONTACT -->
## Authors

 - Alessandro Cudazzo - [@alessandrocuda](https://twitter.com/alessandrocuda) - alessandro@cudazzo.com

 - Giulia Volpi - giuliavolpi25.93@gmail.com

<!-- LICENSE -->
## License
Copyright 2019 ©  <a href="https://alessandrocudazzo.it" target="_blank">Alessandro Cudazzo</a> - <a href="mailto:giuliavolpi25.93@gmail.com">Giulia Volpi</a>
