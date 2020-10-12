# Computational Mathematics for Learning and Data Analysis  - Project 2019-2020

*TRAINING  A NEURAL NETWORK WITH NONLINEAR CONJUGATE GRADIENT AND LIMITED-MEMORY BFGS METHODS*

> [`ISANet`](https://github.com/alessandrocuda/ISANet) lib has been extended to include the NCG FR/PR/HS and beta+ variants, and the L-BFGS methods.


## Abastract
Neural Networks are highly expressive models that have achieved the state of the art performance in many tasks as pattern recognition, natural language processing, and many others. Usually, stochastic momentum methods coupled with the classical Backpropagation algorithm for the gradient computation is used in training a neural network.
In recent years several methods have been developed to accelerate the learning convergence of first-order methods such as Classic Momentum, also known as Polyak's heavy ball method, or the Nesterov momentum.
This work aims to go beyond the first-order methods and analyse some variants of the nonlinear conjugate gradient (NCG) and a specific case of limited-memory quasi-Newton class called L-BFGS as optimization methods. Them are combined with the use of a line search that respects the strong Wolfe conditions to accelerate the learning processes of a feedforward neural network. 

## Introduction
This project has been developed during the Computational Mathematics for learning and data analysis course held by Professor Antonio Frangioni and Professor Federico Poloni. The aims were to extend ISANet lib in order to include the NCG FR/PR/HS and beta+ variants, and the L-BFGS methods (as new optimizers) and study the objective function used during the learning phase from a mathematical and optimisation point of view.

- All the detalis about the project can be found on the full report [here](https://github.com/alessandrocuda/CM_Project_19_20/blob/master/report/CM_19_20_Cudazzo_Volpi.pdf).
- ISANet is is open source library and can be found at this [link](https://github.com/alessandrocuda/ISANet). For more details about the library just visit the repo or <a href="https://alessandrocudazzo.it/ISANet"><strong>explore the docs</strong></a>.

Here we decided to derive three objective functions from the three <a href=https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems> MONK dataset</a> and study the convergence at their local minimum with the NCG and L-BFGS. Objective functions built through the following procedure:
 - A MONK’s training set (1,2,3) is taken, since the features are categorical, the one-hot encoding technique has been applied on them, obtaining 17 binary features.
 - A fixed network topology: 17 (input units) - 4 (hidden units) - 1 (output unit). The sigmoid activation function has been used on both hidden and output layers.

 - The objective function is in the form the MSE plus a L2 regularization term. The L2 term allows to keep the objective function continuous and differentiable (property is already given by the sigmoid activation used in the neural network). Moreover, it makes the level set compact, the gradient and the hessian L-continuous. Here the objective function for each Monk dataset *i* composed by the error and the regularization term with λ = 10−4,

    <a href="https://www.codecogs.com/eqnedit.php?latex=L_i(w)&space;=&space;E_i(\mathbf{w})&space;&plus;&space;10^{-4}\left&space;\|&space;\mathbf{w}&space;\right&space;\|^2,\;&space;\;&space;i&space;\in&space;(1,2,3)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_i(w)&space;=&space;E_i(\mathbf{w})&space;&plus;&space;10^{-4}\left&space;\|&space;\mathbf{w}&space;\right&space;\|^2,\;&space;\;&space;i&space;\in&space;(1,2,3)" title="L_i(w) = E_i(\mathbf{w}) + 10^{-4}\left \| \mathbf{w} \right \|^2,\; \; i \in (1,2,3)" /></a>

    where E_i is the MSE and 10^-4||w||^2 is the regularization term.

The experiments have been divided into two parts. 
 - First, <a href=https://github.com/alessandrocuda/CM_Project_19_20/tree/master/experiments/methods_behaviors>methods behaviors and validation</a> where we have studied the methods' behaviors in order to check the correctness of our implementation and verify that the results described in the theoretical part, in terms of global convergence and local convergence, can be observed from a practical point of view.
 - Instead, in <a href=https://github.com/alessandrocuda/CM_Project_19_20/tree/master/experiments/methods_comparisons>methods comparisons</a> we have compared the efficiency of all the methods from the same starting point.

All our experiments were performed on a Intel CPU with 2 cores and 4 thread at 2.6GHz and with OpenBLAS as optimized math routine for Numpy.
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
np.random.seed(seed=6)
#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
kernel_initializer = 0.003 
kernel_regularizer = 1e-4

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
model.add(1, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

# Define your optimizer, debug parameter helps you to execture step by step by pressing a key on the keyboard.
optimizer = NCG(beta_method="hs+", c1=1e-4, c2=0.6, restart=None, ln_maxiter = 100, norm_g_eps = 3e-5, l_eps = 1e-6, debug = True)
model.set_optimizer(optimizer)
# or you can choose the LBFGS optimizer
#optimizer = LBFGS(m = 30, c1=1e-4, c2=0.9, ln_maxiter = 100, norm_g_eps = 1e-9, l_eps = 1e-9, debug = True)

#start the optimisation phase
# no batch with NCG or LBFGS optimizer
model.fit(X_train,
          Y_train, 
          epochs=600,  
          verbose=2) 
```
what's next? [here](https://github.com/alessandrocuda/CM_Project_19_20/blob/master/examples/README.md) you can find some example scripts.

## References
 - [1] Stiefel Hestenes. Methods of conjugate gradients for solving linear systems.Journal of Research of the National Bureau of Standards, 49:409–436, 12 1952.
 - [2] Reeves Fletcher. Function minimization by conjugate gradients.Computer Journal, pages 149–154, 71964.
 - [3] Stephen J. Wright Jorge Nocedal.Numerical Optimization. Springer Series in Operations Research.Springer, 1999.
 - [4] M. J. D. Powell. Restart procedures for the conjugate gradient method.Mathematical Programming,12(1):241–254, Dec 1977.
 - [5] M. J. D. Powell. Nonconvex minimization calculations and the conjugate gradient method. In David F.Griffiths, editor,Numerical Analysis, pages 122–141, Berlin, Heidelberg, 1984. Springer Berlin Heidel-berg.
 - [6] Jean Charles Gilbert and Jorge Nocedal. Global convergence properties of conjugate gradient methodsfor optimization. SIAM Journal on Optimization, 2:21–42, 02 1992.31
 - [7] G. Zoutendijk,  Nonlinear Programming,  Computational Methods,  In:  J. Abadie Ed.,  Integer,  andNonlinear Programming. North-Holland, Amsterdam. 1970.
 - [8] Mehiddin Al-Baali. Descent property and global convergence of the fletcher–reeves method with inexactline search. IMA Journal of Numerical Analysis, 5, 01 1985.
 - [9] H. Crowder and P. Wolfe.  Linear convergence of the conjugate gradient method.IBM Journal of Research and Development, 16(4):431–433, 1972.
 - [10] M. J. Powell.   Some convergence properties of the conjugate gradient method.Math. Program., 11(1):42–49, December 1976. 
 - [11] Arthur I. Cohen.  Rate of convergence of several conjugate gradient algorithms. SIAM Journal onNumerical Analysis, 9(2):248–259, 1972.
 - [12] W C Davidon. Variable metric method for minimization. 5 1959.
 - [13] R. Fletcher and M. J. D. Powell.   A Rapidly Convergent Descent Method for Minimization. The Computer Journal, 6(2):163–168, 08 1963. 
 - [14] Mokhtar S. Bazaraa. Nonlinear Programming: Theory and Algorithms. Wiley Publishing, 3rd edition,2013.
 - [15] Willard I. Zangwill.  Convergence conditions for nonlinear programming algorithms. ManagementScience, 16(1):1–13, 1969. 
 - [16] Richard H. Byrd and Jorge Nocedal. A tool for the analysis of quasi-newton methods with applicationto unconstrained minimization. SIAM Journal on Numerical Analysis, 26(3):727–739, 1989. 
 - [17] Global convergence  of  a  class  of  quasi-newton  methods  on  convex  problems. SIAM  Journal  onNumerical Analysis, 24(5):1171–1190, 1987. 
 - [18] M. Powell. Some global convergence properties of a variable metric algorithm for minimization withoutexact lin. 1976. 
 - [19] Yu-Hong Dai. Convergence properties of the bfgs algoritm. SIAM Journal on Optimization, 13:693–701,01 2002. 
 - [20] Dong-Hui Li and Masao Fukushima. A modified bfgs method and its global convergence in nonconvexminimization.Journal of Computational and Applied Mathematics, 129(1):15 – 35, 2001. Nonlinear Programming and Variational Inequalities. 
 - [21]John E. Dennis and Jorge J. More.  Quasi-newton methods, motivation and theory.  Technical report,USA, 1974. 
 
 <!-- CONTACT -->
## Authors

 - Alessandro Cudazzo - [@alessandrocuda](https://twitter.com/alessandrocuda) - alessandro@cudazzo.com

 - Giulia Volpi - giuliavolpi25.93@gmail.com

<!-- LICENSE -->
## License
Copyright 2019 ©  <a href="https://alessandrocudazzo.it" target="_blank">Alessandro Cudazzo</a> - <a href="mailto:giuliavolpi25.93@gmail.com">Giulia Volpi</a>
