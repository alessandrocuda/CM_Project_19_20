# Computational Mathematics for Learning and Data Analysis  - Project 2019-2020

*TRAINING  A NEURAL NETWORK WITH NONLINEAR CONJUGATE GRADIENT AND LIMITED-MEMORY BFGS METHODS*

## Abastract
Neural Networks are highly expressive models that have achieved the state of the art performance in many tasks as pattern recognition, natural language processing, and many others. Usually, stochastic momentum methods coupled with the classical Backpropagation algorithm for the gradient computation is used in training a neural network.
In recent years several methods have been developed to accelerate the learning convergence of first-order methods such as Classic Momentum, also known as Polyak's heavy ball method, or the Nesterov momentum.
This work aims to go beyond the first-order methods and analyse some variants of the nonlinear conjugate gradient (NCG) and a specific case of limited-memory quasi-Newton class called L-BFGS as optimization methods. Them are combined with  the use of a line search that respects the strong Wolfe conditions to accelerate the learning processes of a feedforward neural network.  

To achieve this, our library, called ISANet lib, has been extended to include these two new optimizers.

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


## References
TBW
 
 <!-- CONTACT -->
## Authors

 - Alessandro Cudazzo - [@alessandrocuda](https://twitter.com/alessandrocuda) - alessandro@cudazzo.com

 - Giulia Volpi - giuliavolpi25.93@gmail.com

<!-- LICENSE -->
## License
Copyright 2019 ©  <a href="https://alessandrocudazzo.it" target="_blank">Alessandro Cudazzo</a> - <a href="mailto:giuliavolpi25.93@gmail.com">Giulia Volpi</a>
