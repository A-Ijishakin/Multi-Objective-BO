# Multi-Objective Bayesian Optimisation

This is the repo that we will use for the multi-objective bayesian optimisation hackathon project! A pre-requisite of use is mini/anaconda so please download it from here if you do not have it:

[[Miniconda Download](https://docs.anaconda.com/free/miniconda/index.html)] 
---------------- 
### Using the repo
1. Clone the repo by running: 
    ```
    git clone https://github.com/A-Ijishakin/Contrast-DiffAE.git
    ```
2. Then run the command:
    ```
    conda create --name <env> --file <this file>
    ```
    where <env> is the environment name 

3. You will now be able to use the multi_objective_bo.py script, which takes the following system arguments: 
    - test_function: The type of test function we are using, this will be selected from those specified in optim_configs.py/ 
    - input_constraint: This specifies a type of input constraint, which should be defined in optim_configs.py. 
    - output_constraint: The same as above but for output constraints/
    - noise_se: The standard error of the gaussian noise that will be added (if any).
    - w_dragonfly: Specifies whether we are running with or without dragonfly.  

#### Ploting
There is some basic code for plotting the output of BO runs in plot.py. 
The first plot illustrates the log difference in hypervolume between the complete hypervolume and the hypervolume encompassed by candidates suggested by various acquisition functions. This comparison is presented as a function of time. The second plot is the mean pareto front of these candidates. 

They should have the following form: 
[![hv-difference](https://i.postimg.cc/8s1HPMQs/hv-difference.png)](https://postimg.cc/8s1HPMQs)


[![Objectives](https://i.postimg.cc/56yS8JtV/Objectives.png)](https://postimg.cc/56yS8JtV)

