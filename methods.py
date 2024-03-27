import torch 
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list 
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
import warnings
import numpy as np 
import pickle 
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

warnings.filterwarnings("ignore", message="Attempted to use direct, but fortran library could not be imported. Using PDOO optimiser instead of direct.")


def c2dtlz2_constraint(x, function): 
    return - function.evaluate_slack(x) 

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS) 

def to_cpu(tensor):
    return tensor.detach().cpu().numpy()

def load_pickles(files, root='MOO/runs'):
    outputs = []
    for file in files:
        with open(f'{root}/{file}', "rb") as f:
            outputs.append(pickle.load(f))  
    return tuple(outputs) 
    
def generate_initial_data(problem, NOISE_SE=None, n=6, 
                          train_obj_true=None):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x) 

    if NOISE_SE != None:
        train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE 
    else:
        train_obj = train_obj_true
    
    return {'train_x' :  train_x,  'train_obj' : train_obj, 
             'train_obj_true': train_obj_true} 
    
def initialize_model(train_x, train_obj, problem, NOISE_SE=None, 
                     train_con=None):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = [] 
    
    if train_con != None:
        train_y = torch.cat([train_obj, train_con], dim=-1) 
    
    if NOISE_SE != None: 
        for i in range(train_obj.shape[-1]): 
            train_y = train_obj[..., i : i + 1]
            train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
            models.append(
                FixedNoiseGP(
                    train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            ) 
                    
    else:
        train_y = train_obj 
        for i in range(train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)
                )
            ) 
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi_and_get_observation(model, train_x, sampler, problem, standard_bounds, BATCH_SIZE, 
                                       NUM_RESTARTS, RAW_SAMPLES, NOISE_SE=None, output_constraint=None,  
                                       new_obj_true=None, new_con=None, objective=None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    
        
    if output_constraint:
        output_constraint = [lambda Z: Z[..., -1]] 
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]) 
           
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
        constraints = output_constraint, 
        objective = objective 
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE if NOISE_SE != None else new_obj_true 
    
    if output_constraint:
        new_con = output_constraint(problem, new_x) 
    
    return  {'new_x' : new_x, 'new_obj': new_obj, 'new_obj_true': new_obj_true, 
              'new_con' : new_con}  


def optimize_qnehvi_and_get_observation(model, train_x,  sampler, problem, standard_bounds, BATCH_SIZE, 
                                        NUM_RESTARTS, RAW_SAMPLES, NOISE_SE=None, output_constraint=None,
                                        new_obj_true=None, new_con=None, objective=None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation.""" 
    # partition non-dominated space into disjoint rectangles
    
    train_x = normalize(train_x, problem.bounds)  
    
    if output_constraint:
        output_constraint = [lambda Z: Z[..., -1]] 
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]) 
    
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=train_x ,
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
        objective=objective,
        constraints=output_constraint 
    )
    
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE if NOISE_SE != None else new_obj_true 
    
    if output_constraint:
        new_con = output_constraint(problem, new_x) 

        
    return  {'new_x' : new_x, 'new_obj': new_obj, 'new_obj_true': new_obj_true, 
              'new_con' : new_con}  

def optimize_qnparego_and_get_observation(model, train_x, sampler, problem, standard_bounds, BATCH_SIZE, 
                                        NUM_RESTARTS, RAW_SAMPLES, tkwargs, NOISE_SE=None, 
                                        train_obj=None, new_con=None, output_constraint=None):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, problem.bounds)
    
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        
        if not output_constraint: 
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )

        else:
            scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj) 
            # initialize the scalarized objective (w/o constraints)
            objective = GenericMCObjective(
                # the last element of the model outputs is the constraint
                lambda Z, X: scalarization(Z[..., :-1]),
            )
            constraints = [lambda Z: Z[..., -1]] 
             
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
                constraints= output_constraint
            ) 

        acq_func_list.append(acq_func)
        
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x) 
    

    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE if NOISE_SE != None else new_obj_true 
    
    if output_constraint:
        new_con = output_constraint(problem, new_x) 
    
    return {'new_x': new_x, 'new_obj': new_obj, 
            'new_obj_true': new_obj_true, 'new_con': new_con}  