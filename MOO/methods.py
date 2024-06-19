import torch 
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.gp_regression import FixedNoiseGP
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
import numpy as np 
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch.mlls import ExactMarginalLogLikelihood 
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", message="Attempted to use direct, but fortran library could not be imported. Using PDOO optimiser instead of direct.")

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
} 

NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs) 


N_TRIALS = 4 

SMOKE_TEST = True

problem = BraninCurrin(negate=True).to(**tkwargs)

# BATCH_SIZE = 2 
NUM_RESTARTS = 10 if not SMOKE_TEST else 3
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

standard_bounds = torch.ones(2, problem.dim, **tkwargs) * -1 
standard_bounds[1] = 0

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_BATCH = 20 if not SMOKE_TEST else 2
MC_SAMPLES = 128 if not SMOKE_TEST else 16

verbose = True

def produce_mean_and_variance(train_x, train_obj):
    # Compute the mean and variance along the dimension 1 (groups of three)
    train_obj_mean = np.mean(train_obj.reshape(-1, 3, 2), axis=1) 
    train_x = np.mean(train_x.reshape(-1, 3, 20), axis=1)  

    #import the standard scalers
    xScaler, yScaler = StandardScaler(), StandardScaler()

    #scale the data 
    train_x = torch.tensor(xScaler.fit_transform(train_x).astype(np.float32)) 
    train_obj_mean = torch.tensor(yScaler.fit_transform(train_obj_mean).astype(np.float32)) * -1 

    Yvar = torch.tensor(np.mean(np.var(train_obj.reshape(-1, 3, 2), axis=1)).astype(np.float32)) 
    return train_x, train_obj_mean, Yvar, xScaler, yScaler

def add_noise(data, train_y, noise_factor=0.1): 
    # Calculate mean and standard deviation for each column
    # means = data.mean(dim=0)
    stds = np.std(data, axis=0) 

    # Generate Gaussian noise scaled by the standard deviation of each column
    noise = torch.randn_like(torch.tensor(data.astype(np.float32))) * (stds * noise_factor)

    # Add noise to the original data
    noisy_data = data + np.array(noise)   

    xScaler, yScaler = StandardScaler(), StandardScaler()

    noisy_data = torch.tensor(xScaler.fit_transform(noisy_data).astype(np.float32)) 
    train_y = torch.tensor(yScaler.fit_transform(train_y).astype(np.float32)) * -1 

    return noisy_data, train_y, xScaler, yScaler


def denormalize_rows(normalized_tensor, row_min, row_range):
    """
    Denormalizes each row of a tensor from the range [0, 1] back to the original range.
    """
    original_tensor = normalized_tensor * row_range + row_min
    return original_tensor

def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj, train_yvar=None):
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        models.append(
            FixedNoiseGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar,
            ) 
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model 


def calc_likelihoods(model, train_x, train_obj):
    likelihoods = []
    # Assuming train_obj has the same order of objectives as models in ModelListGP
    for i, single_model in enumerate(model.models):
        # Set model and its likelihood to evaluation mode
        single_model.eval()
        single_model.likelihood.eval()

        with torch.no_grad():
            # Get the output from the model
            output = single_model(train_x)
            # We assume train_obj is already appropriately sliced per model
            train_y = train_obj[..., i:i+1]
            # Initialize the Exact Marginal Log Likelihood for the single model
            mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
            # Calculate the MLL
            mll_value = mll(output, train_y)
            # Since we generally minimize the negative MLL, we display the negative of mll_value
            likelihoods.append(mll_value.sum().item()) 
    return np.mean(likelihoods)  


# def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler, BATCH_SIZE=2):
#     """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
#     # partition non-dominated space into disjoint rectangles
#     with torch.no_grad():
#         pred = model.posterior(normalize(train_x, problem.bounds)).mean
#     partitioning = FastNondominatedPartitioning(
#         ref_point=problem.ref_point,
#         Y=pred,
#     )
#     acq_func = qExpectedHypervolumeImprovement(
#         model=model,
#         ref_point=problem.ref_point,
#         partitioning=partitioning,
#         sampler=sampler,
#     )
#     # optimize
#     candidates, _ = optimize_acqf(
#         acq_function=acq_func,
#         bounds=standard_bounds,
#         q=BATCH_SIZE,
#         num_restarts=NUM_RESTARTS,
#         raw_samples=RAW_SAMPLES,  # used for intialization heuristic
#         options={"batch_limit": 5, "maxiter": 200},
#         sequential=True,
#     )
#     # observe new values
#     new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    
#     return new_x 


# # def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, BATCH_SIZE=2):
# #     """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
# #     # partition non-dominated space into disjoint rectangles
#     acq_func = qNoisyExpectedHypervolumeImprovement(
#         model=model,
#         # ref_point=problem.ref_point.tolist(),  # use known reference point
#         X_baseline=train_x,
#         prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
#         sampler=sampler,
#     )
#     # optimize
#     candidates, _ = optimize_acqf(
#         acq_function=acq_func,
#         bounds=standard_bounds,
#         q=BATCH_SIZE,
#         num_restarts=NUM_RESTARTS,
#         raw_samples=RAW_SAMPLES,  # used for intialization heuristic
#         options={"batch_limit": 5, "maxiter": 200},
#         sequential=True,
#     )

#     return candidates  


def optimize_qnparego_and_get_observation(model, train_x, sampler, BATCH_SIZE=2):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []

    for _ in range(BATCH_SIZE):
        weights = sample_simplex(2, **tkwargs).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    
    lower = torch.min(train_x, dim=0).values 
    upper = torch.max(train_x, dim=0).values 

    standard_bounds = torch.stack([lower, upper], dim=0) 
    
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    return candidates  




