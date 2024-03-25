import torch
import warnings 
from botorch.test_functions.multi_objective import BraninCurrin
import time
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from dragonfly import load_config
from argparse import Namespace
from dragonfly.exd.experiment_caller import CPMultiFunctionCaller
from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit 
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from methods import (generate_initial_data, initialize_model, 
                     optimize_qehvi_and_get_observation, optimize_qnehvi_and_get_observation, 
                      optimize_qnparego_and_get_observation) 
from tqdm import tqdm 
import numpy as np 
import pickle 

hvs_all_qparego, hvs_all_qehvi, hvs_all_qnehvi, hvs_all_random, hvs_all_dragonfly = [], [], [], [], [] 
train_obj_true_all_qparego, train_obj_true_all_qehvi, train_obj_true_all_qnehvi, train_obj_true_all_random, train_obj_true_all_dragonfly = [], [], [], [], [] 
train_obj_all_qparego, train_obj_all_qehvi, train_obj_all_qnehvi, train_obj_all_random, train_obj_all_dragonfly = [], [], [], [], [] 
train_x_all_qparego, train_x_all_qehvi, train_x_all_qnehvi, train_x_all_random, train_x_all_dragonfly = [], [], [], [], []

seed_var = 41 

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
} 

date = '21-03-2024'
N_ITER = 20 
MC_SAMPLES = 16 
verbose = True
NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs) 
N_TRIALS = 20 
BATCH_SIZE = 1 
problem = BraninCurrin(negate=True).to(**tkwargs) 
NUM_RESTARTS = 3
RAW_SAMPLES = 4 
standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

for n in tqdm(range(N_TRIALS)):
    seed_var += 1 
    torch.manual_seed(seed_var)
    np.random.seed(seed_var) 
    
    options = Namespace(
                # build_new_model_every=BATCH_SIZE,  
                total_budget = 100, 
                
                # set to batch size
                # init_capital=2,  # number of initialization experiments
                # (-1 is included since Dragonfly generates n+1 experiments)
                gpb_hp_tune_criterion='ml-post_sampling',  # Criterion for tuning GP hyper-parameters.
                # Options: 'ml-post_sampling' (algorithm default), 'ml', 'post_sampling'.
                moors_scalarisation='linear',  # Scalarization approach for multi-objective opt. 'linear' or 'tchebychev'
                acq_opt_max_evals=1,
                # acq_opt_method = 'bo'

            )
    # Create optimizer object
    config_params = {
        'domain': [{'type': 'float', 'min': 0.0, 'max': 1.0},
                {'type': 'float', 'min': 0.0, 'max': 1.0}]
    }
    config = load_config(config_params)

    func_caller = CPMultiFunctionCaller(None, config.domain,
                                        domain_orderings=config.domain_orderings)
    func_caller.num_funcs = 2  # must specify how many functions are being optimized

    wm = SyntheticWorkerManager(1)
    dragonfly_opt = CPMultiObjectiveGPBandit(func_caller, wm, options=options)
    dragonfly_opt.ask_tell_mode = True
    dragonfly_opt.worker_manager = None
    dragonfly_opt._set_up()
    # dragonfly_opt._build_new_model()
    # dragonfly_opt._set_next_gp()
    dragonfly_opt.initialise()
    dragonfly_opt.ask()

    hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_random, hvs_dragonfly = [], [], [], [], []  
    
    # call helper functions to generate initial training data
    train_x_qparego, train_obj_qparego, train_obj_true_qparego = generate_initial_data(problem=problem, 
                    NOISE_SE=NOISE_SE, n=4)

    train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )
    train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )
    train_x_random, train_obj_random, train_obj_true_random = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )

    train_x_dragonfly, train_obj_dragonfly, train_obj_true_dragonfly = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego
    )

    # new_x_dragonfly, new_obj_dragonfly = train_x_dragonfly, train_obj_dragonfly

    #initialize models for the different acquisition functions
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, problem=problem, NOISE_SE=NOISE_SE)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, problem=problem, NOISE_SE=NOISE_SE)
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, problem=problem, NOISE_SE=NOISE_SE)

    # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qparego)
    volume = bd.compute_hypervolume().item()

    hvs_qparego.append(volume)
    hvs_qehvi.append(volume)
    hvs_qnehvi.append(volume)
    hvs_random.append(volume)
    hvs_dragonfly.append(volume)

    # run N_ITER rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_ITER + 1):
        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_qparego)
        fit_gpytorch_mll(mll_qehvi)
        fit_gpytorch_mll(mll_qnehvi)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qparego_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition functions and get new observations
        (
            new_x_qparego,
            new_obj_qparego,
            new_obj_true_qparego,
        ) = optimize_qnparego_and_get_observation(
            model=model_qparego, train_x=train_x_qparego, sampler=qparego_sampler, 
            problem=problem, standard_bounds=standard_bounds, BATCH_SIZE=BATCH_SIZE,
            NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE, 
            tkwargs=tkwargs 
        )
        new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
            model=model_qehvi, train_x=train_x_qehvi, sampler=qehvi_sampler, problem=problem, standard_bounds=standard_bounds, 
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE 
        )
        (
            new_x_qnehvi,
            new_obj_qnehvi,
            new_obj_true_qnehvi,
        ) = optimize_qnehvi_and_get_observation(
            model=model_qnehvi, train_x=train_x_qnehvi, sampler=qnehvi_sampler, 
             problem=problem, standard_bounds=standard_bounds, BATCH_SIZE=BATCH_SIZE, 
                                        NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE) 
        new_x_random, new_obj_random, new_obj_true_random = generate_initial_data(problem=problem, 
                    NOISE_SE=NOISE_SE, n=BATCH_SIZE)

        for (x, y) in zip(train_x_dragonfly, train_obj_dragonfly):
            #dragonfly
            x = [datum.item() for datum in x]
            y = [datum.item() for datum in y]
            dragonfly_opt.tell([(x,
                                y)])

        dragonfly_opt.step_idx += 1

        # Retrieve the Pareto-optimal points
        new_x_dragonfly = torch.tensor(dragonfly_opt.ask()).to('cuda') 

        dragonfly_opt._build_new_model() 
        dragonfly_opt._set_next_gp()

        #compute
        new_obj_true_dragonfly = problem(new_x_dragonfly).to('cuda')
        new_obj_dragonfly = new_obj_true_dragonfly + torch.randn_like(new_obj_true_dragonfly) * NOISE_SE

        # update training points
        train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
        train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
        train_obj_true_qparego = torch.cat([train_obj_true_qparego, new_obj_true_qparego])

        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])

        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

        train_x_dragonfly = torch.cat([train_x_dragonfly, new_x_dragonfly.reshape(-1, 2)])
        train_obj_dragonfly = torch.cat([train_obj_dragonfly, new_obj_dragonfly.reshape(-1, 2)])
        train_obj_true_dragonfly = torch.cat([train_obj_true_dragonfly, new_obj_true_dragonfly.reshape(-1, 2)])

        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])
        train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random])

        # update progress
        for hvs_list, train_obj in zip(
            (hvs_random, hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_dragonfly),
            (
                train_obj_true_random,
                train_obj_true_qparego,
                train_obj_true_qehvi,
                train_obj_true_qnehvi,
                train_obj_true_dragonfly
            ),
        ):
            # compute hypervolume
            bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
            volume = bd.compute_hypervolume().item()
            hvs_list.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, problem=problem, NOISE_SE=NOISE_SE)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, problem=problem, NOISE_SE=NOISE_SE)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, problem=problem, NOISE_SE=NOISE_SE)

        # dragonfly_opt._build_new_model()
        # dragonfly_opt._set_next_gp()

        t1 = time.monotonic()
        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI, dragonfly) = "
                f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}, {hvs_qnehvi[-1]:>4.2f}, {hvs_dragonfly[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")  

    hvs_all_dragonfly.append(hvs_dragonfly), hvs_all_qehvi.append(hvs_qehvi), hvs_all_qnehvi.append(hvs_qnehvi) 
    hvs_all_qparego.append(hvs_qparego), hvs_all_random.append(hvs_random) 
    
    train_obj_true_all_qparego.append(train_obj_true_qparego), train_obj_true_all_qehvi.append(train_obj_true_qehvi) 
    train_obj_true_all_qnehvi.append(train_obj_true_qnehvi), train_obj_true_all_random.append(train_obj_true_random) 
    train_obj_true_all_dragonfly.append(train_obj_true_dragonfly) 
    
    
    train_obj_all_qparego.append(train_obj_qparego), train_obj_all_qehvi.append(train_obj_qehvi), 
    train_obj_all_qnehvi.append(train_obj_qnehvi), train_obj_all_random.append(train_obj_random), 
    train_obj_all_dragonfly.append(train_obj)
    
    train_x_all_qparego.append(train_x_qparego), train_x_all_qehvi.append(train_x_qehvi), 
    train_x_all_qnehvi.append(train_x_qnehvi), train_x_all_random.append(train_x_random), 
    train_x_all_dragonfly.append(train_x_dragonfly) 

qparego = {'hvs': hvs_all_qparego, 'train_obj_true': train_obj_true_all_qparego, 'train_obj': train_obj_all_qparego, 
           'train_x': train_x_all_qparego, 'method': 'qparego'} 

qehvi = {'hvs': hvs_all_qehvi, 'train_obj_true': train_obj_true_all_qehvi, 'train_obj': train_obj_all_qehvi,
         'train_x': train_x_all_qehvi, 'method': 'qehvi'}

qnehvi = {'hvs': hvs_all_qnehvi, 'train_obj_true': train_obj_true_all_qnehvi, 'train_obj': train_obj_all_qnehvi,
          'train_x': train_x_all_qnehvi, 'method': 'qnehvi'}
dragonfly = {'hvs': hvs_all_dragonfly, 'train_obj_true': train_obj_true_all_dragonfly, 'train_obj': train_obj_all_dragonfly,
             'train_x': train_x_all_dragonfly, 'method': 'dragonfly'}
random = {'hvs': hvs_all_random, 'train_obj_true': train_obj_true_all_random, 'train_obj': train_obj_all_random,
          'train_x': train_x_all_random, 'method': 'random'} 

for (name, method) in zip(['qparego', 'qehvi', 'qnehvi', 'dragonfly', 'random'], [qparego, qehvi, qnehvi, dragonfly, random]):
    with open(f"MOO/runs/{date}/{name}.pkl", "wb") as f:
        pickle.dump(method, f) 
    
breakpoint 