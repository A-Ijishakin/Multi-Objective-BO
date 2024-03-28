import torch
from  argparse import ArgumentParser 
import warnings 
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
from botorch.utils.multi_objective.hypervolume import Hypervolume 
from botorch.utils.multi_objective.pareto import is_non_dominated 
from optim_configs import test_functions, input_constraints, output_constraints

#instantiate the argument parser 
parser = ArgumentParser() 
parser.add_argument('--test_function', type=str, default='branincurrin', 
                     help='Test function to optimize') 
parser.add_argument('--input_constraint', type=bool, default=False, 
                     help='The constraint on the test function input (if any)')  
parser.add_argument('--output_constraint', default=None, 
                        help='The constraint on the test function (if any)') 
parser.add_argument('--noise_se', type=list, default=None) 
parser.add_argument('--w_dragonfly', type=bool, default=False) 
parser.add_argument('--gpu', type=bool, default=False)
parser.add_argument('--dtype', type=int, default=torch.double) 
parser.add_argument('--sb_dir', type=str, default='')

args = parser.parse_args() 

if args.test_function == 'ScattBO':
    eval_metric = 'Scatter'


eval_all_qparego, eval_all_qehvi, eval_all_qnehvi, eval_all_random, eval_all_dragonfly = [], [], [], [], [] 
train_obj_true_all_qparego, train_obj_true_all_qehvi, train_obj_true_all_qnehvi, train_obj_true_all_random, train_obj_true_all_dragonfly = [], [], [], [], [] 
train_obj_all_qparego, train_obj_all_qehvi, train_obj_all_qnehvi, train_obj_all_random, train_obj_all_dragonfly = [], [], [], [], [] 
train_x_all_qparego, train_x_all_qehvi, train_x_all_qnehvi, train_x_all_random, train_x_all_dragonfly = [], [], [], [], []

seed_var = 41 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "dtype": args.dtype,
    "device": device,
} 
print(f"Using device {tkwargs['device']}")

########
args.test_function = 'thermoelectrics' #'ScattBO' #'c2dtlz2' 
args.w_dragonfly = True
args.sb_dir = 'ThermoElectrics' #'ScattBO'
########
date = '29-03-2024'
N_ITER = 5 
MC_SAMPLES = 16 
verbose = True
#if there is noise specified than instantiate it's standard error else leave it as None 
NOISE_SE = torch.tensor(args.noise_se, **tkwargs) if args.noise_se else args.noise_se  
N_TRIALS = 20 
BATCH_SIZE = 1
#problem = test_functions[args.test_function].to(**tkwargs) 
problem = test_functions[args.test_function]
NUM_RESTARTS = 3
RAW_SAMPLES = 4 
standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1 
num_constraints = 1 if args.output_constraint else 0 
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

scale = -1 if args.test_function == 'ScattBO' else 1

for n in tqdm(range(N_TRIALS)): 
    if verbose:
        noise_string = f" with noise SE = {[noise.item() for noise in NOISE_SE]}" if NOISE_SE != None else "" 
        o_con_string = f" with output constraint {args.output_constraint}" if args.output_constraint else "" 
        i_con_string = f" with input constraint {args.input_constraint}" if args.input_constraint else ""
        print(f"Running Bayesian Optimisation of {args.test_function}{noise_string}{o_con_string}{i_con_string}")
    
    
    seed_var += 1 
    torch.manual_seed(seed_var)
    np.random.seed(seed_var) 
    
    if args.w_dragonfly:
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
        
        if args.input_constraint:
            domain_constraints = [
            {'name': 'main_constraint', 'constraint': args.input_constraint},
                    ]
        else:
            domain_constraints = None 
        
        # Create optimizer object
        config_params = {
            'domain': [{'type': 'float', 'min': 0.0, 'max': 1.0} for _ in range(problem.dim)],
            'domain_constraints': domain_constraints    
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
        dragonfly_opt.initialise()
        dragonfly_opt.ask()

    eval_qparego, eval_qehvi, eval_qnehvi, eval_random, eval_dragonfly = [], [], [], [], []  
    

    initial_data = generate_initial_data(problem=problem, 
                    NOISE_SE=NOISE_SE, n=4, test_function=args.test_function, root_dir=args.sb_dir)

    train_x_qparego, train_obj_qparego, train_obj_true_qparego = (initial_data['train_x'].to(**tkwargs), 
                                                                  initial_data['train_obj'].to(**tkwargs) * scale, 
                                                                  initial_data['train_obj_true'].to(**tkwargs) * scale)    
    train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego
    ) 
    train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego
    )
    train_x_random, train_obj_random, train_obj_true_random = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego 
    ) 
    if args.w_dragonfly:
        train_x_dragonfly, train_obj_dragonfly, train_obj_true_dragonfly = (
            train_x_qparego,
            train_obj_qparego,
            train_obj_true_qparego
        )  
    
    else:
        train_obj_true_dragonfly, train_con_dragonfly = None, 0  
    if args.output_constraint:
        train_con_qparego =  output_constraints[args.output_constraint](problem, train_x_qparego)  
        train_con_qehvi, train_con_qnehvi, train_con_random  = train_con_qparego, train_con_qparego, train_con_qparego
        if args.w_dragonfly:
            train_con_dragonfly = output_constraints[args.output_constraint](problem, train_x_dragonfly) 
            train_obj_dragonfly = torch.cat([train_obj_dragonfly, train_con_dragonfly], dim=1) 
    
    else:
        train_con_qparego, train_con_qehvi, train_con_qnehvi, train_con_random, train_con_dragonfly = None, None, None, None, None
    
    #initialize models for the different acquisition functions
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, 
                                    problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qparego, tkwargs=tkwargs)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, 
                                    problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qehvi, tkwargs=tkwargs)
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, 
                                    problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qnehvi, tkwargs=tkwargs)
    if args.output_constraint: 
        hv = Hypervolume(ref_point=problem.ref_point) 
        # compute pareto front
        is_feas = (train_con_qparego <= 0).all(dim=-1)
        feas_train_obj = train_obj_qparego[is_feas]
        if feas_train_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            # compute hypervolume
            volume = hv.compute(pareto_y)
        else:
            volume = 0.0
    else:    
        if args.test_function == 'ScattBO':
            volume = torch.mean(train_obj_qparego, dim=0).sum() 
        else:
            # compute hypervolume
            bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qparego)
            volume = bd.compute_hypervolume().item() 
    eval_qparego.append(volume)
    eval_qehvi.append(volume)
    eval_qnehvi.append(volume)
    eval_random.append(volume)
    eval_dragonfly.append(volume)

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
        qn_parego_optimized_and_observed =  optimize_qnparego_and_get_observation(
            model=model_qparego, train_x=train_x_qparego, sampler=qparego_sampler, 
            problem=problem, standard_bounds=standard_bounds, BATCH_SIZE=BATCH_SIZE,
            NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE, 
            test_function=args.test_function,   
            tkwargs=tkwargs, train_obj=train_obj_qparego, output_constraint=output_constraints[args.output_constraint], 
        )
        (
            new_x_qparego,
            new_obj_qparego,
            new_obj_true_qparego,
            new_con_qparego) = (qn_parego_optimized_and_observed['new_x'], qn_parego_optimized_and_observed['new_obj'] * scale, 
                            qn_parego_optimized_and_observed['new_obj_true'] * scale, qn_parego_optimized_and_observed['new_con']) 
                            
        qehvi_optimized_and_observed = optimize_qehvi_and_get_observation(
            model=model_qehvi.to(tkwargs['device']), train_x=train_x_qehvi.to(tkwargs['device']), sampler=qehvi_sampler, 
            problem=problem, standard_bounds=standard_bounds, tkwargs=tkwargs,
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, test_function=args.test_function, 
            NOISE_SE=NOISE_SE, output_constraint=output_constraints[args.output_constraint] 
        )       
        ( 
            new_x_qehvi, 
            new_obj_qehvi, 
            new_obj_true_qehvi, 
            new_con_qparego ) = ( qehvi_optimized_and_observed['new_x'], qehvi_optimized_and_observed['new_obj'] * scale,
                                  qehvi_optimized_and_observed['new_obj_true'] * scale, qehvi_optimized_and_observed['new_con']) 
        
        qnehvi_optimized_and_observed = optimize_qnehvi_and_get_observation( model=model_qnehvi.to(tkwargs['device']), train_x=train_x_qehvi.to(tkwargs['device']), sampler=qnehvi_sampler, 
             problem=problem, standard_bounds=standard_bounds, BATCH_SIZE=BATCH_SIZE, test_function=args.test_function,
                                         tkwargs=tkwargs,
                                        NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE, 
                                        output_constraint=output_constraints[args.output_constraint]) 
        (
            new_x_qnehvi,
            new_obj_qnehvi,
            new_obj_true_qnehvi,
            new_con_qnehvi 
        ) = (qnehvi_optimized_and_observed['new_x'], qnehvi_optimized_and_observed['new_obj'] * scale, 
             qnehvi_optimized_and_observed['new_obj_true'] * scale, qnehvi_optimized_and_observed['new_con']) 
        
        new_random = generate_initial_data(problem=problem, 
                NOISE_SE=NOISE_SE, n=BATCH_SIZE, test_function=args.test_function, root_dir=args.sb_dir) 
        
        new_x_random, new_obj_random, new_obj_true_random = (new_random['train_x'],
                                                             new_random['train_obj'] * scale,
                                                                new_random['train_obj_true'] * scale) 
        
        if args.w_dragonfly:
            for (x, y) in zip(train_x_dragonfly, train_obj_dragonfly):
                #dragonfly 
                x = [datum.item() for datum in x]
                y = [datum.item() for datum in y]
                dragonfly_opt.tell([(x,
                                    y)])

            ##### 
            dragonfly_opt.step_idx += 1

            # Retrieve the Pareto-optimal points
            new_x_dragonfly = torch.tensor(dragonfly_opt.ask()).to(tkwargs['device']).unsqueeze(0) 

            dragonfly_opt._build_new_model() 
            dragonfly_opt._set_next_gp()

            #compute
            new_obj_true_dragonfly = problem(new_x_dragonfly).to(device)
            new_obj_dragonfly = new_obj_true_dragonfly + torch.randn_like(new_obj_true_dragonfly) * NOISE_SE if args.noise_se else new_obj_true_dragonfly
            
            #
            if args.output_constraint:
                new_con_dragonfly = output_constraints[args.output_constraint](problem, new_x_dragonfly) 
                new_obj_dragonfly = torch.cat([new_obj_dragonfly, new_con_dragonfly], dim=-1) 
            
            train_x_dragonfly = torch.cat([train_x_dragonfly, new_x_dragonfly.reshape(-1, problem.dim)])
            train_obj_dragonfly = torch.cat([train_obj_dragonfly, new_obj_dragonfly.reshape(-1, problem.num_objectives + num_constraints)])
            train_obj_true_dragonfly = torch.cat([train_obj_true_dragonfly, new_obj_true_dragonfly.reshape(-1, problem.num_objectives)])


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


        train_x_random = torch.cat([train_x_random, new_x_random.to(**tkwargs)])
        train_obj_random = torch.cat([train_obj_random, new_obj_random.to(**tkwargs)])
        train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random.to(**tkwargs)])

        if args.output_constraint:
            train_con_qparego = torch.cat([train_con_qparego, new_con_qparego])
            train_con_qehvi = torch.cat([train_con_qehvi, new_con_qparego])
            train_con_qnehvi = torch.cat([train_con_qnehvi, new_con_qparego])
            if args.w_dragonfly:
                train_con_dragonfly = torch.cat([train_con_dragonfly, new_con_dragonfly]) 
            
            train_con_random = torch.cat([train_con_random, output_constraints[args.output_constraint](problem, new_x_random)]) 
        
        
        # update progress
        for eval_list, train_obj, train_con in zip(
            (eval_random, eval_qparego, eval_qehvi, eval_qnehvi, eval_dragonfly), 
            (
                train_obj_true_random,
                train_obj_true_qparego,
                train_obj_true_qehvi,
                train_obj_true_qnehvi,
                train_obj_true_dragonfly
            ), 
            
            ( 
             train_con_random, 
             train_con_qparego, 
             train_con_qehvi,
             train_con_qnehvi,
             train_con_dragonfly)
            
        ):   
            if not args.w_dragonfly and train_obj == train_obj_true_dragonfly:
                continue 
            
            if args.output_constraint:
                # compute pareto front
                is_feas = (train_con <= 0).all(dim=-1)
                feas_train_obj = train_obj[is_feas]
                if feas_train_obj.shape[0] > 0:
                    pareto_mask = is_non_dominated(feas_train_obj)
                    pareto_y = feas_train_obj[pareto_mask]
                    # compute feasible hypervolume
                    volume = hv.compute(pareto_y)
                else:
                    volume = 0.0
                eval_list.append(volume)
            
            elif args.test_function == 'ScattBO':
                volume = torch.mean(train_obj[-BATCH_SIZE:], dim=0).sum().item()
                eval_list.append(volume * scale) 
                
            else:
                # compute hypervolume
                bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
                volume = bd.compute_hypervolume().item()
                eval_list.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qparego, tkwargs=tkwargs)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qehvi, tkwargs=tkwargs)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, problem=problem, NOISE_SE=NOISE_SE, train_con=train_con_qnehvi, tkwargs=tkwargs)  

        # dragonfly_opt._build_new_model()
        # dragonfly_opt._set_next_gp()

        t1 = time.monotonic()
        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI, dragonfly) = "
                f"({eval_random[-1]:>4.2f}, {eval_qparego[-1]:>4.2f}, {eval_qehvi[-1]:>4.2f}, {eval_qnehvi[-1]:>4.2f}, {eval_dragonfly[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")  

    eval_all_dragonfly.append(eval_dragonfly), eval_all_qehvi.append(eval_qehvi), eval_all_qnehvi.append(eval_qnehvi) 
    eval_all_qparego.append(eval_qparego), eval_all_random.append(eval_random) 
    
    train_obj_true_all_qparego.append(train_obj_true_qparego), train_obj_true_all_qehvi.append(train_obj_true_qehvi) 
    train_obj_true_all_qnehvi.append(train_obj_true_qnehvi), train_obj_true_all_random.append(train_obj_true_random) 
    train_obj_true_all_dragonfly.append(train_obj_true_dragonfly) 
    
    
    train_obj_all_qparego.append(train_obj_qparego), train_obj_all_qehvi.append(train_obj_qehvi), 
    train_obj_all_qnehvi.append(train_obj_qnehvi), train_obj_all_random.append(train_obj_random), 
    train_obj_all_dragonfly.append(train_obj)
    
    train_x_all_qparego.append(train_x_qparego), train_x_all_qehvi.append(train_x_qehvi), 
    train_x_all_qnehvi.append(train_x_qnehvi), train_x_all_random.append(train_x_random), 
    if args.w_dragonfly:
        train_x_all_dragonfly.append(train_x_dragonfly) 

qparego = {'eval': eval_all_qparego, 'train_obj_true': train_obj_true_all_qparego, 'train_obj': train_obj_all_qparego, 
           'train_x': train_x_all_qparego, 'method': 'qparego'} 
qehvi = {'eval': eval_all_qehvi, 'train_obj_true': train_obj_true_all_qehvi, 'train_obj': train_obj_all_qehvi,
         'train_x': train_x_all_qehvi, 'method': 'qehvi'}
qnehvi = {'eval': eval_all_qnehvi, 'train_obj_true': train_obj_true_all_qnehvi, 'train_obj': train_obj_all_qnehvi,
          'train_x': train_x_all_qnehvi, 'method': 'qnehvi'}
if args.w_dragonfly:
    dragonfly = {'eval': eval_all_dragonfly, 'train_obj_true': train_obj_true_all_dragonfly, 'train_obj': train_obj_all_dragonfly,
             'train_x': train_x_all_dragonfly, 'method': 'dragonfly'}
random = {'eval': eval_all_random, 'train_obj_true': train_obj_true_all_random, 'train_obj': train_obj_all_random,
          'train_x': train_x_all_random, 'method': 'random'} 

for (name, method) in zip(['qparego', 'qehvi', 'qnehvi', 'dragonfly', 'random'], [qparego, qehvi, qnehvi, dragonfly, random]):
    with open(f"MOO/runs/{date}/{name}.pkl", "wb") as f:
        pickle.dump(method, f) 
