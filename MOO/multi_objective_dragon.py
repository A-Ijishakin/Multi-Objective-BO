import torch
import time
from botorch import fit_gpytorch_mll
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from dragonfly import load_config
from argparse import Namespace
from dragonfly.exd.experiment_caller import CPMultiFunctionCaller
from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit 
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from methods import (generate_initial_data, initialize_model, problem, 
                     optimize_qehvi_and_get_observation, optimize_qnehvi_and_get_observation, 
                      optimize_qnparego_and_get_observation, 
                      MC_SAMPLES, BATCH_SIZE, NOISE_SE, N_BATCH, verbose, N_TRIALS) 
import numpy as np 
import json 


torch.manual_seed(43)
np.random.seed(43) 

hvs_all_qparego, hvs_all_qehvi, hvs_all_qnehvi, hvs_all_random, hvs_all_dragonfly = [], [], [], [], [] 
train_obj_true_all_qparego, train_obj_true_all_qehvi, train_obj_true_all_qnehvi, train_obj_true_all_random, train_obj_true_all_dragonfly = [], [], [], [], [] 
train_obj_all_qparego, train_obj_all_qehvi, train_obj_all_qnehvi, train_obj_all_random, train_obj_all_dragonfly = [], [], [], [], [] 
train_x_all_qparego, train_x_all_qehvi, train_x_all_qnehvi, train_x_all_random, train_x_all_dragonfly = [], [], [], [], []


for n in N_TRIALS:
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
    train_x_qparego, train_obj_qparego, train_obj_true_qparego = generate_initial_data(
        n=6
    )
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
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)


    # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qparego)
    volume = bd.compute_hypervolume().item()

    hvs_qparego.append(volume)
    hvs_qehvi.append(volume)
    hvs_qnehvi.append(volume)
    hvs_random.append(volume)
    hvs_dragonfly.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):
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
            model_qparego, train_x_qparego, train_obj_qparego, qparego_sampler
        )
        new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler
        )
        (
            new_x_qnehvi,
            new_obj_qnehvi,
            new_obj_true_qnehvi,
        ) = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
        )
        new_x_random, new_obj_random, new_obj_true_random = generate_initial_data(
            n=BATCH_SIZE
        )

        for (x, y) in zip(train_x_dragonfly, train_obj_dragonfly):
            #dragonfly
            x = [datum.item() for datum in x]
            y = [datum.item() for datum in y]
            dragonfly_opt.tell([(x,
                                y)])

        dragonfly_opt.step_idx += 1

        # Retrieve the Pareto-optimal points
        new_x_dragonfly = torch.tensor(dragonfly_opt.ask(n_points=BATCH_SIZE)).to('cuda')

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
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

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

for method in [qparego, qehvi, qnehvi, dragonfly, random]:
    json.dump(method, open(f"results/{method}.json", "w")) 
    
breakpoint 