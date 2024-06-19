import torch
from botorch import fit_gpytorch_mll 
from botorch.models.transforms import Normalize, Standardize 
from botorch.sampling.normal import SobolQMCNormalSampler
from methods import (initialize_model, produce_mean_and_variance, 
                    calc_likelihoods, 
                    optimize_qnparego_and_get_observation, add_noise, 
                    MC_SAMPLES) 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 

#set the seed for reproducibility 
torch.manual_seed(42)
np.random.seed(42) 

#set the batch size 
BATCH_SIZE = 32 

#load in the data 
path = 'WSSB MasterFile.xlsx'

#load in the data 
data = pd.read_excel(path) 

#specify the training data and objective values 
train_X  = data.iloc[:96].drop(['Campaign', 'date', 'sample_id', 'replica_id', 'exp_id', 'well', 'Growth (OD)', 'Lipid content (FL)'], axis=1).values
train_Y = data.iloc[:96][['Growth (OD)', 'Lipid content (FL)']].values  

#memory allocation for the likelihoods and candidates 
likelihoods, all_candidates = [], []

for i in range(3): 
    train_obj = train_Y 
    if i == 0:
        train_x, train_obj, train_Yvar, xScaler, yScaler = produce_mean_and_variance(train_X, train_Y)  
    elif i == 1: 
        train_x, train_obj, xScaler, yScaler = add_noise(train_X, train_Y) 
    else:
        xScaler, yScaler = StandardScaler(), StandardScaler()
        ids = np.array([0.2, 0.5, 0.8] * 32) 
        train_x = np.concatenate([train_X, ids.reshape(96, 1)], axis=1) 
        train_x = torch.tensor(xScaler.fit_transform(train_X).astype(np.float32))
        train_obj = torch.tensor(yScaler.fit_transform(train_Y).astype(np.float32)) * -1 

    #initialize model 
    mll_qnparego, model_qnparego = initialize_model(train_x, train_obj) 

    #fit the gp modell
    fit_gpytorch_mll(mll_qnparego) 

    likelihoods.append(calc_likelihoods(model_qnparego, train_x, train_obj)) 

    #initialize the sampler
    qnparego_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))  

    #get candidates 
    candidates = optimize_qnparego_and_get_observation(model=model_qnparego, train_x=train_x, 
                                                        sampler=qnparego_sampler, BATCH_SIZE=BATCH_SIZE)

    #denormalize the candidates
    candidates = xScaler.inverse_transform(np.array(candidates).astype(np.float64))   

    all_candidates.append(candidates)

candidate_index = np.argmax(likelihoods) 
candidates = all_candidates[candidate_index]    

columns = data.drop(['Campaign', 'date', 'sample_id', 'replica_id', 'exp_id', 'well', 'Growth (OD)', 'Lipid content (FL)'], axis=1).columns 
results = pd.DataFrame({columns[i]: candidates[:, i] for i in range(len(columns))}) 

results.to_excel('Results.xlsx')



  

