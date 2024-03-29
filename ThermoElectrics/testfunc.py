from .features import featurize, reverse_target_norm
from .model import load_model
import numpy as np
import pandas as pd
import torch

def call_model(features,model_file="ThermoElectrics/gbr_model.joblib"):
    model = load_model(model_file)
    features_arry = np.reshape(features,(1,-1))
    seebeck, sigma, kappa = reverse_target_norm(model.predict(features_arry)[0])
    return seebeck, sigma, kappa

def test_thermoelectric_BaCrSe(ba_comp,cr_comp,se_comp,temperature):
    #se_comp = max(1.0e-6, 1.0 - ba_comp - cr_comp)
    formula = f"Ba{ba_comp}Cr{cr_comp}Se{se_comp}"
    features = featurize(formula,temperature=temperature)
    return call_model(features)

def test_thermoelectric_FeNbMgSn(fe_comp,
                            nb_comp,
                            mg_comp,
                            sn_comp,
                            temperature):
    """
    Given fractional compositions of Fe, Nb, Mg, Sn
    gets featurizes using matminer, then returns gradient
    boost regression model prediction for Seebeck coeff (μV/K)
    sigma (S/m), Kappa(W/mK).

    We want to maximize Seebeck Coeff and sigma, while minimizing kappa.
    """
    assert sum([fe_comp,nb_comp,mg_comp,sn_comp]) == 1.0
    formula = f"Fe{fe_comp}Nb{nb_comp}Mg{mg_comp}Sn{sn_comp}"
    features = featurize(formula,temperature=temperature)
    return call_model(features)

class ThermoelectricBenchmark:
    def __init__(self):
        self.dim = 4
        self.bounds = torch.tensor([[0.5,0.5,0.5,300.0],
                                    [10.0,10.0,10.0,400.0]])
        self.num_objectives = 3 
        self.ref_point = torch.tensor([228.0,372.0, 1.09])
        self.ideal_scenario = torch.tensor([1.0e3, 10**6, 1.0e-1])
        self.max_hv = self.estimate_initial_max_hv()

    def estimate_initial_max_hv(self):
        hv_volume = 1
        for ideal, ref in zip(self.ideal_scenario, self.ref_point):
            diff = abs(ideal - ref) if ideal > ref else 0
            hv_volume *= diff
        return hv_volume

    def update_max_hv(self, new_max_hv):
        self.max_hv = new_max_hv

    def evaluate(self, X):
        # Convert tensor X to numpy array if necessary
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        
        outputs = []

        for i in range(len(X_np)):
            ba_comp, cr_comp, se_comp, temperature = X_np[i]
            output = test_thermoelectric_BaCrSe(ba_comp, cr_comp, se_comp, temperature)
            #  Seebeck & sigma for maximization
            seebeck2 = output[0]**2
            sigma = output[1]
            # Kappa negate for minimization
            kappa = -output[2]
            outputs.append([seebeck2,sigma,kappa])
        
        return torch.tensor(outputs)
    
    def __call__(self, X):
        return self.evaluate(X)

