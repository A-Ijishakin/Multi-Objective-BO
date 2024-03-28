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

def test_thermoelectric_BaCrSe(ba_comp,cr_comp,temperature):
    se_comp = 1.0 - ba_comp - cr_comp
    formula = f"Ba{ba_comp}Cr{cr_comp}Se{se_comp}"
    #print(se_comp+ba_comp+cr_comp)
    #assert sum([ba_comp,cr_comp,se_comp]) == 1.0
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
        self.dim = 3
        self.bounds = torch.tensor([[0.0e0,0.0e0,300.0],
                                    [0.499999e0,0.49999e0,1200.0]])
        self.num_objectives = 3 
        self.ref_point = torch.tensor([-1.0e2,-1.0e2, 1.0e3])

    def evaluate(self, X):
        # Convert tensor X to numpy array if necessary
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        
        outputs = []

        for i in range(len(X_np)):
            ba_comp, cr_comp, temperature = X_np[i]
            output = test_thermoelectric_BaCrSe(ba_comp, cr_comp, temperature)
            #  Seebeck & sigma for maximization
            seebeck2 = output[0]**2
            sigma = output[1]
            # Kappa negate for minimization
            kappa = -output[2]
            outputs.append([seebeck2,sigma,kappa])
        
        return torch.tensor(outputs)
    
    def __call__(self, X):
        return self.evaluate(X)

