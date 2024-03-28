from features import featurize, reverse_target_norm
from model import fit_gbr_model,load_model
import numpy as np

def call_model(features,model_file="gbr_model.joblib"):
    model = load_model(model_file)
    features_arry = np.reshape(features,(1,-1))
    seebeck, sigma, kappa = reverse_target_norm(model.predict(features_arry)[0])
    return seebeck, sigma, kappa

def test_thermoelectric_BaCrSe(ba_comp,cr_comp,temperature):
    se_comp = 1.0 - ba_comp - cr_comp
    assert sum([ba_comp,cr_comp,se_comp]) == 1.0
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


