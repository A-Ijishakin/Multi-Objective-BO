import torch 
from botorch.test_functions.multi_objective import BraninCurrin, C2DTLZ2 
from ScattBO.utils.ScattBO import generate_structure, ScatterBO_small_benchmark, ScatterBO_large_benchmark

scat_bo = ScatterBO_small_benchmark 
scat_bo.dim = 3  
scat_bo.bounds = torch.tensor([(2.0, 15.0, 0.0), (12.0, 80.0, 1.0)]) 
scat_bo.num_objectives = 2
scat_bo.ref_point = torch.tensor([2.5, 2.5]) 

test_functions = {'branincurrin': BraninCurrin(negate=True), 
                  'c2dtlz2': C2DTLZ2(dim=4, num_objectives=2, negate=True),
                  'ScattBO': scat_bo} 

# Define the domain for each parameter
domain = [
    [2, 12],  # pH values range from 2 to 12
    [15, 80],  # pressure values range from 20 to 80
    [0, 1]  # solvent can be 0 ('Ethanol') or 1 ('Methanol') 
] 

input_constraints = {'ScattBO-constraint' : domain}

output_constraints = {'c2-constraint' : lambda problem, x: -problem.evaluate_slack(x), 
                      None : None}



