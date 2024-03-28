import torch 
from botorch.test_functions.multi_objective import BraninCurrin, C2DTLZ2 
from ThermoElectrics.testfunc import ThermoelectricBenchmark


test_functions = {'branincurrin': BraninCurrin(negate=True), 
                  'c2dtlz2': C2DTLZ2(dim=4, num_objectives=2, negate=True),
                  'thermoelectrics': ThermoelectricBenchmark(),  # Your function
} 

# Define the domain for each parameter
domain = [
    [2, 12],  # pH values range from 2 to 12
    [15, 80],  # pressure values range from 20 to 80
    [0, 1]  # solvent can be 0 ('Ethanol') or 1 ('Methanol') 
] 

input_constraints = {'ScattBO-constraint' : domain}

output_constraints = {'c2-constraint' : lambda problem, x: -problem.evaluate_slack(x), 
                      None : None}



