from botorch.test_functions.multi_objective import BraninCurrin, C2DTLZ2 

test_functions = {'branincurrin': BraninCurrin(negate=True), 
                  'c2dtlz2': C2DTLZ2(dim=4, num_objectives=2, negate=True)} 


output_constraints = {'c2-constraint' : lambda problem, x: -problem.evaluate_slack(x)}



input_constraints = {}
