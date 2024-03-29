import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from features import featurize, reverse_target_norm
from joblib import load
from matplotlib.patches import Patch
import matplotlib.lines as mlines

def load_model(filepath):
    return load(filepath)

def call_model(features, model_file="gbr_model.joblib"):
    model = load_model(model_file)
    features_array = np.reshape(features, (1, -1))
    seebeck, sigma, kappa = reverse_target_norm(model.predict(features_array)[0], scalers_path="./target_norm.joblib")
    return seebeck, sigma, kappa

def test_thermoelectric_BaCrSe(ba_comp, cr_comp, se_comp, temperature):
    formula = f"Ba{ba_comp}Cr{cr_comp}Se{se_comp}"
    features = featurize(formula, temperature=temperature)
    return call_model(features)

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.01, 0.98)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    ba_comp, cr_comp = individual[0], individual[1]
    se_comp = 1.0 - ba_comp - cr_comp
    if se_comp < 0 or any(comp < 0 for comp in individual):
        return -1000, -1000, 1000
    temperature = 300
    seebeck, elec_cond, therm_cond = test_thermoelectric_BaCrSe(ba_comp, cr_comp, se_comp, temperature)
    return seebeck**2, elec_cond, therm_cond

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def ensure_valid_individual(func):
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        for child in offspring:
            total_comp = sum(child[:2])
            if total_comp > 1.0 or any(comp < 0 for comp in child):
                child[0] = min(max(child[0], 0), 0.99)
                child[1] = min(max(child[1], 0), 0.99 - child[0])
            child[2] = 1.0 - child[0] - child[1]
        return offspring
    return wrapper

toolbox.register("mate", ensure_valid_individual(toolbox.mate))
toolbox.register("mutate", ensure_valid_individual(toolbox.mutate))

population_size = 150
number_of_generations = 50

population = toolbox.population(n=population_size)
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size*2, cxpb=0.7, mutpb=0.3, ngen=number_of_generations, verbose=True)

fronts = tools.sortNondominated(population, len(population), first_front_only=True)
sorted_front = sorted(fronts[0], key=lambda x: x.fitness.values[0])

# Define consistent colors for Ba, Cr, and Se
colors = ['blue', 'green', 'red']

def draw_pie(dist, xpos, ypos, size, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()

    for r1, r2, color in zip(pie[:-1], pie[1:], colors):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, s=size, color=color)

    return ax

fig, ax = plt.subplots(figsize=(10, 8))

# Get the minimum and maximum thermal conductivity values
therm_cond_values = [ind.fitness.values[2] for ind in sorted_front]
min_therm_cond = min(therm_cond_values)
max_therm_cond = max(therm_cond_values)

for ind in sorted_front:
    seebeck, elec_cond, therm_cond = ind.fitness.values
    wedge_size = 0.01 * (1 / therm_cond)  # Scale wedge size based on thermal conductivity
    pie_components = [ind[0], ind[1], 1.0 - ind[0] - ind[1]]
    draw_pie(pie_components, seebeck, elec_cond, wedge_size * 10000, ax=ax)




species_legend = ax.legend(handles=[Patch(color=colors[i], label=['Ba', 'Cr', 'Se'][i]) for i in range(3)], 
                           title="Species", 
                           bbox_to_anchor=(1.05, 1), 
                           loc='upper left')
ax.add_artist(species_legend)

legend_markers = [mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                                markersize=max_therm_cond, 
                                label=r'$\kappa = {:.2f}$'.format(max_therm_cond)),
                  mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                                markersize=(min_therm_cond + max_therm_cond) / 2, 
                                label=r'$\kappa = {:.2f}$'.format((min_therm_cond + max_therm_cond) / 2)),
                  mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                                markersize=min_therm_cond, 
                                label=r'$\kappa = {:.2f}$'.format(min_therm_cond))]
ax.legend(handles=legend_markers, 
          title='Thermal Conductivity (W/mK)', 
          bbox_to_anchor=(1.05, 0.5), 
          loc='center left')
ax.set_xlabel('Squared Seebeck Coeff. (μV$^2$/K$^2$)')
ax.set_ylabel('Electrical Conductivity (S/m)')
ax.set_title("Pareto Front")
plt.savefig('pareto_evoalg_moo.png',dpi=300,bbox_inches='tight')
