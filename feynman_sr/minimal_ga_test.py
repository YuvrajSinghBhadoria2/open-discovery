"""
Minimal test script for DEAP setup
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import random
from deap import base, creator, tools, algorithms

# Setup DEAP
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

# Mock evaluate function
def evaluate(ind):
    return (-np.random.rand(),)

# Mock crossover and mutation
toolbox = base.Toolbox()
toolbox.register('individual', lambda: creator.Individual([random.random()]))
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxBlend, alpha=0.5)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=3)

# Create population
population = toolbox.population(n=5)

# Run GA
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=True)

# Get best individual
best_individual = tools.selBest(population, k=1)[0]
print(f'Best individual: {best_individual}')
print(f'Best individual fitness: {best_individual.fitness.values}')