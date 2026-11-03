"""
Debug script for GA setup
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import random
from deap import base, creator, tools, algorithms
from grammar import Grammar, TypedExpr, DimVec


def evaluate_individual(individual, grammar, variables, target_values, pinner):
    """Evaluate an individual (expression) using NMSE."""
    expr = individual[0]
    try:
        pred = expr.evaluate(variables)
        if not np.all(np.isfinite(pred)):
            return (-float('inf'),)
        if pred.shape != target_values.shape:
            return (-float('inf'),)
        mse = np.mean((pred - target_values) ** 2)
        var = np.var(target_values)
        if var == 0:
            return (-float('inf'),)
        nmse = mse / var
        return (-nmse,)
    except Exception:
        return (-float('inf'),)

# Setup DEAP
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

def generate_individual(grammar: Grammar, target_dim: DimVec):
    """Generate a random individual (expression) using the grammar."""
    exprs = grammar.generate_dimension(target_dim)
    if exprs:
        expr = exprs[0]
    else:
        expr = TypedExpr.constant(0.5, DimVec.parse('1'))
    return creator.Individual([expr])

# Test
if __name__ == "__main__":
    # Test with feynman-i.12.1
    names = ['x0', 'x1']
    var_dims = {'x0': DimVec.parse('M'), 'x1': DimVec.parse('L T^-1')}
    target_dim = DimVec.parse('M')
    
    # Create grammar
    grammar = Grammar(var_dims, target_dim)
    
    # Mock data
    variables = {name: np.random.rand(100, 1) for name in names}
    target_values = np.random.rand(100, 1)
    
    # Mock pinner
    class MockPinner:
        pass
    
    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register('individual', lambda: generate_individual(grammar, target_dim))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate_individual, grammar=grammar, variables=variables, target_values=target_values, pinner=MockPinner())
    toolbox.register('select', tools.selRandom)
    
    # Create population
    population = toolbox.population(n=5)
    
    # Evaluate population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Print fitness values
    for ind in population:
        print(f'Individual: {ind[0]}, Fitness: {ind.fitness.values}')