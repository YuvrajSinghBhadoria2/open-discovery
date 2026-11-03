# Genetic Algorithm for Feynman Symbolic Regression
# Uses DEAP library for genetic algorithms.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import random
import math
from deap import base, creator, tools, algorithms

from grammar import Grammar, TypedExpr, DimVec
from constants import fit_constants


# --- DEAP singleton: create once, reuse ---
if 'FitnessMin' not in creator.__dict__:
    creator.create('FitnessMin', base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create('Individual', list, fitness=creator.FitnessMin)


def evaluate_individual(individual, grammar, variables, target_values, pinner):
    """Evaluate an individual with constant optimization via Levenberg-Marquardt.
    Returns negative NMSE so GA maximizes fitness (minimizes NMSE)."""
    expr = individual[0]
    try:
        # Check if expression has constants - if not, skip LM optimization
        has_consts = any(c.op.op_name == 'const' for c in expr.get_constants())

        if has_consts:
            # Optimize constants for THIS individual on the fly (fast LM, max 20 iter)
            optimized_expr, const_values = fit_constants(
                expr, variables, target_values, max_iter=20, ftol=1e-8, xtol=1e-8
            )
        else:
            optimized_expr, const_values = expr, {}

        pred = optimized_expr.evaluate(variables)
        if not np.all(np.isfinite(pred)):
            return (-float('inf'),)
        pred = pred.flatten() if pred.ndim > 1 else pred
        if pred.shape != target_values.shape:
            return (-float('inf'),)
        mse = np.mean((pred - target_values) ** 2)
        var = np.var(target_values)
        if var == 0:
            return (-float('inf'),)
        nmse = mse / var
        # Cache optimized expression for crossover/mutation to inherit
        individual.optimized_expr = optimized_expr
        individual.const_values = const_values
        # Return NEGATIVE NMSE so maximizing fitness = minimizing NMSE
        return (-nmse,)
    except Exception:
        return (-float('inf'),)


def generate_individual(grammar: Grammar, target_dim: DimVec):
    """Generate a random individual using the grammar."""
    exprs = grammar.generate_dimension(target_dim, max_depth=6, n_samples=5, max_nodes=20)
    if exprs:
        expr = random.choice(exprs)
    else:
        expr = TypedExpr.constant(0.5, DimVec.parse('1'))
    return creator.Individual([expr])


def collect_subtree_paths(expr: TypedExpr, path: Tuple[int, ...] = (), visited: Optional[set] = None, max_depth: int = 20) -> List[Tuple[int, ...]]:
    """Collect all valid subtree paths in an expression with cycle and depth protection."""
    if visited is None:
        visited = set()
    # Use id() to detect cycles
    expr_id = id(expr)
    if expr_id in visited:
        return []
    visited.add(expr_id)

    # Depth protection
    if len(path) >= max_depth:
        return [path]

    paths = [path]
    for i, child in enumerate(expr.children):
        paths.extend(collect_subtree_paths(child, path + (i,), visited, max_depth))
    return paths


def get_subexpr_at_path(expr: TypedExpr, path: Tuple[int, ...]) -> TypedExpr:
    """Extract subexpression at given path."""
    for idx in path:
        expr = expr.children[idx]
    return expr


def replace_subexpr_at_path(expr: TypedExpr, path: Tuple[int, ...], replacement: TypedExpr, max_nodes: int = 30) -> Optional[TypedExpr]:
    """Replace subexpression at path with replacement, return new expression.
    Returns None if result would exceed max_nodes."""
    if not path:
        return replacement if replacement.count_nodes() <= max_nodes else None
    idx = path[0]
    new_children = list(expr.children)
    replaced = replace_subexpr_at_path(expr.children[idx], path[1:], replacement, max_nodes)
    if replaced is None:
        return None
    new_children[idx] = replaced
    result = TypedExpr(expr.op, tuple(new_children), expr.dim)
    return result if result.count_nodes() <= max_nodes else None


def crossover(individual1, individual2, max_nodes: int = 30):
    """Proper subtree crossover: swap random compatible subtrees between different-shaped trees.
    Modifies individuals in place (DEAP standard pattern)."""
    expr1, expr2 = individual1[0], individual2[0]

    # Collect all subtree paths from both expressions
    paths1 = collect_subtree_paths(expr1, max_depth=15)
    paths2 = collect_subtree_paths(expr2, max_depth=15)

    if not paths1 or not paths2:
        return individual1, individual2

    # Try to find dimensionally compatible swap
    max_attempts = 10
    for _ in range(max_attempts):
        path1 = random.choice(paths1)
        path2 = random.choice(paths2)

        sub1 = get_subexpr_at_path(expr1, path1)
        sub2 = get_subexpr_at_path(expr2, path2)

        # Only swap if dimensions match (grammar constraint)
        if sub1.dim == sub2.dim:
            new1 = replace_subexpr_at_path(expr1, path1, sub2, max_nodes)
            new2 = replace_subexpr_at_path(expr2, path2, sub1, max_nodes)

            # Verify both are valid and not too complex
            if new1 is not None and new2 is not None:
                # Modify in place - keep same Individual objects with fitness
                individual1[0] = new1
                individual2[0] = new2
                # Invalidate fitness - DEAP will re-evaluate
                del individual1.fitness.values
                del individual2.fitness.values
                return individual1, individual2

    # No compatible swap found - return parents unchanged
    return individual1, individual2


def mutate(individual, grammar, target_dim, introduce_constants: bool = True, max_nodes: int = 30):
    """Mutate by replacing a random subtree with a grammar-valid alternative.
    Returns a tuple (individual,) to match DEAP's varAnd expectation."""
    expr = individual[0]
    paths = collect_subtree_paths(expr, max_depth=15)

    if not paths:
        return (individual,)

    # Pick random subtree to replace
    path = random.choice(paths)
    subexpr = get_subexpr_at_path(expr, path)
    required_dim = subexpr.dim

    # Generate valid replacements for this dimension
    replacements = grammar.generate_dimension(required_dim, max_depth=4, n_samples=10, max_nodes=max_nodes)

    if not replacements:
        return (individual,)

    replacement = random.choice(replacements)
    new_expr = replace_subexpr_at_path(expr, path, replacement, max_nodes)

    if new_expr is None:
        return (individual,)

    # Modify in place - keep same Individual object with fitness
    individual[0] = new_expr
    del individual.fitness.values
    return (individual,)


def clone_individual(individual):
    """Custom clone for Individual: shallow copy list, keep same expression (immutable)."""
    clone = creator.Individual(list(individual))
    clone.fitness = creator.FitnessMin(individual.fitness.values) if individual.fitness.valid else creator.FitnessMin()
    return clone


def run_ga_search(equation_name, grammar, target_dim, variables, target_values, pinner, max_generations=200,
                  use_mdl_penalty: bool = True):
    """Run genetic algorithm search for symbolic regression."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    toolbox = base.Toolbox()
    toolbox.register('individual', lambda: generate_individual(grammar, target_dim))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', crossover, max_nodes=30)
    toolbox.register('mutate', mutate, grammar=grammar, target_dim=target_dim, max_nodes=30)
    toolbox.register('evaluate', evaluate_individual, grammar=grammar, variables=variables, target_values=target_values, pinner=pinner)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('clone', clone_individual)

    # Larger population, more generations
    population = toolbox.population(n=200)

    # Pre-evaluate all
    for individual in population:
        if not individual.fitness.valid:
            try:
                fit = toolbox.evaluate(individual)
                individual.fitness.values = fit
            except Exception:
                individual.fitness.values = (-float('inf'),)

    # Standard eaSimple with hall of fame - DEAP handles fitness invalidation correctly
    hall_of_fame = tools.HallOfFame(5)
    hall_of_fame.update(population)

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.6, mutpb=0.3,
        ngen=max_generations,
        halloffame=hall_of_fame,
        verbose=True
    )

    best_individual = hall_of_fame[0]
    if best_individual is None:
        best_individual = tools.selBest(population, k=1)[0]

    # Use cached optimized expression if available
    best_expr = getattr(best_individual, 'optimized_expr', best_individual[0])

    best_fitness = best_individual.fitness.values
    if len(best_fitness) == 0:
        raise ValueError("No fitness values found")

    best_nmse = -float(best_fitness[0])  # Fitness is stored as -NMSE, so negate to get NMSE

    if use_mdl_penalty:
        k = best_expr.complexity()
        n_data = len(target_values) if hasattr(target_values, '__len__') else 1000
        if best_nmse > 0 and best_nmse < 1e10:
            mdl_score = n_data * math.log(best_nmse) + k * math.log(n_data)
            bic_score = n_data * math.log(best_nmse) + k * math.log(n_data)
            aic_score = n_data * math.log(best_nmse) + 2 * k
        else:
            mdl_score = float('inf')
            bic_score = float('inf')
            aic_score = float('inf')
    else:
        mdl_score = best_nmse
        bic_score = best_nmse
        aic_score = best_nmse

    return best_expr, best_nmse, mdl_score, bic_score, aic_score