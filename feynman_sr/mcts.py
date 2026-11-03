"""
Monte Carlo Tree Search for Feynman Symbolic Regression
Combines MCTS with dimensional constraints and hypothesis pinning.
Search space is expressions typed by physical dimensions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
import time
import math
import random
from dataclasses import dataclass, field
import numpy as np

from grammar import Grammar, TypedExpr, OpType, DimVec, Dim, FEYNMAN_OPS
from pinning import HypothesisPinner, PinType, Pin, PinningState
from ga_search import run_ga_search
from constants import fit_constants


@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    expr: TypedExpr
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    depth: int = 0

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        ) if self.parent else float('inf')

    def best_child(self, exploration: float = 1.414) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def add_child(self, child: 'MCTSNode') -> 'MCTSNode':
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child


def evaluate_expression(expr: TypedExpr, variables: Dict[str, np.ndarray],
                        target_values: np.ndarray,
                        pinner: Optional[HypothesisPinner] = None) -> float:
    """Evaluate expression fitness (lower is better: NMSE)"""
    try:
        pred = expr.evaluate(variables)
        if not np.all(np.isfinite(pred)):
            return float('inf')
        if pred.shape != target_values.shape:
            return float('inf')
        mse = np.mean((pred - target_values) ** 2)
        var = np.var(target_values)
        if var == 0:
            return float('inf')
        nmse = mse / var
        return float(nmse)
    except Exception:
        return float('inf')


def mdl_score(expr: TypedExpr, nmse: float, n_data: int) -> float:
    """Minimum Description Length score: MDL = n * log(NMSE) + k * log(n)"""
    k = expr.complexity()
    if nmse <= 0 or not np.isfinite(nmse):
        return float('inf')
    return n_data * math.log(nmse) + k * math.log(n_data)


def bic_score(expr: TypedExpr, nmse: float, n_data: int) -> float:
    """Bayesian Information Criterion"""
    k = expr.complexity()
    if nmse <= 0 or not np.isfinite(nmse):
        return float('inf')
    return n_data * math.log(nmse) + k * math.log(n_data)


def aic_score(expr: TypedExpr, nmse: float, n_data: int) -> float:
    """Akaike Information Criterion"""
    k = expr.complexity()
    if nmse <= 0 or not np.isfinite(nmse):
        return float('inf')
    return n_data * math.log(nmse) + 2 * k


def score_expression(expr: TypedExpr, nmse: float, n_data: int) -> float:
    """Default scoring: use MDL"""
    return mdl_score(expr, nmse, n_data)


def optimize_constants(expr: TypedExpr, variables: Dict[str, np.ndarray],
                        target_values: np.ndarray) -> Tuple[TypedExpr, Dict[int, float]]:
    """Optimize constants in expression using Levenberg-Marquardt"""
    return fit_constants(expr, variables, target_values)


def select_action(node: MCTSNode, grammar: Grammar, pinner: HypothesisPinner,
                   exploration: float = 1.414) -> Optional[MCTSNode]:
    """Select best child using UCB1"""
    if not node.children:
        return None
    return node.best_child(exploration)


def expand_node(node: MCTSNode, grammar: Grammar, pinner: HypothesisPinner,
                 target_dim: DimVec) -> List[MCTSNode]:
    """Generate child nodes by applying valid grammar rules"""
    children = []
    # Find expandable positions (variables that can be replaced)
    expandable_paths = find_expandable_positions(node.expr)

    for path in expandable_paths:
        subexpr = get_subexpr(node.expr, path)
        if subexpr is None:
            continue

        # Get the dimension this subexpression must have
        required_dim = get_required_dimension(node.expr, path, target_dim)
        if required_dim is None:
            continue

        # Generate valid replacements for this dimension
        replacements = grammar.generate_dimension(required_dim, max_depth=3)

        for repl in replacements[:10]:  # Limit branching
            new_expr = replace_subexpr(node.expr, path, repl)
            if new_expr and grammar.check_types(new_expr):
                child = MCTSNode(expr=new_expr, parent=node)
                children.append(node.add_child(child))

    return children


def find_expandable_positions(expr: TypedExpr) -> List[Tuple[int, ...]]:
    """Find paths to nodes that can be expanded (variables, constants)"""
    paths = []

    def traverse(e: TypedExpr, path: Tuple[int, ...]):
        if e.op.op_name in ('var', 'const'):
            paths.append(path)
        for i, child in enumerate(e.children):
            traverse(child, path + (i,))

    traverse(expr, ())
    return paths


def get_subexpr(expr: TypedExpr, path: Tuple[int, ...]) -> Optional[TypedExpr]:
    """Get subexpression at path"""
    current = expr
    for idx in path:
        if idx < len(current.children):
            current = current.children[idx]
        else:
            return None
    return current


def get_required_dimension(expr: TypedExpr, path: Tuple[int, ...],
                            target_dim: DimVec) -> Optional[DimVec]:
    """Infer required dimension at path given target dimension at root"""
    # This is the inverse of dimension propagation
    # For now, we use the grammar's type checking to validate
    # A full implementation would do proper inverse dimensional analysis
    return None  # Let grammar handle it


def replace_subexpr(expr: TypedExpr, path: Tuple[int, ...],
                     replacement: TypedExpr) -> Optional[TypedExpr]:
    """Replace subexpression at path with replacement"""
    if not path:
        return replacement

    def recurse(e: TypedExpr, current_path: Tuple[int, ...]) -> Optional[TypedExpr]:
        if current_path == path:
            return replacement

        if len(current_path) > len(path):
            return None

        next_idx = path[len(current_path)]
        if next_idx >= len(e.children):
            return None

        new_children = []
        for i, child in enumerate(e.children):
            if i == next_idx:
                new_child = recurse(child, current_path + (i,))
                if new_child is None:
                    return None
                new_children.append(new_child)
            else:
                new_children.append(child)

        return TypedExpr(e.op, tuple(new_children), e.dim)

    return recurse(expr, ())


def run_mcts_search(equation_name: str,
                    grammar: Grammar,
                    target_dim: DimVec,
                    variables: Dict[str, np.ndarray],
                    target_values: np.ndarray,
                    pinner: HypothesisPinner,
                    max_iterations: int = 10**9,
                    time_limit: float = 120.0,
                    num_restarts: int = 3) -> Tuple[Optional[TypedExpr], float, Dict[int, float]]:
    """
    Run genetic algorithm search for symbolic regression.

    Returns: (best_expr, best_score, best_constants)
    """
    from ga_search import run_ga_search
    best_expr, best_score = run_ga_search(equation_name, grammar, target_dim, variables, target_values, pinner, max_generations=100)
    return best_expr, best_score, {}