"""
Levenberg-Marquardt constant optimization for Feynman Symbolic Regression.

Fits numeric constants in expressions to minimize NMSE on training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import least_squares
import sympy as sp

from grammar import TypedExpr


@dataclass
class ConstantFitResult:
    """Result of constant fitting"""
    expr: TypedExpr
    constants: Dict[int, float]
    nmse: float
    success: bool
    message: str


def extract_constants(expr: TypedExpr) -> List[Tuple[int, float]]:
    """Extract all constant nodes with their node IDs and current values"""
    constants = []
    node_id = 0

    def traverse(e: TypedExpr):
        nonlocal node_id
        if e.op.op_name == 'const':
            val = float(e.value) if e.value is not None else 1.0
            constants.append((node_id, val))
            node_id += 1
        for child in e.children:
            traverse(child)

    traverse(expr)
    return constants


def set_constants(expr: TypedExpr, const_values: Dict[int, float]) -> TypedExpr:
    """Create new expression with updated constant values"""
    const_iter = iter(sorted(const_values.items()))

    def traverse(e: TypedExpr) -> TypedExpr:
        if e.op.op_name == 'const':
            node_id, _ = next(const_iter)
            new_val = const_values.get(node_id, float(e.value) if e.value is not None else 1.0)
            return TypedExpr.const(new_val, e.dim)
        new_children = tuple(traverse(c) for c in e.children)
        return TypedExpr(e.op, new_children, e.dim)

    return traverse(expr)


def expr_to_callable(expr: TypedExpr, var_names: List[str]):
    """Convert TypedExpr to a callable function for optimization"""
    # Convert to sympy first, then lambdify
    sympy_expr = typed_to_sympy(expr, var_names)
    return sp.lambdify(var_names, sympy_expr, modules='numpy')


def typed_to_sympy(expr: TypedExpr, var_names: List[str]) -> sp.Expr:
    """Convert TypedExpr to sympy expression"""
    const_counter = [0]

    def convert(e: TypedExpr) -> sp.Expr:
        if e.op.op_name == 'const':
            v = float(e.value) if e.value is not None else 1.0
            i = const_counter[0]
            const_counter[0] += 1
            # Add small unique offset to keep constants separate
            v = v + (i + 1) * max(abs(v), 1.0) * 1e-12
            return sp.Float(v)

        if e.op.op_name == 'var':
            return sp.Symbol(e.var_name)

        kids = [convert(c) for c in e.children]
        a, b = (kids + [None, None])[:2]

        op = e.op.op_name
        if op == 'add':
            return a + b
        elif op == 'sub':
            return a - b
        elif op == 'mul':
            return a * b
        elif op == 'div':
            return a / b
        elif op == 'pow':
            return a ** b
        elif op == 'pow2':
            return a ** 2
        elif op == 'pow3':
            return a ** 3
        elif op == 'pow_neg1':
            return 1 / a
        elif op == 'pow_half':
            return sp.sqrt(a)
        elif op == 'neg':
            return -a
        elif op == 'sqrt':
            return sp.sqrt(a)
        elif op == 'sin':
            return sp.sin(a)
        elif op == 'cos':
            return sp.cos(a)
        elif op == 'exp':
            return sp.exp(a)
        elif op == 'log':
            return sp.log(a)
        elif op == 'abs':
            return sp.Abs(a)
        else:
            raise ValueError(f"unknown op {op}")

    return convert(expr)


def fit_constants(expr: TypedExpr,
                  variables: Dict[str, np.ndarray],
                  target_values: np.ndarray,
                  max_iter: int = 400,
                  ftol: float = 1e-14,
                  xtol: float = 1e-14) -> Tuple[TypedExpr, Dict[int, float]]:
    """
    Fit constants in expression using Levenberg-Marquardt.

    Returns: (optimized_expr, constant_values_dict)
    """
    const_list = extract_constants(expr)
    if not const_list:
        return expr, {}

    const_ids = [cid for cid, _ in const_list]
    x0 = np.array([val for _, val in const_list], dtype=float)
    var_names = list(variables.keys())

    # Create a version of the expression with symbolic constants
    sympy_expr = typed_to_sympy(expr, var_names)

    # Extract the constant symbols in order
    const_symbols = []
    for node in sp.preorder_traversal(sympy_expr):
        if isinstance(node, sp.Float) and not isinstance(node, sp.Integer):
            const_symbols.append(node)

    if len(const_symbols) != len(const_ids):
        # Fallback: return original
        return expr, dict(zip(const_ids, x0))

    # Create function with constants as first arguments
    all_args = list(const_symbols) + [sp.Symbol(n) for n in var_names]
    try:
        f = sp.lambdify(all_args, sympy_expr, modules='numpy')
    except Exception:
        return expr, dict(zip(const_ids, x0))

    def residuals(params):
        """Residual function for least_squares"""
        try:
            with np.errstate(all='ignore'):
                pred = f(*params, *[variables[n] for n in var_names])
            pred = np.asarray(pred, dtype=float)
            if pred.shape != target_values.shape:
                pred = np.broadcast_to(pred, target_values.shape).astype(float)
            pred = np.where(np.isfinite(pred), pred, 1e6)
            return pred - target_values
        except Exception:
            return np.full_like(target_values, 1e6)

    try:
        result = least_squares(residuals, x0, method='lm',
                               max_nfev=max_iter, ftol=ftol, xtol=xtol,
                               verbose=0)
        best_params = result.x if np.isfinite(result.cost) else x0
    except Exception:
        best_params = x0

    if not np.all(np.isfinite(best_params)):
        best_params = x0

    const_values = dict(zip(const_ids, [float(v) for v in best_params]))
    optimized_expr = set_constants(expr, const_values)

    return optimized_expr, const_values


def evaluate_with_constants(expr: TypedExpr,
                            variables: Dict[str, np.ndarray],
                            const_values: Dict[int, float]) -> np.ndarray:
    """Evaluate expression with specific constant values"""
    expr_with_consts = set_constants(expr, const_values)
    return expr_with_consts.evaluate(variables)