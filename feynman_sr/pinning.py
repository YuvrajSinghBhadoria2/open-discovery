"""
Hypothesis Pinning for Feynman Symbolic Regression

Adapted from Sudoku "Hypothesis Pinning Search":
- In Sudoku: pin a digit in a cell -> propagate constraints -> prune search
- In Feynman: pin a dimensional assignment for a subexpression -> propagate -> prune 99% of search space

Key idea: At each MCTS node, we can "pin" hypotheses about:
1. The dimensional type of each subexpression
2. Symmetry properties (even/odd, scaling)
3. Conservation laws (energy, momentum)
4. Known physical relationships (e.g., F=ma means force dim = mass * acceleration dim)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from enum import Enum
import numpy as np
from grammar import DimVec, TypedExpr, OpType, Dim, DEFAULT_OPS


class PinType(Enum):
    """Types of hypotheses we can pin"""
    DIMENSION = "dimension"           # This subexpr has dimension X
    SYMMETRY_EVEN = "symmetry_even"   # f(-x) = f(x)
    SYMMETRY_ODD = "symmetry_odd"     # f(-x) = -f(x)
    SCALING = "scaling"               # f(k*x) = k^a * f(x)
    CONSERVATION = "conservation"     # Expression conserves quantity
    MONOTONIC = "monotonic"           # Monotonic in variable
    BOUNDED = "bounded"               # Expression bounded in range
    ZERO_AT_ORIGIN = "zero_at_origin" # f(0) = 0
    PERIODIC = "periodic"             # Periodic in variable


@dataclass
class Pin:
    """A pinned hypothesis at a specific expression node"""
    pin_type: PinType
    node_path: Tuple[int, ...]  # Path from root: (0, 1, 0) = left child of right child of root
    value: Any  # The pinned value (DimVec, bool, float, etc.)
    confidence: float = 1.0  # How confident we are (1.0 = hard constraint)
    source: str = "manual"   # "manual", "deduced", "learned", "data-driven"

    def __hash__(self):
        return hash((self.pin_type, self.node_path, str(self.value)))


@dataclass
class PinningState:
    """Current state of all pinned hypotheses"""
    pins: Dict[Tuple[PinType, Tuple[int, ...]], Pin] = field(default_factory=dict)
    dimension_pins: Dict[Tuple[int, ...], DimVec] = field(default_factory=dict)

    def add_pin(self, pin: Pin) -> bool:
        """Add a pin, return False if conflicts with existing"""
        key = (pin.pin_type, pin.node_path)
        if key in self.pins:
            existing = self.pins[key]
            if existing.value != pin.value:
                return False  # Conflict
        self.pins[key] = pin
        if pin.pin_type == PinType.DIMENSION:
            self.dimension_pins[pin.node_path] = pin.value
        return True

    def get_pin(self, pin_type: PinType, node_path: Tuple[int, ...]) -> Optional[Pin]:
        return self.pins.get((pin_type, node_path))

    def get_dimension(self, node_path: Tuple[int, ...]) -> Optional[DimVec]:
        return self.dimension_pins.get(node_path)

    def copy(self) -> 'PinningState':
        new = PinningState()
        new.pins = self.pins.copy()
        new.dimension_pins = self.dimension_pins.copy()
        return new


class HypothesisPinner:
    """
    Core pinning engine: given partial expression + data, deduce and pin hypotheses
    """

    def __init__(self, grammar, data: Dict[str, np.ndarray] = None):
        self.grammar = grammar
        self.data = data or {}
        self.known_physical_laws = self._load_physical_laws()

    def _load_physical_laws(self) -> List[Dict]:
        """Known physical relationships to use as pinning priors"""
        return [
            {'name': 'F=ma', 'pattern': 'force = mass * acceleration',
             'dims': {'force': DimVec.parse('M L T^-2'), 'mass': DimVec.parse('M'), 'acceleration': DimVec.parse('L T^-2')}},
            {'name': 'E=mc^2', 'pattern': 'energy = mass * velocity^2',
             'dims': {'energy': DimVec.parse('M L^2 T^-2'), 'mass': DimVec.parse('M'), 'velocity': DimVec.parse('L T^-1')}},
            {'name': 'v=f*lambda', 'pattern': 'velocity = frequency * wavelength',
             'dims': {'velocity': DimVec.parse('L T^-1'), 'frequency': DimVec.parse('T^-1'), 'wavelength': DimVec.parse('L')}},
            {'name': 'PV=nRT', 'pattern': 'pressure * volume = energy',
             'dims': {'pressure': DimVec.parse('M L^-1 T^-2'), 'volume': DimVec.parse('L^3'), 'energy': DimVec.parse('M L^2 T^-2')}},
            {'name': 'E=F/q', 'pattern': 'electric_field = force / charge',
             'dims': {'electric_field': DimVec.parse('M L T^-2 Q^-1'), 'force': DimVec.parse('M L T^-2'), 'charge': DimVec.parse('Q')}},
        ]

    def deduce_from_data(self, expr: TypedExpr, target_values: np.ndarray,
                         variables: Dict[str, np.ndarray]) -> List[Pin]:
        """Deduce pins from numerical data analysis"""
        pins = []

        try:
            pred = expr.evaluate(variables)
            if not np.all(np.isfinite(pred)):
                return pins
        except:
            return pins

        for var_name, var_vals in variables.items():
            if var_name not in self._get_variables_in_expr(expr):
                continue

            if self._has_symmetric_range(var_vals):
                mask_pos = var_vals > 0
                mask_neg = var_vals < 0
                if np.sum(mask_pos) > 5 and np.sum(mask_neg) > 5:
                    pos_indices = np.where(mask_pos)[0]
                    neg_indices = np.where(mask_neg)[0]

                    pos_mags = np.abs(var_vals[pos_indices])
                    neg_mags = np.abs(var_vals[neg_indices])

                    if len(pos_mags) > 0 and len(neg_mags) > 0:
                        pred_pos = pred[pos_indices]
                        pred_neg = pred[neg_indices]

                        even_corr = self._symmetry_correlation(pred_pos, pred_neg, even=True)
                        odd_corr = self._symmetry_correlation(pred_pos, pred_neg, even=False)

                        if even_corr > 0.95:
                            pins.append(Pin(PinType.SYMMETRY_EVEN, (), True, confidence=even_corr, source='data'))
                        elif odd_corr > 0.95:
                            pins.append(Pin(PinType.SYMMETRY_ODD, (), True, confidence=odd_corr, source='data'))

        for var_name, var_vals in variables.items():
            if var_name not in self._get_variables_in_expr(expr):
                continue

            scale_exp = self._estimate_scaling_exponent(var_vals, pred)
            if scale_exp is not None:
                pins.append(Pin(PinType.SCALING, (), scale_exp, confidence=0.8, source='data'))

        zero_mask = np.all([np.abs(v) < 1e-6 for v in variables.values()], axis=0)
        if np.any(zero_mask) and np.all(np.abs(pred[zero_mask]) < 1e-6):
            pins.append(Pin(PinType.ZERO_AT_ORIGIN, (), True, confidence=0.9, source='data'))

        return pins

    def _get_variables_in_expr(self, expr: TypedExpr) -> Set[str]:
        vars_set = set()
        def collect(node):
            if node.op.op_name == 'var':
                vars_set.add(node.var_name)
            for child in node.children:
                collect(child)
        collect(expr)
        return vars_set

    def _has_symmetric_range(self, vals: np.ndarray) -> bool:
        return np.min(vals) < -0.1 and np.max(vals) > 0.1 and abs(np.min(vals) + np.max(vals)) < 0.1 * (np.max(vals) - np.min(vals))

    def _symmetry_correlation(self, pos_vals: np.ndarray, neg_vals: np.ndarray, even: bool) -> float:
        if len(pos_vals) == 0 or len(neg_vals) == 0:
            return 0.0

        if even:
            return 1.0 - abs(np.mean(pos_vals) - np.mean(neg_vals)) / (abs(np.mean(pos_vals)) + abs(np.mean(neg_vals)) + 1e-10)
        else:
            return 1.0 - abs(np.mean(pos_vals) + np.mean(neg_vals)) / (abs(np.mean(pos_vals)) + abs(np.mean(neg_vals)) + 1e-10)

    def _estimate_scaling_exponent(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 10:
            return None

        log_x = np.log(x[mask])
        log_y = np.log(y[mask])

        A = np.vstack([log_x, np.ones_like(log_x)]).T
        try:
            a, b = np.linalg.lstsq(A, log_y, rcond=None)[0]
            pred = a * log_x + b
            r2 = 1 - np.sum((log_y - pred)**2) / np.sum((log_y - np.mean(log_y))**2)
            if r2 > 0.8:
                return float(a)
        except:
            pass
        return None

    def deduce_from_structure(self, expr: TypedExpr, target_dim: DimVec) -> List[Pin]:
        """Deduce dimensional pins from expression structure + target dimension"""
        pins = []

        # Pin root dimension to target (ONLY the root - don't invent arbitrary factorizations)
        pins.append(Pin(PinType.DIMENSION, (), target_dim, confidence=1.0, source='target'))

        # Only propagate for unary ops where child dimension is uniquely determined
        # The grammar's type checking is sufficient to constrain valid combinations.
        # For binary ops (MUL, DIV), we do NOT propagate to children because that
        # would invent arbitrary factorizations (e.g., for F=ma, we'd pin left child
        # to M^3 L^-1 T^2 when the actual child is m = M).
        self._propagate_dimensions(expr, (), target_dim, pins)

        return pins

    def _propagate_dimensions(self, expr: TypedExpr, path: Tuple[int, ...],
                              expected_dim: DimVec, pins: List[Pin]):
        """Propagate dimensional constraints down the expression tree - only for unary ops"""
        if expr.op.arity == 0:
            return

        if expr.op.arity == 1:
            child = expr.children[0]
            child_dim = self._infer_child_dim_unary(expr.op, expected_dim)
            if child_dim:
                self._propagate_dimensions(child, path + (0,), child_dim, pins)
        # For binary ops, we do NOT propagate to children because
        # that would invent arbitrary factorizations.
        # The grammar's dimensional typing handles validity.

    def _infer_child_dim_unary(self, op: OpType, parent_dim: DimVec) -> Optional[DimVec]:
        """Infer child dimension from parent dimension for unary op"""
        if op.op_name in ('neg', 'abs'):
            return parent_dim
        elif op.op_name == 'sqrt':
            if all(v % 2 == 0 for v in parent_dim.vec):
                return DimVec(tuple(v * 2 for v in parent_dim.vec))
            return None
        elif op.op_name in ('sin', 'cos', 'exp', 'log'):
            if parent_dim.is_dimensionless():
                return DimVec.zero()
            return None
        elif op.op_name == 'pow2':
            if all(v % 2 == 0 for v in parent_dim.vec):
                return parent_dim / 2
            return None
        elif op.op_name == 'pow3':
            if all(v % 3 == 0 for v in parent_dim.vec):
                return DimVec(tuple(v // 3 for v in parent_dim.vec))
            return None
        elif op.op_name == 'pow_neg1':
            return -parent_dim
        elif op.op_name == 'pow_half':
            return parent_dim * 2
        return parent_dim

    def _infer_child_dims_binary(self, op: OpType, parent_dim: DimVec) -> List[Tuple[DimVec, DimVec]]:
        """NOT USED - we don't propagate to binary children"""
        return []

    def combine_pins(self, pins1: List[Pin], pins2: List[Pin]) -> PinningState:
        """Combine two pin lists, detecting conflicts"""
        state = PinningState()
        for pin in pins1 + pins2:
            if not state.add_pin(pin):
                existing = state.get_pin(pin.pin_type, pin.node_path)
                if existing and existing.confidence < pin.confidence:
                    state.pins[(pin.pin_type, pin.node_path)] = pin
        return state


def create_pinner_for_equation(equation_data: Dict, grammar) -> HypothesisPinner:
    """Factory to create pinner with equation-specific knowledge"""
    pinner = HypothesisPinner(grammar, equation_data.get('data'))

    if 'known_relations' in equation_data:
        for rel in equation_data['known_relations']:
            pass

    return pinner


def pin_from_target_dimension(target_dim: DimVec) -> List[Pin]:
    """Create initial pins from just the target dimension"""
    return [Pin(PinType.DIMENSION, (), target_dim, confidence=1.0, source='target')]