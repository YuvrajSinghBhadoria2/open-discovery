"""
Typed Expression Grammar for Feynman Symbolic Regression
Supports dimensional types: [M]ass, [L]ength, [T]ime, [Q]harge, [Θ]emperature
Optimized for speed - no O(N^2) loops
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np


class Dim(Enum):
    """Base physical dimensions"""
    M = 0  # Mass
    L = 1  # Length
    T = 2  # Time
    Q = 3  # Charge
    THETA = 4  # Temperature

    @classmethod
    def from_str(cls, s: str) -> 'Dim':
        s_upper = s.upper()
        mapping = {
            'M': cls.M, 'MASS': cls.M,
            'L': cls.L, 'LENGTH': cls.L,
            'T': cls.T, 'TIME': cls.T,
            'Q': cls.Q, 'CHARGE': cls.Q,
            # FIX: THETA is its own dimension, not T (time)
            'THETA': cls.THETA, 'Θ': cls.THETA, 'TEMPERATURE': cls.THETA, 'TH': cls.THETA,
            # Keep single-letter T for time
            'T': cls.T, 'TIME': cls.T,
        }
        # Filter out THETA-like names that might be confused
        if s_upper == 'T' or s_upper == 'TIME':
            # Check if it's really time or temperature
            # Single 'T' is time, THETA is temperature
            pass
        if s_upper not in mapping:
            raise ValueError(f"Unknown dimension: {s}")
        return mapping[s_upper]

    def __str__(self):
        return self.name


@dataclass(frozen=True, eq=True)
class DimVec:
    """Dimensional vector: [M, L, T, Q, Θ] exponents"""
    vec: Tuple[int, int, int, int, int]

    def __post_init__(self):
        if len(self.vec) != 5:
            raise ValueError(f"DimVec must have 5 components, got {len(self.vec)}")

    @classmethod
    def zero(cls) -> 'DimVec':
        return cls((0, 0, 0, 0, 0))

    @classmethod
    def from_dim(cls, dim: Dim, exp: int = 1) -> 'DimVec':
        v = [0, 0, 0, 0, 0]
        v[dim.value] = exp
        return cls(tuple(v))

    @classmethod
    def parse(cls, s: str) -> 'DimVec':
        """Parse string like 'M L^-1 T^-2' or '[M][L]^-1[T]^-2' or 'M L T^-2 THETA^-1'"""
        import re
        s = s.replace('[', '').replace(']', '').strip()
        if not s or s == '1':
            return cls.zero()

        vec = [0, 0, 0, 0, 0]
        # Match full dimension names (THETA) or single letters with optional exponents
        for match in re.finditer(r'(THETA|[MLTQΘ])(?:\^(-?\d+))?', s):
            dim_str = match.group(1)
            exp = int(match.group(2)) if match.group(2) else 1
            dim = Dim.from_str(dim_str)
            vec[dim.value] = exp
        return cls(tuple(vec))

    def __add__(self, other: 'DimVec') -> 'DimVec':
        return DimVec(tuple(a + b for a, b in zip(self.vec, other.vec)))

    def __sub__(self, other: 'DimVec') -> 'DimVec':
        return DimVec(tuple(a - b for a, b in zip(self.vec, other.vec)))

    def __mul__(self, scalar: int) -> 'DimVec':
        return DimVec(tuple(a * scalar for a in self.vec))

    def __rmul__(self, scalar: int) -> 'DimVec':
        return self * scalar

    def __truediv__(self, scalar: int) -> 'DimVec':
        if scalar == 0:
            raise ZeroDivisionError
        if not all(a % scalar == 0 for a in self.vec):
            raise ValueError(f"DimVec {self} not divisible by {scalar}")
        return DimVec(tuple(a // scalar for a in self.vec))

    def __neg__(self) -> 'DimVec':
        return DimVec(tuple(-a for a in self.vec))

    def is_dimensionless(self) -> bool:
        return all(v == 0 for v in self.vec)

    def __str__(self) -> str:
        parts = []
        for i, exp in enumerate(self.vec):
            if exp != 0:
                dim = Dim(i)
                if exp == 1:
                    parts.append(str(dim))
                else:
                    parts.append(f"{dim}^{exp}")
        return ' '.join(parts) if parts else '1'

    def __repr__(self):
        return f"DimVec({self})"

    def __deepcopy__(self, memo):
        return self  # Immutable, safe to reuse


DIMENSIONLESS = DimVec.zero()


# Named dimension rule functions
def _dim_rule_none(*args) -> None:
    return None

def _dim_rule_add_sub(a: DimVec, b: DimVec) -> Optional[DimVec]:
    return a if a == b else None

def _dim_rule_mul(a: DimVec, b: DimVec) -> DimVec:
    return a + b

def _dim_rule_div(a: DimVec, b: DimVec) -> DimVec:
    return a - b

def _dim_rule_pow(a: DimVec, b: DimVec) -> Optional[DimVec]:
    return a + a if b.is_dimensionless() else None

def _dim_rule_pow2(a: DimVec) -> DimVec:
    return a + a

def _dim_rule_pow3(a: DimVec) -> DimVec:
    return a + a + a

def _dim_rule_pow_neg1(a: DimVec) -> DimVec:
    return -a

def _dim_rule_pow_half(a: DimVec) -> Optional[DimVec]:
    return a / 2 if all(v % 2 == 0 for v in a.vec) else None

def _dim_rule_neg(a: DimVec) -> DimVec:
    return a

def _dim_rule_sqrt(a: DimVec) -> Optional[DimVec]:
    return a / 2 if all(v % 2 == 0 for v in a.vec) else None

def _dim_rule_trig(a: DimVec) -> Optional[DimVec]:
    return a if a.is_dimensionless() else None

def _dim_rule_abs(a: DimVec) -> DimVec:
    return a


class OpType(Enum):
    """Operator types with their dimensional signatures"""
    CONST = ("const", 0, _dim_rule_none)
    VAR = ("var", 0, _dim_rule_none)
    ADD = ("add", 2, _dim_rule_add_sub)
    SUB = ("sub", 2, _dim_rule_add_sub)
    MUL = ("mul", 2, _dim_rule_mul)
    DIV = ("div", 2, _dim_rule_div)
    POW = ("pow", 2, _dim_rule_pow)
    POW2 = ("pow2", 1, _dim_rule_pow2)
    POW3 = ("pow3", 1, _dim_rule_pow3)
    POW_NEG1 = ("pow_neg1", 1, _dim_rule_pow_neg1)
    POW_HALF = ("pow_half", 1, _dim_rule_pow_half)
    NEG = ("neg", 1, _dim_rule_neg)
    SQRT = ("sqrt", 1, _dim_rule_sqrt)
    SIN = ("sin", 1, _dim_rule_trig)
    COS = ("cos", 1, _dim_rule_trig)
    EXP = ("exp", 1, _dim_rule_trig)
    LOG = ("log", 1, _dim_rule_trig)
    ABS = ("abs", 1, _dim_rule_abs)

    def __init__(self, name: str, arity: int, dim_rule):
        self.op_name = name
        self.arity = arity
        self.dim_rule = dim_rule

    @classmethod
    def from_name(cls, name: str) -> 'OpType':
        for op in cls:
            if op.op_name == name:
                return op
        raise ValueError(f"Unknown operator: {name}")


DEFAULT_OPS = [
    OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.POW,
    OpType.NEG, OpType.SQRT, OpType.SIN, OpType.COS, OpType.EXP, OpType.LOG, OpType.ABS,
]

FEYNMAN_OPS = [
    OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.POW,
    OpType.POW2, OpType.POW3, OpType.POW_NEG1, OpType.POW_HALF,
    OpType.NEG, OpType.SQRT,
    OpType.SIN, OpType.COS, OpType.EXP, OpType.LOG, OpType.ABS,
]

OP_SYMBOLS = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '^',
    'pow2': '^2', 'pow3': '^3', 'pow_neg1': '^-1', 'pow_half': '^0.5',
    'neg': '-', 'sqrt': '√', 'sin': 'sin', 'cos': 'cos',
    'exp': 'exp', 'log': 'log', 'abs': '|·|',
}


@dataclass
class TypedExpr:
    """Typed expression node in the grammar"""
    op: OpType
    children: Tuple['TypedExpr', ...]
    dim: DimVec
    value: Optional[float] = None
    var_name: Optional[str] = None
    const_id: Optional[int] = None

    def __post_init__(self):
        if self.op.arity != len(self.children):
            raise ValueError(f"Operator {self.op.op_name} expects {self.op.arity} children, got {len(self.children)}")

    @classmethod
    def constant(cls, value: float, dim: DimVec = DIMENSIONLESS, const_id: int = None) -> 'TypedExpr':
        return cls(OpType.from_name('const'), (), dim, value=value, const_id=const_id)

    @classmethod
    def variable(cls, name: str, dim: DimVec) -> 'TypedExpr':
        return cls(OpType.from_name('var'), (), dim, var_name=name)

    @classmethod
    def binary(cls, op_name: str, left: 'TypedExpr', right: 'TypedExpr') -> Optional['TypedExpr']:
        op = OpType.from_name(op_name)
        dim = op.dim_rule(left.dim, right.dim)
        if dim is None:
            return None
        return cls(op, (left, right), dim)

    @classmethod
    def unary(cls, op_name: str, child: 'TypedExpr') -> Optional['TypedExpr']:
        op = OpType.from_name(op_name)
        dim = op.dim_rule(child.dim)
        if dim is None:
            return None
        return cls(op, (child,), dim)

    def __str__(self) -> str:
        if self.op.op_name == 'const':
            return str(self.value)
        elif self.op.op_name == 'var':
            return self.var_name or 'x'
        elif self.op.arity == 1:
            sym = OP_SYMBOLS.get(self.op.op_name, self.op.op_name)
            child_str = str(self.children[0])
            if self.op.op_name in ('pow2', 'pow3', 'pow_neg1', 'pow_half'):
                return f"{child_str}{sym}"
            elif self.op.op_name in ('sqrt', 'sin', 'cos', 'exp', 'log', 'abs'):
                return f"{sym}({child_str})"
            else:
                return f"{sym}{child_str}"
        else:
            sym = OP_SYMBOLS.get(self.op.op_name, self.op.op_name)
            left_str = str(self.children[0])
            right_str = str(self.children[1])
            return f"({left_str} {sym} {right_str})"

    def __repr__(self):
        return f"TypedExpr({self}, dim={self.dim})"

    def __deepcopy__(self, memo):
        return self  # Immutable dataclass with tuple children, safe to reuse

    def get_constants(self) -> List['TypedExpr']:
        if self.op.op_name == 'const':
            return [self]
        consts = []
        for child in self.children:
            consts.extend(child.get_constants())
        return consts

    def set_constants(self, values: List[float]) -> 'TypedExpr':
        idx = [0]
        def replace(node: 'TypedExpr') -> 'TypedExpr':
            if node.op.op_name == 'const':
                val = values[idx[0]]
                idx[0] += 1
                return TypedExpr.constant(val, node.dim, node.const_id)
            elif node.op.op_name == 'var':
                return node
            else:
                new_children = tuple(replace(c) for c in node.children)
                return TypedExpr(node.op, new_children, node.dim)
        return replace(self)

    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        if self.op.op_name == 'const':
            return np.full_like(next(iter(variables.values())), self.value)
        elif self.op.op_name == 'var':
            return variables[self.var_name]

        child_vals = [c.evaluate(variables) for c in self.children]

        if self.op.op_name == 'add':
            return child_vals[0] + child_vals[1]
        elif self.op.op_name == 'sub':
            return child_vals[0] - child_vals[1]
        elif self.op.op_name == 'mul':
            return child_vals[0] * child_vals[1]
        elif self.op.op_name == 'div':
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = child_vals[1]
                safe_denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + (denom == 0) * 1e-12, denom)
                result = np.where(np.abs(denom) >= 1e-12, child_vals[0] / safe_denom, np.inf)
            return result
        elif self.op.op_name == 'pow':
            with np.errstate(over='ignore', invalid='ignore'):
                result = np.power(child_vals[0], child_vals[1])
            return result
        elif self.op.op_name == 'pow2':
            return child_vals[0] * child_vals[0]
        elif self.op.op_name == 'pow3':
            return child_vals[0] * child_vals[0] * child_vals[0]
        elif self.op.op_name == 'pow_neg1':
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = child_vals[0]
                safe_denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + (denom == 0) * 1e-12, denom)
                return np.where(np.abs(denom) >= 1e-12, 1.0 / safe_denom, np.inf)
        elif self.op.op_name == 'pow_half':
            with np.errstate(invalid='ignore'):
                return np.where(child_vals[0] >= 0, np.sqrt(child_vals[0]), np.nan)
        elif self.op.op_name == 'neg':
            return -child_vals[0]
        elif self.op.op_name == 'sqrt':
            with np.errstate(invalid='ignore'):
                return np.where(child_vals[0] >= 0, np.sqrt(child_vals[0]), np.nan)
        elif self.op.op_name == 'sin':
            return np.sin(child_vals[0])
        elif self.op.op_name == 'cos':
            return np.cos(child_vals[0])
        elif self.op.op_name == 'exp':
            with np.errstate(over='ignore'):
                return np.exp(child_vals[0])
        elif self.op.op_name == 'log':
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(child_vals[0] > 0, np.log(child_vals[0]), np.nan)
        elif self.op.op_name == 'abs':
            return np.abs(child_vals[0])

        raise ValueError(f"Unknown operator: {self.op.op_name}")

    def count_nodes(self) -> int:
        if self.op.arity == 0:
            return 1
        return 1 + sum(c.count_nodes() for c in self.children)

    def depth(self) -> int:
        if self.op.arity == 0:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def complexity(self) -> int:
        """Alias for count_nodes for compatibility"""
        return self.count_nodes()


class Grammar:
    """Typed grammar for generating dimensionally valid expressions - FAST VERSION"""

    def __init__(self,
                 variables: Dict[str, DimVec],
                 target_dim: DimVec,
                 operators: List[OpType] = None,
                 max_depth: int = 10,
                 allow_dimensionless_constants: bool = True):

        self.variables = variables
        self.target_dim = target_dim
        self.operators = operators or DEFAULT_OPS
        self.max_depth = max_depth
        self.allow_dimensionless_constants = allow_dimensionless_constants

        self.unary_ops = [op for op in self.operators if op.arity == 1]
        self.binary_ops = [op for op in self.operators if op.arity == 2]

        self._build()

    def _build(self):
        all_dims = self._get_all_dims_fast()

        self.valid_unary: Dict[DimVec, List[OpType]] = {}
        for d in all_dims:
            valid_ops = []
            for op in self.unary_ops:
                result = self._unary_result(op, d)
                if result is not None:
                    valid_ops.append(op)
            if valid_ops:
                self.valid_unary[d] = valid_ops
        
        # Add dimensionless constants to the grammar
        self.constants = [TypedExpr.constant(value, DIMENSIONLESS) for value in [0.5, 2, 2*np.pi]]

        self.binary_candidates_for_target: Dict[DimVec, List[Tuple[OpType, DimVec, DimVec]]] = {}
        key_targets = {self.target_dim}

        var_dims = list(self.variables.values())
        for i, d1 in enumerate(var_dims):
            for d2 in var_dims[i:]:
                for op in [OpType.MUL, OpType.DIV]:
                    result = self._binary_result(op, d1, d2)
                    if result and result in all_dims:
                        key_targets.add(result)
                    if op == OpType.DIV:
                        result = self._binary_result(op, d2, d1)
                        if result and result in all_dims:
                            key_targets.add(result)

        for d in var_dims:
            for op in [OpType.POW2, OpType.POW3, OpType.POW_NEG1, OpType.POW_HALF]:
                result = self._unary_result(op, d)
                if result and result in all_dims:
                    key_targets.add(result)

        current_targets = list(key_targets)
        for _ in range(4):
            new_targets = set(current_targets)
            for d1 in var_dims:
                for d2 in current_targets:
                    for op in [OpType.MUL, OpType.DIV]:
                        result = self._binary_result(op, d1, d2)
                        if result and result in all_dims:
                            new_targets.add(result)
                        if op == OpType.DIV:
                            result = self._binary_result(op, d2, d1)
                            if result and result in all_dims:
                                new_targets.add(result)
            for i, d1 in enumerate(current_targets):
                for d2 in current_targets[i:]:
                    for op in [OpType.MUL, OpType.DIV]:
                        result = self._binary_result(op, d1, d2)
                        if result and result in all_dims:
                            new_targets.add(result)
                        if op == OpType.DIV:
                            result = self._binary_result(op, d2, d1)
                            if result and result in all_dims:
                                new_targets.add(result)
            current_targets = list(new_targets)

        priority_targets = [self.target_dim]
        for op in self.unary_ops:
            child_dim = None
            if op == OpType.SQRT:
                child_dim = self.target_dim * 2
            elif op == OpType.POW2:
                if all(v % 2 == 0 for v in self.target_dim.vec):
                    child_dim = self.target_dim / 2
            elif op == OpType.POW3:
                if all(v % 3 == 0 for v in self.target_dim.vec):
                    child_dim = DimVec(tuple(v // 3 for v in self.target_dim.vec))
            elif op == OpType.POW_NEG1:
                child_dim = -self.target_dim
            elif op == OpType.POW_HALF:
                child_dim = self.target_dim * 2
            elif op == OpType.NEG:
                child_dim = None
            elif op in (OpType.SIN, OpType.COS, OpType.EXP, OpType.LOG, OpType.ABS):
                if self.target_dim.is_dimensionless():
                    child_dim = self.target_dim
            if child_dim is not None and child_dim in all_dims and child_dim != self.target_dim:
                priority_targets.append(child_dim)

        seen = set(priority_targets)
        final_targets = list(priority_targets)
        for t in current_targets:
            if t not in seen:
                seen.add(t)
                final_targets.append(t)

        key_targets = final_targets[:300]

        for target in key_targets:
            candidates = []
            for d1 in all_dims:
                d2_mul = target - d1
                if d2_mul in all_dims:
                    candidates.append((OpType.MUL, d1, d2_mul))

                d2_div = d1 - target
                if d2_div in all_dims:
                    candidates.append((OpType.DIV, d1, d2_div))

            if candidates:
                var_dims = set(self.variables.values())

                def is_simple_dim(d: DimVec) -> bool:
                    if d in var_dims:
                        return True
                    if d.is_dimensionless():
                        return True
                    for v in var_dims:
                        for op in [OpType.POW2, OpType.POW3, OpType.POW_NEG1, OpType.POW_HALF]:
                            if self._unary_result(op, v) == d:
                                return True
                    for v1 in var_dims:
                        for v2 in var_dims:
                            if self._binary_result(OpType.MUL, v1, v2) == d:
                                return True
                            if self._binary_result(OpType.DIV, v1, v2) == d:
                                return True
                    return False

                def complexity(c):
                    op, d1, d2 = c
                    nz1 = sum(1 for v in d1.vec if v != 0)
                    nz2 = sum(1 for v in d2.vec if v != 0)
                    base = nz1 + nz2
                    simple_bonus = -10 if (is_simple_dim(d1) and is_simple_dim(d2)) else 0
                    div_bonus = -1 if op == OpType.DIV else 0
                    return base + simple_bonus + div_bonus

                candidates.sort(key=complexity)
                self.binary_candidates_for_target[target] = candidates[:20]

        self.unary_candidates_for_target: Dict[DimVec, List[Tuple[OpType, DimVec]]] = {}
        for target in key_targets:
            candidates = []
            for op in self.unary_ops:
                child_dim = None
                if op == OpType.SQRT:
                    child_dim = target * 2
                elif op == OpType.POW2:
                    if all(v % 2 == 0 for v in target.vec):
                        child_dim = target / 2
                elif op == OpType.POW3:
                    if all(v % 3 == 0 for v in target.vec):
                        child_dim = DimVec(tuple(v // 3 for v in target.vec))
                elif op == OpType.POW_NEG1:
                    child_dim = -target
                elif op == OpType.POW_HALF:
                    child_dim = target * 2
                elif op == OpType.NEG:
                    child_dim = target
                elif op in (OpType.SIN, OpType.COS, OpType.EXP, OpType.LOG, OpType.ABS):
                    if target.is_dimensionless():
                        child_dim = target

                if child_dim is not None and child_dim in all_dims:
                    candidates.append((op, child_dim))

            if candidates:
                self.unary_candidates_for_target[target] = candidates

    def _get_all_dims_fast(self) -> Set[DimVec]:
        dims = set(self.variables.values())
        dims.add(DIMENSIONLESS)

        max_var_exp = 0
        for d in self.variables.values():
            for v in d.vec:
                max_var_exp = max(max_var_exp, abs(v))
        max_exp = min(max(4, max_var_exp * 2), 8)

        def bounded(d: DimVec) -> bool:
            return all(-max_exp <= v <= max_exp for v in d.vec)

        var_dims = list(self.variables.values())

        for d1 in var_dims:
            for op in self.unary_ops:
                result = self._unary_result(op, d1)
                if result and bounded(result):
                    dims.add(result)

        dims_list = list(dims)
        for d1 in dims_list:
            for d2 in dims_list:
                for op in self.binary_ops:
                    result = self._binary_result(op, d1, d2)
                    if result and bounded(result):
                        dims.add(result)

        dims_list = list(dims)
        for d1 in dims_list:
            for d2 in dims_list:
                for op in self.binary_ops:
                    result = self._binary_result(op, d1, d2)
                    if result and bounded(result):
                        dims.add(result)

        if len(dims) > 2000:
            sorted_dims = sorted(dims, key=lambda d: sum(abs(v) for v in d.vec))
            dims = set(sorted_dims[:2000])

        return dims

    def _unary_result(self, op: OpType, dim: DimVec) -> Optional[DimVec]:
        return op.dim_rule(dim)

    def _binary_result(self, op: OpType, dim1: DimVec, dim2: DimVec) -> Optional[DimVec]:
        return op.dim_rule(dim1, dim2)

    def can_apply_unary(self, op: OpType, dim: DimVec) -> bool:
        return self._unary_result(op, dim) is not None

    def can_apply_binary(self, op: OpType, dim1: DimVec, dim2: DimVec) -> bool:
        return self._binary_result(op, dim1, dim2) is not None

    def get_unary_result_dim(self, op: OpType, dim: DimVec) -> Optional[DimVec]:
        return self._unary_result(op, dim)

    def get_binary_result_dim(self, op: OpType, dim1: DimVec, dim2: DimVec) -> Optional[DimVec]:
        return self._binary_result(op, dim1, dim2)

    def generate_random_expr(self, target_dim: DimVec = None, depth: int = 0, rng: np.random.Generator = None, max_nodes: int = 20, _call_count: list = None) -> Optional[TypedExpr]:
        # Hard limit on recursive calls to prevent infinite loops
        if _call_count is None:
            _call_count = [0]
        _call_count[0] += 1
        if _call_count[0] > 1000:
            return None

        if rng is None:
            rng = np.random.default_rng()

        target = target_dim or self.target_dim

        # Get variables matching target dimension exactly
        matching_vars = [v for v, d in self.variables.items() if d == target]
        var_dims = set(self.variables.values())

        # SPECIAL CASE: For dimensionless, strongly prefer simple variables/constants
        # to avoid deeply nested expressions like x0/(x0/(x0/x0))
        if target.is_dimensionless():
            if matching_vars and rng.random() < 0.6:
                return TypedExpr.variable(rng.choice(matching_vars), target)
            if self.allow_dimensionless_constants and rng.random() < 0.3:
                constants = [0.5, 2, 2*np.pi]
                return TypedExpr.constant(rng.choice(constants), target)
            # Try unary operations on dimensionless variables (sin, cos, exp, log, sqrt)
            # This is crucial for physics equations with trig functions
            if depth < 3 and self.unary_candidates_for_target.get(target):
                candidates = self.unary_candidates_for_target[target]
                rng.shuffle(candidates)
                for op, child_dim in candidates[:5]:
                    # Allow child_dim == target for trig/exp/log on dimensionless
                    if child_dim == target and op.op_name in ('sin', 'cos', 'exp', 'log', 'abs', 'sqrt'):
                        child = self.generate_random_expr(child_dim, depth + 1, rng, max_nodes, _call_count)
                        if child is not None and child.op.op_name == 'var':
                            expr = TypedExpr.unary(op.op_name, child)
                            if expr is not None and expr.count_nodes() <= max_nodes:
                                return expr
            # Only rarely try complex dimensionless expressions
            if rng.random() < 0.1 and depth < 2:
                pass  # Fall through to binary/unary below
            else:
                return None

        # At max depth, return variable or constant
        if depth >= self.max_depth:
            if matching_vars and rng.random() < 0.7:
                return TypedExpr.variable(rng.choice(matching_vars), target)
            if target.is_dimensionless() and self.allow_dimensionless_constants:
                return TypedExpr.constant(rng.uniform(-5, 5), target)
            return None

        # PRIORITY 1: Try binary operations that combine variables
        # This is crucial for physics: F = m*a, E = mc^2, etc.
        if target in self.binary_candidates_for_target:
            candidates = self.binary_candidates_for_target[target]
            rng.shuffle(candidates[:20])
            for op, d1, d2 in candidates[:20]:
                d1_is_base = d1 in var_dims or d1.is_dimensionless()
                d2_is_base = d2 in var_dims or d2.is_dimensionless()

                # Case A: Both are base variable dimensions - recurse normally
                if d1_is_base and d2_is_base and d1 != target and d2 != target:
                    if rng.random() < 0.9:
                        left = self.generate_random_expr(d1, depth + 1, rng, max_nodes, _call_count)
                        right = self.generate_random_expr(d2, depth + 1, rng, max_nodes, _call_count)
                        if left is not None and right is not None:
                            expr = TypedExpr.binary(op.op_name, left, right)
                            if expr is not None and expr.count_nodes() <= max_nodes:
                                return expr

                # Case B: One dimension equals target, other is dimensionless (e.g., x0 * x1 where x1 has target dim)
                # Use the matching variable directly for target dim, recurse for dimensionless
                elif (d1 == target and d2.is_dimensionless()) or (d2 == target and d1.is_dimensionless()):
                    if rng.random() < 0.9:
                        if d1 == target and matching_vars:
                            left = TypedExpr.variable(rng.choice(matching_vars), target)
                            right = self.generate_random_expr(d2, depth + 1, rng, max_nodes, _call_count)
                        elif d2 == target and matching_vars:
                            left = self.generate_random_expr(d1, depth + 1, rng, max_nodes, _call_count)
                            right = TypedExpr.variable(rng.choice(matching_vars), target)
                        else:
                            continue
                        if left is not None and right is not None:
                            expr = TypedExpr.binary(op.op_name, left, right)
                            if expr is not None and expr.count_nodes() <= max_nodes:
                                return expr

        # PRIORITY 2: Try unary candidates (sin, cos, sqrt, exp, log, neg)
        if target in self.unary_candidates_for_target:
            candidates = self.unary_candidates_for_target[target]
            for op, child_dim in candidates:
                # Avoid infinite recursion: child_dim should not equal target for most ops
                if child_dim == target and op.op_name not in ('neg', 'abs'):
                    continue
                child = self.generate_random_expr(child_dim, depth + 1, rng, max_nodes, _call_count)
                if child is not None:
                    expr = TypedExpr.unary(op.op_name, child)
                    if expr is not None and expr.count_nodes() <= max_nodes:
                        return expr

        # PRIORITY 3: Variable or constant fallback
        if matching_vars and rng.random() < 0.4:
            return TypedExpr.variable(rng.choice(matching_vars), target)
        if target.is_dimensionless() and self.allow_dimensionless_constants:
            if rng.random() < 0.3:
                constants = [0.5, 2, 2*np.pi]
                return TypedExpr.constant(rng.choice(constants), target)

        # PRIORITY 4: Try binary candidates where neither dim equals target
        if target in self.binary_candidates_for_target:
            candidates = self.binary_candidates_for_target[target]
            filtered = [(op, d1, d2) for op, d1, d2 in candidates[:10]
                       if d1 != target and d2 != target]
            for op, d1, d2 in filtered:
                left = self.generate_random_expr(d1, depth + 1, rng, max_nodes, _call_count)
                right = self.generate_random_expr(d2, depth + 1, rng, max_nodes, _call_count)
                if left is not None and right is not None:
                    expr = TypedExpr.binary(op.op_name, left, right)
                    if expr is not None and expr.count_nodes() <= max_nodes:
                        return expr

        return None
    def validate_expr(self, expr: TypedExpr) -> bool:
        if expr.op.op_name in ('const', 'var'):
            return True
        for child in expr.children:
            if not self.validate_expr(child):
                return False
        if expr.op.arity == 1:
            expected = self.get_unary_result_dim(expr.op, expr.children[0].dim)
        else:
            expected = self.get_binary_result_dim(expr.op, expr.children[0].dim, expr.children[1].dim)
        return expected == expr.dim

    def get_all_dims(self, max_exp: int = 3) -> Set[DimVec]:
        return self._get_all_dims_fast()

    def generate_dimension(self, target_dim: DimVec, max_depth: int = None, max_nodes: int = 20, n_samples: int = 10) -> List[TypedExpr]:
        """Generate multiple expressions of the given target dimension"""
        results = []
        rng = np.random.default_rng()
        depth_limit = max_depth or self.max_depth

        for _ in range(n_samples):
            expr = self.generate_random_expr(target_dim, depth=0, rng=rng, max_nodes=max_nodes)
            if expr is not None:
                results.append(expr)

        return results

    def check_types(self, expr: TypedExpr) -> bool:
        """Check if expression has valid dimensional types"""
        return self.validate_expr(expr)