#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
from fractions import Fraction

import numpy as np


# Private exact-arithmetic utilities. The cache contains oracle-side data only.
def _nr_integer(value, name, low=0, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _nr_indices(values, n, name, sorted_required=True, nonempty=False):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list of indices")
    result = tuple(_nr_integer(x, name, 0, n - 1) for x in values)
    if len(set(result)) != len(result) or (nonempty and not result):
        raise ValueError(f"{name} must contain distinct indices")
    if sorted_required and result != tuple(sorted(result)):
        raise ValueError(f"{name} must be increasing")
    return result


def _nr_real_array(value, name, ndim):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or raw.ndim != ndim or raw.size == 0:
            raise ValueError(f"{name} must be a nonempty real array of dimension {ndim}")
        with np.errstate(over='raise', invalid='raise'):
            array = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} cannot be represented as finite binary64") from exc
    return array


def _nr_bareiss_solve(gram, rhs):
    # Fraction-free elimination for an integer positive-definite Gram matrix.
    n = len(rhs)
    a = [[int(x) for x in row] + [int(rhs[i])] for i, row in enumerate(gram)]
    previous = 1
    for j in range(n - 1):
        pivot = a[j][j]
        if pivot == 0:
            raise ValueError("the matrix or selected column system is singular")
        for i in range(j + 1, n):
            entry = a[i][j]
            for k in range(j + 1, n + 1):
                a[i][k] = (pivot * a[i][k] - entry * a[j][k]) // previous
            a[i][j] = 0
        previous = pivot
    if a[-1][-2] == 0:
        raise ValueError("the matrix or selected column system is singular")
    result = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        result[i] = (Fraction(a[i][-1]) - sum(
            (a[i][j] * result[j] for j in range(i + 1, n)), Fraction(0)
        )) / a[i][i]
    return tuple(result)


def _nr_matrix(C):
    array = _nr_real_array(C, 'C', 2)
    if array.shape[0] != array.shape[1] or np.any(np.diag(array) == 0):
        raise ValueError("C must be square with a nonzero diagonal")
    key = (array.shape, array.tobytes())
    cache = getattr(_nr_matrix, '_cache', None)
    if cache is None:
        cache = {}
        _nr_matrix._cache = cache
    if key in cache:
        return cache[key]
    n = len(array)
    values = [Fraction(float(x)) for x in array.flat]
    den = math.lcm(*(v.denominator for v in values))
    X = np.array([int(v * den) for v in values], dtype=object).reshape(n, n)
    gram = X.T @ X
    _nr_bareiss_solve(gram.tolist(), [0] * n)  # Exact nonsingularity check.
    data = {
        'n': n, 'X': X, 'den': den, 'gram': gram,
        'rows': tuple(tuple(int(j) for j in np.flatnonzero(X[i])) for i in range(n)),
        'cols': tuple(tuple(int(i) for i in np.flatnonzero(X[:, j])) for j in range(n)),
        'fits': {},
    }
    cache[key] = data
    return data


def _nr_pair(value):
    value = Fraction(value)
    return int(value.numerator), int(value.denominator)


def _nr_fraction(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("an exact rational must be a (numerator, denominator) pair")
    p, q = value
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           for x in (p, q)):
        raise ValueError("rational components must be integers")
    p, q = int(p), int(q)
    if q <= 0 or math.gcd(abs(p), q) != 1:
        raise ValueError("rational pairs must be reduced with a positive denominator")
    return Fraction(p, q)


def initial_pattern(C: np.ndarray, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = _nr_matrix(C)
    n, X = data['n'], data['X']
    k = _nr_integer(k, 'k', 0, n - 1)
    # The common positive denominator cannot change numerical nonzero tests.
    power_column = X @ X[:, k]
    J = tuple(j for j in range(n) if j == k or X[j, k] != 0 or power_column[j] != 0)
    I = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    return I, J

from fractions import Fraction

import numpy as np


def fit_column(
    C: np.ndarray,
    k: int,
    J: tuple[int, ...],
) -> tuple[tuple, tuple, tuple[int, int]]:
    data = _nr_matrix(C)
    n, X, den, gram = data['n'], data['X'], data['den'], data['gram']
    k = _nr_integer(k, 'k', 0, n - 1)
    J = _nr_indices(J, n, 'J', nonempty=True)
    if k not in J:
        raise ValueError("J must contain k")
    key = (k, J)
    if key in data['fits']:
        return data['fits'][key]
    I = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    local_gram = [[gram[a, b] for b in J] for a in J]
    local_rhs = [den * X[k, j] for j in J]
    v = _nr_bareiss_solve(local_gram, local_rhs)
    r = [Fraction(0)] * n
    for i in I:
        r[i] = sum((Fraction(int(X[i, j]), den) * z
                    for j, z in zip(J, v) if X[i, j] != 0), Fraction(0)) - int(i == k)
    beta = sum((z * z for z in r), Fraction(0))
    result = (tuple(_nr_pair(z) for z in v), tuple(_nr_pair(z) for z in r), _nr_pair(beta))
    data['fits'][key] = result
    return result

from fractions import Fraction

import numpy as np


def _nr_interval(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("intervals need two rational endpoints and two closure flags")
    lo, hi = _nr_fraction(value[0]), _nr_fraction(value[1])
    lc, hc = (_nr_integer(value[i], 'closure flag', 0, 1) for i in (2, 3))
    if lo > hi or (lo == hi and not (lc and hc)):
        raise ValueError("interval is empty")
    return lo, hi, lc, hc


def _nr_box(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("box must have eta and xi intervals")
    eta, xi = _nr_interval(value[0]), _nr_interval(value[1])
    if eta[0] < 0 or xi[0] <= 0 or xi[1] > 1:
        raise ValueError("box requires eta >= 0 and 0 < xi <= 1")
    return eta, xi


def _nr_encode_box(box):
    return tuple((_nr_pair(b[0]), _nr_pair(b[1]), int(b[2]), int(b[3])) for b in box)


def _nr_intersect(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    lc = (a[2] if lo == a[0] else 1) and (b[2] if lo == b[0] else 1)
    hc = (a[3] if hi == a[1] else 1) and (b[3] if hi == b[1] else 1)
    if lo > hi or (lo == hi and not (lc and hc)):
        return None
    return lo, hi, int(lc), int(hc)


def _nr_box_intersect(a, b):
    eta = _nr_intersect(a[0], b[0])
    if eta is None:
        return None
    xi = _nr_intersect(a[1], b[1])
    return None if xi is None else (eta, xi)


def _nr_box_key(box):
    return tuple(x for interval in box for x in interval)


def partition_decision(
    residual: tuple,
    I: tuple[int, ...],
    used: tuple[int, ...],
    box: tuple,
    c: int = 2,
) -> tuple:
    if not isinstance(residual, (tuple, list)) or not residual:
        raise ValueError("residual must be a nonempty tuple or list")
    r = tuple(_nr_fraction(v) for v in residual)
    I = _nr_indices(I, len(r), 'I', nonempty=True)
    used = _nr_indices(used, len(r), 'used')
    if not set(used).issubset(I) or any(r[i] != 0 for i in range(len(r)) if i not in I):
        raise ValueError("history or residual is inconsistent with I")
    c = _nr_integer(c, 'c', 1)
    eta, xi = _nr_box(box)
    beta = sum((v * v for v in r), Fraction(0))
    result = []
    stop = _nr_intersect(eta, (beta, max(beta, eta[1]), 1, 1))
    if stop is not None:
        result.append((_nr_encode_box((stop, xi)), ()))
    continuing = _nr_intersect(eta, (min(eta[0], beta), beta, 1, 0))
    if continuing is None:
        return tuple(result)
    candidates = sorted((i for i in I if i not in used), key=lambda i: (-r[i] * r[i], i))[:c]
    if not candidates:
        result.append((_nr_encode_box((continuing, xi)), ()))
        return tuple(result)
    ratios = [r[i] * r[i] / beta for i in candidates]
    no_row = _nr_intersect(xi, (ratios[0], max(xi[1], ratios[0]), 0, 1))
    if no_row is not None:
        result.append((_nr_encode_box((continuing, no_row)), ()))
    for count in range(1, len(candidates) + 1):
        low = ratios[count] if count < len(candidates) else Fraction(0)
        region = _nr_intersect(xi, (low, ratios[count - 1], int(count == len(candidates)), 1))
        if region is not None:
            result.append((_nr_encode_box((continuing, region)), tuple(candidates[:count])))
    return tuple(result)

import numpy as np


def expand_pattern(
    C: np.ndarray,
    I: tuple[int, ...],
    J: tuple[int, ...],
    used: tuple[int, ...],
    selected: tuple[int, ...],
) -> tuple:
    data = _nr_matrix(C)
    n = data['n']
    I = _nr_indices(I, n, 'I', nonempty=True)
    J = _nr_indices(J, n, 'J', nonempty=True)
    used = _nr_indices(used, n, 'used')
    selected = _nr_indices(selected, n, 'selected', sorted_required=False)
    closure = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    if I != closure or not set(used).issubset(I) or not set(selected).issubset(set(I) - set(used)):
        raise ValueError("inconsistent row closure, history, or selection")
    J_new = tuple(sorted(set(J).union(*(data['rows'][i] for i in selected))))
    I_new = tuple(sorted(set().union(*(data['cols'][j] for j in J_new))))
    used_new = tuple(sorted(set(used) | set(selected)))
    return I_new, J_new, used_new

import numpy as np


def column_regions(C: np.ndarray, k: int, box: tuple, c: int = 2) -> tuple:
    domain = _nr_box(box)
    c = _nr_integer(c, 'c', 1)
    I, J = initial_pattern(C, k)
    pending = [(I, J, (), _nr_encode_box(domain))]
    leaves = []
    while pending:
        I, J, used, current_box = pending.pop()
        _, residual, _ = fit_column(C, k, J)
        decisions = partition_decision(residual, I, used, current_box, c)
        for subbox, selected in decisions:
            if not selected:
                leaves.append((subbox, J))
            else:
                I_new, J_new, used_new = expand_pattern(C, I, J, used, selected)
                pending.append((I_new, J_new, used_new, subbox))
    leaves.sort(key=lambda item: (_nr_box_key(_nr_box(item[0])), item[1]))
    return tuple(leaves)

import numpy as np


def joint_patterns(partitions: tuple, budget: int) -> tuple:
    budget = _nr_integer(budget, 'budget', 0)
    if not isinstance(partitions, (tuple, list)) or not partitions or len(partitions) % 2:
        raise ValueError("partitions must contain a positive even number of columns")
    n = len(partitions) // 2
    decoded = []
    for part in partitions:
        if not isinstance(part, (tuple, list)) or not part:
            raise ValueError("each column partition must be nonempty")
        entries = []
        for entry in part:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("each region needs a box and a support")
            bb, J = _nr_box(entry[0]), _nr_indices(entry[1], n, 'J', nonempty=True)
            if any(_nr_box_intersect(bb, old[0]) is not None for old in entries):
                raise ValueError("boxes within a column partition must be disjoint")
            entries.append((bb, J))
        decoded.append(entries)
    # This lower bound uses supports of the remaining columns only; it cannot
    # discard a feasible common parameter choice.
    minima = [min(len(J) for _, J in part) for part in decoded]
    remaining = sum(minima[1:])
    cells = [(bb, (J,), len(J)) for bb, J in decoded[0] if len(J) + remaining <= budget]
    for index, part in enumerate(decoded[1:], start=1):
        remaining -= minima[index]
        updated = []
        for box, supports, cost in cells:
            for bb, J in part:
                new_cost = cost + len(J)
                if new_cost + remaining > budget:
                    continue
                common = _nr_box_intersect(box, bb)
                if common is not None:
                    updated.append((common, supports + (J,), new_cost))
        cells = updated
        if not cells:
            return ()
    unique = {}
    for bb, supports, cost in cells:
        old = unique.get(supports)
        if old is None or _nr_box_key(bb) < _nr_box_key(old[0]):
            unique[supports] = (bb, cost)
    return tuple((_nr_encode_box(unique[s][0]), s, unique[s][1]) for s in sorted(unique))

import numpy as np


def _nr_norm(v):
    scale = float(np.max(np.abs(v)))
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(v / scale))


def _nr_gmres(H, g, restart, cycles):
    n = len(g)
    x = np.zeros(n, dtype=float)
    for _ in range(cycles):
        residual = g - H @ x
        beta = _nr_norm(residual)
        if beta == 0:
            continue
        V = np.zeros((n, restart + 1), dtype=float)
        hessenberg = np.zeros((restart + 1, restart), dtype=float)
        V[:, 0] = residual / beta
        dimension = restart
        for j in range(restart):
            w = H @ V[:, j]
            before = _nr_norm(w)
            # Two passes reduce loss of orthogonality in this nonsymmetric problem.
            for _pass in range(2):
                for i in range(j + 1):
                    projection = float(V[:, i] @ w)
                    hessenberg[i, j] += projection
                    w -= projection * V[:, i]
            length = _nr_norm(w)
            hessenberg[j + 1, j] = length
            # Numerical invariant-subspace detection, not a residual tolerance.
            if length <= 32 * np.finfo(float).eps * before:
                dimension = j + 1
                break
            V[:, j + 1] = w / length
        target = np.zeros(dimension + 1)
        target[0] = beta
        correction = np.linalg.lstsq(
            hessenberg[:dimension + 1, :dimension], target, rcond=None
        )[0]
        x += V[:, :dimension] @ correction
    return x


def evaluate_patterns(
    A: np.ndarray,
    rhs: np.ndarray,
    supports: tuple,
    restart: int = 3,
    cycles: int = 2,
) -> np.ndarray:
    data = _nr_matrix(A)
    n = data['n']
    A = _nr_real_array(A, 'A', 2)
    rhs = _nr_real_array(rhs, 'rhs', 2)
    if rhs.shape[0] != n or any(not np.any(b) for b in rhs.T):
        raise ValueError("rhs must have n rows and no zero column")
    restart = _nr_integer(restart, 'restart', 1, n)
    cycles = _nr_integer(cycles, 'cycles', 1)
    if not isinstance(supports, (tuple, list)) or len(supports) != 2 * n:
        raise ValueError("supports must contain exactly 2*n columns")
    matrices = []
    for side, C in enumerate((A, A.T)):
        matrix = np.zeros((n, n))
        for k in range(n):
            J = supports[side * n + k]
            v, _, _ = fit_column(C, k, J)
            try:
                matrix[list(J), k] = [float(_nr_fraction(z)) for z in v]
            except (OverflowError, FloatingPointError) as exc:
                raise ValueError('inverse coefficients exceed binary64 range') from exc
        matrices.append(matrix)
    MR, ML = matrices[0], matrices[1].T
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if not np.all(np.isfinite(MR)) or not np.all(np.isfinite(ML)):
                raise ValueError("nonfinite inverse coefficients")
            HR, HL = A @ MR, ML @ A
            result = np.empty((2, rhs.shape[1]))
            for j, b in enumerate(rhs.T):
                right_y = _nr_gmres(HR, b, restart, cycles)
                left_x = _nr_gmres(HL, ML @ b, restart, cycles)
                result[0, j] = _nr_norm(b - A @ left_x) / _nr_norm(b)
                result[1, j] = _nr_norm(b - A @ (MR @ right_y)) / _nr_norm(b)
            if not np.all(np.isfinite(result)):
                raise ValueError("nonfinite true residuals")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("binary64 Krylov evaluation failed") from exc
    return result

import numpy as np


def solve_shared_budget(
    A: np.ndarray,
    rhs: np.ndarray,
    box: tuple,
    budget: int,
    c: int = 2,
    restart: int = 3,
    cycles: int = 2,
) -> float:
    data = _nr_matrix(A)
    n = data['n']
    A = _nr_real_array(A, 'A', 2)
    rhs = _nr_real_array(rhs, 'rhs', 2)
    if rhs.shape[0] != n or any(not np.any(b) for b in rhs.T):
        raise ValueError("rhs must have n rows and no zero column")
    _nr_box(box)
    budget = _nr_integer(budget, 'budget', 0)
    c = _nr_integer(c, 'c', 1)
    restart = _nr_integer(restart, 'restart', 1, n)
    cycles = _nr_integer(cycles, 'cycles', 1)
    partitions = tuple(
        column_regions(C, k, box, c)
        for C in (A, A.T) for k in range(n)
    )
    configurations = joint_patterns(partitions, budget)
    if not configurations:
        raise ValueError("the storage budget has no feasible configuration")
    best = float('inf')
    for _region, supports, _cost in configurations:
        residuals = evaluate_patterns(A, rhs, supports, restart, cycles)
        if np.any(residuals[1] == 0):
            raise ValueError("a right residual is zero, so a ratio is undefined")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            score = float(np.max(residuals[0] / residuals[1]))
        if not np.isfinite(score):
            raise ValueError("nonfinite residual ratio")
        best = min(best, score)
    return float(best)
SCICODE_GOLD_EOF
