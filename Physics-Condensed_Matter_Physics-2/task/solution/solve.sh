#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def _numeric(value, name, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if real and np.any(a.imag != 0):
        raise ValueError(f"{name} must be real")
    return a.real if real else a


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} is outside the supported range")
    return int(value)


def _temperature(beta):
    b = _numeric(beta, "beta", True)
    if b.ndim != 0 or b <= 0:
        raise ValueError("beta must be positive")
    return float(b)


def _tensor(value, name, antisymmetric):
    a = _numeric(value, name, True)
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"{name} must have shape (3,L,L)")
    n = a.shape[1]
    if n < 2 or n % 2:
        raise ValueError("L must be positive and even, at least two")
    target = -a.swapaxes(-1, -2) if antisymmetric else a.swapaxes(-1, -2)
    if not np.allclose(a, target, atol=1e-12, rtol=0):
        raise ValueError(f"{name} has the wrong exchange symmetry")
    if np.max(np.abs(np.diagonal(a, axis1=1, axis2=2))) > 1e-12:
        raise ValueError(f"{name} must have a zero diagonal")
    return a


def _observable(sites, component, tau, beta, n):
    pair = _numeric(sites, "sites", True)
    if pair.shape != (2,) or np.any(pair != np.floor(pair)):
        raise ValueError("sites must contain two integer indices")
    if np.any((pair < 0) | (pair >= n)):
        raise ValueError("site index is out of range")
    color = _integer(component, "component", 0, 2)
    time = _numeric(tau, "tau", True)
    if time.ndim != 0 or not 0 <= time <= beta:
        raise ValueError("tau must lie in [0,beta]")
    return pair.astype(int), color, float(time)


def _vertices(value, dimension):
    v = _numeric(value, "vertices")
    if v.ndim != 2 or v.shape[1] != 6:
        raise ValueError("vertices must have shape (V,6)")
    for row in v:
        if np.any(row[1:].imag != 0) or np.any(
            row[1:].real != np.floor(row[1:].real)
        ):
            raise ValueError("vertex degrees and indices must be integers")
        degree = int(row[1].real)
        if degree not in (2, 4):
            raise ValueError("vertex degree must be two or four")
        indices = row[2 : 2 + degree].real
        if np.any((indices < 0) | (indices >= dimension)):
            raise ValueError("Majorana index is out of range")
        if np.any(row[2 + degree :] != -1):
            raise ValueError("unused vertex indices must be -1")
    return v


def map_shifted_vertices(
    couplings: np.ndarray, hopping: np.ndarray
) -> np.ndarray:
    j = _tensor(couplings, "couplings", False)
    a = _tensor(hopping, "hopping", True)
    if j.shape != a.shape:
        raise ValueError("couplings and hopping must have identical shapes")
    n = j.shape[1]
    rows = []
    for color in range(3):
        b, c = (color + 1) % 3, (color + 2) % 3
        for left in range(n):
            for right in range(left + 1, n):
                if j[color, left, right] != 0:
                    rows.append(
                        [
                            -j[color, left, right] / 4,
                            4,
                            b * n + left,
                            c * n + left,
                            b * n + right,
                            c * n + right,
                        ]
                    )
    for color in range(3):
        for left in range(n):
            for right in range(left + 1, n):
                if a[color, left, right] != 0:
                    rows.append(
                        [
                            -1j * a[color, left, right],
                            2,
                            color * n + left,
                            color * n + right,
                            -1,
                            -1,
                        ]
                    )
    return np.asarray(rows, complex).reshape(-1, 6)

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def thermal_majorana_kernel(
    hopping: np.ndarray, beta: float, times: np.ndarray
) -> np.ndarray:
    a = _tensor(hopping, "hopping", True)
    beta = _temperature(beta)
    t = _numeric(times, "times", True)
    if t.ndim != 1 or t.size == 0 or np.any((t < 0) | (t > beta)):
        raise ValueError("times must be a nonempty vector in [0,beta]")
    n = a.shape[1]
    delta = t[:, None] - t[None, :]
    wrapped = np.where(delta < 0, delta + beta, delta)
    sign = np.where(delta < 0, -1.0, 1.0)
    result = np.zeros((t.size, t.size, 3 * n, 3 * n), complex)
    for color in range(3):
        energy, basis = np.linalg.eigh(1j * a[color])
        log_weight = (
            np.log(2.0)
            - 2 * wrapped[..., None] * energy
            - np.logaddexp(0.0, -2 * beta * energy)
        )
        block = np.einsum(
            "ik,abk,jk->abij", basis, np.exp(log_weight), basis.conj()
        )
        result[
            :, :, color * n : (color + 1) * n, color * n : (color + 1) * n
        ] = (sign[..., None, None] * block)
    return result

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def wick_contraction_matrices(
    kernel: np.ndarray, indices: np.ndarray, slots: np.ndarray
) -> np.ndarray:
    k = _numeric(kernel, "kernel")
    x = _numeric(indices, "indices", True)
    s = _numeric(slots, "slots", True)
    if (
        k.ndim != 4
        or k.shape[0] != k.shape[1]
        or k.shape[2] != k.shape[3]
        or min(k.shape) == 0
    ):
        raise ValueError(
            ("kernel must have shape (T,T,D,D), with " "T,D positive")
        )
    if x.ndim != 2 or x.shape[1] % 2 or s.shape != (x.shape[1],):
        raise ValueError("indices must have shape (B,2m), slots shape (2m,)")
    if np.any(x != np.floor(x)) or np.any(s != np.floor(s)):
        raise ValueError("indices and slots must be integer-valued")
    if np.any((x < 0) | (x >= k.shape[2])) or np.any(
        (s < 0) | (s >= k.shape[0])
    ):
        raise ValueError("index or slot out of range")
    x, s = x.astype(int), s.astype(int)
    batch, size = x.shape
    result = np.zeros((batch, size, size), complex)
    upper = np.triu_indices(size, 1)
    values = k[s[upper[0]], s[upper[1]], x[:, upper[0]], x[:, upper[1]]]
    result[:, upper[0], upper[1]] = values
    result[:, upper[1], upper[0]] = -values
    return result

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def pfaffian_batch(matrices: np.ndarray) -> np.ndarray:
    a = _numeric(matrices, "matrices").copy()
    if a.ndim != 3 or a.shape[1] != a.shape[2] or a.shape[1] % 2:
        raise ValueError("matrices must have shape (B,2m,2m)")
    if not np.allclose(a, -a.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("matrices must be skew-symmetric")
    batch, size, unused = a.shape
    result = np.zeros(batch, complex)
    active = np.arange(batch)
    weight = np.ones(batch, complex)
    for k in range(0, size, 2):
        rows = np.arange(len(active))
        if not len(active):
            break
        pivot = k + 1 + np.argmax(np.abs(a[:, k, k + 1 :]), axis=1)
        saved = a[:, k + 1, :].copy()
        a[:, k + 1, :] = a[rows, pivot, :]
        a[rows, pivot, :] = saved
        saved = a[:, :, k + 1].copy()
        a[:, :, k + 1] = a[rows, :, pivot]
        a[rows, :, pivot] = saved
        weight *= np.where(pivot == k + 1, 1, -1)
        value = a[:, k, k + 1]
        weight *= value
        keep = value != 0
        a, active, weight, value = (
            a[keep],
            active[keep],
            weight[keep],
            value[keep],
        )
        u = a[:, k, k + 2 :].copy()
        v = a[:, k + 1, k + 2 :].copy()
        a[:, k + 2 :, k + 2 :] -= (
            u[:, :, None] * v[:, None, :] - v[:, :, None] * u[:, None, :]
        ) / value[:, None, None]
    result[active] = weight
    return result

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def absolute_wick_sums(matrices: np.ndarray) -> np.ndarray:
    """Sum magnitudes of individual perfect-pairing products.

    Parameters
    ----------
    matrices : complex ndarray, shape (B, 2m, 2m)
        Finite skew-symmetric matrices (transpose, not adjoint), with
        zero diagonal, within absolute tolerance 1e-12. B or m may be
        zero. Repeated fields and singular matrices are permitted.

    Returns
    -------
    sums : real ndarray, shape (B,)
        For each matrix K, sum over all perfect matchings P of
        product(abs(K[u, v]) for (u, v) in P), with u < v.
        The empty matching has weight one. Take absolute values before
        summing matchings. Inputs are unchanged.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd dimension, nonzero
        diagonal or failure of skew symmetry at the stated tolerance.
    """
    matrix = _numeric(matrices, "matrices")
    if (
        matrix.ndim != 3
        or matrix.shape[1] != matrix.shape[2]
        or matrix.shape[1] % 2
    ):
        raise ValueError("matrices must have shape (B, 2m, 2m)")
    if not np.allclose(
        matrix + matrix.swapaxes(1, 2), 0, atol=1e-12, rtol=0
    ) or not np.allclose(
        np.diagonal(matrix, axis1=1, axis2=2), 0, atol=1e-12, rtol=0
    ):
        raise ValueError("matrices must be skew with zero diagonal")
    weights = np.abs(matrix)
    batch, size, _ = weights.shape

    @lru_cache(None)
    def _match(mask):
        if mask == 0:
            return np.ones(batch)
        first = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << first)
        remaining = rest
        total = np.zeros(batch)
        while remaining:
            bit = remaining & -remaining
            second = bit.bit_length() - 1
            edge = weights[:, first, second]
            if np.any(edge):
                total += edge * _match(rest ^ bit)
            remaining ^= bit
        return total

    return _match((1 << size) - 1).copy()

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def _colored_pairing_weights(matrices, indices, size):
    """Factor color-diagonal Wick sums, retaining permutation parity."""
    batch, width = indices.shape
    colors = indices // size
    counts = np.stack([(colors == c).sum(axis=1) for c in range(3)], axis=1)
    result = np.zeros((batch, 2), complex)
    if batch == 0:
        return result
    permutation = np.argsort(colors, axis=1, kind="stable")
    upper = np.triu_indices(width, 1)
    parity = (colors[:, upper[0]] > colors[:, upper[1]]).sum(axis=1) % 2
    sorted_matrices = np.take_along_axis(
        matrices, permutation[:, :, None], axis=1
    )
    sorted_matrices = np.take_along_axis(
        sorted_matrices, permutation[:, None, :], axis=2
    )
    for count in np.unique(counts, axis=0):
        if np.any(count % 2):
            continue
        selected = np.flatnonzero(np.all(counts == count, axis=1))
        signed = (1 - 2 * parity[selected]).astype(complex)
        unsigned = np.ones(len(selected))
        start = 0
        for number in count:
            block = sorted_matrices[
                selected, start : start + number, start : start + number
            ]
            signed *= pfaffian_batch(block)
            unsigned *= absolute_wick_sums(block)
            start += number
        result[selected, 0] = signed
        result[selected, 1] = unsigned
    return result


def shifted_wick_weights(
    hopping: np.ndarray,
    beta: float,
    vertices: np.ndarray,
    vertex_times: np.ndarray,
    sites: np.ndarray,
    component: int,
    tau: float,
) -> np.ndarray:
    """Evaluate signed and pairing-absolute insertion weights.

    Parameters
    ----------
    hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal skew reference with even L >= 2, as in
        thermal_majorana_kernel; the reference is held fixed.
    beta : float
        Finite positive inverse temperature.
    vertices : complex ndarray, shape (V, 6)
        Rows [coefficient, degree, i1, i2, i3, i4] as in
        map_shifted_vertices. Degrees are 2 or 4, active indices are
        integers in [0, 3L), and unused indices are -1. Empty tables
        and repeated active indices are allowed. Retain distinct rows.
    vertex_times : real ndarray, shape (n,) or (B, n)
        Finite times in [0, beta], 0 <= n <= 3. The second form batches
        independent time tuples, and B may be zero. Unsorted times and
        ties are permitted. Preserve field order inside every even
        block and initial block order for ties.
    sites : ndarray, shape (2,)
        Integer-valued sites j, l in [0, L).
    component : int
        Spin color a in {0, 1, 2}, with cyclic partners b, c.
    tau : float
        Finite external time in [0, beta].

    Returns
    -------
    weights : complex ndarray, shape (4,) or (B, 4)
        Columns are signed numerator, signed vacuum, unsigned numerator,
        unsigned vacuum. Sum every ordered choice of n vertex rows.
        The numerator inserts S_j^a(tau) and S_l^a(0) after the vertex
        blocks; S_j^a = -(i/2) rho_j^b rho_j^c. Signed weights use the
        normalized H0 Wick average. Unsigned weights sum magnitudes of
        each separate vertex-choice/perfect-pairing contribution, with
        spin prefactor magnitude 1/4. Absolute values precede all sums.
        All contractions, including vacuum components, are retained.
        No Dyson sign, integration, factorial or series division is
        included. At n=0 both vacuum weights are one. Inputs unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, vertex table, times, sites,
        component or tau, including nonfinite/out-of-range values.
    """
    a = _tensor(hopping, "hopping", True)
    beta = _temperature(beta)
    size = a.shape[1]
    vertices = _vertices(vertices, 3 * size)
    times = _numeric(vertex_times, "vertex_times", True)
    if (
        times.ndim not in (1, 2)
        or times.shape[-1] > 3
        or np.any((times < 0) | (times > beta))
    ):
        raise ValueError("vertex_times must have shape (n,) or (B,n)")
    pair, color, tau = _observable(sites, component, tau, beta, size)
    single = times.ndim == 1
    times = times.reshape(1, -1) if single else times
    order = times.shape[1]
    b, c = (color + 1) % 3, (color + 2) % 3
    external = np.array(
        [
            b * size + pair[0],
            c * size + pair[0],
            b * size + pair[1],
            c * size + pair[1],
        ]
    )
    groups = {d: np.flatnonzero(vertices[:, 1].real == d) for d in (2, 4)}
    layouts = []
    for degrees in product((2, 4), repeat=order):
        choices = [groups[d] for d in degrees]
        if any(len(choice) == 0 for choice in choices):
            continue
        chosen = (
            np.array(list(product(*choices)), int).reshape(-1, order)
            if order
            else np.empty((1, 0), int)
        )
        coefficient = (
            np.prod(vertices[chosen, 0], axis=1)
            if order
            else np.ones(1, complex)
        )
        parts = [
            vertices[chosen[:, k], 2 : 2 + degree].real.astype(int)
            for k, degree in enumerate(degrees)
        ]
        internal = (
            np.concatenate(parts, axis=1) if parts else np.empty((1, 0), int)
        )
        indices = np.c_[internal, np.broadcast_to(external, (len(chosen), 4))]
        slots = np.r_[
            np.repeat(np.arange(order), degrees),
            order,
            order,
            order + 1,
            order + 1,
        ].astype(int)
        layouts.append((indices, slots, coefficient, internal.shape[1]))
    result = np.zeros((len(times), 4), complex)
    for start in range(0, len(times), 32):
        time_chunk = times[start : start + 32]
        kernels = [
            thermal_majorana_kernel(a, beta, np.r_[t, tau, 0])
            for t in time_chunk
        ]
        for indices, slots, coefficient, width in layouts:
            count = len(coefficient)
            matrices = np.concatenate(
                [
                    wick_contraction_matrices(k, indices, slots)
                    for k in kernels
                ]
            )
            all_indices = np.tile(indices, (len(kernels), 1))
            numerator = _colored_pairing_weights(matrices, all_indices, size)
            vacuum = _colored_pairing_weights(
                matrices[:, :width, :width], all_indices[:, :width], size
            )
            numerator = numerator.reshape(len(kernels), count, 2)
            vacuum = vacuum.reshape(len(kernels), count, 2)
            target = result[start : start + len(kernels)]
            target[:, 0] += -0.25 * (numerator[:, :, 0] @ coefficient)
            target[:, 1] += vacuum[:, :, 0] @ coefficient
            target[:, 2] += 0.25 * (numerator[:, :, 1] @ np.abs(coefficient))
            target[:, 3] += vacuum[:, :, 1] @ np.abs(coefficient)
    return result[0] if single else result

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def ordered_time_rule(
    beta: float, tau: float, order: int, quadrature: int
) -> np.ndarray:
    """Construct a folded-half-period ordered-simplex Gauss rule.

    Parameters
    ----------
    beta : float
        Finite positive interval length.
    tau : float
        Finite external time in [0, beta].
    order : int
        Number n of insertion times, 0 <= n <= 3.
    quadrature : int
        Gauss-Legendre nodes per variable, 2 <= q <= 24.

    Returns
    -------
    rule : real ndarray, shape (R, n+1)
        Rows [t1, ..., tn, weight], with descending physical times in
        [0, beta]. For n=0 return [[1.0]]. Otherwise set h=beta/2 and
        f=tau modulo h. Enumerate bit tuples in {0,1}**n lexicographically.
        For each tuple enumerate k=0,...,n folded variables above f,
        skipping zero-volume regions. Enumerate the ascending unit-
        interval Gauss nodes in lexicographic tensor-product order.
        Generate descending folded variables u on [f,h] for the first
        k positions and [0,f] for the rest. In each block start at its
        upper bound, map x to lower+(upper-lower)*x, multiply the weight
        by (upper-lower) times its unit Gauss weight, and replace upper
        with the new point. Set physical t to the descending sort of
        u+h*bits. No extra factorial. Total weight is beta**n/n!.
        These panels resolve pairwise half-period crossings and the
        external time. They do not guarantee exact integration for
        arbitrary hopping; convergence remains a numerical question.

    Raises
    ------
    ValueError
        For nonfinite beta/tau, nonpositive beta, tau outside [0,beta],
        or noninteger/out-of-range order or quadrature.
    """
    beta = _temperature(beta)
    time = _numeric(tau, "tau", True)
    if time.ndim != 0 or not 0 <= time <= beta:
        raise ValueError("tau must lie in [0,beta]")
    order = _integer(order, "order", 0, 3)
    q = _integer(quadrature, "quadrature", 2, 24)
    if order == 0:
        return np.ones((1, 1))
    half = beta / 2
    folded = float(time) % half
    nodes, weights = leggauss(q)
    nodes, weights = (nodes + 1) / 2, weights / 2
    result = []
    for bits in product((0, 1), repeat=order):
        for above in range(order + 1):
            bounds = [(folded, half)] * above + [(0.0, folded)] * (
                order - above
            )
            if any(lo == hi for lo, hi in bounds):
                continue
            for picks in product(range(q), repeat=order):
                times = []
                weight = 1.0
                upper = half
                for k, pick in enumerate(picks):
                    lower, initial_upper = bounds[k]
                    if k == 0 or k == above:
                        upper = initial_upper
                    width = upper - lower
                    weight *= width * weights[pick]
                    upper = lower + width * nodes[pick]
                    times.append(upper + half * bits[k])
                result.append(sorted(times, reverse=True) + [weight])
    return np.asarray(result, float)

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_coefficient_series(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> np.ndarray:
    """Integrate signed and pairing-absolute connected coefficients.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2, as in map_shifted_vertices.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0, beta].
    sites : ndarray, shape (2,)
        Integer-valued sites j,l in [0,L).
    component : int
        Color a in {0,1,2} of both external spins.
    order : int
        Highest insertion order p in {0,1,2,3}.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    series : complex ndarray, shape (6, p+1)
        Rows [X, Z, C, X_abs, Z_abs, C_abs]. X_n and Z_n are
        normalized-H0 numerator/vacuum Dyson coefficients of xi**n
        for H(xi)=H0+xi*(H_Maj-H0). C=X/Z as a formal power series.
        X_abs,n and Z_abs,n integrate individual pairing magnitudes
        before any vertex/pairing sums; use the exact row decomposition
        from map_shifted_vertices and no Dyson minus sign. C_abs is
        the formal quotient X_abs/Z_abs. This removes vacuum components
        while preserving magnitudes of separate connected diagrams.
        A component must include both external spin blocks because
        each interaction vertex has even population of every color.
        Z_0=Z_abs,0=1. Apply the supplied quadrature to every order,
        including lower orders entering the quotient. No projection or
        reference refitting is applied. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, observable, order or quadrature
        under the contracts above and of the producing functions.
    """
    j = _tensor(couplings, "couplings", False)
    a = _tensor(hopping, "hopping", True)
    if j.shape != a.shape:
        raise ValueError("couplings and hopping must have identical shapes")
    beta = _temperature(beta)
    pair, color, tau = _observable(sites, component, tau, beta, j.shape[1])
    order = _integer(order, "order", 0, 3)
    _integer(quadrature, "quadrature", 2, 24)
    vertices = map_shifted_vertices(j, a)
    result = np.zeros((6, order + 1), complex)
    for power in range(order + 1):
        rule = ordered_time_rule(beta, tau, power, quadrature)
        moments = rule[:, -1] @ shifted_wick_weights(
            a, beta, vertices, rule[:, :-1], pair, color, tau
        )
        result[:2, power] = (-1) ** power * moments[:2]
        result[3:5, power] = moments[2:]
        for row in (0, 3):
            result[row + 2, power] = result[row, power] - sum(
                result[row + 1, k] * result[row + 2, power - k]
                for k in range(1, power + 1)
            )
    return result

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_average_sign(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> float:
    """Return the real average phase of connected order-p diagrams.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2. Use map_shifted_vertices without
        regrouping its monomial rows or changing the reference.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0,beta].
    sites : ndarray, shape (2,)
        Integer-valued sites in [0,L) for the two external spins.
    component : int
        Spin color in {0,1,2}.
    order : int
        Selected insertion order p in {0,1,2,3}; not a truncated sum.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    percentage : float
        100*Re(C_p)/C_abs,p from connected_coefficient_series with the
        supplied quadrature. C_abs,p sums magnitudes of individual
        connected pairings before summation and integration. For real
        diagram weights this is the average sign in percent; otherwise
        it is the real average phase in percent. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid inputs under connected_coefficient_series, a
        nonfinite coefficient/result, or C_abs,p <= 1e-12.
    """
    series = connected_coefficient_series(
        couplings, hopping, beta, tau, sites, component, order, quadrature
    )
    denominator = series[5, -1].real
    if not np.all(np.isfinite(series)) or denominator <= 1e-12:
        raise ValueError("connected absolute weight must exceed 1e-12")
    result = 100 * series[2, -1].real / denominator
    if not np.isfinite(result):
        raise ValueError("average phase must be finite")
    return float(result)
SCICODE_GOLD_EOF
