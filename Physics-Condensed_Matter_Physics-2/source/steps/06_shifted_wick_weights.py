"""
Evaluate both numerator and vacuum insertion weights, with signed Wick sums and sums of individual magnitudes kept separate. Support single time tuples and batches while retaining all residual vertices.

Quadratic counterterms and quartic spin vertices enter the same colored Gaussian expansion; individual diagrams retain separate weights.

For each ordered vertex choice with coefficient $c$, use $N=-c\operatorname{Pf}(K_N)/4$, $D=c\operatorname{Pf}(K_D)$, $N_{\rm abs}=|c|\operatorname{haf}(|K_N|)/4$, and $D_{\rm abs}=|c|\operatorname{haf}(|K_D|)$. Sum these four quantities over vertex choices before any time integration.

Returns
-------
complex ndarray (4,) or (B,4)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


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
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
            signed *= _oracle_pfaffian_batch(block)
            unsigned *= _oracle_absolute_wick_sums(block)
            start += number
        result[selected, 0] = signed
        result[selected, 1] = unsigned
    return result


def _oracle_shifted_wick_weights(
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
            _oracle_thermal_majorana_kernel(a, beta, np.r_[t, tau, 0])
            for t in time_chunk
        ]
        for indices, slots, coefficient, width in layouts:
            count = len(coefficient)
            matrices = np.concatenate(
                [
                    _oracle_wick_contraction_matrices(k, indices, slots)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])

vertices = np.array(
    [
        [-0.225, 4, 4, 8, 5, 9],
        [-0.275, 4, 9, 1, 10, 2],
        [-0.175, 4, 2, 6, 3, 7],
        [-0.23j, 2, 0, 1, -1, -1],
        [0.31j, 2, 4, 6, -1, -1],
        [-0.19j, 2, 8, 11, -1, -1],
    ],
    complex,
)


def with_preservation(fn, *args):
    before = [
        arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args
    ]
    result = fn(*args)
    preserved = all(np.array_equal(arg, old) for arg, old in zip(args, before))
    return np.r_[np.asarray(result).ravel(), float(preserved)]
""",
            "call": (
                "with_preservation(\n    shifted_wick_weights,\n    "
                "h,\n    2.4,\n    vertices,\n    np.array([1.91, 0.83, "
                "0.17]),\n    sites,\n    2,\n    0.83,\n)"
            ),
            "gold_call": (
                "with_preservation(\n    "
                "_oracle_shifted_wick_weights,\n    h,\n    2.4,\n    "
                "vertices,\n    np.array([1.91, 0.83, 0.17]),\n    "
                "sites,\n    2,\n    0.83,\n)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 2, 2))
h = np.zeros_like(j)
for color, value in enumerate([0.7, 1.2, -0.3]):
    j[color, 0, 1] = j[color, 1, 0] = value
for color, value in enumerate([0.19, -0.27, 0.13]):
    h[color, 0, 1] = value
    h[color, 1, 0] = -value
sites = np.array([0, 1])

vertices = np.array(
    [
        [0.2 + 0.4j, 2, 0, 2, -1, -1],
        [-0.3j, 4, 0, 0, 3, 3],
        [0.5, 2, 1, 1, -1, -1],
    ],
    complex,
)
times = np.array([[0.2, 0.2], [0.8, 0.3], [0.3, 0.8]])
""",
            "call": (
                "shifted_wick_weights(h, 2.1, vertices, times, sites, "
                "0, 0.68)"
            ),
            "gold_call": (
                "_oracle_shifted_wick_weights(h, 2.1, vertices, "
                "times, sites, 0, 0.68)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])
""",
            "call": (
                "shifted_wick_weights(h, 2.4, np.empty((0, 6)), "
                "np.array([0.0]), sites, 2, 0.83)"
            ),
            "gold_call": (
                "_oracle_shifted_wick_weights(\n    h, 2.4, "
                "np.empty((0, 6)), np.array([0.0]), sites, 2, 0.83\n)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])
""",
            "call": (
                "shifted_wick_weights(h, 2.4, np.empty((0, 6)), "
                "np.empty(0), sites, 2, 0.83)"
            ),
            "gold_call": (
                "_oracle_shifted_wick_weights(\n    h, 2.4, "
                "np.empty((0, 6)), np.empty(0), sites, 2, 0.83\n)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])

vertices = np.array(
    [
        [-0.225, 4, 4, 8, 5, 9],
        [-0.275, 4, 9, 1, 10, 2],
        [-0.175, 4, 2, 6, 3, 7],
        [-0.23j, 2, 0, 1, -1, -1],
        [0.31j, 2, 4, 6, -1, -1],
        [-0.19j, 2, 8, 11, -1, -1],
    ],
    complex,
)
""",
            "call": (
                "shifted_wick_weights(h, 2.4, vertices, np.empty((0, "
                "3)), sites, 2, 0.83)"
            ),
            "gold_call": (
                "_oracle_shifted_wick_weights(\n    h, 2.4, vertices, "
                "np.empty((0, 3)), sites, 2, 0.83\n)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])

vertices = np.array(
    [
        [-0.225, 4, 4, 8, 5, 9],
        [-0.275, 4, 9, 1, 10, 2],
        [-0.175, 4, 2, 6, 3, 7],
        [-0.23j, 2, 0, 1, -1, -1],
        [0.31j, 2, 4, 6, -1, -1],
        [-0.19j, 2, 8, 11, -1, -1],
    ],
    complex,
)


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "expect_value_error(\n    shifted_wick_weights, h, "
                "2.4, vertices, np.array([2.5]), sites, 2, 0.83\n)"
            ),
            "gold_call": (
                "expect_value_error(\n    "
                "_oracle_shifted_wick_weights,\n    h,\n    2.4,\n    "
                "vertices,\n    np.array([2.5]),\n    sites,\n    2,\n    "
                "0.83,\n)"
            ),
            "tol": 1e-08,
        },
    ]
