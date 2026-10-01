"""
Find the globally admissible logical sector and reconstruct its least history.

Return the globally selected logical sector, its energy, all sector minima, and a physical history. The tie tolerance is one whole-history budget for both sector choice and history choice. Among admissible options choose the smallest logical identifier, then the lexicographically smallest time-major physical history. The final syndrome is exact. Ordinary floating-point roundoff is allowed at equality. Measurement-error energies use the same zero-offset spin normalization at inverse temperature one, and the returned history energy and sector minima are sums of the supplied data energies and those measurement energies.

Returns
-------
A float vector contains the selected logical sector, history energy, four sector minima and the complete physical history.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_spacetime_sectors(
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    branches: "np.ndarray",
    pm: float,
    tie_tol: float,
) -> "np.ndarray":
    r"""Find the globally admissible logical sector and reconstruct its least history.

    Parameters
    ----------
    constraints : np.ndarray
        Binary temporal matrix of shape $(Tm+2,Tn+(T-1)m)$ with repeated spatial blocks.
    detectors : np.ndarray
        Binary detector record of shape $(T,m)$, $1\le T\le8$, $1\le m\le3$.
    branches : np.ndarray
        Finite float array $(T,2^n,n+2)$ of exact ordered words and jump labels;
        its energy column may contain any finite values, with $m+2\le n\le10$.
    pm : float
        Independent measurement error rate in $(0,1/2]$.
    tie_tol : float
        Nonnegative finite global energy window $\delta$.

    Returns
    -------
    result : np.ndarray
        Float vector of length $6+Tn+(T-1)m$: chosen logical identifier, chosen
        history energy, four exact sector minima, then all data bits and measurement
        bits.

    Raises
    ------
    ValueError
        If detector/constraint dimensions or repeated blocks are inconsistent,
        spatial syndrome/logical rows are dependent, branch words or labels are
        invalid, or pm or tie_tol lies outside its domain.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _detector_layout(constraints, detectors):
    a = _binary(constraints, 2, "constraints")
    d = _binary(detectors, 2, "detectors")
    tmax, m = d.shape
    if not 1 <= tmax <= 8 or not 1 <= m <= 3 or a.shape[0] != tmax * m + 2:
        raise ValueError("detector dimensions are invalid")
    n, rem = divmod(a.shape[1] - (tmax - 1) * m, tmax)
    if rem or not m + 2 <= n <= 10:
        raise ValueError("constraint column count is invalid")
    h = a[:m, :n]
    logical = a[-2:, :n]
    expected = np.zeros_like(a)
    for t in range(tmax):
        expected[t * m : (t + 1) * m, t * n : (t + 1) * n] = h
        expected[-2:, t * n : (t + 1) * n] = logical
        for j in (t - 1, t):
            if 0 <= j < tmax - 1:
                expected[
                    t * m : (t + 1) * m, tmax * n + j * m : tmax * n + (j + 1) * m
                ] ^= np.eye(m, dtype=int)
    if (
        not np.array_equal(a, expected)
        or len(_rref(np.concatenate((h, logical)))[1]) != m + 2
    ):
        raise ValueError(
            "constraints must encode the stated independent chain and logical rows"
        )
    return a, d, tmax, m, n


def _emissions(detectors, pm):
    tmax, m = detectors.shape
    records = np.bitwise_xor.accumulate(detectors, axis=0)
    syndrome = _words(m)[np.arange(1 << (m + 2)) >> 2]
    errors = records[:, None, :] ^ syndrome[None, :, :]
    coupling = 0.5 * np.log((1 - pm) / pm)
    cost = coupling * np.sum(2 * errors - 1, axis=2)
    cost[-1] = 0
    return records, errors, cost


def _dyadic_costs(data, measurement, tolerance):
    shapes = (data.shape, measurement.shape)
    values = np.concatenate((data.ravel(), measurement.ravel(), [float(tolerance)]))
    ratios = [float(v).as_integer_ratio() for v in values]
    exponent = max(den.bit_length() - 1 for _, den in ratios)
    integers = [num << (exponent - (den.bit_length() - 1)) for num, den in ratios]
    cut = data.size
    last = cut + measurement.size
    return (
        np.array(integers[:cut], dtype=object).reshape(shapes[0]),
        np.array(integers[cut:last], dtype=object).reshape(shapes[1]),
        integers[-1],
        1 << exponent,
    )


def _oracle_solve_spacetime_sectors(
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    branches: "np.ndarray",
    pm: float,
    tie_tol: float,
) -> "np.ndarray":
    a, d, tmax, m, n = _detector_layout(constraints, detectors)
    br = _real(branches, "branches")
    pm = _prob(pm, 0.5, "pm")
    tol = _real(tie_tol, "tie_tol")
    if tol.ndim or tol < 0:
        raise ValueError("tie_tol must be a nonnegative finite scalar")
    words = _words(n)
    label_bits = np.concatenate((a[:m, :n], a[-2:, :n]))
    labels = ((words @ label_bits.T) % 2) @ (1 << np.arange(m + 1, -1, -1))
    if (
        br.shape != (tmax, 1 << n, n + 2)
        or not np.all(br[:, :, 0] == labels)
        or not np.all(br[:, :, 2:] == words)
    ):
        raise ValueError(
            "branches must contain the exact ordered words and quotient labels"
        )
    records, measurement, cost = _emissions(d, pm)
    states = np.arange(1 << (m + 2))
    terminal_syndrome = int(records[-1] @ (1 << np.arange(m - 1, -1, -1)))
    data_cost, measurement_cost, window, scale = _dyadic_costs(br[:, :, 1], cost, tol)
    reduced = np.full((tmax, len(states)), np.inf, dtype=object)
    for t in range(tmax):
        np.minimum.at(reduced[t], labels, data_cost[t])
    transition = states[:, None] ^ states[None, :]
    suffixes = np.full((4, tmax + 1, len(states)), np.inf, dtype=object)
    for logical in range(4):
        suffixes[logical, -1, 4 * terminal_syndrome + logical] = 0
        for t in range(tmax - 1, -1, -1):
            suffixes[logical, t] = np.min(
                reduced[t, transition]
                + measurement_cost[t, None, :]
                + suffixes[logical, t + 1, None, :],
                axis=1,
            )
    minima = suffixes[:, 0, 0]
    global_minimum = int(minima.min())
    logical = int(np.flatnonzero(minima <= global_minimum + window)[0])
    suffix = suffixes[logical]
    state = 0
    accumulated = 0
    chosen = []
    measures = []
    for t in range(tmax):
        next_states = state ^ labels
        trial = (
            accumulated
            + data_cost[t]
            + measurement_cost[t, next_states]
            + suffix[t + 1, next_states]
        )
        allowed = np.flatnonzero(trial <= global_minimum + window)
        if not allowed.size:
            raise ValueError("no numerically admissible reconstruction")
        word = int(allowed[0])
        state = int(next_states[word])
        accumulated += data_cost[t, word] + measurement_cost[t, state]
        chosen.extend(words[word])
        if t < tmax - 1:
            measures.extend(measurement[t, state])
    errors = np.array(chosen + measures, dtype=int)
    target = np.concatenate((d.ravel(), _words(2)[logical]))
    if not np.array_equal((a @ errors) % 2, target):
        raise ValueError("reconstructed path violates its detector or logical sector")
    return np.concatenate(
        ([logical, accumulated / scale], [int(v) / scale for v in minima], errors)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
tie_tol_ref_input = copy.deepcopy(tie_tol)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_ref_input, tie_tol_ref_input)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array([[[1, 0, 1]], [[0, 1, 1]]], dtype=int)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
tie_tol_ref_input = copy.deepcopy(tie_tol)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_ref_input, tie_tol_ref_input)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
stabilizers = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
logicals = np.array(
    [
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 0, 1, 0], [1, 1, 0, 0]],
    ],
    dtype=int,
)
detectors = np.array([[[1], [0]], [[0], [1]]], dtype=int)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
tie_tol_ref_input = copy.deepcopy(tie_tol)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_ref_input, tie_tol_ref_input)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
stabilizers = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
logicals = np.array(
    [
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 0, 1, 0], [1, 1, 0, 0]],
    ],
    dtype=int,
)
detectors = np.array([[[1], [0]], [[0], [1]]], dtype=int)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
stabilizers_ref_input = copy.deepcopy(stabilizers)
tie_tol_ref_input = copy.deepcopy(tie_tol)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
p = 0.75
pm = 0.5
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.75
pm_g = 0.5
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_g, counts_ref_input, True)
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_g, tie_tol_ref_input)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
stabilizers = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
logicals = np.array(
    [
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 0, 1, 0], [1, 1, 0, 0]],
    ],
    dtype=int,
)
detectors = np.array([[[1], [0]], [[0], [1]]], dtype=int)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
tie_tol = 0.6
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
tie_tol_g = 0.6
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_ref_input, tie_tol_g)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
tie_tol_ref_input = copy.deepcopy(tie_tol)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)
branches[0, 0, 0] = 999


def _raises(fn):
    try:
        fn(
            constraints[0].copy(),
            detectors[0].copy(),
            branches.copy(),
            pm,
            tie_tol,
        )
    except ValueError:
        return 1
    return 0

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
branches_bad_g = np.where(np.arange(branches_g.size).reshape(branches_g.shape) == 0, 999.0, branches_g)

def _raises_gold(fn):
    try:
        fn(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_bad_g.copy(), pm_ref_input, tie_tol_ref_input)
    except ValueError:
        return 1
    return 0
"""
            )
            )
            )
            ),
            "call": ("_raises(solve_spacetime_sectors)"),
            "gold_call": ('_raises_gold(_oracle_solve_spacetime_sectors)'),
            "tol": 0,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

h = np.array([[1, 1, 1, 1]])
stabilizers = np.array([[1, 1, 0, 0]])
logicals = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
h_ref_input = copy.deepcopy(h)
logicals_ref_input = copy.deepcopy(logicals)
stabilizers_ref_input = copy.deepcopy(stabilizers)
q = build_binary_quotient(h, stabilizers, logicals)
a = build_detector_constraints(q, 1, 1)
counts = np.array([[930, 70], [30, 970]])
b = build_conditional_branches(
    q, 1, np.zeros((1, 4), dtype=int), 0.08, counts, True
)
w = b[0, :, 2:]
b[0, :, 1] = (2 * w - 1) @ np.array(
    [1.0, 1.0 + 5.2e-13, 20.0, 20.0]
)
d = np.array([[1]])

# Independent reference inputs from the same primitive instance.
q_g = _oracle_build_binary_quotient(h_ref_input, stabilizers_ref_input, logicals_ref_input)
a_g = _oracle_build_detector_constraints(q_g, 1, 1)
counts_g = np.array([[930, 70], [30, 970]])
b_g = _oracle_build_conditional_branches(q_g, 1, np.zeros((1, 4), dtype=int), 0.08, counts_g, True)
w_g = b_g[0, :, 2:]
b_mod_g = np.concatenate([b_g[:, :, :1], ((2 * w_g - 1) @ np.array([1.0, 1.0 + 5.2e-13, 20.0, 20.0]))[None, :, None], b_g[:, :, 2:]], axis=2)
d_g = np.array([[1]])
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(a.copy(), d.copy(), b.copy(), 0.06, 1e-12)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(a_g.copy(), d_g.copy(), b_mod_g.copy(), 0.06, 1e-12)'
            ),
            "tol": 1e-10,
        },
        {
            "setup": (
                (
                (
                r"""import copy
import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
m = checks.shape[1]
n = checks.shape[2]
T = detectors.shape[1]
T_ref_input = copy.deepcopy(T)
checks_ref_input = copy.deepcopy(checks)
counts_ref_input = copy.deepcopy(counts)
detectors_ref_input = copy.deepcopy(detectors)
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
pm_ref_input = copy.deepcopy(pm)
stabilizers_ref_input = copy.deepcopy(stabilizers)
quotients = np.stack(
    [
        build_binary_quotient(
            checks[c], stabilizers[c], logicals[c]
        )
        for c in range(2)
    ]
)
constraints = np.stack(
    [
        build_detector_constraints(q, m, T)
        for q in quotients
    ]
)
opposite = np.zeros((T, n), dtype=int)
branches = build_conditional_branches(
    quotients[0], m, opposite, p, counts, True
)
tie_tol = 1e-9

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
branches_g = _oracle_build_conditional_branches(quotients_g[0], m_ref_input, opposite_g, p_ref_input, counts_ref_input, True)
tie_tol_g = 1e-09
"""
            )
            )
            ),
            "call": (
                "solve_spacetime_sectors(constraints[0].copy(), detectors"
                "[0].copy(), branches.copy(), pm, tie_tol)"
            ),
            "gold_call": (
                '_oracle_solve_spacetime_sectors(constraints_g[0].copy(), detectors_ref_input[0].copy(), branches_g.copy(), pm_ref_input, tie_tol_g)'
            ),
            "tol": 1e-09,
        },
    ]
