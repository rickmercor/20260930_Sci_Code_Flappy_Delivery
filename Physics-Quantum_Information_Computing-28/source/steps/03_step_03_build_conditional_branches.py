"""
Construct energy-labelled physical transitions under conditional reliability.

Return every physical-word transition with its quotient jump and conditional data energy. Rows of the supplied reliability table refer to estimated bits and columns to true bits. Enumerate words lexicographically in increasing edge order, including nonminimum words within a quotient sector. The initial estimate uses the marginal data model; measurement energy is separate. Report energies at inverse temperature one in the zero-offset spin normalization: with spin $(-1)^{e_i}$ on each edge, a word's energy is linear in its spins with no constant term, so a word and its bitwise complement have opposite energies.

Returns
-------
A float array of shape (T, 2ⁿ, n+2) contains each quotient jump, energy and ordered physical word.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_conditional_branches(
    quotient: "np.ndarray",
    check_count: int,
    opposite: "np.ndarray",
    p: float,
    counts: "np.ndarray",
    initial: bool,
) -> "np.ndarray":
    r"""Construct energy-labelled physical transitions under conditional reliability.

    Parameters
    ----------
    quotient : np.ndarray
        Binary inverse pair of shape $(2,n,n)$.
    check_count : int
        Independent check count $m\in\{1,2,3\}$.
    opposite : np.ndarray
        Binary opposite-component data history of shape $(T,n)$, $1\le T\le8$.
    p : float
        Total depolarizing rate in $(0,3/4]$.
    counts : np.ndarray
        Nonnegative finite $(2,2)$ table with positive row sums and $a,b\ge1/2$.
    initial : bool
        Whether to use the unconditioned component marginal.

    Returns
    -------
    result : np.ndarray
        Float array of shape $(T,2^n,n+2)$ with rows in physical-word order.
        Columns contain quotient jump identifier, data energy, then $n$ physical bits.

    Raises
    ------
    ValueError
        If the inverse pair, dimensions, binary history, rate, counts or Boolean
        initialization flag violates the stated contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_conditional_branches(
    quotient: "np.ndarray",
    check_count: int,
    opposite: "np.ndarray",
    p: float,
    counts: "np.ndarray",
    initial: bool,
) -> "np.ndarray":
    q, m = _quotient(quotient, check_count)
    opposite = _binary(opposite, 2, "opposite")
    tmax, n = opposite.shape
    if (
        not 1 <= tmax <= 8
        or n != q.shape[1]
        or not isinstance(initial, (bool, np.bool_))
    ):
        raise ValueError("opposite shape or initial flag is invalid")
    p = _prob(p, 0.75, "p")
    c = _real(counts, "counts")
    if c.shape != (2, 2) or np.any(c < 0) or np.any(c.sum(axis=1) <= 0):
        raise ValueError("counts must be nonnegative 2 by 2 with positive row sums")
    a, b = c[1, 1] / c[1].sum(), c[0, 0] / c[0].sum()
    if min(a, b) < 0.5:
        raise ValueError("both fitted reliabilities must be at least one half")
    low = p / (3 - 2 * p)
    probability = (
        np.full(opposite.shape, 2 * p / 3)
        if initial
        else np.where(opposite, a / 2 + (1 - a) * low, b * low + (1 - b) / 2)
    )
    coupling = 0.5 * np.log((1 - probability) / probability)
    words = _words(n)
    labels = ((words @ q[1, :, : m + 2]) % 2) @ (1 << np.arange(m + 1, -1, -1))
    result = np.empty((tmax, 1 << n, n + 2))
    result[:, :, 0] = labels
    result[:, :, 1] = coupling @ (2 * words - 1).T
    result[:, :, 2:] = words
    return result

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
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
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

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
"""
            )
            )
            ),
            "call": (
                "build_conditional_branches(quotients[0].copy(), m, oppos"
                "ite.copy(), p, counts.copy(), True)"
            ),
            "gold_call": (
                '_oracle_build_conditional_branches(quotients_g[0].copy(), m_ref_input, opposite_g.copy(), p_ref_input, counts_ref_input.copy(), True)'
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
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
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
opposite[:, ::2] = 1

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
opposite_g[:, ::2] = 1
"""
            )
            )
            ),
            "call": (
                "build_conditional_branches(quotients[0].copy(), m, oppos"
                "ite.copy(), p, counts.copy(), False)"
            ),
            "gold_call": (
                '_oracle_build_conditional_branches(quotients_g[0].copy(), m_ref_input, opposite_g.copy(), p_ref_input, counts_ref_input.copy(), False)'
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
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
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
p = 0.75

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.75
"""
            )
            )
            ),
            "call": (
                "build_conditional_branches(quotients[0].copy(), m, oppos"
                "ite.copy(), p, counts.copy(), False)"
            ),
            "gold_call": (
                '_oracle_build_conditional_branches(quotients_g[0].copy(), m_ref_input, opposite_g.copy(), p_g, counts_ref_input.copy(), False)'
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
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
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
counts = np.array([[1, 0], [0, 1]])
opposite[:, 1::2] = 1

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
counts_g = np.array([[1, 0], [0, 1]])
opposite_g[:, 1::2] = 1
"""
            )
            )
            ),
            "call": (
                "build_conditional_branches(quotients[0].copy(), m, oppos"
                "ite.copy(), p, counts.copy(), False)"
            ),
            "gold_call": (
                '_oracle_build_conditional_branches(quotients_g[0].copy(), m_ref_input, opposite_g.copy(), p_ref_input, counts_g.copy(), False)'
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
logicals_ref_input = copy.deepcopy(logicals)
m_ref_input = copy.deepcopy(m)
n_ref_input = copy.deepcopy(n)
p_ref_input = copy.deepcopy(p)
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
counts[0] = 0


def _raises(fn):
    try:
        fn(
            quotients[0].copy(),
            m,
            opposite.copy(),
            p,
            counts.copy(),
            False,
        )
    except ValueError:
        return 1
    return 0

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
counts_ref_input[0] = 0

def _raises_gold(fn):
    try:
        fn(quotients_g[0].copy(), m_ref_input, opposite_g.copy(), p_ref_input, counts_ref_input.copy(), False)
    except ValueError:
        return 1
    return 0
"""
            )
            )
            )
            ),
            "call": ("_raises(build_conditional_branches)"),
            "gold_call": ('_raises_gold(_oracle_build_conditional_branches)'),
            "tol": 0,
        },
    ]
