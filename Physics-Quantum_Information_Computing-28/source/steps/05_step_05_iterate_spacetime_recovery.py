"""
Propagate reliability between two complete spacetime recovery problems.

Return the completed alternating recovery history within the supplied sweep budget. The first data estimate uses the marginal model; subsequent estimates use the supplied conditional reliability table. Convergence means equality of both complete physical data histories after a completed sweep, not equality of logical labels alone. Return the last completed iterate at the budget.

Returns
-------
A float array contains the complete ordered channel recovery records for every completed sweep.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def iterate_spacetime_recovery(
    quotients: "np.ndarray",
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> "np.ndarray":
    r"""Propagate reliability between two complete spacetime recovery problems.

    Parameters
    ----------
    quotients : np.ndarray
        Binary array $(2,2,n,n)$, channels $X,Z$, each basis followed by its inverse.
    constraints : np.ndarray
        Binary array $(2,Tm+2,Tn+(T-1)m)$ matching the quotients.
    detectors : np.ndarray
        Binary array $(2,T,m)$, $X$-detected and $Z$-detected records.
    p : float
        Total data depolarizing rate in $(0,3/4]$.
    pm : float
        Measurement flip probability in $(0,1/2]$.
    counts : np.ndarray
        Fixed $(2,2)$ estimate/true table with valid nonnegative entries and
        reliabilities.
    max_sweeps : int
        Positive completed-sweep budget, at most eight.
    tie_tol : float
        Finite nonnegative whole-history energy window.

    Returns
    -------
    result : np.ndarray
        Float array $(N,2,6+Tn+(T-1)m)$ containing every completed sweep,
        with channel records in the single-solve format and repeated terminal sweep
        included.

    Raises
    ------
    ValueError
        If channel dimensions, quotient/constraint identities, binary records,
        rates, counts, sweep budget or energy tolerance violates its contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_iterate_spacetime_recovery(
    quotients: "np.ndarray",
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> "np.ndarray":
    qq = _binary(quotients, 4, "quotients")
    aa = _binary(constraints, 3, "constraints")
    dd = _binary(detectors, 3, "detectors")
    if qq.shape[0] != 2 or aa.shape[0] != 2 or dd.shape[0] != 2:
        raise ValueError("channel axis must contain X then Z")
    _, _, tmax, m, n = _detector_layout(aa[0], dd[0])
    if dd.shape != (2, tmax, m) or qq.shape != (2, 2, n, n):
        raise ValueError("channel dimensions disagree")
    for channel in range(2):
        expected = _oracle_build_detector_constraints(qq[channel], m, tmax)
        if not np.array_equal(aa[channel], expected):
            raise ValueError("constraints disagree with quotient")
    budget = _int(max_sweeps, 1, 8, "max_sweeps")
    tol = _real(tie_tol, "tie_tol")
    if tol.ndim or tol < 0:
        raise ValueError("tie_tol must be a nonnegative finite scalar")
    previous = np.zeros((2, tmax, n), dtype=int)
    history = []
    for sweep in range(budget):
        pair = []
        current = previous.copy()
        for channel in range(2):
            opposite = previous[1] if channel == 0 else current[0]
            branches = _oracle_build_conditional_branches(
                qq[channel], m, opposite, p, counts, sweep == 0 and channel == 0
            )
            record = _oracle_solve_spacetime_sectors(
                aa[channel], dd[channel], branches, pm, tie_tol
            )
            pair.append(record)
            current[channel] = record[6 : 6 + tmax * n].reshape(tmax, n).astype(int)
        history.append(np.stack(pair))
        if sweep > 0 and np.array_equal(current, previous):
            break
        previous = current
    return np.stack(history)

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

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
"""
            )
            )
            ),
            "call": (
                "iterate_spacetime_recovery(quotients.copy(), constraints"
                ".copy(), detectors.copy(), p, pm, counts.copy(), 4, tie_"
                "tol)"
            ),
            "gold_call": (
                '_oracle_iterate_spacetime_recovery(quotients_g.copy(), constraints_g.copy(), detectors_ref_input.copy(), p_ref_input, pm_ref_input, counts_ref_input.copy(), 4, tie_tol_ref_input)'
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

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
"""
            )
            )
            ),
            "call": (
                "iterate_spacetime_recovery(quotients.copy(), constraints"
                ".copy(), detectors.copy(), p, pm, counts.copy(), 1, tie_"
                "tol)"
            ),
            "gold_call": (
                '_oracle_iterate_spacetime_recovery(quotients_g.copy(), constraints_g.copy(), detectors_ref_input.copy(), p_ref_input, pm_ref_input, counts_ref_input.copy(), 1, tie_tol_ref_input)'
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

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
"""
            )
            )
            ),
            "call": (
                "iterate_spacetime_recovery(quotients.copy(), constraints"
                ".copy(), detectors.copy(), p, pm, counts.copy(), 4, tie_"
                "tol)"
            ),
            "gold_call": (
                '_oracle_iterate_spacetime_recovery(quotients_g.copy(), constraints_g.copy(), detectors_ref_input.copy(), p_ref_input, pm_ref_input, counts_ref_input.copy(), 4, tie_tol_ref_input)'
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

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.75
pm_g = 0.5
"""
            )
            )
            ),
            "call": (
                "iterate_spacetime_recovery(quotients.copy(), constraints"
                ".copy(), detectors.copy(), p, pm, counts.copy(), 4, tie_"
                "tol)"
            ),
            "gold_call": (
                '_oracle_iterate_spacetime_recovery(quotients_g.copy(), constraints_g.copy(), detectors_ref_input.copy(), p_g, pm_g, counts_ref_input.copy(), 4, tie_tol_ref_input)'
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


def _raises(fn):
    try:
        fn(
            quotients.copy(),
            constraints.copy(),
            detectors.copy(),
            p,
            pm,
            counts.copy(),
            0,
            tie_tol,
        )
    except ValueError:
        return 1
    return 0

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)

def _raises_gold(fn):
    try:
        fn(quotients_g.copy(), constraints_g.copy(), detectors_ref_input.copy(), p_ref_input, pm_ref_input, counts_ref_input.copy(), 0, tie_tol_ref_input)
    except ValueError:
        return 1
    return 0
"""
            )
            )
            )
            ),
            "call": ("_raises(iterate_spacetime_recovery)"),
            "gold_call": ('_raises_gold(_oracle_iterate_spacetime_recovery)'),
            "tol": 0,
        },
    ]
