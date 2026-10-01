"""
Evaluate the rate curvature of the frozen recovery’s conditional failure odds.

Return the curvature of the conditional failure log odds for the recovery sector selected at the nominal rate. Keep that discrete sector fixed as the rate varies, along with the detector record, measurement flip probability, and fitted reliability table. The final exact measurement fixes the syndrome; the selected cumulative logical sector defines success. Combine the earlier numerical steps.

Returns
-------
A finite float gives the second derivative of the frozen recovery’s conditional failure log odds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_odds_curvature(
    checks: "np.ndarray",
    stabilizers: "np.ndarray",
    logicals: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> float:
    r"""Evaluate the rate curvature of the frozen recovery’s conditional failure odds.

    Parameters
    ----------
    checks : np.ndarray
        Binary $(2,m,n)$ checks, ordered by the error component they detect.
    stabilizers : np.ndarray
        Binary $(2,n-m-2,n)$ component stabilizer generators.
    logicals : np.ndarray
        Binary $(2,2,n)$ ordered component logical generators.
    detectors : np.ndarray
        Binary $(2,T,m)$ fixed detection-event records.
    p : float
        Nominal total depolarizing probability in $(0,3/4]$.
    pm : float
        Fixed measurement probability in $(0,1/2]$.
    counts : np.ndarray
        Fixed nonnegative $(2,2)$ estimated/true table with positive row sums
        and both fitted reliabilities at least one half.
    max_sweeps : int
        Completed-sweep budget from one to eight.
    tie_tol : float
        Nonnegative finite whole-history degeneracy window.

    Returns
    -------
    result : float
        One finite dimensionless second derivative of conditional failure log odds.

    Raises
    ------
    ValueError
        If an upstream generator, record, dimension, probability, reliability or
        iteration contract fails, or the selected success mass is outside $(0,1)$.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_log_odds_curvature(
    checks: "np.ndarray",
    stabilizers: "np.ndarray",
    logicals: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> float:
    h = _binary(checks, 3, "checks")
    s = _binary(stabilizers, 3, "stabilizers")
    logical_rows = _binary(logicals, 3, "logicals")
    d = _binary(detectors, 3, "detectors")
    if any(a.shape[0] != 2 for a in (h, s, logical_rows, d)):
        raise ValueError("all channel axes must contain X then Z")
    if d.shape[2] != h.shape[1]:
        raise ValueError("detector and check dimensions disagree")
    quotients = np.stack(
        [_oracle_build_binary_quotient(h[c], s[c], logical_rows[c]) for c in range(2)]
    )
    constraints = np.stack(
        [
            _oracle_build_detector_constraints(q, h.shape[1], d.shape[1])
            for q in quotients
        ]
    )
    history = _oracle_iterate_spacetime_recovery(
        quotients, constraints, d, p, pm, counts, max_sweeps, tie_tol
    )
    kernel = _oracle_build_joint_transition_jets(quotients, h.shape[1], p)
    posterior = _oracle_propagate_conditioned_jets(constraints, d, kernel, pm)
    ix, iz = history[-1, :, 0].astype(int)
    success, first, second = posterior[:, ix, iz]
    if not 0 < success < 1:
        raise ValueError(
            "selected success probability must lie strictly between zero and one"
        )
    return float(
        -second / (success * (1 - success))
        + (1 - 2 * success) * first**2 / (success**2 * (1 - success) ** 2)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                r"""import numpy as np

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
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 4, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 4, tie_tol)"
            ),
            "tol": 6.519281775072303e-09,
        },
        {
            "setup": (
                r"""import numpy as np

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
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 4, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 4, tie_tol)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                r"""import numpy as np

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
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 4, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 4, tie_tol)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                r"""import numpy as np

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
p = 0.75
pm = 0.5
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 4, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 4, tie_tol)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                r"""import numpy as np

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
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 1, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 1, tie_tol)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                r"""import numpy as np

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
detectors = detectors[:, :, :2]


def _raises(fn):
    try:
        fn(
            checks.copy(),
            stabilizers.copy(),
            logicals.copy(),
            detectors.copy(),
            p,
            pm,
            counts.copy(),
            4,
            tie_tol,
        )
    except ValueError:
        return 1
    return 0
"""
            ),
            "call": ("_raises(compute_log_odds_curvature)"),
            "gold_call": ("_raises(_oracle_compute_log_odds_curvature)"),
            "tol": 0,
        },
        {
            "setup": (
                r"""import numpy as np

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
tie_tol = 1e-9
"""
            ),
            "call": (
                "compute_log_odds_curvature(checks.copy(), stabilizers.co"
                "py(), logicals.copy(), detectors.copy(), p, pm, counts.c"
                "opy(), 4, tie_tol)"
            ),
            "gold_call": (
                "_oracle_compute_log_odds_curvature(checks.copy(), stabil"
                "izers.copy(), logicals.copy(), detectors.copy(), p, pm, "
                "counts.copy(), 4, tie_tol)"
            ),
            "tol": 1e-08,
        },
    ]
