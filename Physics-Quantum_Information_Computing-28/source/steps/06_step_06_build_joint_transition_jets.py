"""
Form the correlated Pauli quotient kernel and its first two rate derivatives.

Return the joint quotient transition probabilities and their first two ordinary rate derivatives for the supplied correlated data-fault model. Pack the joint identifier with the X-channel quotient preceding the Z-channel quotient. The derivative arrays have total masses one, zero, and zero. All quotient coordinates use the supplied binary bases.

Returns
-------
A float array contains the joint quotient transition probabilities and their first two ordinary derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_joint_transition_jets(
    quotients: "np.ndarray", check_count: int, p: float
) -> "np.ndarray":
    r"""Form the correlated Pauli quotient kernel and its first two rate derivatives.

    Parameters
    ----------
    quotients : np.ndarray
        Binary inverse pairs of shape $(2,2,n,n)$ for the two physical components.
    check_count : int
        Independent check count $m$ from one to three.
    p : float
        Total depolarizing probability in $(0,3/4]$.

    Returns
    -------
    result : np.ndarray
        Float array $(3,2^{2(m+2)})$ with kernel value, first derivative and
        second derivative with respect to the same scalar $p$.

    Raises
    ------
    ValueError
        If the two inverse pairs, their common dimensions, check count or rate
        violates the stated contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_joint_transition_jets(
    quotients: "np.ndarray", check_count: int, p: float
) -> "np.ndarray":
    qq = _binary(quotients, 4, "quotients")
    if qq.shape[0] != 2:
        raise ValueError("quotients must have two channels")
    qx, m = _quotient(qq[0], check_count)
    qz, _ = _quotient(qq[1], check_count)
    if qx.shape != qz.shape:
        raise ValueError("channel quotient dimensions disagree")
    p = _prob(p, 0.75, "p")
    n = qx.shape[1]
    width = m + 2
    count = 1 << (2 * width)
    weights = 1 << np.arange(width - 1, -1, -1)
    xlabels = qx[1, :, :width] @ weights
    zlabels = qz[1, :, :width] @ weights
    result = np.zeros((3, count))
    result[0, 0] = 1
    states = np.arange(count)
    for edge in range(n):
        shifts = (
            0,
            int(xlabels[edge]) << width,
            (int(xlabels[edge]) << width) ^ int(zlabels[edge]),
            int(zlabels[edge]),
        )
        updated = np.zeros_like(result)
        for pauli, shift in enumerate(shifts):
            probability = 1 - p if pauli == 0 else p / 3
            derivative = -1.0 if pauli == 0 else 1 / 3
            values = result[:, states ^ shift]
            updated[0] += probability * values[0]
            updated[1] += probability * values[1] + derivative * values[0]
            updated[2] += probability * values[2] + 2 * derivative * values[1]
        result = updated
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
p = 0.08

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.08
"""
            )
            )
            ),
            "call": ("build_joint_transition_jets(quotients.copy(), m, p)"),
            "gold_call": (
                '_oracle_build_joint_transition_jets(quotients_g.copy(), m_ref_input, p_g)'
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
            "call": ("build_joint_transition_jets(quotients.copy(), m, p)"),
            "gold_call": (
                '_oracle_build_joint_transition_jets(quotients_g.copy(), m_ref_input, p_g)'
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
detectors = np.array([[[1]], [[0]]], dtype=int)
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
p = 0.17

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.17
"""
            )
            )
            ),
            "call": ("build_joint_transition_jets(quotients.copy(), m, p)"),
            "gold_call": (
                '_oracle_build_joint_transition_jets(quotients_g.copy(), m_ref_input, p_g)'
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
detectors = np.array([[[1]], [[0]]], dtype=int)
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
p = 1e-05

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 1e-05
"""
            )
            )
            ),
            "call": ("build_joint_transition_jets(quotients.copy(), m, p)"),
            "gold_call": (
                '_oracle_build_joint_transition_jets(quotients_g.copy(), m_ref_input, p_g)'
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


def _raises(fn):
    try:
        fn(quotients.copy(), m, 0.0)
    except ValueError:
        return 1
    return 0

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)

def _raises_gold(fn):
    try:
        fn(quotients_g.copy(), m_ref_input, 0.0)
    except ValueError:
        return 1
    return 0
"""
            )
            )
            )
            ),
            "call": ("_raises(build_joint_transition_jets)"),
            "gold_call": ('_raises_gold(_oracle_build_joint_transition_jets)'),
            "tol": 0,
        },
    ]
