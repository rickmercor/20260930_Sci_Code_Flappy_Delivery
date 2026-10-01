"""
Condition joint logical-sector probability derivatives on the full temporal record.

Return conditional probabilities for all sixteen terminal logical sectors and their first two ordinary rate derivatives. Hold the detector record, measurement flip probability, and logical-sector definition fixed under differentiation. The final measurement is exact; no posterior class is selected by this function.

Returns
-------
A float array of shape (3, 4, 4) contains normalized logical-sector probabilities and their first two rate derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_conditioned_jets(
    constraints: "np.ndarray", detectors: "np.ndarray", kernel: "np.ndarray", pm: float
) -> "np.ndarray":
    r"""Condition joint logical-sector probability derivatives on the full temporal
    record.

    Parameters
    ----------
    constraints : np.ndarray
        Two matching binary temporal matrices of shape $(2,Tm+2,Tn+(T-1)m)$.
    detectors : np.ndarray
        Fixed binary records of shape $(2,T,m)$, with one to eight intervals.
    kernel : np.ndarray
        Finite joint derivative array $(3,2^{2(m+2)})$; row zero is nonnegative
        and row sums equal $(1,0,0)$ to absolute tolerance $10^{-9}$.
    pm : float
        Fixed measurement flip probability in $(0,1/2]$.

    Returns
    -------
    result : np.ndarray
        Float array $(3,4,4)$ of conditional sector probabilities, first derivatives
        and second derivatives, with logical axes ordered $X,Z$.

    Raises
    ------
    ValueError
        If dimensions, temporal blocks, binary records, kernel normalization or
        measurement rate is invalid, or the detector record has zero evidence.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xor_convolution(a, b):
    # Positive zeroth-order sums avoid cancellation in rare detector fibers.
    index = np.arange(a.size)
    out = np.zeros_like(a)
    for shift in range(a.size):
        if b[shift] != 0:
            out += b[shift] * a[index ^ shift]
    return out


def _oracle_propagate_conditioned_jets(
    constraints: "np.ndarray", detectors: "np.ndarray", kernel: "np.ndarray", pm: float
) -> "np.ndarray":
    aa = _binary(constraints, 3, "constraints")
    dd = _binary(detectors, 3, "detectors")
    if aa.shape[0] != 2 or dd.shape[0] != 2:
        raise ValueError("two channels are required")
    _, dx, tmax, m, n = _detector_layout(aa[0], dd[0])
    _, dz, tz, mz, nz = _detector_layout(aa[1], dd[1])
    if (tmax, m, n) != (tz, mz, nz):
        raise ValueError("channel dimensions disagree")
    pm = _prob(pm, 0.5, "pm")
    width = m + 2
    size = 1 << (2 * width)
    k = _real(kernel, "kernel")
    if (
        k.shape != (3, size)
        or np.any(k[0] < 0)
        or not np.allclose(k.sum(axis=1), [1, 0, 0], atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "kernel must contain normalized probability and two derivative rows"
        )
    records = np.bitwise_xor.accumulate(dd, axis=1)
    states = np.arange(size)
    x = states >> width
    z = states & ((1 << width) - 1)
    sx = _words(m)[x >> 2]
    sz = _words(m)[z >> 2]
    values = np.zeros((3, size))
    values[0, 0] = 1
    for t in range(tmax):
        prediction = np.zeros_like(values)
        prediction[0] = _xor_convolution(values[0], k[0])
        prediction[1] = _xor_convolution(values[1], k[0]) + _xor_convolution(
            values[0], k[1]
        )
        prediction[2] = (
            _xor_convolution(values[2], k[0])
            + 2 * _xor_convolution(values[1], k[1])
            + _xor_convolution(values[0], k[2])
        )
        if t < tmax - 1:
            mistakes = np.sum(sx ^ records[0, t], axis=1) + np.sum(
                sz ^ records[1, t], axis=1
            )
            evidence = pm**mistakes * (1 - pm) ** (2 * m - mistakes)
        else:
            evidence = np.all(sx == records[0, t], axis=1) & np.all(
                sz == records[1, t], axis=1
            )
        weighted = prediction * evidence
        scale = weighted.sum(axis=1)
        if scale[0] <= 0:
            raise ValueError("detector record has zero evidence")
        values[0] = weighted[0] / scale[0]
        values[1] = (weighted[1] - values[0] * scale[1]) / scale[0]
        values[2] = (
            weighted[2] - 2 * values[1] * scale[1] - values[0] * scale[2]
        ) / scale[0]
    labels = 4 * (x & 3) + (z & 3)
    return np.stack(
        [np.bincount(labels, weights=row, minlength=16).reshape(4, 4) for row in values]
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
kernel = build_joint_transition_jets(
    quotients, m, p
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
kernel_g = _oracle_build_joint_transition_jets(quotients_g, m_ref_input, p_ref_input)
"""
            )
            )
            ),
            "call": (
                "propagate_conditioned_jets(constraints.copy(), detectors"
                ".copy(), kernel.copy(), pm)"
            ),
            "gold_call": (
                '_oracle_propagate_conditioned_jets(constraints_g.copy(), detectors_ref_input.copy(), kernel_g.copy(), pm_ref_input)'
            ),
            "tol": 1e-08,
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
kernel = build_joint_transition_jets(
    quotients, m, p
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
kernel_g = _oracle_build_joint_transition_jets(quotients_g, m_ref_input, p_ref_input)
"""
            )
            )
            ),
            "call": (
                "propagate_conditioned_jets(constraints.copy(), detectors"
                ".copy(), kernel.copy(), pm)"
            ),
            "gold_call": (
                '_oracle_propagate_conditioned_jets(constraints_g.copy(), detectors_ref_input.copy(), kernel_g.copy(), pm_ref_input)'
            ),
            "tol": 1e-08,
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
kernel = build_joint_transition_jets(
    quotients, m, p
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
kernel_g = _oracle_build_joint_transition_jets(quotients_g, m_ref_input, p_ref_input)
"""
            )
            )
            ),
            "call": (
                "propagate_conditioned_jets(constraints.copy(), detectors"
                ".copy(), kernel.copy(), pm)"
            ),
            "gold_call": (
                '_oracle_propagate_conditioned_jets(constraints_g.copy(), detectors_ref_input.copy(), kernel_g.copy(), pm_ref_input)'
            ),
            "tol": 1e-08,
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
detectors_ref_input = copy.deepcopy(detectors)
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
pm = 0.5
kernel = build_joint_transition_jets(
    quotients, m, p
)

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
p_g = 0.75
pm_g = 0.5
kernel_g = _oracle_build_joint_transition_jets(quotients_g, m_ref_input, p_g)
"""
            )
            )
            ),
            "call": (
                "propagate_conditioned_jets(constraints.copy(), detectors"
                ".copy(), kernel.copy(), pm)"
            ),
            "gold_call": (
                '_oracle_propagate_conditioned_jets(constraints_g.copy(), detectors_ref_input.copy(), kernel_g.copy(), pm_g)'
            ),
            "tol": 1e-08,
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
kernel = build_joint_transition_jets(
    quotients, m, p
)
kernel[0] *= 2


def _raises(fn):
    try:
        fn(
            constraints.copy(),
            detectors.copy(),
            kernel.copy(),
            pm,
        )
    except ValueError:
        return 1
    return 0

# Independent reference inputs from the same primitive instance.
quotients_g = np.stack([_oracle_build_binary_quotient(checks_ref_input[c_g], stabilizers_ref_input[c_g], logicals_ref_input[c_g]) for c_g in range(2)])
constraints_g = np.stack([_oracle_build_detector_constraints(q_g, m_ref_input, T_ref_input) for q_g in quotients_g])
opposite_g = np.zeros((T_ref_input, n_ref_input), dtype=int)
kernel_g = _oracle_build_joint_transition_jets(quotients_g, m_ref_input, p_ref_input)
kernel_bad_g = kernel_g * np.array([[2.0], [1.0], [1.0]])

def _raises_gold(fn):
    try:
        fn(constraints_g.copy(), detectors_ref_input.copy(), kernel_bad_g.copy(), pm_ref_input)
    except ValueError:
        return 1
    return 0
"""
            )
            )
            )
            ),
            "call": ("_raises(propagate_conditioned_jets)"),
            "gold_call": ('_raises_gold(_oracle_propagate_conditioned_jets)'),
            "tol": 0,
        },
    ]
