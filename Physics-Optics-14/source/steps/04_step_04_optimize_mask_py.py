"""
Independent-pixel APS trajectory

Each candidate starts from the supplied phase with fresh zero first and second moment accumulators. Apply the scheduled APS phase derivative for i=1,...,iterations. Adam has beta1=0.9, beta2=0.999 and epsilon=1e-8 outside the square root; use bias-corrected moments and learning_rate. All pixel phases update simultaneously, in their unwrapped real representation. This step composes aps_gradient.

Returns
-------
float ndarray (N,). Final unwrapped phases in radians, in source order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimize_mask(
    initial_phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    weights: "np.ndarray",
    iterations: int,
    transition: float,
    width: float,
    learning_rate: float,
) -> "np.ndarray":
    """Independent-pixel APS trajectory.

    Each candidate starts from the supplied phase with fresh zero first and
    second moment accumulators. Apply the scheduled APS phase derivative
    for i=1,...,iterations. Adam has beta1=0.9, beta2=0.999 and
    epsilon=1e-8 outside the square root; use bias-corrected moments and
    learning_rate. All pixel phases update simultaneously, in their
    unwrapped real representation. This step composes aps_gradient.

    Parameters
    initial_phase : float ndarray (N,): starting unwrapped cell phases in
    radians.
    operator : complex ndarray (M,N): source-to-observation field operator.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.
    weights : float ndarray (3,): nonnegative maxima [w_RMSE,w_SD,w_Eff].
    iterations : int >=0: number of complete simultaneous phase updates.
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.
    learning_rate : positive float: Adam step size.

    Returns
    float ndarray (N,). Final unwrapped phases in radians, in source order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_optimize_mask(
    initial_phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    weights: "np.ndarray",
    iterations: int,
    transition: float,
    width: float,
    learning_rate: float,
) -> "np.ndarray":
    phase = np.asarray(initial_phase, dtype=float).copy()
    if (
        phase.ndim != 1
        or phase.size == 0
        or not np.isfinite(phase).all()
        or not np.isfinite(
            [iterations, transition, width, learning_rate]
        ).all()
        or int(iterations) != iterations
        or iterations < 0
        or min(width, learning_rate) <= 0
    ):
        raise ValueError("Invalid phase or optimization controls.")
    m = np.zeros_like(phase)
    v = np.zeros_like(phase)
    for i in range(1, int(iterations) + 1):
        g = _oracle_aps_gradient(
            phase,
            operator,
            amplitude,
            target,
            bright,
            pixel_area,
            incident_power,
            i,
            weights,
            transition,
            width,
        )[1:]
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        phase -= (
            learning_rate
            * (m / (1 - 0.9**i))
            / (np.sqrt(v / (1 - 0.999**i)) + 1e-8)
        )
    return phase

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized differential cases."""
    return [
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 0
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 1
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 17
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 9
w = np.array([1.0, 0.4, 0.7])
tr = 4.0
d = 1.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 90
w = np.array([1.0, 1.0, 0.2])
tr = 12.0
d = 4.0
alpha = 0.02
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 4
w = np.array([0.0, 0.0, 1e-12])
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-10,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
A[1] = 0
k = 11
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array(
    [
        [0.5 + 0.3j, -0.2 + 0.4j, 0.3 - 0.7j],
        [0.8 - 0.1j, 0.1 + 0.6j, -0.4 + 0.2j],
        [-0.3 + 0.2j, 0.7 - 0.5j, 0.6 + 0.1j],
        [0.2 + 0.8j, -0.5 - 0.2j, 0.4 - 0.3j],
    ]
)
A = np.array([1.0, 0.7, 0.9])
ph = np.array([0.2, -0.6, 1.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
k = 13
w = np.zeros(3)
tr = 20.0
d = 7.0
alpha = 0.01
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, w, k, tr, d, alpha))
"""
            ),
            "call": "optimize_mask(*candidate_args)",
            "gold_call": "_oracle_optimize_mask(*oracle_args)",
            "tol": 1e-06,
        },
    ]
