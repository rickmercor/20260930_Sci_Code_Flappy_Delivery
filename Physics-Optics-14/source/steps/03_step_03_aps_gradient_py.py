"""
Scheduled phase-probability objective

At one-based iteration i, s=(1+tanh((i-transition)/width))/2. The objective is s*w[0]*R+(1-s)*w[1]*S+(1-s)*w[2]*(1-eta), evaluated on E=H@(amplitude*exp(1j*phase)). Return its phase derivative, retaining the intensity normalization’s dependence on phase. The same iteration index controls the objective and the Adam update.

Returns
-------
float ndarray (N+1,). Entry 0 is the scheduled objective; entries 1...N are its phase derivatives in source order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aps_gradient(
    phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    iteration: int,
    weights: "np.ndarray",
    transition: float,
    width: float,
) -> "np.ndarray":
    """Scheduled phase-probability objective.

    At one-based iteration i, s=(1+tanh((i-transition)/width))/2. The
    objective is s*w[0]*R+(1-s)*w[1]*S+(1-s)*w[2]*(1-eta), evaluated on
    E=H@(amplitude*exp(1j*phase)). Return its phase derivative, retaining
    the intensity normalization’s dependence on phase. The same iteration
    index controls the objective and the Adam update.

    Parameters
    phase : float ndarray (N,): cell phases in radians, in source order.
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
    iteration : int >=1: one-based objective and update index.
    weights : float ndarray (3,): nonnegative maxima [w_RMSE,w_SD,w_Eff].
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.

    Returns
    float ndarray (N+1,). Entry 0 is the scheduled objective; entries 1...N
    are its phase derivatives in source order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_aps_gradient(
    phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    iteration: int,
    weights: "np.ndarray",
    transition: float,
    width: float,
) -> "np.ndarray":
    phase = np.asarray(phase, dtype=float)
    operator = np.asarray(operator, dtype=complex)
    amplitude = np.asarray(amplitude, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        phase.ndim != 1
        or amplitude.shape != phase.shape
        or operator.shape != (len(target), len(phase))
        or weights.shape != (3,)
    ):
        raise ValueError("Incompatible optical arrays or weights.")
    if (
        not all(
            np.isfinite(x).all()
            for x in [
                phase,
                operator,
                amplitude,
                weights,
                iteration,
                transition,
                width,
            ]
        )
        or np.any(amplitude < 0)
        or np.any(weights < 0)
        or iteration < 1
        or int(iteration) != iteration
        or width <= 0
    ):
        raise ValueError("Invalid phase, schedule, amplitude or weights.")
    p = amplitude * np.exp(1j * phase)
    field = operator @ p
    met = _oracle_image_metrics(
        np.abs(field) ** 2, target, bright, pixel_area, incident_power
    )
    s = 0.5 * (1 + np.tanh((iteration - transition) / width))
    coef = np.array(
        [s * weights[0], (1 - s) * weights[1], -(1 - s) * weights[2]]
    )
    di = coef @ met[:, 1:]
    gradient = 2 * np.imag(np.conj(p) * (operator.conj().T @ (di * field)))
    return np.r_[coef @ met[:, 0] + (1 - s) * weights[2], gradient]

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
i = 1
w = np.array([1.0, 1.0, 0.1])
tr = 175.0
d = 40.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
i = 800
w = np.array([1.0, 0.4, 0.9])
tr = 175.0
d = 40.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
i = 175
w = np.array([1.0, 0.3, 0.8])
tr = 175.0
d = 40.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
i = 1
w = np.array([0.0, 0.0, 1.0])
tr = 175.0
d = 40.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
ph = ph + 2.4
i = 13
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
A[1] = 0
i = 13
w = np.array([1.0, 0.4, 0.7])
tr = 20.0
d = 7.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
H[1] *= 3
i = 25
w = np.array([1.0, 0.9, 0.7])
tr = 20.0
d = 7.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
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
H = H.conj()
i = 2
w = np.array([1.0, 0.2, 0.8])
tr = 20.0
d = 7.0
candidate_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
oracle_args = deepcopy((ph, H, A, T, B, 0.2, 3.0, i, w, tr, d))
"""
            ),
            "call": "aps_gradient(*candidate_args)",
            "gold_call": "_oracle_aps_gradient(*oracle_args)",
            "tol": 1e-06,
        },
    ]
