"""
Select and certify the hologram

Compose every preceding public step, directly or transitively, with all intermediate results used. Design each candidate at design_z; certify its quantized, perturbed mask at planes. A candidate is eligible when minimum plane efficiency >= eta_min and its exposure interval has positive width. Maximize log(upper/lower), choosing the earliest candidate at an exact tie. Return -1.0 if none is eligible. The single scalar is the selected logarithmic exposure latitude. This design-selection and reliability certificate is a task-defined extension of the source method.

Returns
-------
float. Selected dimensionless log(upper/lower), or -1.0 when no candidate is eligible.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certify_hologram(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    amplitude: "np.ndarray",
    initial_phase: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    wavelength: float,
    design_z: float,
    planes: "np.ndarray",
    source_area: float,
    pixel_area: float,
    weights: "np.ndarray",
    rotation_sigma: "np.ndarray",
    levels: int,
    iterations: int,
    eta_min: float,
    dose_cap: float,
    tail_probability: float = 0.05,
    transition: float = 175.0,
    width: float = 40.0,
    learning_rate: float = 0.005,
) -> float:
    """Select and certify the hologram.

    Compose every preceding public step, directly or transitively, with all
    intermediate results used. Design each candidate at design_z; certify
    its quantized, perturbed mask at planes. A candidate is eligible when
    minimum plane efficiency >= eta_min and its exposure interval has
    positive width. Maximize log(upper/lower), choosing the earliest
    candidate at an exact tie. Return -1.0 if none is eligible. The single
    scalar is the selected logarithmic exposure latitude. This
    design-selection and reliability certificate is a task-defined
    extension of the source method.

    Parameters
    source_xy : float ndarray (N,2): source-cell centers [x,y],
    micrometres.
    target_xy : float ndarray (M,2): observation centers [x,y],
    micrometres, in target order.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    initial_phase : float ndarray (N,): starting unwrapped cell phases in
    radians.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    At least one dark observation is also required.
    wavelength : positive float: wavelength in micrometres.
    design_z : positive float: design-plane distance in micrometres.
    planes : float ndarray (Z,): positive certification distances in
    micrometres, ordered as supplied.
    source_area : positive float: one source-cell area in square
    micrometres.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    weights : float ndarray (C,3): candidate rows [w_RMSE,w_SD,w_Eff] in
    selection order.
    rotation_sigma : float ndarray (N,): nonnegative standard deviation of
    nanobrick rotation error in radians.
    levels : int >=2: number of uniformly spaced manufactured phase levels.
    iterations : int >=0: number of complete simultaneous phase updates.
    eta_min : nonnegative float: minimum allowed worst-plane diffraction
    efficiency.
    dose_cap : positive float: maximum dimensionless normalized exposure.
    tail_probability : float in (0,0.5): allowed marginal error probability
    (default 0.05).
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.
    learning_rate : positive float: Adam step size.

    Returns
    float. Selected dimensionless log(upper/lower), or -1.0 when no
    candidate is eligible.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certify_hologram(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    amplitude: "np.ndarray",
    initial_phase: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    wavelength: float,
    design_z: float,
    planes: "np.ndarray",
    source_area: float,
    pixel_area: float,
    weights: "np.ndarray",
    rotation_sigma: "np.ndarray",
    levels: int,
    iterations: int,
    eta_min: float,
    dose_cap: float,
    tail_probability: float = 0.05,
    transition: float = 175.0,
    width: float = 40.0,
    learning_rate: float = 0.005,
) -> float:
    amplitude = np.asarray(amplitude, dtype=float)
    weights = np.asarray(weights, dtype=float)
    planes = np.asarray(planes, dtype=float)
    if (
        weights.ndim != 2
        or weights.shape[1] != 3
        or len(weights) == 0
        or planes.ndim != 1
        or len(planes) == 0
        or not np.isfinite([eta_min, tail_probability]).all()
        or eta_min < 0
        or not 0 < tail_probability < 0.5
    ):
        raise ValueError(
            "Invalid candidate set, efficiency floor or tail probability."
        )
    pin = source_area * np.sum(amplitude**2)
    H = _oracle_rs_operator(
        source_xy, target_xy, wavelength, design_z, source_area
    )
    propagation = [
        _oracle_rs_operator(source_xy, target_xy, wavelength, z, source_area)
        for z in planes
    ]
    best = -1.0
    for w in weights:
        phi = _oracle_optimize_mask(
            initial_phase,
            H,
            amplitude,
            target,
            bright,
            pixel_area,
            pin,
            w,
            iterations,
            transition,
            width,
            learning_rate,
        )
        moments = _oracle_fabrication_moments(phi, levels, rotation_sigma)
        stats = np.array(
            [
                _oracle_field_statistics(op, amplitude, moments)
                for op in propagation
            ]
        )
        qs = np.array(
            [
                _oracle_intensity_quantiles(
                    s, np.array([tail_probability, 1 - tail_probability])
                )
                for s in stats
            ]
        )
        lo, hi, eta = _oracle_exposure_window(
            stats, qs, bright, pixel_area, pin, dose_cap
        )
        if eta >= eta_min and hi > lo:
            score = float(np.log(hi / lo))
            if score > best:
                best = score
    return float(best)

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

yy, xx = np.indices((16, 16))
src = np.column_stack((xx.ravel() - 7.5, yy.ravel() - 7.5)) * 0.25
vv, uu = np.indices((7, 7))
u = uu - 3
v = vv - 3
dst = np.column_stack((u.ravel(), v.ravel())) * 0.5
B = (
    (np.abs(u) <= 2) & (np.abs(v) <= 2) & ~((u >= 0) & (v >= 0))
    | (u == 2) & (v == 2)
).ravel()
T = B.astype(float)
j = np.arange(256)
A = np.exp(-((src[:, 0] - 0.11) ** 2 + (src[:, 1] + 0.07) ** 2) / 2.2**2)
ph = 0.35 * np.sin(0.71 * j) + 0.17 * np.cos(0.33 * j)
rot = 0.044 + 0.014 * (1 + np.sin(0.27 * j))
W = np.array(
    [
        [1.0, 0.12, 1.0],
        [1.0, 0.35, 1.0],
        [1.0, 1.0, 0.1],
        [1.0, 0.7, 0.65],
        [1.0, 0.3, 0.2],
        [1.0, 0.0, 1.0],
    ]
)
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    4.0,
    np.array([3.95, 4.0, 4.05]),
    0.0625,
    0.25,
    W,
    rot,
    128,
    600,
    0.22,
    40.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    350,
    0.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    350,
    2.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    350,
    0.0,
    0.01,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.0)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    370,
    0.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    2,
    410,
    0.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.633,
    1.3,
    np.array([1.26, 1.33]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    400,
    0.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

yy, xx = np.indices((8, 8))
src = (
    np.column_stack((xx.ravel() - (8 - 1) / 2, yy.ravel() - (8 - 1) / 2))
    * 0.25
)
vv, uu = np.indices((5, 5))
u = uu - (5 - 1) / 2
v = vv - (5 - 1) / 2
dst = np.column_stack((u.ravel(), v.ravel())) * 0.45
B = ((u < 0) & (np.abs(v) <= 1) | (v < 0) & (np.abs(u) <= 1)).ravel()
T = B.astype(float)
j = np.arange(64)
A = np.exp(-((src[:, 0] - 0.05) ** 2 + (src[:, 1] + 0.09) ** 2) / 1.7**2)
ph = 0.25 * np.sin(0.63 * j + 0.19) + 0.19 * np.cos(0.29 * j)
rot = np.full(64, 0.045)
W = np.array([[1.0, 0.2, 0.9], [1.0, 0.8, 0.1]])
args = (
    src,
    dst,
    A,
    ph,
    T,
    B,
    0.405,
    2.0,
    np.array([1.96, 2.03]),
    0.0625,
    0.2025,
    W,
    rot,
    32,
    400,
    0.0,
    500.0,
)
candidate_args = deepcopy((*args,))
oracle_args = deepcopy((*args,))
"""
            ),
            "call": "certify_hologram(*candidate_args)",
            "gold_call": "_oracle_certify_hologram(*oracle_args)",
            "tol": 2e-05,
        },
    ]
