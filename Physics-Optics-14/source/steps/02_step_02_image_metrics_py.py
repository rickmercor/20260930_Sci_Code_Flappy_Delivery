"""
Normalized image-quality differentials

For intensity I, bright set B, and target T, mu=mean(I[B]); R=sqrt(mean((I/mu-T)^2)); S=sqrt(mean((I-mu)^2)); eta=A_p sum(I[B])/P_in. Both means inside R and S range over every observation sample. Differentiate with respect to each raw intensity, including the dependence of mu. At R=0 or S=0 use the zero subgradient for that norm. These are the constructed mask’s explicit observation domains for the source Methods objective.

Returns
-------
float ndarray (3,M+1). Rows [RMSE,SD,eta]; column 0 is the metric and columns 1...M its derivatives with respect to raw intensity in observation order. RMSE and eta are dimensionless; SD has intensity units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def image_metrics(
    intensity: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
) -> "np.ndarray":
    """Normalized image-quality differentials.

    For intensity I, bright set B, and target T, mu=mean(I[B]);
    R=sqrt(mean((I/mu-T)^2)); S=sqrt(mean((I-mu)^2)); eta=A_p
    sum(I[B])/P_in. Both means inside R and S range over every observation
    sample. Differentiate with respect to each raw intensity, including the
    dependence of mu. At R=0 or S=0 use the zero subgradient for that norm.
    These are the constructed mask’s explicit observation domains for the
    source Methods objective.

    Parameters
    intensity : float ndarray (M,): nonnegative raw intensities; positive
    bright-region mean.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.

    Returns
    float ndarray (3,M+1). Rows [RMSE,SD,eta]; column 0 is the metric and
    columns 1...M its derivatives with respect to raw intensity in
    observation order. RMSE and eta are dimensionless; SD has intensity
    units.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_image_metrics(
    intensity: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
) -> "np.ndarray":
    intensity = np.asarray(intensity, dtype=float)
    target = np.asarray(target, dtype=float)
    bright = np.asarray(bright, dtype=bool)
    if (
        intensity.ndim != 1
        or len(intensity) == 0
        or target.shape != intensity.shape
        or bright.shape != intensity.shape
        or not bright.any()
    ):
        raise ValueError(
            "Matching nonempty vectors and a nonempty bright region are"
            " required."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [intensity, target, pixel_area, incident_power]
        )
        or np.any(intensity < 0)
        or min(pixel_area, incident_power) <= 0
    ):
        raise ValueError(
            "Nonnegative intensity and positive finite area/power are"
            " required."
        )
    n = len(intensity)
    b = bright.astype(float) / bright.sum()
    mu = b @ intensity
    if mu <= 0:
        raise ValueError("Bright-region mean must be positive.")
    q = intensity / mu - target
    rmse = np.sqrt(q @ q / n)
    d = intensity - mu
    sd = np.sqrt(d @ d / n)
    eta = pixel_area * intensity[bright].sum() / incident_power
    dr = (
        (q / mu - b * (q @ intensity) / mu**2) / (n * rmse)
        if rmse > 0
        else np.zeros(n)
    )
    ds = (d - b * d.sum()) / (n * sd) if sd > 0 else np.zeros(n)
    de = pixel_area / incident_power * bright
    return np.column_stack(([rmse, sd, eta], np.array([dr, ds, de])))

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

intensities = np.array([2.0, 0.3, 3.0, 0.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.array([0.2, 1.3, 0.4])
T = np.array([0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.ones(4) * 2
T = np.ones(4)
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.array([2.0, 0.0, 2.0, 0.0])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.ones(4)
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = 10000.0 * np.array([2.0, 0.3, 3.0, 0.1])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.array([0.4, 0.7, 1.2, 0.1])
T = np.array([0.5, 1.0, 0.75, 0.0])
B = np.array([1, 1, 1, 0], bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

intensities = np.array([1.0, 9.0, 1.2, 5.0])
T = np.array([1.0, 0.0, 1.0, 0.0])
B = T.astype(bool)
candidate_args = deepcopy((intensities, T, B, 0.2, 3.0))
oracle_args = deepcopy((intensities, T, B, 0.2, 3.0))
"""
            ),
            "call": "image_metrics(*candidate_args)",
            "gold_call": "_oracle_image_metrics(*oracle_args)",
            "tol": 1e-06,
        },
    ]
