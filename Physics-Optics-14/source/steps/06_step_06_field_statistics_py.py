"""
Local noncircular field statistics

The fabricated field at observation m is the sum of H_mn*amplitude[n]*Z_n over independent unit-modulus phasors. moments[k-1,n]=E[Z_n**k] for k=1,2. Calculate each observation’s own mean and real/imaginary covariance before applying the source’s bivariate-Gaussian closure. The supplied moments may vary between cells. Cross-observation correlations are immaterial to the marginal reliability criterion.

Returns
-------
float ndarray (M,5). Columns [mean U,mean V,var U,cov(U,V),var V], observation order. Means have field-amplitude units and covariance entries have intensity units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def field_statistics(
    operator: "np.ndarray", amplitude: "np.ndarray", moments: "np.ndarray"
) -> "np.ndarray":
    """Local noncircular field statistics.

    The fabricated field at observation m is the sum of
    H_mn*amplitude[n]*Z_n over independent unit-modulus phasors.
    moments[k-1,n]=E[Z_n**k] for k=1,2. Calculate each observation’s own
    mean and real/imaginary covariance before applying the source’s
    bivariate-Gaussian closure. The supplied moments may vary between
    cells. Cross-observation correlations are immaterial to the marginal
    reliability criterion.

    Parameters
    operator : complex ndarray (M,N): source-to-observation field operator.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    moments : complex ndarray (2,N): rows E[Z] and E[Z^2], in source order;
    admissible unit-phasor moments.

    Returns
    float ndarray (M,5). Columns [mean U,mean V,var U,cov(U,V),var V],
    observation order. Means have field-amplitude units and covariance
    entries have intensity units.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_field_statistics(
    operator: "np.ndarray", amplitude: "np.ndarray", moments: "np.ndarray"
) -> "np.ndarray":
    operator = np.asarray(operator, dtype=complex)
    amplitude = np.asarray(amplitude, dtype=float)
    moments = np.asarray(moments, dtype=complex)
    if (
        operator.ndim != 2
        or min(operator.shape) == 0
        or amplitude.shape != (operator.shape[1],)
        or moments.shape != (2, operator.shape[1])
    ):
        raise ValueError("Incompatible field dimensions.")
    if not all(
        np.isfinite(x).all() for x in [operator, amplitude, moments]
    ) or np.any(amplitude < 0):
        raise ValueError("Invalid optical inputs.")
    d = 1 - np.abs(moments[0]) ** 2
    p = moments[1] - moments[0] ** 2
    if np.any(d < -1e-12) or np.any(np.abs(p) > d + 1e-12):
        raise ValueError("Moments do not describe a unit-modulus phasor.")
    # Repair roundoff outside d >= |p|, but preserve every positive d.
    # A small central moment can become significant after optical scaling.
    d = np.maximum(d, 0.0)
    magnitude = np.abs(p)
    outside = magnitude > d
    p[outside] *= d[outside] / magnitude[outside]
    h = operator * amplitude
    mu = h @ moments[0]
    c = np.abs(h) ** 2 @ d
    pseudo = h * h @ p
    return np.array(
        [
            mu.real,
            mu.imag,
            (c + pseudo.real) / 2,
            pseudo.imag / 2,
            (c - pseudo.real) / 2,
        ]
    ).T

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

H = np.array([[1.0, 1j], [0.3 + 0.2j, -0.4j]])
A = np.ones(2)
m = np.array([[1.0, 1.0], [1.0, 1.0]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1.0, 1j], [2.0, -1.0]])
A = np.array([1.0, 0.5])
m = np.zeros((2, 2), complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1.0, 2.0]], dtype=complex)
A = np.ones(2)
m = np.array([[0.8, 0.8], [0.8**4, 0.8**4]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1.0 + 2j, 2.0 - 1j]])
A = np.array([1.0, 0.4])
m = np.array([[0.9, 0.7], [0.9**4, 0.7**4]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1.0, -1.0]], dtype=complex)
A = np.ones(2)
m = np.array([[0.8, 0.8], [0.8**4, 0.8**4]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1.0, 1j], [1j, 1.0]])
A = np.ones(2)
m = np.array([[0.9, 0.6j], [0.9**4, -(0.6**4)]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[0.3 + 0.2j, 0.4 - 0.5j]])
A = np.array([0.2, 2.0])
m = np.array([[0.85, 0.85], [0.85**4, 0.85**4]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

H = np.array([[1j, 0.0, 1.0]])
A = np.array([0.7, 0.0, 1.0])
m = np.array([[0.6, 0.8, 0.9], [0.6**4, 0.8**4, 0.9**4]], complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
"""
            ),
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": """from copy import deepcopy
import numpy as np

H = np.array([[1e8, -1e8]], dtype=complex)
A = np.ones(2)
first = 1.0 - 2.0**-50
m = np.array([[first, first], [1.0, 1.0]], dtype=complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
""",
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": """from copy import deepcopy
import numpy as np

H = np.array([[1e8 + 1e8j, -1e8 - 1e8j]], dtype=complex)
A = np.ones(2)
first = 1.0 - 2.0**-50
m = np.array([[first, first], [1.0, 1.0]], dtype=complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
""",
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": """from copy import deepcopy
import numpy as np

H = np.array([[1e8, -1e8]], dtype=complex)
A = np.ones(2)
first = 1.0 - 2.0**-50
m = np.array(
    [[first, first], [first**2, first**2]], dtype=complex
)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
""",
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": """from copy import deepcopy
import numpy as np

H = np.array(
    [[1e8, 1e8j], [2.0 - 1j, 0.5 + 0.25j]], dtype=complex
)
A = np.ones(2)
m = np.array([[1j, -1.0], [-1.0, 1.0]], dtype=complex)
candidate_args = deepcopy((H, A, m))
oracle_args = deepcopy((H, A, m))
""",
            "call": "field_statistics(*candidate_args)",
            "gold_call": "_oracle_field_statistics(*oracle_args)",
            "tol": 1e-06,
        },
    ]
