"""
Infer the finite pole-plus-zero count from the numerical rank of a Hankel probe.

The finite-spectrum construction identifies the total number d of poles plus zeros with
the rank of the structured Hankel matrix. Numerically, count singular values satisfying
sigma_i > rtol * sigma_max.

Parameters
----------
hankel_probe : np.ndarray
    Nonempty finite real square matrix.
rtol : float
    Positive finite relative singular-value threshold.

Returns
-------
d : int
    Numerical rank of the probe.

Raises
------
ValueError
    If the probe is not a finite real nonempty square matrix or rtol is not
    strictly positive and finite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def infer_finite_spectrum_size(hankel_probe: "np.ndarray", rtol: float) -> int:
    """Return the numerical Hankel rank using sigma_i > rtol*sigma_max."""
    return 0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_infer_finite_spectrum_size(hankel_probe: "np.ndarray", rtol: float) -> int:
    """Return the numerical Hankel rank using sigma_i > rtol*sigma_max."""
    raw = np.asarray(hankel_probe)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("hankel_probe must be real-valued")
    try:
        matrix = np.asarray(raw, dtype=float)
        tol = float(rtol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hankel_probe and rtol must be real numeric values") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("hankel_probe must be a finite nonempty square matrix")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rtol must be strictly positive and finite")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > tol * singular_values[0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431,-0.331672505490807096899101287418561408111207028195519199,-0.321904529323547068016655331121902730433961992623601257581247]); H=np.array([[c[1+i+j] for j in range(5)] for i in range(5)])",
            "call": "infer_finite_spectrum_size(H,1e-10)",
            "gold_call": "_oracle_infer_finite_spectrum_size(H,1e-10)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.5,0.25,0.125,0.0625,0.03125]); H=np.array([[c[1+i+j] for j in range(2)] for i in range(2)])",
            "call": "infer_finite_spectrum_size(H,1e-10)",
            "gold_call": "_oracle_infer_finite_spectrum_size(H,1e-10)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.1,0.12,0.10825,0.087,0.065700625,0.04773825,0.03379793140625]); H=np.array([[c[1+i+j] for j in range(3)] for i in range(3)])",
            "call": "infer_finite_spectrum_size(H,1e-10)",
            "gold_call": "_oracle_infer_finite_spectrum_size(H,1e-10)",
        },
        {
            "setup": "import numpy as np\nH=np.zeros((2,2), dtype=float)",
            "call": "infer_finite_spectrum_size(H,1e-10)",
            "gold_call": "_oracle_infer_finite_spectrum_size(H,1e-10)",
            "tol": 0.0,
        },
    ]
