"""
Compute the variance coefficient for a signed uniform quantizer.



For signed bit width `$b >= 2$`, the largest positive index is

`$q_b = 2**(b - 1) - 1$`. Non-overloading subtractive dither gives a uniform

error over one step, so a group with range `$R$` has variance

`$c_b * R**2$`, where `$c_b = 1 / (12 * q_b**2)$`.

Returns
-------
one finite positive float equal to 1 / (12 * (2**(bits - 1) - 1)**2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_dither_variance_coefficient(bits: int) -> float:
    """Return the signed-quantizer dither variance coefficient.

    Raises ``ValueError`` unless ``bits`` is an integer of at least 2 and the
    resulting coefficient is finite and strictly positive. NumPy integer
    scalars such as ``np.int64(11)`` must be accepted as integers. Boolean
    values, including ``np.bool_``, are not accepted as integers.

    Parameters
    ----------
    bits : int
        Signed quantizer bit width.

    Returns
    -------
    float
        The finite coefficient ``1 / (12 * (2**(bits - 1) - 1)**2)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_dither_variance_coefficient(bits: int) -> float:
    """Reference coefficient with strict signed-bit validation."""
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)):
        raise ValueError("bits must be an integer at least 2")
    if int(bits) < 2:
        raise ValueError("bits must be an integer at least 2")
    level = 2 ** (int(bits) - 1) - 1
    try:
        coefficient = 1.0 / (12.0 * float(level) ** 2)
    except OverflowError as exc:
        raise ValueError("bits must produce a finite positive coefficient") from exc
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("bits must produce a finite positive coefficient")
    return float(coefficient)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return multiple precisions, NumPy integers, and two invalid cases."""
    return [
        {
            "setup": "bits = 2\n",
            "call": "compute_dither_variance_coefficient(bits)",
            "gold_call": "_oracle_compute_dither_variance_coefficient(bits)",
            "tol": 1e-15,
        },
        {
            "setup": "bits = 8\n",
            "call": "compute_dither_variance_coefficient(bits)",
            "gold_call": "_oracle_compute_dither_variance_coefficient(bits)",
            "tol": 1e-15,
        },
        {
            "setup": "bits = 3\n",
            "call": "compute_dither_variance_coefficient(bits)",
            "gold_call": "_oracle_compute_dither_variance_coefficient(bits)",
            "tol": 1e-15,
        },
        {
            "setup": "import numpy as np\nbits = np.int64(11)\n",
            "call": "compute_dither_variance_coefficient(bits)",
            "gold_call": "_oracle_compute_dither_variance_coefficient(bits)",
            "tol": 1e-15,
        },
        {
            "setup": "bits = 16\n",
            "call": "compute_dither_variance_coefficient(bits)",
            "gold_call": "_oracle_compute_dither_variance_coefficient(bits)",
            "tol": 1e-15,
        },
        {
            "setup": """bits = True
def run_model():
    try:
        compute_dither_variance_coefficient(bits)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dither_variance_coefficient(bits)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
