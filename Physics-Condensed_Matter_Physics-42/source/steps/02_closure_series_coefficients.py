"""
Step 02: the closure coefficients of the gradient expansion, as functions of
the renormalised steering strength.

Step 02: the closure coefficients of the gradient expansion, as functions of
the renormalised steering strength.

## Scope

Closing the heading statistics at first order in spatial gradients introduces
four scalar coefficient functions of the renormalised steering strength, on
top of the zeroth-order alignment. None of the four admits a closed form in
elementary functions; each is a power series in the squared steering strength
with rational coefficients, and this step returns their values.

## Layout, ordering and dtype

kappa_values is a 1-D array of renormalised steering strengths, one per
region, ordered from the origin outwards. The return has shape (n, 5) and
dtype float64, with n = kappa_values.size and the same row order:

out[j, 0]  the zeroth-order alignment coefficient
out[j, 1]  the coefficient of the isotropic part of the density gradient
out[j, 2]  the coefficient of the part of the density gradient projected
along the policy direction
out[j, 3]  the coefficient of the policy divergence
out[j, 4]  the coefficient of the transverse-policy divergence

## Conventions this step fixes

* Column 0 is the same function of its argument as the mean cosine returned by
  step 01 is of the concentration parameter, and takes the value 0 at
  argument 0.
* Columns 1 to 4 are the four coefficients in the order just listed, which is
  the order in which they appear in the expansion of the position-dependent
  mean heading.
* Every series is truncated at exactly order 20 in the strength, that is,
  eleven terms in ascending powers of the squared strength, which is the order
  to which the source tabulates the coefficients. The returned value is that
  truncated polynomial itself, not the limiting function, and the same
  truncation is applied to all four. For arguments up to 0.7 the last retained
  term is below 2e-8 of the sum for every coefficient; at the upper end of the
  admissible range it is below 5e-4.
* Argument 0 is legal: the coefficients then take their zero-strength values,
  which are exact rational numbers and must be returned without any division
  by zero.
* Arguments are dimensionless and non-negative. Values above 1.2 lie outside
  the admissible range and are rejected.

## The four series, written out

Each of columns 1 to 4 is the truncated polynomial

sum over n = 0, 1, ..., 10 of  (N_n / D_n) * x**(2 * n)

in the renormalised strength x, with integer numerators N_n and denominators
D_n listed below in ascending order of n. n = 0 is the constant term; a
leading numerator of 0 means the series starts at order x**2. The signs are
part of the numerators. Column 1 therefore begins 1/2 - 5 x**2/32 + 23 x**4/576
- ..., column 2 begins -x**2/8 + x**4/16 - ..., and columns 3 and 4 both begin
3 x**2/32. Evaluate the sum in float64; no term may be dropped or added.

Column 1, isotropic density-gradient coefficient:
N: 1, -5, 23, -677, 7313, -218491, 863897, -874088357, 27545803997, -423385249313, 19488418951523
D: 2, 32, 576, 73728, 3686400, 530841600, 10404495360, 53271016243200, 8629904631398400, 690392370511872000, 167074953663873024000

Column 2, coefficient of the density gradient projected along the policy:
N: 0, -1, 1, -131, 25, -41851, 20209, -33334307, 133205867, -19173165917, 3470122403
D: 1, 8, 16, 6144, 4096, 26542080, 53084160, 380507258880, 6849130659840, 4566087106560000, 3913788948480000

Column 3, coefficient of the policy divergence:
N: 0, 3, -29, 1325, -19553, 247777, -397169, 473737993, -154836151097, 13477269234097, -693144302217667
D: 1, 32, 576, 73728, 3686400, 176947200, 1156055040, 5919001804800, 8629904631398400, 3451961852559360000, 835374768319365120000

Column 4, coefficient of the transverse-policy divergence:
N: 0, 3, -17, 545, -6173, 190111, -767717, 788938397, -25160566037, 390389770937, -18107708487467
D: 1, 32, 576, 73728, 3686400, 530841600, 10404495360, 53271016243200, 8629904631398400, 690392370511872000, 167074953663873024000

## Inputs

kappa_values : array_like
1-D, at least one element, every entry finite and in [0, 1.2].

## Returns

coeffs : numpy.ndarray of shape (n, 5), dtype float64.

Returns
-------
coeffs : numpy.ndarray of shape (n, 5), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def closure_series_coefficients(kappa_values):
    """Return the five closure coefficients for each renormalised strength.

    For each entry of kappa_values, return the zeroth-order alignment
    coefficient followed by the four first-order gradient-expansion
    coefficients, in the order fixed above.

    Parameters
    ----------
    kappa_values : array_like
        1-D array of renormalised steering strengths, ordered from the origin
        outwards. Must be non-empty and finite, with every entry in the closed
        interval [0, 1.2].

    Returns
    -------
    coeffs : numpy.ndarray
        Array of shape (n, 5) and dtype float64 whose row j holds, in order,
        the zeroth-order alignment coefficient and the four first-order
        coefficients evaluated at kappa_values[j].

    Raises
    ------
    ValueError
        If kappa_values is not a non-empty 1-D array of real numbers, if any
        entry is not finite, or if any entry lies outside [0, 1.2].
    """
    return coeffs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import i0e, i1e

# Rational series coefficients of the four gradient-expansion functions, in
# ascending powers of the squared steering strength, truncated at order 20.
_C1_NUM = (1, -5, 23, -677, 7313, -218491, 863897, -874088357,
           27545803997, -423385249313, 19488418951523)
_C1_DEN = (2, 32, 576, 73728, 3686400, 530841600, 10404495360,
           53271016243200, 8629904631398400, 690392370511872000,
           167074953663873024000)
_C2_NUM = (0, -1, 1, -131, 25, -41851, 20209, -33334307, 133205867,
           -19173165917, 3470122403)
_C2_DEN = (1, 8, 16, 6144, 4096, 26542080, 53084160, 380507258880,
           6849130659840, 4566087106560000, 3913788948480000)
_C3_NUM = (0, 3, -29, 1325, -19553, 247777, -397169, 473737993,
           -154836151097, 13477269234097, -693144302217667)
_C3_DEN = (1, 32, 576, 73728, 3686400, 176947200, 1156055040,
           5919001804800, 8629904631398400, 3451961852559360000,
           835374768319365120000)
_C4_NUM = (0, 3, -17, 545, -6173, 190111, -767717, 788938397,
           -25160566037, 390389770937, -18107708487467)
_C4_DEN = (1, 32, 576, 73728, 3686400, 530841600, 10404495360,
           53271016243200, 8629904631398400, 690392370511872000,
           167074953663873024000)


def _series_value(num, den, x):
    """Evaluate sum_n (num[n]/den[n]) * x**(2n) for an array x."""
    x = np.asarray(x, dtype=np.float64)
    z = x * x
    total = np.zeros_like(x)
    power = np.ones_like(x)
    for n, d in zip(num, den):
        total = total + (float(n) / float(d)) * power
        power = power * z
    return total


def _oracle_closure_series_coefficients(kappa_values):
    """Reference implementation of closure_series_coefficients."""
    try:
        kap = np.asarray(kappa_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("kappa_values must be an array of real numbers")
    if kap.ndim != 1:
        raise ValueError("kappa_values must be one-dimensional")
    if kap.size < 1:
        raise ValueError("kappa_values must have at least one entry")
    if not np.all(np.isfinite(kap)):
        raise ValueError("kappa_values must be finite")
    if np.any(kap < 0.0) or np.any(kap > 1.2):
        raise ValueError("every entry of kappa_values must lie in [0, 1.2]")

    c0 = np.zeros_like(kap)
    nz = kap > 0.0
    c0[nz] = i1e(kap[nz]) / i0e(kap[nz])

    coeffs = np.empty((kap.size, 5), dtype=np.float64)
    coeffs[:, 0] = c0
    coeffs[:, 1] = _series_value(_C1_NUM, _C1_DEN, kap)
    coeffs[:, 2] = _series_value(_C2_NUM, _C2_DEN, kap)
    coeffs[:, 3] = _series_value(_C3_NUM, _C3_DEN, kap)
    coeffs[:, 4] = _series_value(_C4_NUM, _C4_DEN, kap)
    return coeffs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`closure_series_coefficients`."""
    cases = []

    # Normal case: the two non-trivial renormalised strengths of the benchmark
    # configuration.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 0.5, 1.0]))\n"
            "kappa_values = 0.48 * stats[:, 1]"
        ),
        "call": "closure_series_coefficients(kappa_values)",
        "gold_call": "_oracle_closure_series_coefficients(kappa_values)",
    })

    # Normal case: a spread of strengths across the admissible range, all five
    # columns order unity or larger.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([0.1, 0.35, 0.7, 1.05])"
        ),
        "call": "closure_series_coefficients(kappa_values)",
        "gold_call": "_oracle_closure_series_coefficients(kappa_values)",
    })

    # Boundary case: the largest admissible argument, where the truncated
    # polynomial departs most from the limiting function.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([1.2])"
        ),
        "call": "closure_series_coefficients(kappa_values)",
        "gold_call": "_oracle_closure_series_coefficients(kappa_values)",
    })

    # Edge case: zero strength, where the four gradient coefficients collapse
    # to their exact rational limits and the alignment coefficient vanishes.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([0.0])"
        ),
        "call": "closure_series_coefficients(kappa_values)",
        "gold_call": "_oracle_closure_series_coefficients(kappa_values)",
    })

    # Edge case: a very small but non-zero strength, where the leading
    # behaviour of each column is its lowest retained power.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([1e-4, 1e-2])"
        ),
        "call": "closure_series_coefficients(kappa_values)",
        "gold_call": "_oracle_closure_series_coefficients(kappa_values)",
    })

    # Invalid input: a negative strength.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([0.3, -0.1])\n"
            "def run_model():\n"
            "    try:\n"
            "        closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a strength beyond the admissible range.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([2.5])\n"
            "def run_model():\n"
            "    try:\n"
            "        closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: an empty array.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_values = np.array([])\n"
            "def run_model():\n"
            "    try:\n"
            "        closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_closure_series_coefficients(kappa_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    return cases
