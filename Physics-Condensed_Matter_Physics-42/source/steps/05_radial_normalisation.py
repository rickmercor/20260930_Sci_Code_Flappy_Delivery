"""
Step 05: normalise the matched profile and split the total weight between the
regions.

Step 05: normalise the matched profile and split the total weight between the
regions.

Scope
-----
The matched profile of step 04 is fixed only up to one overall constant. This
step supplies that constant by integrating the profile over the plane, and
reports how the total probability divides between the regions.

Layout, ordering and dtype
--------------------------
transport is the table returned by step 03, amplitudes the (n,) array of step 04
and radii the (n - 1,) interface radii, all origin-outwards. The return has
shape (n + 1,) and dtype float64:

    out[0]      the normalisation integral of the matched profile
    out[1 + j]  the share of the total carried by region j

Conventions this step fixes
--------------------------
* The normalisation integral is taken with the plane's radial measure, that
  is, the profile weighted by one power of r and integrated over r from 0 to
  infinity, WITHOUT the factor of two pi. Downstream steps therefore divide by
  out[0] alone.
* Region j runs from radii[j - 1] to radii[j], with the innermost starting at
  r = 0 and the outermost ending at infinity.
* The shares out[1:] sum to exactly 1 up to rounding and are returned in the
  same origin-outwards order.
* Convergence requires the outermost inverse decay length to be strictly
  positive; a non-positive outermost value is rejected rather than returned as
  an infinity.
* A region whose inverse decay length is zero is integrated as a pure power of
  r over its finite range.

Inputs
------
transport : array_like, the step-03 table; two-dimensional with at
    least two columns, finite, column 0 non-negative and its last
    entry strictly positive.
amplitudes : array_like, shape (n,), finite and strictly positive.
radii : array_like, shape (n - 1,), finite, strictly positive, strictly
    increasing.

Returns
-------
summary : numpy.ndarray of shape (n + 1,), dtype float64.

Returns
-------
summary : numpy.ndarray of shape (n + 1,), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def radial_normalisation(transport, amplitudes, radii):
    """Return the normalisation integral and the per-region shares.

    Parameters
    ----------
    transport : array_like
        The table returned by step 03, origin-outwards, with
        the inverse decay length in column 0 and the exponent in column 1. The
        outermost inverse decay length must be strictly positive.
    amplitudes : array_like
        Array of shape (n,) as returned by step 04, finite and strictly
        positive.
    radii : array_like
        Array of shape (n - 1,) holding the interface radii, finite, strictly
        positive and strictly increasing.

    Returns
    -------
    summary : numpy.ndarray
        Array of shape (n + 1,) and dtype float64 whose first entry is the
        normalisation integral of the matched profile under the plane's radial
        measure, and whose remaining n entries are the shares of the total
        carried by each region, origin-outwards.

    Raises
    ------
    ValueError
        If transport is not a finite (n, 2) array with a non-negative column 0
        and a strictly positive outermost inverse decay length; if amplitudes
        is not a finite, strictly positive one-dimensional array of length n;
        if radii is not a finite, strictly positive, strictly increasing
        one-dimensional array of length n - 1; or if the resulting
        normalisation integral is not finite and strictly positive.
    """
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad as _quad
from scipy.special import exp1, gamma as _gamma_fn, gammaincc


def _upper_incomplete_gamma(order, x):
    """Upper incomplete gamma for any real order, with x > 0.

    scipy covers order > 0 only. Order 0 is the exponential integral, and
    negative orders follow from the standard downward recurrence
    Gamma(a, x) = (Gamma(a + 1, x) - x**a * exp(-x)) / a.
    """
    if order > 0.0:
        return _gamma_fn(order) * gammaincc(order, x)
    if order == 0.0:
        return float(exp1(x))
    return (_upper_incomplete_gamma(order + 1.0, x)
            - x ** order * np.exp(-x)) / order


def _finite_panel_moment(index, inverse_length, lower, upper):
    """Integral of r**index * exp(-inverse_length * r) over a finite panel
    whose lower edge is strictly positive.

    The panel is mapped onto [0, 1], integrated adaptively there and scaled
    back by its width. Differencing two upper incomplete gammas instead loses
    the panel when the region barely decays, because both values then sit on
    Gamma(order) itself and their difference falls below the rounding of
    either one.
    """
    width = float(upper) - float(lower)
    if width <= 0.0:
        return 0.0

    def integrand(t):
        r = float(lower) + width * t
        return r ** index * np.exp(-float(inverse_length) * r)

    return width * _quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13,
                         limit=200)[0]


def _radial_power_moment(exponent, inverse_length, power, lower, upper):
    """Integral of r**(exponent + power) * exp(-inverse_length * r) over
    [lower, upper]. upper may be numpy.inf."""
    order = float(exponent) + float(power) + 1.0
    if inverse_length > 0.0:
        if lower > 0.0 and not np.isinf(upper):
            return _finite_panel_moment(order - 1.0, inverse_length,
                                        lower, upper)
        scale = 1.0 / float(inverse_length)
        if lower > 0.0:
            lo = _upper_incomplete_gamma(order, float(lower) / scale)
        elif order > 0.0:
            lo = _gamma_fn(order)
        else:
            raise ValueError(
                "the integral diverges at the origin for a non-positive order")
        hi = (0.0 if np.isinf(upper)
              else _upper_incomplete_gamma(order, float(upper) / scale))
        return scale ** order * (lo - hi)
    if np.isinf(upper):
        raise ValueError(
            "an unbounded region needs a strictly positive inverse decay length")
    if lower <= 0.0 and order <= 0.0:
        raise ValueError(
            "the integral diverges at the origin for a non-positive order")
    if order == 0.0:
        return float(np.log(float(upper) / float(lower)))
    return (float(upper) ** order - float(lower) ** order) / order


def _check_profile(transport, amplitudes, radii):
    """Validate a matched profile: transport table, amplitudes and radii."""
    try:
        tr, rad = _check_transport_and_radii(transport, radii)
    except ValueError as exc:
        raise ValueError("invalid profile: %s" % exc)
    try:
        amp = np.asarray(amplitudes, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("amplitudes must be an array of real numbers")
    if amp.ndim != 1:
        raise ValueError("amplitudes must be one-dimensional")
    if amp.size != tr.shape[0]:
        raise ValueError("amplitudes must have one entry per region")
    if not np.all(np.isfinite(amp)):
        raise ValueError("amplitudes must be finite")
    if np.any(amp <= 0.0):
        raise ValueError("amplitudes must be strictly positive")
    if tr[-1, 0] <= 0.0:
        raise ValueError(
            "the outermost inverse decay length must be strictly positive")
    return tr, amp, rad


def _region_edges(rad, n):
    """[(lower, upper)] for each region, innermost first, outermost unbounded."""
    lowers = np.concatenate(([0.0], rad))
    uppers = np.concatenate((rad, [np.inf]))
    return list(zip(lowers[:n], uppers[:n]))


def _oracle_radial_normalisation(transport, amplitudes, radii):
    """Reference implementation of radial_normalisation."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)
    n = tr.shape[0]

    pieces = np.empty(n, dtype=np.float64)
    for j, (lo, hi) in enumerate(_region_edges(rad, n)):
        pieces[j] = amp[j] * _radial_power_moment(tr[j, 1], tr[j, 0], 1.0, lo, hi)

    total = float(np.sum(pieces))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the profile has no finite, positive normalisation")
    summary = np.empty(n + 1, dtype=np.float64)
    summary[0] = total
    summary[1:] = pieces / total
    return summary

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`radial_normalisation`."""
    cases = []

    # Normal case: the benchmark's three regions.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 0.5, 1.0]))\n"
            "transport = _oracle_region_transport_parameters(\n"
            "    0.48, stats[:, 1], 2.5)\n"
            "radii = np.array([0.06, 0.55])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    # Normal case: four regions, one of which decays much faster than the rest.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.05, 0.03],\n"
            "                      [1.5, 0.09],\n"
            "                      [3.0, 0.2]])\n"
            "radii = np.array([0.1, 0.8, 3.0])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    # Boundary case: a single unbounded region, whose share must be exactly 1.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.4, 0.07]])\n"
            "radii = np.array([])\n"
            "amplitudes = np.array([1.0])"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    # Edge case: a flat interior region carrying essentially none of the
    # weight, so the shares span several orders of magnitude.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.004])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    # Edge case: two adjacent regions with identical transport parameters, so
    # the split must follow the geometry alone.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.5, 0.05],\n"
            "                      [0.5, 0.05]])\n"
            "radii = np.array([1.0])\n"
            "amplitudes = np.array([1.0, 1.0])"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    # Invalid input: a non-decaying outermost region, which is not normalisable.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.0, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a non-positive amplitude.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([0.0, 1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: an amplitude array of the wrong length.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Regression: an annulus whose decay length is enormous, so its share of
    # the total is small but strictly positive.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 1e-8, 1.0]))\n"
            "transport = _oracle_region_transport_parameters(\n"
            "    0.48, stats[:, 1], 2.5)\n"
            "radii = np.array([0.06, 0.55])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)"
        ),
        "call": "radial_normalisation(transport, amplitudes, radii)",
        "gold_call": (
            "_oracle_radial_normalisation(transport, amplitudes, radii)"
        ),
    })

    return cases
