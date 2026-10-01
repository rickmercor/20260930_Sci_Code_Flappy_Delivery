"""
Step 03: the two transport parameters that fix the stationary radial profile
inside each region.

Step 03: the two transport parameters that fix the stationary radial profile
inside each region.

## Scope

Within a region where the sensory modulation is constant, the stationary
radial profile of the agent's position is fixed by exactly two numbers: an
inverse decay length and a short-distance exponent. This step returns that
pair for every region.

## Layout, ordering and dtype

mean_cos_values is the column of stationary mean cosines returned by step 01,
one entry per region, in the same origin-outwards order. The return has shape
(n, 5) and dtype float64, with the same row order:

out[j, 0]  inverse decay length of region j
out[j, 1]  short-distance exponent of region j
out[j, 2]  renormalised steering strength of region j
out[j, 3]  effective diffusivity of region j
out[j, 4]  decay length of region j, the reciprocal of column 0

The later steps read only the first two columns; the last three are the
intermediates those two are built from, or their reciprocal, reported so they
can be checked.

## Conventions this step fixes

* The renormalised steering strength of a region is the bare steering strength
  multiplied by that region's stationary mean cosine, and it is that
  renormalised strength -- never the bare one -- at which the coefficients of
  step 02 are evaluated.
* Both returned parameters are expressed in the length and time units of the
  problem statement, so the inverse decay length has units of inverse length
  and the exponent is dimensionless.
* A region whose mean cosine is zero must return exactly (0.0, 0.0) in the
  first two columns: the profile there is flat in the plane, with no decay and
  no anomalous power. Its decay length is undefined and column 4 is reported as
  exactly 0.0 there by convention, never as an infinity.
* The Peclet number enters only through its reciprocal.
* The transverse-policy coefficient of step 02 plays no part here, because the
  policy of this problem has no transverse divergence.
* The coefficients of step 02 are defined only for renormalised strengths in
  the closed interval [0, 1.2]. A bare strength whose renormalised value
  leaves that interval in any region is outside the admissible range and
  raises ValueError rather than returning a number, so the admissible bare
  strength depends on the mean cosines handed in.

## Inputs

kappa_tilde : float
Bare steering strength. Must be finite and strictly positive.
mean_cos_values : array_like
1-D, at least one element, every entry finite and in [0, 1], ordered from
the origin outwards.
peclet : float
Peclet number. Must be finite and strictly positive.

## Returns

transport : numpy.ndarray of shape (n, 5), dtype float64.

Returns
-------
transport : numpy.ndarray of shape (n, 5), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def region_transport_parameters(kappa_tilde, mean_cos_values, peclet):
    """Return the inverse decay length and exponent for every region.

    Parameters
    ----------
    kappa_tilde : float
        The bare steering strength of the problem statement. Must be finite
        and strictly positive.
    mean_cos_values : array_like
        1-D array of the per-region stationary mean cosines of the sensor's
        angular error, ordered from the origin outwards. Must be non-empty and
        finite, with every entry in [0, 1].
    peclet : float
        The Peclet number of the problem statement. Must be finite and
        strictly positive.

    Returns
    -------
    transport : numpy.ndarray
        Array of shape (n, 5) and dtype float64 whose row j holds, in order,
        the inverse decay length, the short-distance exponent, the renormalised
        steering strength, the effective diffusivity and the decay length of
        region j. A region with zero mean cosine returns exactly zero in
        columns 0, 1, 2 and 4.

    Raises
    ------
    ValueError
        If kappa_tilde or peclet is not a finite, strictly positive real
        number; if mean_cos_values is not a non-empty 1-D array of real
        numbers; if any entry is not finite; if any entry lies outside the
        closed interval [0, 1]; or if the renormalised strength of any region,
        the bare strength times that region's mean cosine, leaves the closed
        interval [0, 1.2] on which the closure coefficients are defined.
    """
    return transport  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_region_transport_parameters(kappa_tilde, mean_cos_values, peclet):
    """Reference implementation of region_transport_parameters."""
    for name, value in (("kappa_tilde", kappa_tilde), ("peclet", peclet)):
        if isinstance(value, bool):
            raise ValueError("%s must be a real number, got a bool" % name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be a real number" % name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and strictly positive" % name)
    kappa_tilde = float(kappa_tilde)
    peclet = float(peclet)

    try:
        mc = np.asarray(mean_cos_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("mean_cos_values must be an array of real numbers")
    if mc.ndim != 1:
        raise ValueError("mean_cos_values must be one-dimensional")
    if mc.size < 1:
        raise ValueError("mean_cos_values must have at least one entry")
    if not np.all(np.isfinite(mc)):
        raise ValueError("mean_cos_values must be finite")
    if np.any(mc < 0.0) or np.any(mc > 1.0):
        raise ValueError("every entry of mean_cos_values must lie in [0, 1]")

    kappa_eff = kappa_tilde * mc
    coeffs = _oracle_closure_series_coefficients(kappa_eff)
    c0k = coeffs[:, 0]
    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]

    effective_diffusivity = 1.0 / peclet + c1 + c2

    transport = np.empty((mc.size, 5), dtype=np.float64)
    transport[:, 0] = c0k / effective_diffusivity
    transport[:, 1] = c3 / effective_diffusivity
    transport[:, 2] = kappa_eff
    transport[:, 3] = effective_diffusivity
    steered = c0k > 0.0
    transport[:, 4] = 0.0
    transport[steered, 4] = effective_diffusivity[steered] / c0k[steered]
    return transport

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`region_transport_parameters`."""
    cases = []

    # Normal case: the benchmark configuration's three regions.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 0.5, 1.0]))\n"
            "kappa_tilde = 0.48\n"
            "mean_cos_values = stats[:, 1]\n"
            "peclet = 2.5"
        ),
        "call": (
            "region_transport_parameters(kappa_tilde, mean_cos_values, peclet)"
        ),
        "gold_call": (
            "_oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)"
        ),
    })

    # Normal case: a stronger steering strength and a lower Peclet number, so
    # the effective diffusivity is dominated by its translational part.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 1.2\n"
            "mean_cos_values = np.array([0.25, 0.6, 0.95])\n"
            "peclet = 2.0"
        ),
        "call": (
            "region_transport_parameters(kappa_tilde, mean_cos_values, peclet)"
        ),
        "gold_call": (
            "_oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)"
        ),
    })

    # Boundary case: a single region at perfect sensing, where the renormalised
    # strength equals the bare one.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.9\n"
            "mean_cos_values = np.array([1.0])\n"
            "peclet = 50.0"
        ),
        "call": (
            "region_transport_parameters(kappa_tilde, mean_cos_values, peclet)"
        ),
        "gold_call": (
            "_oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)"
        ),
    })

    # Edge case: two unsteered regions beside a steered one, so the first two
    # rows must vanish exactly rather than merely become small while the third
    # stays order unity.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.66\n"
            "mean_cos_values = np.array([0.0, 0.0, 0.85])\n"
            "peclet = 8.0"
        ),
        "call": (
            "region_transport_parameters(kappa_tilde, mean_cos_values, peclet)"
        ),
        "gold_call": (
            "_oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)"
        ),
    })

    # Edge case: a very large Peclet number, where the reciprocal contributes
    # almost nothing to the effective diffusivity.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.5\n"
            "mean_cos_values = np.array([0.3, 0.9])\n"
            "peclet = 1.0e6"
        ),
        "call": (
            "region_transport_parameters(kappa_tilde, mean_cos_values, peclet)"
        ),
        "gold_call": (
            "_oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)"
        ),
    })

    # Invalid input: a non-positive Peclet number.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.66\n"
            "mean_cos_values = np.array([0.0, 0.44, 0.8])\n"
            "peclet = -3.0\n"
            "def run_model():\n"
            "    try:\n"
            "        region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a mean cosine above one.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.66\n"
            "mean_cos_values = np.array([0.0, 1.3])\n"
            "peclet = 8.0\n"
            "def run_model():\n"
            "    try:\n"
            "        region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a non-positive bare steering strength.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "kappa_tilde = 0.0\n"
            "mean_cos_values = np.array([0.0, 0.44, 0.8])\n"
            "peclet = 8.0\n"
            "def run_model():\n"
            "    try:\n"
            "        region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_region_transport_parameters("
            "kappa_tilde, mean_cos_values, peclet)\n"
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
