"""
Step 07: the inverse-square radial moment restricted to each region beyond the
innermost.

Step 07: the inverse-square radial moment restricted to each region beyond the
innermost.

Scope
-----
The quantity the pipeline finally needs from the stationary profile is not the
profile itself but its average of one over the squared radius, taken region by
region so that each region can afterwards be weighted differently. This step
returns those restricted averages.

Layout, ordering and dtype
--------------------------
the step-03 table, amplitudes (n,), radii (n - 1,) and normalisation come from
steps 03 to 05, origin-outwards, and n must be at least 2. The return has
shape (n - 1,) and dtype float64:

    out[j]  the average of the indicator of region j + 1 divided by the
            squared radius, taken over the normalised stationary profile

so out[0] belongs to the region just outside the innermost one, and out[-1] to
the unbounded outermost region.

Conventions this step fixes
--------------------------
* Each entry is a restricted average, not a conditional one: it is the
  integral over that region alone, divided by the normalisation of the WHOLE
  profile, so the entries sum to the average of one over the squared radius
  over everything outside the innermost region.
* The measure is the same one step 05 normalised: the matched profile times
  one power of r, without any factor of two pi. Combined with the one over r
  squared, each integrand is the matched profile divided by r.
* The innermost region is deliberately excluded and has no entry, since its
  inner edge is the origin and the integral there need not converge.
* Region j + 1 runs from radii[j] to radii[j + 1], the outermost ending at
  infinity.

Inputs
------
transport : array_like, the step-03 table, with n >= 2 rows.
amplitudes : array_like, shape (n,), as step 04.
radii : array_like, shape (n - 1,), the interface radii.
normalisation : float, entry 0 of the step-05 return; finite and strictly
    positive.

Returns
-------
moments : numpy.ndarray of shape (n - 1,), dtype float64.

Returns
-------
moments : numpy.ndarray of shape (n - 1,), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def conditional_inverse_square_moments(transport, amplitudes, radii,
                                       normalisation):
    """Return the inverse-square moment restricted to each outer region.

    Parameters
    ----------
    transport : array_like
        The table returned by step 03, origin-outwards, with
        n >= 2 and a strictly positive outermost inverse decay length.
    amplitudes : array_like
        Array of shape (n,) as returned by step 04.
    radii : array_like
        Array of shape (n - 1,) holding the interface radii, strictly positive
        and strictly increasing.
    normalisation : float
        The normalisation integral, that is, entry 0 of the step-05 return.
        Must be finite and strictly positive.

    Returns
    -------
    moments : numpy.ndarray
        Array of shape (n - 1,) and dtype float64 whose entry j is the average
        of the indicator of region j + 1 divided by the squared radius, taken
        over the normalised stationary profile.

    Raises
    ------
    ValueError
        If transport, amplitudes or radii fail the shape, finiteness,
        positivity or ordering requirements of steps 03 to 05; if transport
        has fewer than two rows; or if normalisation is not a finite, strictly
        positive real number.
    """
    return moments  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_conditional_inverse_square_moments(transport, amplitudes, radii,
                                               normalisation):
    """Reference implementation of conditional_inverse_square_moments."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)
    n = tr.shape[0]
    if n < 2:
        raise ValueError("at least two regions are required")

    if isinstance(normalisation, bool):
        raise ValueError("normalisation must be a real number, got a bool")
    try:
        norm = float(normalisation)
    except (TypeError, ValueError):
        raise ValueError("normalisation must be a real number")
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalisation must be finite and strictly positive")

    edges = _region_edges(rad, n)
    moments = np.empty(n - 1, dtype=np.float64)
    for j in range(1, n):
        lo, hi = edges[j]
        moments[j - 1] = (
            amp[j] * _radial_power_moment(tr[j, 1], tr[j, 0], -1.0, lo, hi)
            / norm)
    return moments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the cases for :func:`conditional_inverse_square_moments`."""
    cases = []

    # Normal case: the benchmark's annulus and outer region.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 0.5, 1.0]))\n"
            "transport = _oracle_region_transport_parameters(\n"
            "    0.48, stats[:, 1], 2.5)\n"
            "radii = np.array([0.06, 0.55])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])"
        ),
        "call": (
            "conditional_inverse_square_moments(transport, amplitudes, radii, "
            "normalisation)"
        ),
        "gold_call": (
            "_oracle_conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)"
        ),
    })

    # Normal case: four regions, so three moments spanning a wide range.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.05, 0.03],\n"
            "                      [1.5, 0.09],\n"
            "                      [3.0, 0.2]])\n"
            "radii = np.array([0.1, 0.8, 3.0])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])"
        ),
        "call": (
            "conditional_inverse_square_moments(transport, amplitudes, radii, "
            "normalisation)"
        ),
        "gold_call": (
            "_oracle_conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)"
        ),
    })

    # Boundary case: the minimum of two regions, so a single moment over an
    # unbounded outer region.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.6, 0.05]])\n"
            "radii = np.array([0.25])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])"
        ),
        "call": (
            "conditional_inverse_square_moments(transport, amplitudes, radii, "
            "normalisation)"
        ),
        "gold_call": (
            "_oracle_conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)"
        ),
    })

    # Edge case: an inner interface pushed very close to the origin, where the
    # restricted moment grows logarithmically and is order unity or larger.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02],\n"
            "                      [0.5, 0.04]])\n"
            "radii = np.array([0.004, 0.6])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])"
        ),
        "call": (
            "conditional_inverse_square_moments(transport, amplitudes, radii, "
            "normalisation)"
        ),
        "gold_call": (
            "_oracle_conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)"
        ),
    })

    # Edge case: an annulus whose transport parameters match the outer
    # region's exactly, so the two moments must join as one smooth integral.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.4, 0.03],\n"
            "                      [0.4, 0.03]])\n"
            "radii = np.array([0.09, 0.7])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])"
        ),
        "call": (
            "conditional_inverse_square_moments(transport, amplitudes, radii, "
            "normalisation)"
        ),
        "gold_call": (
            "_oracle_conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)"
        ),
    })

    # Invalid input: a single region, which has no outer region to average
    # over.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.4, 0.07]])\n"
            "radii = np.array([])\n"
            "amplitudes = np.array([1.0])\n"
            "normalisation = 5.0\n"
            "def run_model():\n"
            "    try:\n"
            "        conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_conditional_inverse_square_moments(transport, "
            "amplitudes, radii, normalisation)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a negative normalisation.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "normalisation = -2.0\n"
            "def run_model():\n"
            "    try:\n"
            "        conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_conditional_inverse_square_moments(transport, "
            "amplitudes, radii, normalisation)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: an interface radius of zero.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.0])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "normalisation = 5.0\n"
            "def run_model():\n"
            "    try:\n"
            "        conditional_inverse_square_moments(transport, amplitudes, "
            "radii, normalisation)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_conditional_inverse_square_moments(transport, "
            "amplitudes, radii, normalisation)\n"
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
