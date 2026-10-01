"""
Step 06: evaluate the normalised radial density at requested radii.

Step 06: evaluate the normalised radial density at requested radii.

Scope
-----
Steps 03 to 05 determine the stationary profile completely. This step is the
readout: given a list of radii it returns the normalised radial density there,
so the profile can be inspected and checked against its normalisation.

Layout, ordering and dtype
--------------------------
the step-03 table, amplitudes (n,), radii (n - 1,) and normalisation are the
outputs of steps 03 to 05, all origin-outwards. r_values is a 1-D array of
query radii in any order. The return has shape (r_values.size,) and dtype
float64, matching r_values entry for entry.

Conventions this step fixes
--------------------------
* "Radial density" means the density of the radial coordinate, that is, the
  matched profile multiplied by one power of r and divided by the
  normalisation integral of step 05, WITHOUT any factor of two pi. It
  integrates to 1 over r from 0 to infinity.
* A query radius equal to an interface radius is assigned to the OUTER of the
  two regions that meet there. Because the profile is continuous the two
  assignments agree, so this only fixes the tie.
* A query radius of exactly 0 returns 0.0, since the radial measure vanishes
  there.
* Query radii must be finite and non-negative; they need not be sorted and may
  repeat.

Inputs
------
transport : array_like, the step-03 table.
amplitudes : array_like, shape (n,), as step 04.
radii : array_like, shape (n - 1,), the interface radii.
normalisation : float, the first entry of the step-05 return; finite and
    strictly positive.
r_values : array_like, 1-D, at least one entry, finite and non-negative.

Returns
-------
density : numpy.ndarray of shape (r_values.size,), dtype float64.

Returns
-------
density : numpy.ndarray of shape (r_values.size,), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def radial_density_profile(transport, amplitudes, radii, normalisation,
                           r_values):
    """Return the normalised radial density at each requested radius.

    Parameters
    ----------
    transport : array_like
        The table returned by step 03, origin-outwards.
    amplitudes : array_like
        Array of shape (n,) as returned by step 04.
    radii : array_like
        Array of shape (n - 1,) holding the interface radii, strictly positive
        and strictly increasing.
    normalisation : float
        The normalisation integral, that is, entry 0 of the step-05 return.
        Must be finite and strictly positive.
    r_values : array_like
        One-dimensional array of query radii, non-empty, finite and
        non-negative. Need not be sorted and may contain repeats.

    Returns
    -------
    density : numpy.ndarray
        Array of shape (r_values.size,) and dtype float64 holding the
        normalised radial density at each query radius, in the order given.

    Raises
    ------
    ValueError
        If transport, amplitudes or radii fail the shape, finiteness,
        positivity or ordering requirements of steps 03 to 05; if
        normalisation is not a finite, strictly positive real number; or if
        r_values is not a non-empty one-dimensional array of finite,
        non-negative real numbers.
    """
    return density  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radial_density_profile(transport, amplitudes, radii, normalisation,
                                   r_values):
    """Reference implementation of radial_density_profile."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)

    if isinstance(normalisation, bool):
        raise ValueError("normalisation must be a real number, got a bool")
    try:
        norm = float(normalisation)
    except (TypeError, ValueError):
        raise ValueError("normalisation must be a real number")
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalisation must be finite and strictly positive")

    try:
        rv = np.asarray(r_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("r_values must be an array of real numbers")
    if rv.ndim != 1:
        raise ValueError("r_values must be one-dimensional")
    if rv.size < 1:
        raise ValueError("r_values must have at least one entry")
    if not np.all(np.isfinite(rv)):
        raise ValueError("r_values must be finite")
    if np.any(rv < 0.0):
        raise ValueError("r_values must be non-negative")

    # Region index of each query radius; an interface radius goes outward.
    index = np.searchsorted(rad, rv, side="right") if rad.size else np.zeros(
        rv.size, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)

    density = np.zeros(rv.size, dtype=np.float64)
    positive = rv > 0.0
    if np.any(positive):
        j = index[positive]
        r = rv[positive]
        density[positive] = (amp[j] * r ** (tr[j, 1] + 1.0)
                             * np.exp(-tr[j, 0] * r) / norm)
    return density

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`radial_density_profile`."""
    cases = []

    # Normal case: the benchmark profile sampled across all three regions.
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
            "normalisation = float(summary[0])\n"
            "r_values = np.array([0.02, 0.07, 0.2, 0.45, 1.0, 4.0])"
        ),
        "call": (
            "radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
        "gold_call": (
            "_oracle_radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
    })

    # Normal case: unsorted and repeated query radii on a two-region profile.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.6, 0.05]])\n"
            "radii = np.array([0.25])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])\n"
            "r_values = np.array([2.0, 0.1, 0.25, 0.1, 0.9])"
        ),
        "call": (
            "radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
        "gold_call": (
            "_oracle_radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
    })

    # Boundary case: radii exactly at both interfaces, where the tie-break
    # matters and continuity must make it invisible.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.17215285455, 0.00946790716],\n"
            "                      [0.22827700882, 0.01601833317]])\n"
            "radii = np.array([0.06, 0.55])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])\n"
            "r_values = np.array([0.06, 0.55])"
        ),
        "call": (
            "radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
        "gold_call": (
            "_oracle_radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
    })

    # Edge case: the origin, where the radial measure vanishes.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = _oracle_matched_density_amplitudes(transport, radii)\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])\n"
            "r_values = np.array([0.0, 0.5])"
        ),
        "call": (
            "radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
        "gold_call": (
            "_oracle_radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
    })

    # Edge case: a single region, so every radius falls in the same branch.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.4, 0.07]])\n"
            "radii = np.array([])\n"
            "amplitudes = np.array([1.0])\n"
            "summary = _oracle_radial_normalisation(transport, amplitudes, radii)\n"
            "normalisation = float(summary[0])\n"
            "r_values = np.array([0.5, 2.0, 8.0])"
        ),
        "call": (
            "radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
        "gold_call": (
            "_oracle_radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)"
        ),
    })

    # Invalid input: a negative query radius.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "normalisation = 5.0\n"
            "r_values = np.array([0.5, -1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_density_profile(transport, amplitudes, "
            "radii, normalisation, r_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a non-positive normalisation.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "normalisation = 0.0\n"
            "r_values = np.array([0.5])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_density_profile(transport, amplitudes, "
            "radii, normalisation, r_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a two-dimensional query array.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.02]])\n"
            "radii = np.array([0.07])\n"
            "amplitudes = np.array([1.0, 1.0])\n"
            "normalisation = 5.0\n"
            "r_values = np.array([[0.5, 1.0]])\n"
            "def run_model():\n"
            "    try:\n"
            "        radial_density_profile(transport, amplitudes, radii, "
            "normalisation, r_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_radial_density_profile(transport, amplitudes, "
            "radii, normalisation, r_values)\n"
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
