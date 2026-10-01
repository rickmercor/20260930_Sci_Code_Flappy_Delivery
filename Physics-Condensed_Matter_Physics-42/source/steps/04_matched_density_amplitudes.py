"""
Step 04: match the per-region radial profiles across the interfaces.

Step 04: match the per-region radial profiles across the interfaces.

Scope
-----
Each region carries its own pair of transport parameters, so the stationary
profile is written region by region as

    p_j(r) = A_j * r**exponent_j * exp(-inverse_length_j * r)

and the amplitudes A_j are fixed, up to one overall constant, by requiring the
profile to join across every interface. This step returns those amplitudes.

Layout, ordering and dtype
--------------------------
transport is the array returned by step 03, ordered from the origin
outwards, column 0 the inverse decay length and column 1 the exponent. Any
further columns it carries are ignored here. radii
is the 1-D array of the n - 1 interface radii, strictly increasing, with
radii[j] separating region j from region j + 1. The return has shape (n,) and
dtype float64, in the same origin-outwards order.

Conventions this step fixes
--------------------------
* The overall constant is fixed by setting the OUTERMOST amplitude to exactly
  1.0; every other amplitude is expressed relative to it. No normalisation of
  the profile is applied here -- that is step 05.
* The quantity that is continuous across an interface is p_j(r) itself, not
  its logarithmic derivative, which jumps.
* The parameterisation above is the definition of "amplitude" for this
  pipeline: the exponent multiplies a bare power of r, not of r scaled by any
  radius, and the exponential carries r itself, not r minus an interface
  radius.
* n = 1 is legal: radii is then empty and the single amplitude is 1.0.
* Amplitudes are computed through logarithms so that widely separated decay
  lengths do not overflow.

Inputs
------
transport : array_like
    Shape (n, m) with n >= 1 and m >= 2, finite, column 0 non-negative. Only
    the first two columns are read.
radii : array_like
    1-D of length n - 1, finite, strictly positive and strictly increasing.

Returns
-------
amplitudes : numpy.ndarray of shape (n,), dtype float64, outermost entry 1.0.

Returns
-------
amplitudes : numpy.ndarray of shape (n,), dtype float64, outermost entry 1.0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def matched_density_amplitudes(transport, radii):
    """Return the per-region amplitudes that join the profile at every radius.

    Parameters
    ----------
    transport : array_like
        The table returned by step 03, ordered from the origin outwards, with
        the inverse decay length in column 0 and the exponent in column 1. Must
        be two-dimensional with at least two columns, finite, and with a
        non-negative column 0. Columns beyond the second are ignored.
    radii : array_like
        One-dimensional array of the n - 1 interface radii, finite, strictly
        positive and strictly increasing, with radii[j] separating region j
        from region j + 1.

    Returns
    -------
    amplitudes : numpy.ndarray
        Array of shape (n,) and dtype float64 holding the amplitude of each
        region in origin-outwards order, with the outermost entry exactly 1.0.

    Raises
    ------
    ValueError
        If transport is not a finite two-dimensional array with at least two
        columns and at least one row; if column 0 of transport contains a
        negative
        entry; if radii is not a finite one-dimensional array of length one
        less than the number of rows of transport; if the entries of radii are
        not strictly positive and strictly increasing; or if the matched
        amplitudes overflow or underflow because the transport parameters and
        radii are too widely separated to join.
    """
    return amplitudes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_transport_and_radii(transport, radii):
    """Shared validation of a (n, 2) transport table and its n - 1 radii."""
    try:
        tr = np.asarray(transport, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("transport must be an array of real numbers")
    if tr.ndim != 2 or tr.shape[1] < 2:
        raise ValueError("transport must be two-dimensional with at least two "
                         "columns")
    if tr.shape[0] < 1:
        raise ValueError("transport must have at least one row")
    if not np.all(np.isfinite(tr)):
        raise ValueError("transport must be finite")
    if np.any(tr[:, 0] < 0.0):
        raise ValueError("inverse decay lengths must be non-negative")

    try:
        rad = np.asarray(radii, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("radii must be an array of real numbers")
    if rad.ndim != 1:
        raise ValueError("radii must be one-dimensional")
    if rad.size != tr.shape[0] - 1:
        raise ValueError(
            "radii must have exactly one entry fewer than transport has rows")
    if not np.all(np.isfinite(rad)):
        raise ValueError("radii must be finite")
    if rad.size > 0:
        if np.any(rad <= 0.0):
            raise ValueError("radii must be strictly positive")
        if np.any(np.diff(rad) <= 0.0):
            raise ValueError("radii must be strictly increasing")
    return tr, rad


def _oracle_matched_density_amplitudes(transport, radii):
    """Reference implementation of matched_density_amplitudes."""
    try:
        tr, rad = _check_transport_and_radii(transport, radii)
    except ValueError as exc:
        raise ValueError("matched_density_amplitudes: %s" % exc)
    n = tr.shape[0]

    log_amp = np.zeros(n, dtype=np.float64)          # outermost log-amplitude 0
    for j in range(n - 2, -1, -1):
        r = rad[j]
        log_shape_out = tr[j + 1, 1] * np.log(r) - tr[j + 1, 0] * r
        log_shape_in = tr[j, 1] * np.log(r) - tr[j, 0] * r
        log_amp[j] = log_amp[j + 1] + log_shape_out - log_shape_in
    amplitudes = np.exp(log_amp)
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
        raise ValueError(
            "the matched amplitudes overflowed or underflowed; the transport "
            "parameters and radii are too widely separated to match")
    return amplitudes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`matched_density_amplitudes`."""
    cases = []

    # Normal case: the benchmark's three regions, flat interior.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "stats = _oracle_sensor_alignment_statistics(\n"
            "    0.30, np.array([0.0, 0.5, 1.0]))\n"
            "transport = _oracle_region_transport_parameters(\n"
            "    0.48, stats[:, 1], 2.5)\n"
            "radii = np.array([0.06, 0.55])"
        ),
        "call": "matched_density_amplitudes(transport, radii)",
        "gold_call": "_oracle_matched_density_amplitudes(transport, radii)",
    })

    # Normal case: four regions with widely separated decay lengths, which is
    # where a recursion that multiplies raw exponentials loses the inner
    # amplitudes to underflow rather than carrying them in logarithms. The inner
    # amplitudes here are of order 1e-14, far below the harness's absolute
    # comparison tolerance, so this case is graded on their logarithms:
    # that turns a relative error into an absolute one and keeps every
    # entry discriminating.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.05, 0.03],\n"
            "                      [1.5, 0.09],\n"
            "                      [12.0, 0.2]])\n"
            "radii = np.array([0.1, 0.8, 3.0])"
        ),
        "call": "np.log(matched_density_amplitudes(transport, radii))",
        "gold_call": (
            "np.log(_oracle_matched_density_amplitudes(transport, radii))"
        ),
    })

    # Boundary case: two regions meeting at a single interface.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.6, 0.05]])\n"
            "radii = np.array([0.25])"
        ),
        "call": "matched_density_amplitudes(transport, radii)",
        "gold_call": "_oracle_matched_density_amplitudes(transport, radii)",
    })

    # Edge case: a single region, so the amplitude is exactly 1.0 and radii is
    # empty.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.4, 0.07]])\n"
            "radii = np.array([])"
        ),
        "call": "matched_density_amplitudes(transport, radii)",
        "gold_call": "_oracle_matched_density_amplitudes(transport, radii)",
    })

    # Edge case: two adjacent regions with identical transport parameters, so
    # the interface is invisible and both amplitudes must come out equal.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.3, 0.04],\n"
            "                      [0.3, 0.04]])\n"
            "radii = np.array([0.08, 0.5])"
        ),
        "call": "matched_density_amplitudes(transport, radii)",
        "gold_call": "_oracle_matched_density_amplitudes(transport, radii)",
    })

    # Invalid input: radii that are not strictly increasing.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.15, 0.013],\n"
            "                      [0.28, 0.018]])\n"
            "radii = np.array([0.45, 0.07])\n"
            "def run_model():\n"
            "    try:\n"
            "        matched_density_amplitudes(transport, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_matched_density_amplitudes(transport, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: the wrong number of interface radii.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [0.15, 0.013],\n"
            "                      [0.28, 0.018]])\n"
            "radii = np.array([0.07])\n"
            "def run_model():\n"
            "    try:\n"
            "        matched_density_amplitudes(transport, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_matched_density_amplitudes(transport, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a negative inverse decay length.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "transport = np.array([[0.0, 0.0],\n"
            "                      [-0.15, 0.013]])\n"
            "radii = np.array([0.07])\n"
            "def run_model():\n"
            "    try:\n"
            "        matched_density_amplitudes(transport, radii)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_matched_density_amplitudes(transport, radii)\n"
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
