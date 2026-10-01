"""
Step 01: the stationary angular statistics of the internal sensor, region by
region.

Step 01: the stationary angular statistics of the internal sensor, region by
region.

Scope
-----
The agent's sensory alignment is modulated by a radial profile that takes one
constant value in each of several radial regions. This step is handed those
constant values and must return, for each of them, the four scalars that the
rest of the pipeline consumes: the sensor's concentration parameter, the first
two stationary trigonometric moments of the sensor's angular error, and that
region's contribution weight.

Layout, ordering and dtype
--------------------------
gamma_values is a 1-D array of the modulation amplitudes, one per region, in
the order the regions are laid out from the origin outwards. The return has
shape (n, 4) and dtype float64, with n = gamma_values.size and the same row
order:

    out[j, 0]  concentration parameter of region j
    out[j, 1]  stationary mean of the cosine of the sensor's angular error
    out[j, 2]  stationary mean of the squared sine of the same angle
    out[j, 3]  contribution weight of region j

"Angular error" means the sensor readout minus the direction preferred by the
steering policy at the agent's position, and every moment is taken in the
statistical steady state at fixed position.

Conventions this step fixes
---------------------------
* The concentration parameter is the ratio of the region's modulation
  amplitude to the sensor accuracy given in the problem statement, in that
  order.
* The moments are pure numbers, not percentages, and are returned as plain
  float64 rather than as a structured or object array.
* The contribution weight of a region is its modulation amplitude multiplied
  by that region's own mean cosine, in that order, and is therefore zero
  wherever the amplitude is zero.
* A region with zero modulation amplitude is legal and must be handled: the
  concentration is then zero, and the two moments take their zero-alignment
  values.
* The sensor accuracy is a single positive scalar shared by every region; the
  modulation acts on the alignment rate alone.

Inputs
------
sigma_phi_sq : float
    Sensor accuracy. Must be finite and strictly positive.
gamma_values : array_like
    1-D, at least one element, every entry finite and in the closed interval
    [0, 1].

Returns
-------
stats : numpy.ndarray of shape (n, 4), dtype float64.

Returns
-------
stats : numpy.ndarray of shape (n, 4), dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sensor_alignment_statistics(sigma_phi_sq, gamma_values):
    """Return the per-region stationary angular statistics of the sensor.

    For each modulation amplitude in gamma_values, return the sensor's
    concentration parameter, the stationary mean cosine and mean squared sine
    of the sensor's angular error -- all evaluated at fixed position in the
    statistical steady state -- and that region's contribution weight.

    Parameters
    ----------
    sigma_phi_sq : float
        The sensor accuracy of the problem statement. Must be finite and
        strictly positive.
    gamma_values : array_like
        1-D array of the per-region modulation amplitudes, ordered from the
        origin outwards. Must be non-empty, finite, and every entry must lie
        in [0, 1].

    Returns
    -------
    stats : numpy.ndarray
        Array of shape (n, 4) and dtype float64 whose row j holds, in order,
        the concentration parameter of region j, the stationary mean cosine of
        the sensor's angular error, the stationary mean squared sine of that
        angle, and the contribution weight of region j.

    Raises
    ------
    ValueError
        If sigma_phi_sq is not a finite, strictly positive real number; if
        gamma_values is not a non-empty 1-D array of real numbers; if any
        entry of gamma_values is not finite; or if any entry lies outside the
        closed interval [0, 1].
    """
    return stats  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import i0e, i1e


def _bessel_ratio(x):
    """I_1(x) / I_0(x), evaluated stably and with the value 0 at x = 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x > 0.0
    out[nz] = i1e(x[nz]) / i0e(x[nz])
    return out


def _oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values):
    """Reference implementation of sensor_alignment_statistics."""
    if isinstance(sigma_phi_sq, bool):
        raise ValueError("sigma_phi_sq must be a real number, got a bool")
    try:
        sigma_phi_sq = float(sigma_phi_sq)
    except (TypeError, ValueError):
        raise ValueError("sigma_phi_sq must be a real number")
    if not np.isfinite(sigma_phi_sq) or sigma_phi_sq <= 0.0:
        raise ValueError("sigma_phi_sq must be finite and strictly positive")

    try:
        gam = np.asarray(gamma_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("gamma_values must be an array of real numbers")
    if gam.ndim != 1:
        raise ValueError("gamma_values must be one-dimensional")
    if gam.size < 1:
        raise ValueError("gamma_values must have at least one entry")
    if not np.all(np.isfinite(gam)):
        raise ValueError("gamma_values must be finite")
    if np.any(gam < 0.0) or np.any(gam > 1.0):
        raise ValueError("every entry of gamma_values must lie in [0, 1]")

    conc = gam / sigma_phi_sq
    mean_cos = _bessel_ratio(conc)

    # <sin^2> = c0(s)/s, which tends to 1/2 as s -> 0.
    mean_sin_sq = np.full(conc.shape, 0.5, dtype=np.float64)
    nz = conc > 0.0
    mean_sin_sq[nz] = mean_cos[nz] / conc[nz]

    stats = np.empty((gam.size, 4), dtype=np.float64)
    stats[:, 0] = conc
    stats[:, 1] = mean_cos
    stats[:, 2] = mean_sin_sq
    stats[:, 3] = gam * mean_cos
    return stats

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`sensor_alignment_statistics`."""
    cases = []

    # Normal case: the three regions of the benchmark profile, all moments
    # order unity.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.30\n"
            "gamma_values = np.array([0.0, 0.5, 1.0])"
        ),
        "call": "sensor_alignment_statistics(sigma_phi_sq, gamma_values)",
        "gold_call": (
            "_oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values)"
        ),
    })

    # Normal case: a sharper sensor and a different set of amplitudes, so the
    # mean cosine approaches unity and the mean squared sine becomes small.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.05\n"
            "gamma_values = np.array([0.1, 0.5, 0.9, 1.0])"
        ),
        "call": "sensor_alignment_statistics(sigma_phi_sq, gamma_values)",
        "gold_call": (
            "_oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values)"
        ),
    })

    # Boundary case: a single region at full modulation.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 1.25\n"
            "gamma_values = np.array([1.0])"
        ),
        "call": "sensor_alignment_statistics(sigma_phi_sq, gamma_values)",
        "gold_call": (
            "_oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values)"
        ),
    })

    # Edge case: zero modulation, where the concentration vanishes and the two
    # moments must fall back on their zero-alignment values rather than divide
    # by zero.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.30\n"
            "gamma_values = np.array([0.0, 0.0])"
        ),
        "call": "sensor_alignment_statistics(sigma_phi_sq, gamma_values)",
        "gold_call": (
            "_oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values)"
        ),
    })

    # Edge case: a very poor sensor, where the concentration is small but
    # non-zero and the moments sit close to their zero-alignment values.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 400.0\n"
            "gamma_values = np.array([0.25, 1.0])"
        ),
        "call": "sensor_alignment_statistics(sigma_phi_sq, gamma_values)",
        "gold_call": (
            "_oracle_sensor_alignment_statistics(sigma_phi_sq, gamma_values)"
        ),
    })

    # Invalid input: a non-positive sensor accuracy.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.0\n"
            "gamma_values = np.array([0.0, 0.5, 1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        sensor_alignment_statistics(sigma_phi_sq, gamma_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensor_alignment_statistics("
            "sigma_phi_sq, gamma_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a modulation amplitude outside [0, 1].
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.30\n"
            "gamma_values = np.array([0.0, 1.4, 1.0])\n"
            "def run_model():\n"
            "    try:\n"
            "        sensor_alignment_statistics(sigma_phi_sq, gamma_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensor_alignment_statistics("
            "sigma_phi_sq, gamma_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a two-dimensional gamma array.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "sigma_phi_sq = 0.30\n"
            "gamma_values = np.array([[0.0, 0.35], [0.5, 1.0]])\n"
            "def run_model():\n"
            "    try:\n"
            "        sensor_alignment_statistics(sigma_phi_sq, gamma_values)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_sensor_alignment_statistics("
            "sigma_phi_sq, gamma_values)\n"
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
