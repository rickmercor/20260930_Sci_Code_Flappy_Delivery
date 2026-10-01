"""
Evaluate a normalized asymmetric multi-cycle terahertz vector potential at an array of times. The routine takes times and pulse widths in picoseconds, frequency in terahertz and phase in radians, then returns a dimensionless NumPy array with the same shape as the times. Invalid input raises ValueError: the pulse center, both widths, frequency and phase must be finite scalars, both widths must be strictly greater than zero and every time must be finite.

The asymmetric envelope is fitted to a real terahertz pulse whose rise and fall have different widths. The drive enters the gap dynamics through the square of this vector potential, so the relevant drive frequency is twice the optical one.

Returns
-------
numpy.ndarray giving the dimensionless normalized vector potential, same shape as the input times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pulse_envelope(times, t_center, sigma_before, sigma_after, frequency, phase):
    """Return the normalized asymmetric terahertz vector potential.

    Parameters
    ----------
    times : array_like
        Times in picoseconds.
    t_center : float
        Pulse center in picoseconds.
    sigma_before : float
        Positive envelope width before the pulse center in picoseconds.
    sigma_after : float
        Positive envelope width at and after the pulse center in picoseconds.
    frequency : float
        Carrier frequency in terahertz.
    phase : float
        Carrier phase in radians.

    Returns
    -------
    numpy.ndarray
        Dimensionless normalized vector potential with the same shape as ``times``.

    Raises
    ------
    ValueError
        If any scalar parameter is not finite, either width is not strictly
        positive, or any entry of ``times`` is not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pulse_envelope(times, t_center, sigma_before, sigma_after, frequency, phase):
    values = (
        (t_center, "t_center"),
        (sigma_before, "sigma_before"),
        (sigma_after, "sigma_after"),
        (frequency, "frequency"),
        (phase, "phase"),
    )
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    t_center, sigma_before, sigma_after, frequency, phase = converted
    if sigma_before <= 0.0:
        raise ValueError("sigma_before must be strictly greater than zero")
    if sigma_after <= 0.0:
        raise ValueError("sigma_after must be strictly greater than zero")
    try:
        t = np.asarray(times, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must all be finite") from exc
    if not np.all(np.isfinite(t)):
        raise ValueError("times must all be finite")
    sigma = np.where(t < t_center, sigma_before, sigma_after)
    envelope = np.exp(-((t - t_center) ** 2) / sigma**2)
    carrier = np.cos(2.0 * np.pi * frequency * t + phase)
    return np.asarray(envelope * carrier, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n",
            "call": "round(float(np.asarray(pulse_envelope(np.array([0.0]), 0.0, 0.4, 0.8, 0.25, 0.0))[0]), 12)",
            "gold_call": "round(float(np.asarray(_oracle_pulse_envelope(np.array([0.0]), 0.0, 0.4, 0.8, 0.25, 0.0))[0]), 12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(float(v), 10) for v in np.asarray(pulse_envelope(np.array([-0.5, 0.5]), 0.0, 0.4, 0.8, 0.25, 0.0))]",
            "gold_call": "[round(float(v), 10) for v in np.asarray(_oracle_pulse_envelope(np.array([-0.5, 0.5]), 0.0, 0.4, 0.8, 0.25, 0.0))]",
        },
        {
            "setup": "import numpy as np\n",
            "call": "int(bool(np.all(np.abs(np.asarray(pulse_envelope(np.array([-8.0, 8.0]), 0.0, 0.5, 1.0, 0.4, 0.0))) < 1e-20)))",
            "gold_call": "int(bool(np.all(np.abs(np.asarray(_oracle_pulse_envelope(np.array([-8.0, 8.0]), 0.0, 0.5, 1.0, 0.4, 0.0))) < 1e-20)))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(float(v), 10) for v in np.asarray(pulse_envelope(np.array([-1.0, 0.0, 1.0]), 0.0, 1.0, 1.0, 0.0, 0.0))]",
            "gold_call": "[round(float(v), 10) for v in np.asarray(_oracle_pulse_envelope(np.array([-1.0, 0.0, 1.0]), 0.0, 1.0, 1.0, 0.0, 0.0))]",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        pulse_envelope(np.array([0.0, np.nan]), 0.0, 0.4, 0.8, 0.25, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_pulse_envelope(np.array([0.0, np.nan]), 0.0, 0.4, 0.8, 0.25, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
