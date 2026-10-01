"""
Implement wave_amplitude: the rms amplitude of the outward Alfvenic

fluctuation along a line, in the presence of an inhomogeneous background

flow and reflection-driven damping.

How much amplitude the outward fluctuation retains as it travels is set by

the way the flow and the Alfven speed change between the footpoint and the

station: the rms Elsasser amplitude z+ there follows from its footpoint value

and from the Alfven Mach number M_A = U / v_A at both ends. Reflection then

attenuates it by the factor exp(-I(s)), with I(s) the damping accumulated

along the line: integrate `damping` over arclength by the cumulative

trapezoid rule on the given stations, so that I is zero at the first station.



Inputs

------

zp0_cms: float > 0, rms amplitude at the first station, cm/s

va: (N,) Alfven speed in cm/s

u: (N,) outflow speed in cm/s

damping: (N,) damping rate in 1/cm

s_cm: (N,) arclength of each station in cm, increasing, s_cm[0] = 0



Returns

-------

zp: (N,) float, rms outward amplitude in cm/s

Returns
-------
zp : np.ndarray     (N,) rms outward amplitude in cm/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def wave_amplitude(zp0_cms: float, va: np.ndarray, u: np.ndarray,
                   damping: np.ndarray, s_cm: np.ndarray) -> np.ndarray:
    '''Outward Elsasser amplitude at each station along the line.

    Parameters
    ----------
    zp0_cms : float
        Positive rms amplitude of the outward fluctuation at the first
        station, in cm/s.
    va : np.ndarray
        (N,) Alfven speed in cm/s, all positive.
    u : np.ndarray
        (N,) outflow speed in cm/s, all positive.
    damping : np.ndarray
        (N,) damping rate in 1/cm.
    s_cm : np.ndarray
        (N,) strictly increasing arclength in cm with s_cm[0] = 0.

    Returns
    -------
    zp : np.ndarray
        (N,) rms outward amplitude in cm/s.

    Raises
    ------
    ValueError
        If zp0_cms is not a positive finite number, if va, u, damping and s_cm
        are not all 1-D arrays, if they do not all have the same length, if any
        entry of va or u is not positive, or if s_cm does not start at 0 and
        increase strictly.
    '''
    return zp  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wave_amplitude(
    zp0_cms: float,
    va: np.ndarray,
    u: np.ndarray,
    damping: np.ndarray,
    s_cm: np.ndarray
) -> np.ndarray:
    z0 = float(zp0_cms)
    if not np.isfinite(z0) or z0 <= 0.0:
        raise ValueError("zp0_cms must be a positive finite number")

    va_ = np.asarray(va, dtype=float)
    u_ = np.asarray(u, dtype=float)
    dmp = np.asarray(damping, dtype=float)
    s = np.asarray(s_cm, dtype=float)

    if not (va_.ndim == u_.ndim == dmp.ndim == s.ndim == 1):
        raise ValueError("va, u, damping and s_cm must be 1-D arrays")
    if not (va_.shape == u_.shape == dmp.shape == s.shape):
        raise ValueError("va, u, damping and s_cm must have the same length")
    if va_.size == 0:
        raise ValueError("va, u, damping and s_cm must not be empty")
    if (not np.all(np.isfinite(va_))
            or not np.all(np.isfinite(u_))
            or not np.all(np.isfinite(dmp))
            or not np.all(np.isfinite(s))):
        raise ValueError(
            "va, u, damping and s_cm must contain only finite values"
        )
    if np.any(va_ <= 0.0) or np.any(u_ <= 0.0):
        raise ValueError("va and u must be positive")
    if s[0] != 0.0 or np.any(np.diff(s) <= 0.0):
        raise ValueError("s_cm must start at 0 and increase strictly")

    ma = u_ / va_
    g = np.sqrt(ma) + 1.0 / np.sqrt(ma)
    integral = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (dmp[1:] + dmp[:-1]) * np.diff(s)),
    ])

    return z0 * (g[0] / g) * np.exp(-integral)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: accelerating wind, flat Alfven speed, uniform damping ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 8.0, 101) * 6.957e10
va = np.full(101, 5.5e7)
u = 1e5 * (10.0 + 640.0 * (1.0 - np.exp(-np.linspace(0.0, 8.0, 101) / 3.0)))
damping = np.full(101, 2.0e-12)
""",
            "call": "wave_amplitude(3e6, va, u, damping, s)",
            "gold_call": "_oracle_wave_amplitude(3e6, va, u, damping, s)",
        },
        # --- Boundary: no damping and constant Mach number -> constant amplitude ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 1.0, 11) * 6.957e10
va = np.full(11, 4.0e7)
u = np.full(11, 2.0e7)
damping = np.zeros(11)
""",
            "call": "wave_amplitude(3e6, va, u, damping, s)",
            "gold_call": "_oracle_wave_amplitude(3e6, va, u, damping, s)",
        },
        # --- Edge: crossing the Alfven point (M_A passes through 1) ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 4.0, 41) * 6.957e10
va = np.full(41, 3.0e7)
u = np.linspace(0.5e7, 6.0e7, 41)
damping = np.full(41, 5.0e-13)
""",
            "call": "wave_amplitude(2e6, va, u, damping, s)",
            "gold_call": "_oracle_wave_amplitude(2e6, va, u, damping, s)",
        },
        # --- Invalid: arclength not starting at zero ---
        {
            "setup": """import numpy as np
s = np.linspace(1.0, 2.0, 11) * 6.957e10
va = np.full(11, 4.0e7)
u = np.full(11, 2.0e7)
damping = np.zeros(11)
def run_model():
    try:
        wave_amplitude(3e6, va, u, damping, s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wave_amplitude(3e6, va, u, damping, s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive amplitude ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 1.0, 11) * 6.957e10
va = np.full(11, 4.0e7)
u = np.full(11, 2.0e7)
damping = np.zeros(11)
def run_model():
    try:
        wave_amplitude(0.0, va, u, damping, s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wave_amplitude(0.0, va, u, damping, s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
