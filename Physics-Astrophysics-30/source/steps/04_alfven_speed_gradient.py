"""
Implement alfven_speed_gradient: the Alfven speed profile along a line and

its logarithmic arclength gradient.

The first column is the Alfven speed in cgs (B in gauss, rho in g cm^-3,

v_A in cm/s). The second column is its logarithmic derivative along the

line, K_vA = d ln v_A / ds, with s the arclength in cm. On an equally

spaced station grid that derivative uses second-order central differences in

the interior and first-order one-sided differences at the two endpoints (the

default numpy.gradient convention for uniform spacing, edge_order=1).



Inputs

------

bmag: (N,) field magnitude in gauss at the stations

rho: (N,) mass density in g cm^-3 at the stations

ds_cm: float > 0, station spacing in cm



Returns

-------

out: (N, 2) float, columns [v_A in cm/s, K_vA in 1/cm]

Returns
-------
out : np.ndarray     (N, 2) array; column 0 is v_A in cm/s, column 1 is K_vA in 1/cm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def alfven_speed_gradient(bmag: np.ndarray, rho: np.ndarray, ds_cm: float) -> np.ndarray:
    '''Alfven speed and its logarithmic arclength derivative.

    Parameters
    ----------
    bmag : np.ndarray
        (N,) magnetic field magnitude in gauss, N >= 3, all positive.
    rho : np.ndarray
        (N,) mass density in g cm^-3, same length, all positive.
    ds_cm : float
        Positive spacing between consecutive stations, in cm.

    Returns
    -------
    out : np.ndarray
        (N, 2) array; column 0 is v_A in cm/s, column 1 is K_vA in 1/cm.

    Raises
    ------
    ValueError
        If bmag and rho are not 1-D arrays of the same length, if they hold
        fewer than 3 stations, if any entry of bmag or rho is not positive, or
        if ds_cm is not a positive finite number.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_alfven_speed_gradient(
    bmag: np.ndarray,
    rho: np.ndarray,
    ds_cm: float
) -> np.ndarray:
    b = np.asarray(bmag, dtype=float)
    d = np.asarray(rho, dtype=float)

    if b.ndim != 1 or d.ndim != 1 or b.shape != d.shape:
        raise ValueError("bmag and rho must be 1-D arrays of the same length")
    if b.size < 3:
        raise ValueError("need at least 3 stations")
    if (not np.all(np.isfinite(b))
            or not np.all(np.isfinite(d))
            or np.any(b <= 0.0)
            or np.any(d <= 0.0)):
        raise ValueError(
            "bmag and rho must contain only finite positive values"
        )
    if not np.isfinite(float(ds_cm)) or float(ds_cm) <= 0.0:
        raise ValueError("ds_cm must be a positive finite number")

    va = b / np.sqrt(4.0 * np.pi * d)
    kva = np.gradient(np.log(va), float(ds_cm))

    out = np.empty((b.size, 2))
    out[:, 0] = va
    out[:, 1] = kva
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: power-law field over power-law density ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 8.0, 201)
bmag = 3.0 * (1.0 + s / 0.8) ** (-1.9)
rho = 2e-16 * (1.0 + s) ** (-4.0)
ds_cm = (s[1] - s[0]) * 6.957e10
""",
            "call": "alfven_speed_gradient(bmag, rho, ds_cm)",
            "gold_call": "_oracle_alfven_speed_gradient(bmag, rho, ds_cm)",
        },
        # --- Boundary: constant v_A, the gradient must vanish identically ---
        {
            "setup": """import numpy as np
bmag = np.full(11, 2.5)
rho = np.full(11, 1e-16)
ds_cm = 1e9
""",
            "call": "alfven_speed_gradient(bmag, rho, ds_cm)",
            "gold_call": "_oracle_alfven_speed_gradient(bmag, rho, ds_cm)",
        },
        # --- Edge: exponential v_A, K_vA constant in the interior ---
        {
            "setup": """import numpy as np
s = np.linspace(0.0, 2.0, 41)
bmag = np.exp(0.7 * s)
rho = np.full(41, 4.0 * np.pi)
ds_cm = s[1] - s[0]
""",
            "call": "alfven_speed_gradient(bmag, rho, ds_cm)",
            "gold_call": "_oracle_alfven_speed_gradient(bmag, rho, ds_cm)",
        },
        # --- Invalid: mismatched lengths ---
        {
            "setup": """import numpy as np
bmag = np.ones(10)
rho = np.ones(9) * 1e-16
def run_model():
    try:
        alfven_speed_gradient(bmag, rho, 1e9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_alfven_speed_gradient(bmag, rho, 1e9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive spacing ---
        {
            "setup": """import numpy as np
bmag = np.ones(10)
rho = np.ones(10) * 1e-16
def run_model():
    try:
        alfven_speed_gradient(bmag, rho, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_alfven_speed_gradient(bmag, rho, 0.0)
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
