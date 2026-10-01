"""
Implement plasma_background, the spherically symmetric plasma state of the

open corona evaluated along a field line.

The mass density falls as a power law of heliocentric distance,

rho(r) = rho0 (r/r0)^(-alpha_rho), and the outflow speed follows the

saturating profile U(r) = u0 + (uinf - u0) (1 - exp(-(r - r0)/lu)) for

r >= r0, with U(r) = u0 inside r0. The input positions are Cartesian in

R_sun; the returned speeds are converted to cm/s (1 km = 1e5 cm).



Inputs

------

positions: (N, 3) Cartesian positions in R_sun

bg_params: dict with rho0 (g cm^-3), r0 (R_sun), alpha_rho,

    u0 (km/s), uinf (km/s), lu (R_sun)



Returns

-------

out: (N, 2) float, columns [rho in g cm^-3, U in cm/s]

Returns
-------
out : np.ndarray     (N, 2) array; column 0 is rho in g cm^-3, column 1 is U in cm/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def plasma_background(positions: np.ndarray, bg_params: dict) -> np.ndarray:
    '''Evaluate the plasma density and outflow speed at each position.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) Cartesian positions in units of R_sun.
    bg_params : dict
        Keys "rho0" (g cm^-3, > 0), "r0" (R_sun, > 0), "alpha_rho",
        "u0" (km/s), "uinf" (km/s), "lu" (R_sun, > 0).

    Returns
    -------
    out : np.ndarray
        (N, 2) array; column 0 is rho in g cm^-3, column 1 is U in cm/s.

    Raises
    ------
    ValueError
        If positions does not have shape (N, 3), if bg_params is missing any of
        the six required keys, if rho0, r0 or lu is not positive, or if any
        position has non-positive heliocentric distance.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_plasma_background(
    positions: np.ndarray,
    bg_params: dict
) -> np.ndarray:
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.all(np.isfinite(P)):
        raise ValueError("positions must contain only finite values")

    bg = bg_params
    for key in ("rho0", "r0", "alpha_rho", "u0", "uinf", "lu"):
        if key not in bg:
            raise ValueError(f"bg_params missing key {key!r}")

    rho0 = float(bg["rho0"])
    r0 = float(bg["r0"])
    lu = float(bg["lu"])
    if (not np.isfinite(rho0) or not np.isfinite(r0)
            or not np.isfinite(lu)
            or rho0 <= 0.0 or r0 <= 0.0 or lu <= 0.0):
        raise ValueError("rho0, r0 and lu must be positive finite numbers")

    r = np.linalg.norm(P, axis=1)
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError(
            "positions must have finite positive heliocentric distance"
        )

    rho = rho0 * (r / r0) ** (-float(bg["alpha_rho"]))
    x = np.maximum(r - r0, 0.0)
    u_kms = (
        float(bg["u0"])
        + (float(bg["uinf"]) - float(bg["u0"]))
        * (1.0 - np.exp(-x / lu))
    )

    out = np.empty((len(P), 2))
    out[:, 0] = rho
    out[:, 1] = u_kms * 1.0e5
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: stations along a radial line ---
        {
            "setup": """import numpy as np
bg = {"rho0": 2e-16, "r0": 1.0, "alpha_rho": 4.0, "u0": 10.0, "uinf": 650.0, "lu": 3.0}
pts = np.column_stack([np.zeros(9), np.zeros(9), np.linspace(1.0, 9.0, 9)])
""",
            "call": "plasma_background(pts, bg)",
            "gold_call": "_oracle_plasma_background(pts, bg)",
        },
        # --- Boundary: positions at and below the reference radius r0 ---
        {
            "setup": """import numpy as np
bg = {"rho0": 2e-16, "r0": 1.0, "alpha_rho": 4.0, "u0": 10.0, "uinf": 650.0, "lu": 3.0}
pts = np.array([[0.6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0001]])
""",
            "call": "plasma_background(pts, bg)",
            "gold_call": "_oracle_plasma_background(pts, bg)",
        },
        # --- Normal: order-unity density scale, so the density column is graded
        #     independently of the much larger speed column ---
        {
            "setup": """import numpy as np
bg = {"rho0": 1.0, "r0": 2.0, "alpha_rho": 2.0, "u0": 1.0, "uinf": 4.0, "lu": 1.0}
pts = np.column_stack([np.zeros(5), np.zeros(5), np.array([1.0, 2.0, 3.0, 5.0, 8.0])])
""",
            "call": "plasma_background(pts, bg)",
            "gold_call": "_oracle_plasma_background(pts, bg)",
        },
        # --- Edge: off-axis positions where r != z ---
        {
            "setup": """import numpy as np
bg = {"rho0": 5e-17, "r0": 1.0, "alpha_rho": 3.5, "u0": 5.0, "uinf": 500.0, "lu": 2.0}
rng = np.random.default_rng(3)
pts = rng.random((6, 3)) + np.array([0.5, 0.5, 1.0])
""",
            "call": "plasma_background(pts, bg)",
            "gold_call": "_oracle_plasma_background(pts, bg)",
        },
        # --- Invalid: non-positive rho0 ---
        {
            "setup": """import numpy as np
bg = {"rho0": 0.0, "r0": 1.0, "alpha_rho": 4.0, "u0": 10.0, "uinf": 650.0, "lu": 3.0}
pts = np.array([[0.0, 0.0, 2.0]])
def run_model():
    try:
        plasma_background(pts, bg)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_plasma_background(pts, bg)
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
