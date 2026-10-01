"""
Implement evaluate_field, the model coronal magnetic field of one open region.

Each region is described in a Sun-centred Cartesian frame whose third axis is

the region axis. With u3 = z - z_foot the height above the footpoint and

u_perp = (x, y) the transverse offset, the field is



    B_perp = M(u3) . u_perp,      B_z = B3(u3),

    M(u3)  = (t(u3)/2) I + [[p(u3), q(u3)], [q(u3), -p(u3)]],



where B3(u3) = B0 (1 + u3/hB)^(-aB) is the axial profile and p, q are

Gaussian deformation profiles p(u3) = p0 exp(-((u3-up)/wp)^2),

q(u3) = q0 exp(-((u3-uq)/wq)^2). The trace t(u3) is NOT a free function: it

is fixed by requiring div B = 0 exactly for every choice of p and q.



Inputs

------

points: (N, 3) positions in R_sun

region_params: dict with keys B0 (gauss), hB, aB, z_foot,

    p0, up, wp, q0, uq, wq (gauss/R_sun for p0, q0; R_sun for the others)



Returns

-------

B: (N, 3) float, the field in gauss

Returns
-------
B : np.ndarray     (N, 3) magnetic field in gauss at each input position.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_field(points: np.ndarray, region_params: dict) -> np.ndarray:
    '''Evaluate the divergence-free model region field in gauss.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Cartesian positions in units of R_sun.
    region_params : dict
        Keys "B0", "hB", "aB", "z_foot", "p0", "up", "wp", "q0", "uq", "wq",
        all floats. B0 in gauss, p0 and q0 in gauss per R_sun, the length
        scales in R_sun. B0 and hB must be positive.

    Returns
    -------
    B : np.ndarray
        (N, 3) magnetic field in gauss at each input position.

    Raises
    ------
    ValueError
        If points does not have shape (N, 3), if region_params is missing any
        of the ten required keys, if B0 or hB is not positive, or if any point
        fails 1 + (z - z_foot) / hB > 0.
    '''
    return B  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_field(points: np.ndarray, region_params: dict) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(P)):
        raise ValueError("points must contain only finite values")

    rp = region_params
    for key in ("B0", "hB", "aB", "z_foot", "p0", "up", "wp",
                "q0", "uq", "wq"):
        if key not in rp:
            raise ValueError(f"region_params missing key {key!r}")

    B0 = float(rp["B0"])
    hB = float(rp["hB"])
    aB = float(rp["aB"])
    if (not np.isfinite(B0) or not np.isfinite(hB)
            or B0 <= 0.0 or hB <= 0.0):
        raise ValueError("B0 and hB must be positive finite numbers")

    u3 = P[:, 2] - float(rp["z_foot"])
    base = 1.0 + u3 / hB
    if not np.all(np.isfinite(base)) or np.any(base <= 0.0):
        raise ValueError(
            "points lie outside the field domain "
            "(1 + u3/hB must be finite and positive)"
        )

    Bz = B0 * base ** (-aB)
    # div B = dBperp_x/dx + dBperp_y/dy + dBz/dz = t + dBz/du3 = 0
    t = (aB * B0 / hB) * base ** (-aB - 1.0)
    p = float(rp["p0"]) * np.exp(
        -((u3 - float(rp["up"])) / float(rp["wp"])) ** 2
    )
    q = float(rp["q0"]) * np.exp(
        -((u3 - float(rp["uq"])) / float(rp["wq"])) ** 2
    )

    B = np.empty_like(P)
    B[:, 0] = (0.5 * t + p) * P[:, 0] + q * P[:, 1]
    B[:, 1] = q * P[:, 0] + (0.5 * t - p) * P[:, 1]
    B[:, 2] = Bz
    return B

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: batch of generic points, deformed region ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.055, "up": 7.0, "wp": 1.6, "q0": 0.035, "uq": 7.5, "wq": 1.8}
rng = np.random.default_rng(11)
pts = np.column_stack([0.2 * rng.standard_normal(8), 0.2 * rng.standard_normal(8),
                       1.0 + 8.0 * rng.random(8)])
""",
            "call": "evaluate_field(pts, rp)",
            "gold_call": "_oracle_evaluate_field(pts, rp)",
        },
        # --- Boundary: points exactly on the region axis (B_perp must vanish) ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.12, "up": 3.0, "wp": 1.0, "q0": 0.07, "uq": 3.5, "wq": 1.0}
pts = np.column_stack([np.zeros(5), np.zeros(5), np.array([1.0, 2.0, 4.0, 6.0, 9.0])])
""",
            "call": "evaluate_field(pts, rp)",
            "gold_call": "_oracle_evaluate_field(pts, rp)",
        },
        # --- Edge: purely axial region (p0 = q0 = 0), footpoint height included ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
pts = np.array([[0.05, -0.02, 1.0], [0.3, 0.1, 3.5], [-0.2, 0.4, 9.0]])
""",
            "call": "evaluate_field(pts, rp)",
            "gold_call": "_oracle_evaluate_field(pts, rp)",
        },
        # --- Invalid: wrong point shape ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
pts = np.zeros((4, 2))
def run_model():
    try:
        evaluate_field(pts, rp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_field(pts, rp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive B0 ---
        {
            "setup": """import numpy as np
rp = {"B0": -1.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
pts = np.array([[0.0, 0.0, 2.0]])
def run_model():
    try:
        evaluate_field(pts, rp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_field(pts, rp)
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
