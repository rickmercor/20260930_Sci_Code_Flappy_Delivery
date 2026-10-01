"""
Implement deformation_rate: the transverse deformation rate of the field

direction along a line.

A region whose field is structured across itself, and not only along its own

length, shears the field direction as one moves perpendicular to it. This

step returns that shear at each station as a single non-negative rate per

unit length, in 1/R_sun. It is a local quantity, fixed by the field direction

and its first spatial derivatives at the station alone, and its value does

not depend on how a basis perpendicular to the field is oriented. Take the

spatial derivatives of the step-01 field by central differences with

half-step h_fd, in R_sun.



Inputs

------

region_params: dict, as in step 01

positions: (N, 3) station positions in R_sun

h_fd: float > 0, differencing half-step in R_sun



Returns

-------

smag: (N,) float, the transverse deformation rate in 1/R_sun

Returns
-------
smag : np.ndarray     (N,) non-negative deformation rate in 1/R_sun; independent of the     choice of perpendicular basis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deformation_rate(region_params: dict, positions: np.ndarray,
                     h_fd: float) -> np.ndarray:
    '''Transverse deformation rate of the field direction at each station.

    Parameters
    ----------
    region_params : dict
        Field parameters accepted by evaluate_field (step 01).
    positions : np.ndarray
        (N, 3) station positions in units of R_sun.
    h_fd : float
        Positive central-differencing half-step in R_sun.

    Returns
    -------
    smag : np.ndarray
        (N,) non-negative deformation rate in 1/R_sun; independent of the
        choice of perpendicular basis.

    Raises
    ------
    ValueError
        If positions does not have shape (N, 3), or if h_fd is not a positive
        finite number. A region_params dict rejected by step 01 propagates that
        ValueError.
    '''
    return smag  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deformation_rate(region_params: dict, positions: np.ndarray,
                           h_fd: float) -> np.ndarray:
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    h = float(h_fd)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h_fd must be a positive finite number")
    n = len(P)

    off = np.zeros((6, 3))
    for j in range(3):
        off[2 * j, j] = h
        off[2 * j + 1, j] = -h
    B = _oracle_evaluate_field((P[:, None, :] + off[None, :, :]).reshape(-1, 3),
                             region_params).reshape(n, 6, 3)
    G = np.empty((n, 3, 3))
    for j in range(3):
        G[:, :, j] = (B[:, 2 * j] - B[:, 2 * j + 1]) / (2.0 * h)

    B0 = _oracle_evaluate_field(P, region_params)
    bmag = np.linalg.norm(B0, axis=1)
    bh = B0 / bmag[:, None]
    proj = np.eye(3)[None] - bh[:, :, None] * bh[:, None, :]
    gb = np.einsum("nij,njk->nik", proj, G) / bmag[:, None, None]
    Gp = np.einsum("nij,njk->nik", gb, proj)

    # any orthonormal perpendicular pair; |S| does not depend on the choice
    e = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs(np.einsum("ni,ni->n", e, bh)) > 0.9
    e[flip] = np.array([1.0, 0.0, 0.0])
    e1 = e - np.einsum("ni,ni->n", e, bh)[:, None] * bh
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(bh, e1)
    E = np.stack([e1, e2], axis=1)
    M = np.einsum("nai,nij,nbj->nab", E, Gp, E)
    S = 0.5 * (M + M.transpose(0, 2, 1))
    S -= 0.5 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(2)[None]
    return np.sqrt(np.einsum("nab,nab->n", S, S) / 2.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: on-axis stations of a deformed region ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.055, "up": 7.0, "wp": 1.6, "q0": 0.035, "uq": 7.5, "wq": 1.8}
pts = np.column_stack([np.zeros(41), np.zeros(41), np.linspace(1.0, 9.0, 41)])
""",
            "call": "deformation_rate(rp, pts, 2e-5)",
            "gold_call": "_oracle_deformation_rate(rp, pts, 2e-5)",
        },
        # --- Normal: off-axis stations ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.12, "up": 3.0, "wp": 1.0, "q0": 0.07, "uq": 3.5, "wq": 1.0}
rng = np.random.default_rng(7)
pts = np.column_stack([0.1 * rng.standard_normal(12), 0.1 * rng.standard_normal(12),
                       1.0 + 8.0 * rng.random(12)])
""",
            "call": "deformation_rate(rp, pts, 2e-5)",
            "gold_call": "_oracle_deformation_rate(rp, pts, 2e-5)",
        },
        # --- Boundary: purely axial region, |S| must vanish at every station ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
pts = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(1.0, 9.0, 11)])
""",
            "call": "deformation_rate(rp, pts, 2e-5)",
            "gold_call": "_oracle_deformation_rate(rp, pts, 2e-5)",
        },
        # --- Invalid: non-positive half-step ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
pts = np.array([[0.0, 0.0, 2.0]])
def run_model():
    try:
        deformation_rate(rp, pts, -1e-5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_deformation_rate(rp, pts, -1e-5)
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
