"""
Project the laser's dimensionless intensity profile onto every transverse eigenmode returned by the eigenbasis step, using whichever physical direction is relevant to that mode's own parity.

The quantity projected here is the laser's dimensionless intensity profile inside the beam, which is the source term of the pipeline's dimensionless heat problem. The absorption coefficient, the heat-conversion fraction and the incident intensity are all contained in the temperature scale, so the profile itself is projected, with no further factor. It varies independently in two directions. Through the depth of the beam, light entering at the illuminated face X = -1/2 loses intensity as it travels deeper into the absorbing material, at a rate proportional to the local intensity itself, with the constant of proportionality set by the given attenuation coefficient beta; the dimensionless intensity equals 1 at the illuminated face, and working out how it varies with depth from this rate law is required, not given. Across the beam's lateral extent, the laser has a Gaussian intensity profile centred on the mid-plane X = 0, equal to 1 at the centre and falling to 1/e of that value at a distance w from it. Only the odd-symmetry eigenfunctions (parity tag 1.0) have a nonzero depth-direction moment arm, so only they can contribute a depth-direction bending effect, and their relevant projection A is the L2 inner product of their eigenfunction with the depth-wise attenuation profile described above. Only the even-symmetry eigenfunctions (parity tag 0.0) have a nonzero cross-sectional average, so only they can contribute to the lateral extent of the source, and their relevant projection B is the L2 inner product of their eigenfunction with the lateral Gaussian profile described above. Neither profile is given here as an explicit formula, and neither integral is given in closed form; both the profile expressions and their projection integrals must be worked out directly from the physical description above, whether analytically or by accurate numerical quadrature.

Every row of the input eigenbasis must receive exactly one projection value in the output, computed using the integral appropriate to that row's own parity tag: the depth-wise (attenuation) integral for rows with parity 1.0, and the lateral (Gaussian) integral for rows with parity 0.0. The output must be aligned row-for-row with the input eigenbasis (same order, same length), not grouped or reordered by parity.

Returns
-------
np.ndarray of shape (N,), float: the projection value for each row of the input eigenbasis, in the same order as the input.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_heat_source_onto_modes(modes: np.ndarray, beta: float, w: float) -> np.ndarray:
    """Per-mode projection of the laser's dimensionless intensity profile, branching internally by parity.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis (the two parity counts need not be equal):
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only).
    beta : float
        Dimensionless optical (depth-wise) attenuation coefficient, a
        finite number > 0.
    w : float
        Dimensionless laser radius, a finite number > 0.

    Returns
    -------
    proj : np.ndarray
        Array of shape (N,): the projection value for each row of modes,
        in the same order as modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least 2 rows; if any eigenvalue or normalization constant in modes
        is not finite and > 0; if any parity tag is not exactly 0.0 or
        1.0; or if beta or w is not a finite number > 0.
    """
    return proj  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_project_heat_source_onto_modes(modes: np.ndarray, beta: float, w: float) -> np.ndarray:
    import numpy as np
    from scipy.integrate import quad

    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 2:
        raise ValueError("modes must have shape (N, 3) with N >= 2")
    nu_col, c_col, parity_col = modes[:, 0], modes[:, 1], modes[:, 2]
    if not np.all(np.isfinite(nu_col)) or np.any(nu_col <= 0.0):
        raise ValueError("every eigenvalue in modes must be finite and > 0")
    if not np.all(np.isfinite(c_col)) or np.any(c_col <= 0.0):
        raise ValueError("every normalization constant in modes must be finite and > 0")
    if not np.all(np.isin(parity_col, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    if not (isinstance(beta, (int, float, np.floating)) and np.isfinite(beta) and float(beta) > 0.0):
        raise ValueError("beta must be a finite number > 0")
    if not (isinstance(w, (int, float, np.floating)) and np.isfinite(w) and float(w) > 0.0):
        raise ValueError("w must be a finite number > 0")

    beta = float(beta)
    w = float(w)

    proj = np.empty(modes.shape[0], dtype=float)
    for i in range(modes.shape[0]):
        nu, C, parity = nu_col[i], c_col[i], parity_col[i]
        if parity == 1.0:
            val, _ = quad(lambda X: np.exp(-beta * (X + 0.5)) * C * np.sin(nu * X), -0.5, 0.5)
        else:
            val, _ = quad(lambda X: np.exp(-X ** 2 / w ** 2) * C * np.cos(nu * X), -0.5, 0.5)
        proj[i] = val

    return proj

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark combined eigenbasis (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
beta, w = 0.55, 2.0
""",
            "call": "project_heat_source_onto_modes(modes.copy(), beta, w)",
            "gold_call": "_oracle_project_heat_source_onto_modes(modes.copy(), beta, w)",
        },
        # --- Valid: a much stronger attenuation coefficient and a narrower laser ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.3, 1.10, 0.0],
    [2.8, 1.30, 1.0],
    [6.6, 1.38, 0.0],
    [8.1, 1.39, 1.0],
])
beta, w = 4.0, 0.3
""",
            "call": "project_heat_source_onto_modes(modes.copy(), beta, w)",
            "gold_call": "_oracle_project_heat_source_onto_modes(modes.copy(), beta, w)",
        },
        # --- Boundary: the minimum 2-row eigenbasis (one even, one odd) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
beta, w = 1.2, 1.0
""",
            "call": "project_heat_source_onto_modes(modes.copy(), beta, w)",
            "gold_call": "_oracle_project_heat_source_onto_modes(modes.copy(), beta, w)",
        },
        # --- Consistency: an odd-family row's projection matches direct numerical quadrature
        # of the Beer-Lambert integral, and an even-family row's matches the Gaussian integral ---
        {
            "setup": """import numpy as np
from scipy.integrate import quad
modes = np.array([
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
beta, w = 0.55, 2.0
def check(fn):
    p = np.asarray(fn(modes.copy(), beta, w), dtype=float)
    nu_e, C_e = modes[0, 0], modes[0, 1]
    val_e, _ = quad(lambda X: np.exp(-X**2/w**2)*C_e*np.cos(nu_e*X), -0.5, 0.5)
    nu_o, C_o = modes[1, 0], modes[1, 1]
    val_o, _ = quad(lambda X: np.exp(-beta*(X+0.5))*C_o*np.sin(nu_o*X), -0.5, 0.5)
    return int(abs(p[0]-val_e) < 1e-6 and abs(p[1]-val_o) < 1e-6)
""",
            "call": "check(project_heat_source_onto_modes)",
            "gold_call": "check(_oracle_project_heat_source_onto_modes)",
        },
        # --- Invalid: a parity tag that is neither 0.0 nor 1.0 ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 0.5],
])
def run_model():
    try:
        project_heat_source_onto_modes(modes.copy(), 0.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_project_heat_source_onto_modes(modes.copy(), 0.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive beta ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
def run_model():
    try:
        project_heat_source_onto_modes(modes.copy(), 0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_project_heat_source_onto_modes(modes.copy(), 0.0, 1.0)
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
