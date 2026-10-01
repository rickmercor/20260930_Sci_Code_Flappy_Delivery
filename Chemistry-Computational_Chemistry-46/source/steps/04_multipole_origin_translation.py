"""
Re-express a site's permanent dipole and quadrupole about a new origin, given the shift vector R = R_new - R_i (new origin minus the site's own origin) and the site charge q.

A multipole expansion is defined about a chosen origin, and changing that origin mixes the ranks: translating by R leaves the charge unchanged, shifts the dipole by -q R, and shifts the quadrupole by terms linear in the dipole (and, for a charged site, quadratic in R).  For a charge-neutral site the dipole is origin-independent and the quadrupole picks up only the dipole cross terms, so the transform preserves the trace -- a traceless quadrupole stays traceless.

Returns
-------
return np.zeros(12, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def translate_multipole(mu, theta, R, q: float = 0.0) -> np.ndarray:
    '''Translate a site's permanent dipole and quadrupole to a new origin.

    Parameters
    ----------
    mu : array_like, shape (3,)
        Dipole about the site's own origin.
    theta : array_like
        Quadrupole about the site's own origin, either a (3,3) traceless matrix or the
        5-vector [Theta_xx, Theta_yy, Theta_xy, Theta_xz, Theta_yz].
    R : array_like, shape (3,)
        Shift vector R = R_new - R_i (new common origin minus this site's origin).
    q : float, optional
        Site charge (default 0.0).

    Returns
    -------
    shifted : np.ndarray, shape (12,)
        [mu'_x, mu'_y, mu'_z] followed by the row-major (3,3) shifted quadrupole
        Theta'_ab.

    Raises
    ------
    ValueError
        If ``mu`` or ``R`` is not a 3-vector; if ``mu`` / ``R`` / ``q`` is not finite;
        or if ``theta`` is not a (3, 3) matrix or a 5-vector.   
    '''
    return np.zeros(12, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _quad_to_matrix(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or [xx, yy, xy, xz, yz] into a symmetric (3,3) matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        return 0.5 * (theta + theta.T)
    if theta.shape == (5,):
        xx, yy, xy, xz, yz = theta
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
    raise ValueError("theta must be (3,3) or a 5-vector [xx, yy, xy, xz, yz]")

def _oracle_translate_multipole(mu, theta, R, q: float = 0.0) -> np.ndarray:
    """Reference dipole-shift + quadrupole-shift to a new common origin."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    R = np.asarray(R, dtype=float).reshape(-1)
    if mu.size != 3 or R.size != 3:
        raise ValueError("mu and R must be 3-vectors")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(R))) or not np.isfinite(q):
        raise ValueError("mu, R and q must be finite")
    th = _quad_to_matrix(theta)
    q = float(q)
    Rx, Ry, Rz = R
    mx, my, mz = mu

    mu_new = mu - q * R

    txx = th[0, 0] - 2 * mx * Rx + my * Ry + mz * Rz + q * (Rx ** 2 - 0.5 * (Ry ** 2 + Rz ** 2))
    tyy = th[1, 1] - 2 * my * Ry + mx * Rx + mz * Rz + q * (Ry ** 2 - 0.5 * (Rx ** 2 + Rz ** 2))
    tzz = th[2, 2] - 2 * mz * Rz + mx * Rx + my * Ry + q * (Rz ** 2 - 0.5 * (Rx ** 2 + Ry ** 2))
    txy = th[0, 1] - 1.5 * (mx * Ry + my * Rx) + 1.5 * q * Rx * Ry
    txz = th[0, 2] - 1.5 * (mx * Rz + mz * Rx) + 1.5 * q * Rx * Rz
    tyz = th[1, 2] - 1.5 * (my * Rz + mz * Ry) + 1.5 * q * Ry * Rz

    th_new = np.array([[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]], dtype=float)

    out = np.empty(12, dtype=float)
    out[:3] = mu_new
    out[3:] = th_new.ravel()
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": """import numpy as np
mu = np.array([0.42, -0.28, 0.09], dtype=float)
theta = np.array([0.75, -0.45, 0.22, -0.12, 0.05], dtype=float)
R = np.array([1.3, -2.1, 0.4], dtype=float)
q = 0.0
""",
            "call": "translate_multipole(mu, theta, R, q)",
            "gold_call": "_oracle_translate_multipole(mu, theta, R, q)",
        },
        {
            "setup": """import numpy as np
mu = np.array([0.1, 0.2, -0.3]); theta = np.array([0.2, 0.15, -0.05, 0.1, -0.08])
R = np.zeros(3); q = 0.0
""",
            "call": "translate_multipole(mu, theta, R, q)",
            "gold_call": "_oracle_translate_multipole(mu, theta, R, q)",
        },
        {
            "setup": """import numpy as np
mu = np.array([0.42, -0.28, 0.09], dtype=float)
theta = np.array([0.75, -0.45, 0.22, -0.12, 0.05], dtype=float)
R = np.array([0.5, -0.7, 1.1], dtype=float); q = 0.35
""",
            "call": "translate_multipole(mu, theta, R, q)",
            "gold_call": "_oracle_translate_multipole(mu, theta, R, q)",
        },
        {
            "setup": """import numpy as np
mu = np.zeros(3); theta = np.zeros((3, 3))
def run_model():
    try:
        translate_multipole(mu, theta, np.array([1.0, 2.0])); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_translate_multipole(mu, theta, np.array([1.0, 2.0])); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
mu = np.zeros(3); R = np.zeros(3)
def run_model():
    try:
        translate_multipole(mu, np.zeros(4), R); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_translate_multipole(mu, np.zeros(4), R); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
