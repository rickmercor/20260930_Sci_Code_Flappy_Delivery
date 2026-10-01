"""
Step 08 - Critical point of the liquid-liquid demixing boundary.

The demixing boundary is a dome in the plane of protein volume fraction against temperature. Its apex is the critical point, the highest temperature at which the solution still separates into two fluid phases. Below it the reduced chemical potential carries a van der Waals loop and two spinodal roots bound the mechanically unstable region; as the temperature rises those two roots approach each other and merge exactly at the critical point. The critical coordinates therefore satisfy

( d mu / d phi ) = 0    and    ( d^2 mu / d phi^2 ) = 0

evaluated together at (phi_c, T_c), with temperature entering only through the dimensionless depth beta*eps_SW = eps_SW / T and k_B = 1 so that eps_SW and T are both in kelvin. Two properties of the answer are worth knowing in advance. The critical volume fraction is set by the interaction range alone: at fixed lambda it does not move when eps_SW or the degeneracy factor change, because those only rescale the temperature axis. The critical temperature, by contrast, depends on both, and it is sensitive to the well factor of Step 06 at the level of a few kelvin. Locating the merge point by bracketing on the existence of the two spinodal roots converges only as fast as the volume-fraction scan used to detect them, which is not accurate enough for a result that later steps consume. Finish the job on the two conditions above so that the answer is independent of any scan resolution.

Returns
-------
np.ndarray of shape (2,) and dtype float, holding [phi_c, T_c] with T_c in kelvin
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_point(lam: float, eps_sw: float, alpha: float) -> "np.ndarray":
    '''Critical volume fraction and critical temperature of the demixing dome.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (2,), [phi_c, T_c], the critical protein volume fraction
        and the critical temperature in kelvin.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or either
        argument is not finite, or no critical point can be bracketed between
        eps_sw / 50 and 5 * eps_sw in temperature.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bisect_root(f, lo, hi, iters=300):
    """Bisection refined to machine precision on a bracketed sign change."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return float(lo)
    if fhi == 0.0:
        return float(hi)
    if flo * fhi > 0.0:
        raise ValueError("no sign change in bracket")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return float(mid)
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if (hi - lo) <= 5e-17 * max(1.0, abs(hi)):
            break
    return float(0.5 * (lo + hi))


def _validate_state(lam, eps_sw, alpha, temperature=1.0):
    """Shared validation for the physical state point."""
    import numpy as np
    for v in (lam, eps_sw, alpha, temperature):
        if not np.isfinite(v):
            raise ValueError("all state arguments must be finite")
    if float(lam) <= 1.0:
        raise ValueError("lam must be > 1")
    if float(eps_sw) <= 0.0:
        raise ValueError("eps_sw must be > 0")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")


def _spinodal_roots(T, lam, eps_sw, alpha, grid):
    """Inner and outer spinodal volume fractions at temperature T, or None above T_c."""
    import numpy as np
    x = float(eps_sw) / float(T)
    f = lambda p: _oracle_reduced_potentials(p, lam, x, alpha)[3]
    v = np.array([f(p) for p in grid], dtype=float)
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    if idx.size < 2:
        return None
    return (_bisect_root(f, float(grid[int(idx[0])]), float(grid[int(idx[0]) + 1])),
            _bisect_root(f, float(grid[int(idx[-1])]), float(grid[int(idx[-1]) + 1])))


def _critical_residual(pc, T, lam, eps_sw, alpha):
    """The two critical conditions: first and second phi-derivatives of the potential."""
    import numpy as np
    v = _oracle_reduced_potentials(pc, lam, float(eps_sw) / float(T), alpha)
    return np.array([v[3], v[4]], dtype=float)


def _oracle_critical_point(lam: float, eps_sw: float, alpha: float) -> "np.ndarray":
    import numpy as np
    _validate_state(lam, eps_sw, alpha)
    eps_sw = float(eps_sw)
    grid = np.linspace(1e-6, 0.70, 2001)
    lo, hi = eps_sw / 50.0, 5.0 * eps_sw
    if _spinodal_roots(lo, lam, eps_sw, alpha, grid) is None:
        raise ValueError("no spinodal at the lower temperature bracket")
    if _spinodal_roots(hi, lam, eps_sw, alpha, grid) is not None:
        raise ValueError("spinodal still present at the upper temperature bracket")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _spinodal_roots(mid, lam, eps_sw, alpha, grid) is not None:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= 1e-13 * max(1.0, hi):
            break
    r = _spinodal_roots(lo, lam, eps_sw, alpha, grid)
    phi_c, t_c = 0.5 * (r[0] + r[1]), lo

    # polish on the two critical conditions so the answer is scan-independent
    hp, ht = 1e-7, 1e-5
    for _ in range(80):
        f0 = _critical_residual(phi_c, t_c, lam, eps_sw, alpha)
        jac = np.empty((2, 2))
        jac[:, 0] = (_critical_residual(phi_c + hp, t_c, lam, eps_sw, alpha)
                     - _critical_residual(phi_c - hp, t_c, lam, eps_sw, alpha)) / (2.0 * hp)
        jac[:, 1] = (_critical_residual(phi_c, t_c + ht, lam, eps_sw, alpha)
                     - _critical_residual(phi_c, t_c - ht, lam, eps_sw, alpha)) / (2.0 * ht)
        try:
            step = np.linalg.solve(jac, f0)
        except np.linalg.LinAlgError:
            break
        phi_c -= step[0]
        t_c -= step[1]
        if abs(step[0]) < 1e-15 and abs(step[1]) < 1e-12:
            break
    return np.array([float(phi_c), float(t_c)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the additive system ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 0.0293
""",
            "call": "critical_point(lam, eps_sw, alpha)",
            "gold_call": "_oracle_critical_point(lam, eps_sw, alpha)",
        },
        # --- Normal: the reference system, same range so the same phi_c ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1687.0, 0.0372
""",
            "call": "critical_point(lam, eps_sw, alpha)",
            "gold_call": "_oracle_critical_point(lam, eps_sw, alpha)",
        },
        # --- Boundary: isotropic limit, where the critical temperature is
        #     an order of magnitude higher ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 1.0
""",
            "call": "critical_point(lam, eps_sw, alpha)",
            "gold_call": "_oracle_critical_point(lam, eps_sw, alpha)",
        },
        # --- Boundary: longer range moves phi_c down and T_c up ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.5, 1687.0, 0.0372
""",
            "call": "critical_point(lam, eps_sw, alpha)",
            "gold_call": "_oracle_critical_point(lam, eps_sw, alpha)",
        },
        # --- Edge: alpha outside (0, 1] must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        critical_point(1.3, 1732.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_critical_point(1.3, 1732.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: non-positive well depth must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        critical_point(1.3, -5.0, 0.03)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_critical_point(1.3, -5.0, 0.03)
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
