"""
Step 09 - Liquid-liquid coexistence (binodal) volume fractions.

Below the critical temperature the protein solution separates into a protein-poor ("gas", I) and a protein-rich ("liquid", II) fluid phase. Because the solvent is absorbed into the background, coexistence is fixed by equality of the reduced protein chemical potential and of the reduced osmotic pressure between the two phases,

mu(phi_I, beta) = mu(phi_II, beta) pi(phi_I, beta) = pi(phi_II, beta)

with phi_I < phi_II. This is the common-tangent construction on a(phi). Matching only the chemical potential leaves the boundary undetermined; the osmotic pressure rather than the total pressure is the mechanical variable, because mu describes insertion of a protein with isochoric removal of solvent. Below the critical temperature mu(phi) is not monotonic: it carries a van der Waals loop, so the equal-chemical-potential condition alone has three solutions. Exactly one pair also shares the osmotic pressure, and that pair is the binodal. A solver must therefore return the non-trivial pair, separated by more than 1e-6 in volume fraction, rather than the coincident root phi_I = phi_II that trivially satisfies both equations. Choose a scheme that converges on that pair to full double precision and whose answer does not depend on the resolution of any scan used to start it. Temperature enters only through the dimensionless depth beta*eps_SW = eps_SW / T, with k_B = 1 so that eps_SW and T are both in kelvin.

Returns
-------
np.ndarray of shape (2,) and dtype float, holding [phi_I, phi_II] with phi_I < phi_II
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def llps_binodal(lam: float, eps_sw: float, alpha: float,
                 temperature: float) -> "np.ndarray":
    '''Coexisting protein volume fractions of the liquid-liquid binodal.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0.

    Returns
    -------
    out : np.ndarray
        Array of shape (2,), [phi_I, phi_II], the protein-poor and protein-rich
        coexisting volume fractions, with phi_I < phi_II.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or temperature
        <= 0, or any argument is not finite, or the state point admits no
        liquid-liquid coexistence that can be resolved on the volume-fraction
        window (0, 0.74). The latter happens at or above the critical
        temperature, where the loop disappears, and also for extremely deep
        quenches, where the protein-poor branch underflows below the window.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np




def _binodal_outer_roots(m, lam, x, alpha, grid, i1, i2):
    """The dilute and dense volume fractions whose chemical potential equals m."""
    mu_of = lambda p: _oracle_reduced_potentials(p, lam, x, alpha)[1]
    pI = _bisect_root(lambda p: mu_of(p) - m, float(grid[0]), float(grid[i1]))
    pII = _bisect_root(lambda p: mu_of(p) - m, float(grid[i2]), float(grid[-1]))
    return pI, pII


def _binodal_pressure_gap(m, lam, x, alpha, grid, i1, i2):
    """Osmotic-pressure difference between the two branches at equal chemical potential."""
    pi_of = lambda p: _oracle_reduced_potentials(p, lam, x, alpha)[2]
    pI, pII = _binodal_outer_roots(m, lam, x, alpha, grid, i1, i2)
    return pi_of(pII) - pi_of(pI)


def _oracle_llps_binodal(lam: float, eps_sw: float, alpha: float,
                         temperature: float) -> "np.ndarray":
    import numpy as np
    _validate_state(lam, eps_sw, alpha, temperature)
    x = float(eps_sw) / float(temperature)
    mu_of = lambda p: _oracle_reduced_potentials(p, lam, x, alpha)[1]

    grid = np.linspace(1e-8, 0.74, 4001)
    mus = np.array([mu_of(p) for p in grid], dtype=float)
    d = np.diff(mus)
    dec = np.where(d < 0.0)[0]
    if dec.size == 0:
        raise ValueError("no van der Waals loop: state is at or above the critical temperature")
    i1 = int(dec[0])
    inc = np.where(d[i1:] > 0.0)[0]
    if inc.size == 0:
        raise ValueError("no van der Waals loop: state is at or above the critical temperature")
    i2 = i1 + int(inc[0])
    mu_hi, mu_lo = float(mus[i1]), float(mus[i2])
    if not (mu_lo < mu_hi):
        raise ValueError("degenerate loop: state is at or above the critical temperature")

    args = (lam, x, alpha, grid, i1, i2)
    pad = 1e-10 * max(1.0, abs(mu_hi - mu_lo))
    m_star = _bisect_root(lambda m: _binodal_pressure_gap(m, *args), mu_lo + pad, mu_hi - pad)
    pI, pII = _binodal_outer_roots(m_star, *args)
    if not (pII - pI > 1e-6):
        raise ValueError("no distinct two-phase solution at this state point")
    return np.array([pI, pII], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: additive system at 250 K ---
        {
            "setup": """import numpy as np
lam = 1.3
eps_sw = 1732.0
alpha = 0.0293
temperature = 250.0
""",
            "call": "llps_binodal(lam, eps_sw, alpha, temperature)",
            "gold_call": "_oracle_llps_binodal(lam, eps_sw, alpha, temperature)",
        },
        # --- Normal: reference system at 250 K ---
        {
            "setup": """import numpy as np
lam = 1.3
eps_sw = 1687.0
alpha = 0.0372
temperature = 250.0
""",
            "call": "llps_binodal(lam, eps_sw, alpha, temperature)",
            "gold_call": "_oracle_llps_binodal(lam, eps_sw, alpha, temperature)",
        },
        # --- Boundary: deep quench, where the two branches are far apart ---
        {
            "setup": """import numpy as np
lam = 1.3
eps_sw = 1687.0
alpha = 0.0372
temperature = 235.0
""",
            "call": "llps_binodal(lam, eps_sw, alpha, temperature)",
            "gold_call": "_oracle_llps_binodal(lam, eps_sw, alpha, temperature)",
        },
        # --- Boundary: shallow quench close to the critical point ---
        {
            "setup": """import numpy as np
lam = 1.3
eps_sw = 1732.0
alpha = 0.0293
temperature = 262.0
""",
            "call": "llps_binodal(lam, eps_sw, alpha, temperature)",
            "gold_call": "_oracle_llps_binodal(lam, eps_sw, alpha, temperature)",
        },
        # --- Edge: above the critical temperature there is no two-phase solution ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        llps_binodal(1.3, 1732.0, 0.0293, 400.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_llps_binodal(1.3, 1732.0, 0.0293, 400.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: non-positive temperature must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        llps_binodal(1.3, 1732.0, 0.0293, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_llps_binodal(1.3, 1732.0, 0.0293, 0.0)
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
