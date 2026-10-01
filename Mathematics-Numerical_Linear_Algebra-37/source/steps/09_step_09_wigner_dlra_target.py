"""
Run the complete pipeline end to end and return the single scalar it targets. Build the periodic phase-space grid from the node count and half-width, sample the harmonic difference potential on the spatial grid and on the grid dual to the wave-vector grid, as the right-hand-side step defines it, verify through the earlier steps that it has the separation rank the potential implies and that the recovered factors reproduce it, use those factors for the potential term throughout, confirm that the supplied first-stage weight admits a tableau, form the initial low-rank state from the rotated squeezed Gaussian, advance it by the requested number of uniform steps, and evaluate the reconstruction at the requested phase-space point. Raise ValueError if the node count, rank or step count is not a positive integer, if any real parameter is not finite, if the half-width or final time is not positive, if the supplied first-stage weight admits no tableau, if the difference potential does not have that separation rank, or if the evaluation point does not lie on the grid.

The final step composes the earlier ones. The scalar reported is a single value of the reconstructed quasi-distribution at one phase-space point and one time, which does not depend on the sign or scaling conventions internal to the factorisation.

Returns
-------
float, the reconstructed low-rank quasi-distribution evaluated at (x_eval, k_eval) at time t_final, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wigner_dlra_target(n_points: int,
                       half_width: float,
                       theta: float,
                       rank: int,
                       t_final: float,
                       n_steps: int,
                       b1: float,
                       x_eval: float,
                       k_eval: float) -> float:
    '''Run the full pipeline and return the reconstructed value at one point.

    Parameters
    ----------
    n_points : int
        Number of grid nodes per direction.
    half_width : float
        Half-width of the periodic interval, used for both coordinates.
    theta : float
        Rotation angle of the initial squeezed Gaussian.
    rank : int
        Retained rank, held fixed throughout.
    t_final : float
        Final time.
    n_steps : int
        Number of uniform time steps.
    b1 : float
        Final weight assigned to the first stage.
    x_eval : float
        Spatial coordinate of the evaluation point, lying on the grid.
    k_eval : float
        Wave-vector coordinate of the evaluation point, lying on the grid.

    Returns
    -------
    value : float
        Reconstructed quasi-distribution at the evaluation point and final
        time, as a native Python float.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wigner_dlra_target(n_points: int,
                               half_width: float,
                               theta: float,
                               rank: int,
                               t_final: float,
                               n_steps: int,
                               b1: float,
                               x_eval: float,
                               k_eval: float) -> float:
    for nm, val in (("n_points", n_points), ("rank", rank), ("n_steps", n_steps)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < 1:
            raise ValueError(nm + " must be a positive integer")
    n, r, nt = int(n_points), int(rank), int(n_steps)
    for nm, val in (("half_width", half_width), ("theta", theta),
                    ("t_final", t_final), ("x_eval", x_eval), ("k_eval", k_eval)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) \
           or not np.isfinite(float(val)):
            raise ValueError(nm + " must be a finite real number")
    L, th, T = float(half_width), float(theta), float(t_final)
    if L <= 0.0 or T <= 0.0:
        raise ValueError("half_width and t_final must be positive")

    period = 2.0 * L
    g = -L + (period / n) * np.arange(n)

    V = lambda t: 0.5 * t ** 2
    y = (2.0 * np.pi / period) * np.arange(-(n // 2), n - n // 2)
    D_V = _oracle_difference_potential_matrix(V, g, y)
    fac = _oracle_separation_factors(D_V, 1e-12)
    if fac.shape[1] != 1:
        raise ValueError("the difference potential is not of separation rank one")
    if not np.allclose(fac[:n, 0][:, None] * fac[n:, 0][None, :], D_V,
                       rtol=0.0, atol=1e-9 * max(np.abs(D_V).max(), 1.0)):
        raise ValueError("separated factors do not reproduce the difference potential")

    D = _oracle_spectral_derivative_matrix(n, period)

    u = np.cos(th) * g[:, None] - np.sin(th) * g[None, :]
    v = np.sin(th) * g[:, None] + np.cos(th) * g[None, :]
    f0 = np.exp(-(u - 1.0) ** 2 / 2.0 - 2.0 * v ** 2) / np.pi
    state = _oracle_initial_low_rank_state(f0, r)

    h = T / nt
    tab = _oracle_two_stage_tableau(b1)
    a, w1, w2 = float(tab[0]), float(tab[1]), float(tab[2])
    F1 = _oracle_wigner_rhs(state, D, g, g, fac)
    stage = _oracle_projector_splitting_step(state, (a * h) * F1)
    F2 = _oracle_wigner_rhs(stage, D, g, g, fac)
    state = _oracle_projector_splitting_step(state, h * (w1 * F1 + w2 * F2))
    for _ in range(nt - 1):
        state = _oracle_robust_psrk_step(state, D, g, g, fac, h, b1)

    ix = int(np.argmin(np.abs(g - float(x_eval))))
    ik = int(np.argmin(np.abs(g - float(k_eval))))
    tol = 1e-9 * period
    if abs(g[ix] - float(x_eval)) > tol or abs(g[ik] - float(k_eval)) > tol:
        raise ValueError("evaluation point does not lie on the grid")

    U = state[:n, :]
    S = state[n:n + r, :]
    Vf = state[n + r:, :]
    return float((U @ S @ Vf.T)[ix, ik])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the full production configuration ---
        {
            "setup": """import numpy as np
""",
            "call": "wigner_dlra_target(128, 10.0, np.pi/5, 12, 2.0, 600, -1.0, -1.25, 0.78125)",
            "gold_call": "_oracle_wigner_dlra_target(128, 10.0, np.pi/5, 12, 2.0, 600, -1.0, -1.25, 0.78125)",
        },
        # --- normal: the member printed as the reference algorithm ---
        {
            "setup": """import numpy as np
""",
            "call": "wigner_dlra_target(128, 10.0, np.pi/5, 12, 2.0, 600, 0.0, -1.25, 0.78125)",
            "gold_call": "_oracle_wigner_dlra_target(128, 10.0, np.pi/5, 12, 2.0, 600, 0.0, -1.25, 0.78125)",
        },
        # --- edge: coarse grid, low rank, short horizon ---
        {
            "setup": """import numpy as np
""",
            "call": "wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.5, 60, -1.0, -1.25, 0.9375)",
            "gold_call": "_oracle_wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.5, 60, -1.0, -1.25, 0.9375)",
        },
        # --- edge: evaluated at the fixed point of the rigid motion, where the
        #     exact solution is stationary in time ---
        {
            "setup": """import numpy as np
""",
            "call": "wigner_dlra_target(64, 10.0, 0.35, 8, 1.0, 200, -3.0, 0.0, 0.0)",
            "gold_call": "_oracle_wigner_dlra_target(64, 10.0, 0.35, 8, 1.0, 200, -3.0, 0.0, 0.0)",
        },
        # --- edge: single time step, extreme negative weight ---
        {
            "setup": """import numpy as np
""",
            "call": "wigner_dlra_target(32, 6.0, np.pi/7, 4, 0.001, 1, -1e6, -0.375, 1.125)",
            "gold_call": "_oracle_wigner_dlra_target(32, 6.0, np.pi/7, 4, 0.001, 1, -1e6, -0.375, 1.125)",
        },
        # --- boundary: agreement with the closed-form rigid motion improves at
        #     second order as the step size falls, reported as integers ---
        {
            "setup": """import numpy as np
def probe(fn):
    n, L, th, r, T = 64, 10.0, np.pi/5, 20, 0.5
    g = -L + (2.0*L/n)*np.arange(n)
    ix, ik = 34, 36
    u = np.cos(T)*g[ix] - np.sin(T)*g[ik]
    v = np.sin(T)*g[ix] + np.cos(T)*g[ik]
    uu = np.cos(th)*u - np.sin(th)*v
    vv = np.sin(th)*u + np.cos(th)*v
    exact = np.exp(-(uu - 1.0)**2/2.0 - 2.0*vv**2)/np.pi
    errs = []
    for nt in [25, 50, 100]:
        errs.append(abs(fn(n, L, th, r, T, nt, -1.0, g[ix], g[ik]) - exact))
    out = []
    for i in range(2):
        out.append(int(round(np.log2(errs[i]/errs[i+1]))))
    out.append(int(errs[-1] < errs[0]))
    return out
def run_model():
    return probe(wigner_dlra_target)
def run_gold():
    return probe(_oracle_wigner_dlra_target)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: evaluation point off the grid ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.1, 10, -1.0, -1.2, 0.9375)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.1, 10, -1.0, -1.2, 0.9375)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: weight admitting no tableau ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.1, 10, 2.0, -1.25, 0.9375)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_dlra_target(64, 10.0, np.pi/5, 6, 0.1, 10, 2.0, -1.25, 0.9375)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive rank ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        wigner_dlra_target(64, 10.0, np.pi/5, 0, 0.1, 10, -1.0, -1.25, 0.9375)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_dlra_target(64, 10.0, np.pi/5, 0, 0.1, 10, -1.0, -1.25, 0.9375)
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
