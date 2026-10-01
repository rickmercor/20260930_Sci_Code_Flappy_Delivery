"""
Run the augmented-Bregman mirror-descent SPX-VIX calibration (Algorithm 1) to reconcile a reference tensor toward the given target marginals and (approximately) the conditional martingale/dispersion constraints.

Algorithm 1 of the source paper (a global multi-maturity SPX/VIX calibration scheme) reconciles the reference tensor from the previous step toward a joint law over (S1, V1, S2, V2, S3) that: (a) matches prescribed target marginals for S1, V1, and V2 exactly, and (b) satisfies, as closely as a fixed iterative budget allows, a conditional forward-martingale constraint (E[S_{i+1} | S_i, V_i] = S_i) and a conditional log-contract dispersion constraint (E[-(2/tau) log(S_{i+1}/S_i) | S_i, V_i] = V_i^2) at each of the two transition dates. The scheme alternates an entropic mirror step (penalizing the conditional constraint residuals, with a penalty weight that grows across a fixed outer schedule) with an exact per-axis rescale that pins the three marginal-matching rows back to their targets after every inner sweep. The algorithm's own clipping bound, step size, and penalty-growth schedule are not given numeric values in the source paper; an implementer must choose and justify them.

Returns
-------
pi : np.ndarray, (nS, nV, nS, nV, nS) calibrated probability tensor, marginals for S1/V1/V2 matching muS/muV/muV exactly.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def spx_vix_calibration(S_grid: np.ndarray, V_grid: np.ndarray, tau: float,
                         muS: np.ndarray, muV: np.ndarray,
                         Kout: int = 6, Kin: int = 40) -> np.ndarray:
    """Run the augmented-Bregman mirror-descent SPX-VIX calibration
    (Algorithm 1) to reconcile a reference tensor toward the given target
    marginals and (approximately) the conditional martingale/dispersion
    constraints.

    Parameters
    ----------
    S_grid : np.ndarray
        1D array of nS candidate index levels.
    V_grid : np.ndarray
        1D array of nV candidate volatility levels.
    tau : float
        Accrual fraction per step.
    muS : np.ndarray
        1D array of length nS, the target S1 marginal (must sum to 1).
    muV : np.ndarray
        1D array of length nV, the shared target V1 and V2 marginal (must
        sum to 1).
    Kout : int
        Number of outer sweeps (penalty-growth steps).
    Kin : int
        Number of inner sweeps per outer step.

    Returns
    -------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) calibrated probability tensor.

    Raises
    ------
    ValueError
        If muS or muV does not sum to 1 (within 1e-6), or if Kout < 1 or
        Kin < 1.
    """
    nS, nV = len(S_grid), len(V_grid)
    pi = np.ones((nS, nV, nS, nV, nS))
    pi = pi / pi.sum()  # placeholder
    return pi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spx_vix_calibration(S_grid: np.ndarray, V_grid: np.ndarray, tau: float,
                                 muS: np.ndarray, muV: np.ndarray,
                                 Kout: int = 6, Kin: int = 40) -> np.ndarray:
    import numpy as np
    S_grid = np.asarray(S_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)
    muS = np.asarray(muS, dtype=float)
    muV = np.asarray(muV, dtype=float)
    if abs(muS.sum() - 1.0) > 1e-6 or abs(muV.sum() - 1.0) > 1e-6:
        raise ValueError("muS and muV must each sum to 1")
    if Kout < 1 or Kin < 1:
        raise ValueError("Kout and Kin must be >= 1")

    nS, nV = len(S_grid), len(V_grid)
    alpha, c = 0.5, 1.0
    lam0, lam_gamma, lam_max = 1.0, 1.5, 1e3

    def lognormal_kernel(Si, Vi):
        import numpy as np
        m = -0.5 * Vi ** 2 * tau
        s2 = Vi ** 2 * tau
        x = np.log(S_grid / Si)
        dens = np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
        return dens / dens.sum()

    pi = np.zeros((nS, nV, nS, nV, nS))
    for i0 in range(nS):
        for iv1 in range(nV):
            k1 = lognormal_kernel(S_grid[i0], V_grid[iv1])
            for i1 in range(nS):
                for iv2 in range(nV):
                    k2 = lognormal_kernel(S_grid[i1], V_grid[iv2])
                    pi[i0, iv1, i1, iv2, :] = k1[i1] * k2
    pi = pi / pi.sum()

    def marginalize(pi, axis_keep):
        import numpy as np
        axes = tuple(a for a in range(pi.ndim) if a not in axis_keep)
        return pi.sum(axis=axes)

    def project_marginal(pi, axis, target, eps=1e-13):
        import numpy as np
        cur = marginalize(pi, (axis,))
        scale = np.where(cur > eps, target / np.maximum(cur, eps), 1.0)
        shape = [1] * pi.ndim
        shape[axis] = pi.shape[axis]
        pi = pi * scale.reshape(shape)
        return pi / pi.sum()

    L_mat = -(2.0 / tau) * np.log(S_grid[None, :] / S_grid[:, None])

    lam = lam0
    for _ in range(Kout):
        for _ in range(Kin):
            p1 = pi.sum(axis=(3, 4))
            p2 = pi.sum(axis=(0, 1))
            mart1 = np.zeros((nS, nV)); disp1 = np.zeros((nS, nV))
            mart2 = np.zeros((nS, nV)); disp2 = np.zeros((nS, nV))
            for i0 in range(nS):
                for iv1 in range(nV):
                    w = p1[i0, iv1, :]
                    if w.sum() < 1e-14:
                        continue
                    w = w / w.sum()
                    L = -(2.0 / tau) * np.log(S_grid / S_grid[i0])
                    mart1[i0, iv1] = np.sum(w * S_grid) - S_grid[i0]
                    disp1[i0, iv1] = np.sum(w * L) - V_grid[iv1] ** 2
            for i1 in range(nS):
                for iv2 in range(nV):
                    w = p2[i1, iv2, :]
                    if w.sum() < 1e-14:
                        continue
                    w = w / w.sum()
                    L = -(2.0 / tau) * np.log(S_grid / S_grid[i1])
                    mart2[i1, iv2] = np.sum(w * S_grid) - S_grid[i1]
                    disp2[i1, iv2] = np.sum(w * L) - V_grid[iv2] ** 2

            grad = np.zeros_like(pi)
            for i0 in range(nS):
                dS = S_grid - S_grid[i0]
                dL = L_mat[i0, :] - V_grid[:, None] ** 2
                for iv1 in range(nV):
                    g1 = lam * mart1[i0, iv1] * dS + lam * disp1[i0, iv1] * dL[iv1, :]
                    grad[i0, iv1, :, :, :] += g1[:, None, None]
            for i1 in range(nS):
                dS = S_grid - S_grid[i1]
                dL = L_mat[i1, :] - V_grid[:, None] ** 2
                for iv2 in range(nV):
                    g2 = lam * mart2[i1, iv2] * dS + lam * disp2[i1, iv2] * dL[iv2, :]
                    grad[:, :, i1, iv2, :] += g2[None, :]
            eta = min(alpha, c / (np.max(np.abs(grad)) + 1e-300))
            pi = pi * np.exp(-eta * grad)
            pi = pi / pi.sum()
            pi = project_marginal(pi, 0, muS)
            pi = project_marginal(pi, 1, muV)
            pi = project_marginal(pi, 3, muV)
        lam = min(lam * lam_gamma, lam_max)
    return pi

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: the actual task instance -- check the 3 marginals ---
        {
            "setup": """import numpy as np
S_grid = np.array([0.7046880897187134, 0.8394570207692074, 1.0, 1.191246216612358, 1.4190675485932571])
V_grid = np.array([0.12, 0.22, 0.32])
tau = 30 / 365
muS = np.array([0.06, 0.2, 0.48, 0.2, 0.06])
muS = muS / muS.sum()
muV = np.array([0.3, 0.45, 0.25])
def run_model():
    pi = spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=6, Kin=40)
    m1 = pi.sum(axis=(1, 2, 3, 4))
    mv1 = pi.sum(axis=(0, 2, 3, 4))
    mv2 = pi.sum(axis=(0, 1, 2, 4))
    return np.concatenate([m1, mv1, mv2])
def run_gold():
    pi = _oracle_spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=6, Kin=40)
    m1 = pi.sum(axis=(1, 2, 3, 4))
    mv1 = pi.sum(axis=(0, 2, 3, 4))
    mv2 = pi.sum(axis=(0, 1, 2, 4))
    return np.concatenate([m1, mv1, mv2])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-3,
        },
        # --- Boundary case: tiny 2x1 grid, uniform target -- marginals must
        # converge to the trivial uniform target ---
        {
            "setup": """import numpy as np
S_grid = np.array([0.9, 1.1])
V_grid = np.array([0.25])
tau = 30 / 365
muS = np.array([0.5, 0.5])
muV = np.array([1.0])
def run_model():
    pi = spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=4, Kin=20)
    m1 = pi.sum(axis=(1, 2, 3, 4))
    mv1 = pi.sum(axis=(0, 2, 3, 4))
    mv2 = pi.sum(axis=(0, 1, 2, 4))
    return np.concatenate([m1, mv1, mv2])
def run_gold():
    pi = _oracle_spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=4, Kin=20)
    m1 = pi.sum(axis=(1, 2, 3, 4))
    mv1 = pi.sum(axis=(0, 2, 3, 4))
    mv2 = pi.sum(axis=(0, 1, 2, 4))
    return np.concatenate([m1, mv1, mv2])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-3,
        },
        # --- Edge case: muS not summing to 1 should raise ---
        {
            "setup": """import numpy as np
S_grid = np.array([0.9, 1.0, 1.1])
V_grid = np.array([0.2])
tau = 30 / 365
muS = np.array([0.5, 0.5, 0.5])
muV = np.array([1.0])
def run_model():
    try:
        spx_vix_calibration(S_grid, V_grid, tau, muS, muV)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spx_vix_calibration(S_grid, V_grid, tau, muS, muV)
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
