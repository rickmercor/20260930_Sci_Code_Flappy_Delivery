"""
The quantities of ecological interest are sums over patches. The total number of settled individuals of species a is the sum of its N settled components, so its variance is the sum of the corresponding N by N block of Sigma, and the covariance between the totals of two species is the sum of their cross block. For competitors sharing sites that covariance is typically strongly negative, because space one species gains is space the other loses; and when the two species exchange space slowly, that slow mode dominates the variance of each total.

Everything here depends on J and B only. Eliminating the explorers first and adding noise to the settled numbers alone is a different linear process whenever explorers are not much faster than settled turnover, and gives a different covariance.

To leading order in the inverse site numbers, the numbers of settled individuals and explorers fluctuate about the deterministic steady state as a linear Gaussian process: the deviations xi obey d xi = J xi dt + dW with the drift Jacobian J and noise of covariance B dt, both evaluated at the steady state. When the steady state is stable, every eigenvalue of J has a negative real part and the deviations settle into a stationary distribution whose covariance Sigma solves the Lyapunov equation

J Sigma + Sigma J^T + B = 0,

equivalently Sigma = integral from 0 to infinity of exp(J t) B exp(J^T t) dt. The covariance scales with the site numbers, so the standard deviations of numbers of individuals grow as the square root of the landscape size while the relative fluctuations shrink.

Returns
-------
dict, the stationary fluctuations of the linearised chain, keyed by covariance, total_sd (length S), total_correlation (S by S), patch_sd (S by N settled standard deviations), and relaxation_rate (minus the largest real part of the eigenvalues of the Jacobian).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_covariance(
    jacobian: np.ndarray,
    jump_covariance: np.ndarray,
    n_species: int,
    n_patches: int,
) -> dict:
    """Solve for the stationary covariance of the linearised chain and summarise the fluctuations of the settled totals.

    Parameters
    ----------
    jacobian : np.ndarray
        Drift Jacobian of the chain at a stable steady state.
    jump_covariance : np.ndarray
        Jump covariance of the chain at that state.
    n_species : int
        Number of species.
    n_patches : int
        Number of patches.

    Returns
    -------
    dict
        Under the keys covariance, total_sd, total_correlation, patch_sd and relaxation_rate.

    Raises
    ------
    ValueError
        When the matrices fail to be finite and square of size 2SN, when the jump covariance fails to be symmetric and positive semi-definite, or when the Jacobian has an eigenvalue with real part not below zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def _oracle_stationary_covariance(
    jacobian: np.ndarray,
    jump_covariance: np.ndarray,
    n_species: int,
    n_patches: int,
) -> dict:
    """Reference implementation."""
    for value in (n_species, n_patches):
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("n_species and n_patches must be integers of at least one")
    s, n = int(n_species), int(n_patches)
    dim = 2 * s * n
    j = np.asarray(jacobian, dtype=float)
    b = np.asarray(jump_covariance, dtype=float)
    for arr in (j, b):
        if arr.shape != (dim, dim) or not np.all(np.isfinite(arr)):
            raise ValueError("jacobian and jump_covariance must be finite and 2SN by 2SN")
    scale = max(1.0, float(np.max(np.abs(b))))
    if float(np.max(np.abs(b - b.T))) > 1e-10 * scale:
        raise ValueError("jump_covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(0.5 * (b + b.T)))) < -1e-9 * scale:
        raise ValueError("jump_covariance must be positive semi-definite")
    abscissa = float(np.max(np.linalg.eigvals(j).real))
    if abscissa >= 0.0:
        raise ValueError("the Jacobian is not stable")

    sigma = solve_continuous_lyapunov(j, -b)
    sigma = 0.5 * (sigma + sigma.T)
    totals = np.zeros((s, s))
    for a in range(s):
        for c in range(s):
            totals[a, c] = sigma[a * n:(a + 1) * n, c * n:(c + 1) * n].sum()
    sd = np.sqrt(np.diag(totals))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = totals / np.outer(sd, sd)
    patch = np.sqrt(np.clip(np.diag(sigma)[:s * n], 0.0, None)).reshape(s, n)
    return {
        "covariance": sigma,
        "total_sd": sd,
        "total_correlation": corr,
        "patch_sd": patch,
        "relaxation_rate": -abscissa,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
        if isinstance(x, (tuple, list)):
            out = []
            for v in x:
                out.extend(flat(v))
            return tuple(out)
        if hasattr(x, "tolist"):
            return flat(x.tolist())
        if isinstance(x, bool):
            return (int(x),)
        return (x,)
    """

    SETUP = """
    import numpy as np
    rng = np.random.default_rng(99)
    s, n = 2, 3
    dim = 2 * s * n
    A = rng.normal(0.0, 0.3, (dim, dim))
    J = A - np.diag(np.full(dim, 1.5))
    G = rng.normal(0.0, 1.0, (dim, dim))
    B = G @ G.T
    def digest(out):
        return (np.round(out["covariance"], 8), np.round(out["total_sd"], 8), np.round(out["total_correlation"], 8),
                np.round(out["patch_sd"], 8), round(out["relaxation_rate"], 9))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # a seeded stable Jacobian with a full-rank jump covariance for two species on three patches
            "setup": SETUP + FLAT,
            "call": "flat(digest(stationary_covariance(J, B, s, n)))",
            "gold_call": "flat(digest(_oracle_stationary_covariance(J, B, s, n)))",
        },
        {
            # closed forms: one species on one patch with uncoupled settled and explorer components has variances
            # B_kk / (2 |J_kk|); and the Lyapunov residual of the seeded case vanishes
            "setup": SETUP + """
def closed(fn):
    one = fn(np.diag([-0.4, -2.0]), np.diag([3.2, 7.0]), 1, 1)
    out = fn(J, B, s, n)
    res = J @ out["covariance"] + out["covariance"] @ J.T + B
    return (np.round(np.diag(one["covariance"]), 12), round(float(one["total_sd"][0]) ** 2, 12),
            int(float(np.max(np.abs(res))) < 1e-9), round(one["relaxation_rate"], 12))
""" + FLAT,
            "call": "flat(closed(stationary_covariance))",
            "gold_call": "flat(closed(_oracle_stationary_covariance))",
        },
        {
            # the settled-total variance is the sum of the settled block, so a Jacobian and noise that couple only
            # explorers leave the settled totals with zero variance; scaling B scales Sigma
            "setup": SETUP + """
def structure(fn):
    Bx = np.zeros((dim, dim)); Bx[s * n:, s * n:] = B[s * n:, s * n:]
    Jd = np.diag(np.full(dim, -1.0))
    a = fn(Jd, Bx, s, n)
    b = fn(J, 4.0 * B, s, n); c = fn(J, B, s, n)
    return (np.round(a["total_sd"], 12), np.round(b["total_sd"] / c["total_sd"], 10))
""" + FLAT,
            "call": "flat(structure(stationary_covariance))",
            "gold_call": "flat(structure(_oracle_stationary_covariance))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(jacobian=J, jump_covariance=B, n_species=s, n_patches=n)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
ASYM = B.copy(); ASYM[0, 1] += 1.0
""",
            "call": "(verdict(stationary_covariance, jacobian=J + np.eye(dim) * 3.0), verdict(stationary_covariance, jump_covariance=ASYM), verdict(stationary_covariance, jump_covariance=-B), verdict(stationary_covariance, n_species=3), verdict(stationary_covariance, n_patches=0), verdict(stationary_covariance, jacobian=J * np.nan), verdict(stationary_covariance))",
            "gold_call": "(verdict(_oracle_stationary_covariance, jacobian=J + np.eye(dim) * 3.0), verdict(_oracle_stationary_covariance, jump_covariance=ASYM), verdict(_oracle_stationary_covariance, jump_covariance=-B), verdict(_oracle_stationary_covariance, n_species=3), verdict(_oracle_stationary_covariance, n_patches=0), verdict(_oracle_stationary_covariance, jacobian=J * np.nan), verdict(_oracle_stationary_covariance))",
        },
    ]
