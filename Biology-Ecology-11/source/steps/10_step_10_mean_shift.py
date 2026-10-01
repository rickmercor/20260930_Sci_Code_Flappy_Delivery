"""
Writing <n> = n* + delta about the steady state n*, and keeping the leading order, J delta + h = 0 with h = (1/2) H : Sigma. The only non-zero second derivatives are those of the settled drift of species a in patch i with respect to X_ai and any S_bi in the same patch, each equal to -lambda_a / M_i, so that

h_{S_ai} = -(lambda_a / M_i) sum_b Sigma_{X_ai, S_bi},

and h vanishes in every explorer component. The explorer drift is linear, because every colonisation attempt removes the explorer whether or not it succeeds. The mean shift of the total of species a is the sum of its settled components of delta.

The sign follows the covariance between explorers and settled individuals in each patch: where more explorers coincide with more occupied sites, colonisation is less efficient on average than at the mean state, and the settled means fall below the deterministic values.

Fluctuations do not only spread the numbers of individuals about the deterministic state; they also move the mean. In the long-lived state the mean numbers differ from the deterministic steady state by an amount that stays finite as the site numbers grow, although it vanishes relative to the numbers themselves.

The first moments of the chain obey d<n>/dt = <A(n)> exactly, with A the drift. Every propensity of the stated reactions is linear in the numbers of individuals except the successful colonisation, lambda_a X_ai (M_i - sum_b S_bi) / M_i, which is bilinear. The drift is therefore quadratic, and its average is the drift at the mean plus one half of its Hessian contracted with the covariance, with no further terms:

<A(n)> = A(<n>) + (1/2) H : Sigma.

Returns
-------
dict, the leading-order shift of the mean from the deterministic state, keyed by shift (length 2SN), total_shift (length S, settled totals), and forcing (the vector h).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_shift(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    colonisation_rate: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Compute the leading-order shift of the mean numbers of individuals away from the deterministic steady state.

    Parameters
    ----------
    jacobian : np.ndarray
        Drift Jacobian of the chain at the steady state.
    covariance : np.ndarray
        Stationary covariance of the chain.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys shift, total_shift and forcing.

    Raises
    ------
    ValueError
        When the matrices fail to be finite and of size 2SN consistent with the rates and sites, when the covariance is not symmetric, or when the Jacobian is singular.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_mean_shift(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    colonisation_rate: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    lam = np.asarray(colonisation_rate, dtype=float)
    m = np.asarray(sites, dtype=float)
    if lam.ndim != 1 or lam.size < 1 or not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise ValueError("colonisation_rate must be finite, above zero and one per species")
    if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite and above zero")
    s, n = lam.size, m.size
    dim = 2 * s * n
    j = np.asarray(jacobian, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    for arr in (j, sigma):
        if arr.shape != (dim, dim) or not np.all(np.isfinite(arr)):
            raise ValueError("jacobian and covariance must be finite and 2SN by 2SN")
    if float(np.max(np.abs(sigma - sigma.T))) > 1e-8 * max(1.0, float(np.max(np.abs(sigma)))):
        raise ValueError("covariance must be symmetric")
    if not math.isfinite(np.linalg.cond(j)) or np.linalg.cond(j) > 1e14:
        raise ValueError("the Jacobian is singular")

    forcing = np.zeros(dim)
    for a in range(s):
        for i in range(n):
            x_index = (s + a) * n + i
            coupled = sum(sigma[x_index, b * n + i] for b in range(s))
            forcing[a * n + i] = -(lam[a] / m[i]) * coupled
    shift = -np.linalg.solve(j, forcing)
    total = np.array([shift[a * n:(a + 1) * n].sum() for a in range(s)])
    return {"shift": shift, "total_shift": total, "forcing": forcing}

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
    rng = np.random.default_rng(1010)
    s, n = 2, 4
    dim = 2 * s * n
    J = rng.normal(0.0, 0.2, (dim, dim)) - 1.2 * np.eye(dim)
    G = rng.normal(0.0, 3.0, (dim, dim)); SIG = G @ G.T
    LAM = np.array([1.0, 0.8]); M = np.array([120.0, 40.0, 80.0, 200.0])
    def digest(out):
        return (np.round(out["shift"], 9), np.round(out["total_shift"], 9), np.round(out["forcing"], 10))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # a seeded stable Jacobian and covariance for two species on four patches
            "setup": SETUP + FLAT,
            "call": "flat(digest(mean_shift(J, SIG, LAM, M)))",
            "gold_call": "flat(digest(_oracle_mean_shift(J, SIG, LAM, M)))",
        },
        {
            # exactness of the second-order average for a quadratic drift: the shift equals minus J^(-1) times
            # one half of the Hessian of the stated colonisation drift contracted with Sigma, the Hessian built here
            # by finite differences of that drift at an arbitrary point
            "setup": SETUP + """
def drift_col(y):
    Sy = y[:s * n].reshape(s, n); Xy = y[s * n:].reshape(s, n)
    occ = Sy.sum(axis=0)
    out = np.zeros(dim)
    for a in range(s):
        succ = LAM[a] * Xy[a] * (M - occ) / M
        out[a * n:(a + 1) * n] += succ
        out[(s + a) * n:(s + a + 1) * n] -= LAM[a] * Xy[a]
    return out
def check(fn):
    y0 = np.abs(rng.normal(10.0, 2.0, dim)); h = 1e-3
    H = np.zeros((dim, dim, dim))
    for k in range(dim):
        for l in range(dim):
            ek = np.zeros(dim); ek[k] = h; el = np.zeros(dim); el[l] = h
            H[:, k, l] = (drift_col(y0 + ek + el) - drift_col(y0 + ek - el) - drift_col(y0 - ek + el) + drift_col(y0 - ek - el)) / (4 * h * h)
    forcing = 0.5 * np.einsum("ikl,kl->i", H, SIG)
    out = fn(J, SIG, LAM, M)
    return (int(float(np.max(np.abs(forcing - out["forcing"]))) < 1e-6), int(float(np.max(np.abs(-np.linalg.solve(J, forcing) - out["shift"]))) < 1e-5))
""" + FLAT,
            "call": "flat(check(mean_shift))",
            "gold_call": "flat(check(_oracle_mean_shift))",
        },
        {
            # boundary: with no covariance between explorers and settled individuals of the same patch the forcing
            # vanishes, and the shift scales linearly with the covariance and is invariant when every site number
            # and the covariance grow by the same factor
            "setup": SETUP + """
def scaling(fn):
    D = np.diag(np.diag(SIG))
    a = fn(J, D, LAM, M)
    b = fn(J, 3.0 * SIG, LAM, M); c = fn(J, SIG, LAM, M)
    d = fn(J, 5.0 * SIG, LAM, 5.0 * M)
    return (float(np.max(np.abs(a["shift"]))), np.round(b["total_shift"] / c["total_shift"], 10), np.round(d["total_shift"] - c["total_shift"], 10))
""" + FLAT,
            "call": "flat(scaling(mean_shift))",
            "gold_call": "flat(scaling(_oracle_mean_shift))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(jacobian=J, covariance=SIG, colonisation_rate=LAM, sites=M)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
ASYM = SIG.copy(); ASYM[0, 1] += 5.0
""",
            "call": "(verdict(mean_shift, jacobian=np.zeros((dim, dim))), verdict(mean_shift, covariance=ASYM), verdict(mean_shift, colonisation_rate=np.array([1.0])), verdict(mean_shift, sites=M[:3]), verdict(mean_shift, colonisation_rate=np.array([1.0, -0.8])), verdict(mean_shift))",
            "gold_call": "(verdict(_oracle_mean_shift, jacobian=np.zeros((dim, dim))), verdict(_oracle_mean_shift, covariance=ASYM), verdict(_oracle_mean_shift, colonisation_rate=np.array([1.0])), verdict(_oracle_mean_shift, sites=M[:3]), verdict(_oracle_mean_shift, colonisation_rate=np.array([1.0, -0.8])), verdict(_oracle_mean_shift))",
        },
    ]
