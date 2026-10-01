"""
Compute the fraction of a single Crystal Ball (CB) line shape that falls in each bin of a dimuon invariant-mass histogram, with the CB normalized to unit integral over the histogram window $[e_0, e_K]$, where $e_0<e_1<\dots<e_K$ are the bin edges (`edges[0]` to `edges[-1]`).



The CB function has a Gaussian core and a power-law tail on the low-mass side:



$$t=\dfrac{m-\mu}{\sigma},\qquad \mathrm{CB}(t)=\exp\Big(-\dfrac{t^2}{2}\Big)\ \ \mathrm{for}\ t>-\alpha,\qquad \mathrm{CB}(t)=A\,(B-t)^{-n}\ \ \mathrm{for}\ t\le-\alpha,$$



$$A=\Big(\dfrac{n}{\alpha}\Big)^{n}\exp\Big(-\dfrac{\alpha^2}{2}\Big),\qquad B=\dfrac{n}{\alpha}-\alpha.$$



The bin fractions are exact integrals over each bin divided by the exact integral over the window, and they must stay accurate to about $10^{-9}$ relative precision per bin in every regime the fit can visit: windows lying many $\sigma$ above the mean ($|t|$ up to about 30), windows entirely inside the power-law tail, and every $n>0$, including $n=1$ and values of $n$ arbitrarily close to (but not equal to) 1. The window integral is finite only because the window is finite.

The CMS paper describes each Upsilon peak with Crystal Ball functions: a Gaussian core, set by the tracker momentum resolution, joined to a power-law tail on the low-mass side, which absorbs final-state radiation (a muon that radiates a photon reconstructs a lower dimuon mass). The exact functional form used in this task is the one written in this step's description and in the problem statement, where continuity of the function and of its first derivative at $t=-\alpha$ fixes the tail constants $A$ and $B$. The tail integral scales like $(B-t)^{1-n}/(1-n)$, which diverges for $n\le1$ over an infinite range, so a shape with $n=1$ is normalizable only on a finite window. During a fit the optimizer can push the mean and width far from the data, so the shape must be integrated to near machine precision in every regime: bins far in the Gaussian upper tail, where tiny probabilities must not cancel against $O(1)$ constants, and tails with $n$ arbitrarily close to 1, where the general and logarithmic forms of the primitive must join smoothly.

Returns
-------
A float array of shape $(K,)$ gives the bin fractions of the window-normalized Crystal Ball; the entries sum to 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erf, erfc

def crystal_ball_bin_fractions(edges: np.ndarray, mu: float, sigma: float,
                               alpha: float, n: float) -> np.ndarray:
    r'''Window-normalized bin fractions of a low-side-tail Crystal Ball shape.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges
        $e_0<\dots<e_K$ (GeV), with $K\geq1$.
    mu : float
        Peak position $\mu$ of the Gaussian core (GeV).
    sigma : float
        Width $\sigma>0$ of the Gaussian core (GeV).
    alpha : float
        Transition point $\alpha>0$ of the power-law tail, in units of
        $\sigma$; the tail is on the low-mass side, $t\leq-\alpha$ with
        $t=(m-\mu)/\sigma$.
    n : float
        Power-law exponent $n>0$ of the tail; $n=1$ and $n$ arbitrarily
        close to 1 must be supported.

    Returns
    -------
    fractions : np.ndarray
        Shape $(K,)$: the integral of the CB over each bin divided by its
        integral over $[e_0, e_K]$; the entries sum to 1. Each entry must be
        accurate to about $10^{-9}$ relative precision, also when the window
        lies far in the Gaussian upper tail ($|t|$ up to about 30).

    Raises
    ------
    ValueError
        If edges is not a 1D finite array of length $\geq2$ that is strictly
        increasing, if mu is not finite, if $\sigma\leq0$, $\alpha\leq0$ or
        $n\leq0$ (or any of them is not finite), or if the integral of the
        shape over the window underflows to zero in double precision.
    '''
    return fractions  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, erfc

_SQRT_HALF_PI = np.sqrt(np.pi / 2.0)
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _cb_tail_integral(a, b, alpha, n):
    r"""Integral of $A\,(B-t)^{-n}$ over $[a,b]$ with $b\leq-\alpha$, stable for all $n>0$."""
    A = (n / alpha) ** n * np.exp(-0.5 * alpha * alpha)
    B = n / alpha - alpha
    U = np.log(B - a)          # U >= V
    V = np.log(B - b)
    c = 1.0 - n
    d = U - V
    if c == 0.0:
        g = d
    else:
        g = np.expm1(c * d) / c
    return A * np.exp(c * V) * g


def _cb_core_integral(a, b):
    r"""Integral of $\exp(-t^2/2)$ over $[a,b]$ with $a\geq-\alpha$, via $\operatorname{erfc}$ (no cancellation)."""
    from scipy.special import erfc
    return _SQRT_HALF_PI * (erfc(a * _INV_SQRT2) - erfc(b * _INV_SQRT2))


def _oracle_crystal_ball_bin_fractions(edges: np.ndarray, mu: float, sigma: float,
                                       alpha: float, n: float) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    for name, v in (("mu", mu), ("sigma", sigma), ("alpha", alpha), ("n", n)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite")
    if sigma <= 0 or alpha <= 0 or n <= 0:
        raise ValueError("sigma, alpha and n must be > 0")
    alpha = float(alpha)
    n = float(n)
    t = (edges - float(mu)) / float(sigma)
    a, b = t[:-1], t[1:]
    tc = -alpha
    out = np.zeros(a.size)
    tail = a < tc
    if np.any(tail):
        out[tail] += _cb_tail_integral(a[tail], np.minimum(b[tail], tc), alpha, n)
    core = b > tc
    if np.any(core):
        out[core] += _cb_core_integral(np.maximum(a[core], tc), b[core])
    total = out.sum()
    if not (total > 0.0) or not np.isfinite(total):
        raise ValueError("window integral underflows or is not finite")
    return (out / total).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: narrow peak, n = 1 ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
mu, sigma, alpha, n = 9.4507, 0.0553, 2.0, 1.0
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), mu, sigma, alpha, n)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, mu, sigma, alpha, n)",
        },
        # --- Normal: wider peak, n = 2 ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
mu, sigma, alpha, n = 10.3454, 0.0937, 2.0, 2.0
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), mu, sigma, alpha, n)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, mu, sigma, alpha, n)",
        },
        # --- Boundary: a single bin covering the window must return [1.0] ---
        {
            "setup": """import numpy as np
edges = np.array([8.5, 11.5])
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.06, 2.0, 1.0)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.06, 2.0, 1.0)",
        },
        # --- Edge: window entirely inside the power-law tail, non-integer n < 1 ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.0, 9.0, 11)
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.06, 1.5, 0.7)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.06, 1.5, 0.7)",
        },
        # --- Edge: uneven bins straddling the core/tail junction ---
        {
            "setup": """import numpy as np
edges = np.array([9.0, 9.2, 9.33, 9.34, 9.45, 9.7, 10.0])
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.06, 2.0, 1.0)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.06, 2.0, 1.0)",
        },
        # --- Edge: window far in the Gaussian upper tail (t from ~10 to ~28) ---
        {
            "setup": """import numpy as np
edges = np.linspace(10.0, 11.0, 26)
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.055, 2.0, 1.0)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.055, 2.0, 1.0)",
        },
        # --- Edge: n arbitrarily close to 1 (n = 1 + 1e-13), window reaching deep into the tail ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.055, 2.0, 1.0 + 1e-13)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.055, 2.0, 1.0 + 1e-13)",
        },
        # --- Edge: tail-only window with n just below 1 (n = 1 - 1e-12) ---
        {
            "setup": """import numpy as np
edges = np.linspace(7.0, 9.3, 24)
""",
            "call": "crystal_ball_bin_fractions(edges.copy(), 9.45, 0.055, 2.0, 1.0 - 1e-12)",
            "gold_call": "_oracle_crystal_ball_bin_fractions(edges, 9.45, 0.055, 2.0, 1.0 - 1e-12)",
        },
        # --- Invalid: sigma <= 0 ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
def run_model():
    try:
        crystal_ball_bin_fractions(edges.copy(), 9.45, 0.0, 2.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystal_ball_bin_fractions(edges, 9.45, 0.0, 2.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-increasing edges ---
        {
            "setup": """import numpy as np
edges = np.array([8.5, 9.0, 9.0, 11.5])
def run_model():
    try:
        crystal_ball_bin_fractions(edges.copy(), 9.45, 0.06, 2.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystal_ball_bin_fractions(edges, 9.45, 0.06, 2.0, 1.0)
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
