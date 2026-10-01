"""
Generate the deterministic pseudo-data dimuon mass histogram of the task. The generator is deliberately not the fit model: each Upsilon(nS) is a plain Gaussian (no radiative tail) and the continuum is a falling exponential. With bin edges $e_0<e_1<\dots<e_K$, the expected count in bin $i$ is



$$\nu_i=\sum_s N_s\Big[\Phi\Big(\dfrac{e_{i+1}-m_s}{w_s}\Big)-\Phi\Big(\dfrac{e_i-m_s}{w_s}\Big)\Big]+N_{\mathrm{bkg}}\,\dfrac{e^{-c\,e_i}-e^{-c\,e_{i+1}}}{e^{-c\,e_0}-e^{-c\,e_K}},$$



where $\Phi$ is the standard normal distribution function. The Gaussian integrals are not renormalized to the window; the exponential is normalized on the window. Every expected count must be computed to full relative precision, including bins many standard deviations away from a peak: an expectation that is tiny but non-zero and one that is exactly zero are not equivalent inputs to `rng.poisson` (a zero rate does not consume a random draw), so they lead to different pseudo-data. The observed counts are drawn with exactly one call `rng.poisson(nu)` on the whole array of $\nu_i$ (ordered by increasing mass), with `rng = np.random.default_rng(seed)`.

Closure-style studies inject a known truth into a fit to measure how the analysis model responds. Here the truth deliberately differs from the fit model (pure Gaussian peaks and an exponential continuum instead of double Crystal Balls and a polynomial), which is the same logic CMS uses when it refits with alternative shapes to evaluate yield systematics. Poisson fluctuations make the histogram look like real data while a fixed seed keeps it reproducible. Reproducibility then depends on computing every expected count exactly, because the random stream consumed by the Poisson sampler depends on whether a rate is exactly zero.

Returns
-------
An int64 array of shape $(K,)$ gives the Poisson-fluctuated pseudo-data counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erf, erfc

def generate_pseudo_data(edges: np.ndarray, yields: np.ndarray, means: np.ndarray,
                         widths: np.ndarray, n_bkg: float, exp_slope: float,
                         seed: int) -> np.ndarray:
    r'''Poisson pseudo-data from three Gaussian peaks plus an exponential continuum.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges (GeV).
    yields : np.ndarray
        Shape $(3,)$, expected numbers of events $N_s\geq0$ of each Gaussian
        peak, applied to the untruncated Gaussian integral over each bin.
    means : np.ndarray
        Shape $(3,)$, Gaussian means $m_s$ (GeV).
    widths : np.ndarray
        Shape $(3,)$, Gaussian standard deviations $w_s>0$ (GeV).
    n_bkg : float
        Expected number $N_{\mathrm{bkg}}\geq0$ of background events in the
        window.
    exp_slope : float
        Slope $c>0$ of the background density $\propto e^{-c\,m}$, in
        $\mathrm{GeV^{-1}}$.
    seed : int
        Seed for np.random.default_rng.

    Returns
    -------
    counts : np.ndarray
        Shape $(K,)$, int64, drawn as rng.poisson(nu) in a single call on the
        array of expected counts $\nu_i$ ordered by increasing mass, where
        every $\nu_i$ is accurate to full relative precision (also in far
        Gaussian tails).

    Raises
    ------
    ValueError
        If edges is not a finite strictly increasing 1D array of length
        $\geq2$, if yields, means or widths do not have shape $(3,)$ or
        contain non-finite values, if any yield is negative or any width is
        $\leq0$, if $N_{\mathrm{bkg}}<0$, or if $c\leq0$.
    '''
    return counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, erfc

def _gauss_bin_integrals(edges, m, s):
    r"""Probability of $\mathcal{N}(m,s^2)$ in each bin, without cancellation in either tail."""
    from scipy.special import erf, erfc
    z = (edges - m) / (np.sqrt(2.0) * s)
    za, zb = z[:-1], z[1:]
    out = np.empty(za.size)
    up = za >= 0.0                 # bin entirely above the mean
    lo = zb <= 0.0                 # bin entirely below the mean
    mid = ~(up | lo)
    out[up] = 0.5 * (erfc(za[up]) - erfc(zb[up]))
    out[lo] = 0.5 * (erfc(-zb[lo]) - erfc(-za[lo]))
    out[mid] = 0.5 * (erf(zb[mid]) - erf(za[mid]))
    return out


def _oracle_generate_pseudo_data(edges: np.ndarray, yields: np.ndarray, means: np.ndarray,
                                 widths: np.ndarray, n_bkg: float, exp_slope: float,
                                 seed: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    yields = np.asarray(yields, dtype=float)
    means = np.asarray(means, dtype=float)
    widths = np.asarray(widths, dtype=float)
    for arr in (yields, means, widths):
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError("yields, means and widths must be finite with shape (3,)")
    if np.any(yields < 0) or np.any(widths <= 0):
        raise ValueError("yields must be >= 0 and widths > 0")
    if not np.isfinite(n_bkg) or n_bkg < 0:
        raise ValueError("n_bkg must be >= 0")
    if not np.isfinite(exp_slope) or exp_slope <= 0:
        raise ValueError("exp_slope must be > 0")
    nu = np.zeros(edges.size - 1)
    for s in range(3):
        nu += yields[s] * _gauss_bin_integrals(edges, means[s], widths[s])
    e = np.exp(-exp_slope * edges)
    nu += n_bkg * (e[:-1] - e[1:]) / (e[0] - e[-1])
    rng = np.random.default_rng(int(seed))
    return rng.poisson(nu).astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exact task configuration ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
""",
            "call": "generate_pseudo_data(edges.copy(), np.array([2800.0, 1300.0, 900.0]), masses.copy() - 0.012, np.array([0.072, 0.077, 0.080]), 7000.0, 0.4, 13600)",
            "gold_call": "_oracle_generate_pseudo_data(edges, np.array([2800.0, 1300.0, 900.0]), masses - 0.012, np.array([0.072, 0.077, 0.080]), 7000.0, 0.4, 13600)",
        },
        # --- Boundary: all yields zero -> all counts zero ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "generate_pseudo_data(edges.copy(), np.zeros(3), np.array([9.4, 10.0, 10.3]), np.array([0.07, 0.07, 0.07]), 0.0, 0.4, 1)",
            "gold_call": "_oracle_generate_pseudo_data(edges, np.zeros(3), np.array([9.4, 10.0, 10.3]), np.array([0.07, 0.07, 0.07]), 0.0, 0.4, 1)",
        },
        # --- Edge: peak outside the window (un-truncated Gaussians), few bins ---
        {
            "setup": """import numpy as np
edges = np.array([8.5, 9.0, 11.5])
""",
            "call": "generate_pseudo_data(edges.copy(), np.array([500.0, 0.0, 300.0]), np.array([8.4, 10.0, 12.0]), np.array([0.2, 0.1, 0.5]), 50.0, 2.0, 7)",
            "gold_call": "_oracle_generate_pseudo_data(edges, np.array([500.0, 0.0, 300.0]), np.array([8.4, 10.0, 12.0]), np.array([0.2, 0.1, 0.5]), 50.0, 2.0, 7)",
        },
        # --- Edge: no background, peaks whose far tails give tiny-but-nonzero rates ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "generate_pseudo_data(edges.copy(), np.array([400.0, 300.0, 200.0]), np.array([9.2, 10.0, 11.3]), np.array([0.05, 0.06, 0.04]), 0.0, 0.4, 99)",
            "gold_call": "_oracle_generate_pseudo_data(edges, np.array([400.0, 300.0, 200.0]), np.array([9.2, 10.0, 11.3]), np.array([0.05, 0.06, 0.04]), 0.0, 0.4, 99)",
        },
        # --- Invalid: zero width ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
def run_model():
    try:
        generate_pseudo_data(edges.copy(), np.ones(3), np.array([9.4, 10.0, 10.3]), np.array([0.07, 0.0, 0.07]), 10.0, 0.4, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_pseudo_data(edges, np.ones(3), np.array([9.4, 10.0, 10.3]), np.array([0.07, 0.0, 0.07]), 10.0, 0.4, 1)
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
