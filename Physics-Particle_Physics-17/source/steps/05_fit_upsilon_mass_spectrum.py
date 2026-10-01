"""
Extract the Upsilon(nS) yields from a dimuon invariant-mass histogram with the fit of the CMS 13.6 TeV analysis: the signal model of $upsilon_signal_expected_counts$ plus the continuum background of $quadratic_background_bin_fractions$, fitted with the statistical method used in the paper. The free parameters, in this order, are



$$[N_{1S},\ N_{2S},\ N_{3S},\ N_{\mathrm{bkg}},\ \mu_1,\ \sigma_1,\ b_1,\ b_2],$$



where $N_{\mathrm{bkg}}$ is the background yield in the window. Parameter points where the model is undefined (it raises `ValueError`) are excluded from the fit. The global optimum of the paper's fit statistic is unique for the task's data; any minimizer that reaches it is acceptable.

For a histogram whose bin contents are independent Poisson variables, yields that are themselves Poisson-distributed are extracted with an extended likelihood, in which the total number of events is not fixed. CMS performs this fit with RooFit in every $(p_{\mathrm{T}}, |y|)$ bin, with the three signal yields, the Upsilon(1S) mean and width, the background yield and two background shape coefficients as free parameters. Because the pseudo-data were not generated with the fit model, the fitted Upsilon(3S) yield differs from the injected value, and the size of that difference depends on the exact fit model and statistic.

Returns
-------
A float array of shape $(8,)$ gives the best-fit $[N_{1S}, N_{2S}, N_{3S}, N_{\mathrm{bkg}}, \mu_1, \sigma_1, b_1, b_2]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import minimize

def fit_upsilon_mass_spectrum(counts: np.ndarray, edges: np.ndarray, masses: np.ndarray,
                              start: np.ndarray) -> np.ndarray:
    r'''Fit the CMS three-$\Upsilon$ plus background model to a mass histogram.

    Parameters
    ----------
    counts : np.ndarray
        Shape $(K,)$, observed non-negative bin contents; non-integer values
        (for example Asimov data) are allowed.
    edges : np.ndarray
        Shape $(K+1,)$, strictly increasing bin edges (GeV).
    masses : np.ndarray
        Shape $(3,)$, world-average $\Upsilon(nS)$ masses (GeV).
    start : np.ndarray
        Shape $(8,)$, starting point
        $[N_{1S}, N_{2S}, N_{3S}, N_{\mathrm{bkg}}, \mu_1, \sigma_1, b_1, b_2]$
        at which the model is defined.

    Returns
    -------
    best_params : np.ndarray
        Shape $(8,)$, parameters at the optimum of the paper's fit statistic.

    Raises
    ------
    ValueError
        If counts is not 1D with len(edges) $-\,1$ finite non-negative
        entries, if start does not have shape $(8,)$ or is not finite, or if
        the model is undefined at the starting point (negative
        $N_{\mathrm{bkg}}$, invalid signal or background parameters, or a
        non-positive expected count in any bin).
    '''
    return best_params  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize


def _extended_binned_nll(params, counts, edges, masses):
    r"""$\sum_i(\nu_i-n_i\ln\nu_i)$ of the CMS model; raises ValueError where undefined."""
    params = np.asarray(params, dtype=float)
    if params.shape != (8,) or not np.all(np.isfinite(params)):
        raise ValueError("params must be finite with shape (8,)")
    N1, N2, N3, NB, mu1, s1, b1, b2 = params
    if NB < 0:
        raise ValueError("N_bkg must be non-negative")
    sig = _oracle_upsilon_signal_expected_counts(edges, np.array([N1, N2, N3]), mu1, s1, masses)
    nu = sig.sum(axis=0) + NB * _oracle_quadratic_background_bin_fractions(edges, b1, b2)
    if np.any(nu <= 0):
        raise ValueError("expected counts must be positive in every bin")
    return float(np.sum(nu - counts * np.log(nu)))


def _oracle_fit_upsilon_mass_spectrum(counts: np.ndarray, edges: np.ndarray, masses: np.ndarray,
                                      start: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    start = np.asarray(start, dtype=float)
    if counts.ndim != 1 or counts.size != edges.size - 1:
        raise ValueError("counts must be 1D with len(edges) - 1 entries")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and non-negative")
    if start.shape != (8,) or not np.all(np.isfinite(start)):
        raise ValueError("start must be finite with shape (8,)")
    _extended_binned_nll(start, counts, edges, masses)  # raises if undefined

    def objective(p):
        try:
            return _extended_binned_nll(p, counts, edges, masses)
        except ValueError:
            return 1e30

    opts = {"maxiter": 20000, "maxfev": 20000, "xatol": 1e-10, "fatol": 1e-12}
    x, fbest = start.copy(), objective(start)
    for _ in range(5):
        res = minimize(objective, x, method="Nelder-Mead", options=opts)
        improved = fbest - res.fun
        x, fbest = res.x, res.fun
        if improved < 1e-9:
            break
    return np.asarray(x, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: task pseudo-data (candidate input from the public generator,
        #     reference input from the oracle generator) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
yields = np.array([2800.0, 1300.0, 900.0])
means = masses - 0.012
widths = np.array([0.072, 0.077, 0.080])
counts_model = generate_pseudo_data(edges.copy(), yields.copy(), means.copy(), widths.copy(), 7000.0, 0.4, 13600)
counts_gold = _oracle_generate_pseudo_data(edges, yields, means, widths, 7000.0, 0.4, 13600)
start = np.array([2500.0, 1200.0, 800.0, 7000.0, 9.45, 0.06, 0.0, 0.0])
""",
            "call": "fit_upsilon_mass_spectrum(counts_model, edges.copy(), masses.copy(), start.copy())[:4]",
            "gold_call": "_oracle_fit_upsilon_mass_spectrum(counts_gold, edges, masses, start)[:4]",
            "tol": 1e-3,
        },
        # --- Boundary: Asimov data built from the fit model itself -> truth recovered ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
truth = np.array([3000.0, 1400.0, 1000.0, 6000.0, 9.452, 0.057, -0.5, 0.2])
counts_model = (upsilon_signal_expected_counts(edges.copy(), truth[:3].copy(), truth[4], truth[5], masses.copy()).sum(axis=0)
                + truth[3] * quadratic_background_bin_fractions(edges.copy(), truth[6], truth[7]))
counts_gold = (_oracle_upsilon_signal_expected_counts(edges, truth[:3], truth[4], truth[5], masses).sum(axis=0)
               + truth[3] * _oracle_quadratic_background_bin_fractions(edges, truth[6], truth[7]))
start = np.array([2500.0, 1200.0, 800.0, 7000.0, 9.45, 0.06, 0.0, 0.0])
""",
            "call": "fit_upsilon_mass_spectrum(counts_model, edges.copy(), masses.copy(), start.copy())",
            "gold_call": "_oracle_fit_upsilon_mass_spectrum(counts_gold, edges, masses, start)",
            "tol": 1e-3,
        },
        # --- Edge: low-statistics sample (tenfold fewer events, different seed) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
yields = np.array([280.0, 130.0, 90.0])
means = masses - 0.012
widths = np.array([0.072, 0.077, 0.080])
counts_model = generate_pseudo_data(edges.copy(), yields.copy(), means.copy(), widths.copy(), 700.0, 0.4, 2022)
counts_gold = _oracle_generate_pseudo_data(edges, yields, means, widths, 700.0, 0.4, 2022)
start = np.array([250.0, 120.0, 80.0, 700.0, 9.45, 0.06, 0.0, 0.0])
""",
            "call": "fit_upsilon_mass_spectrum(counts_model, edges.copy(), masses.copy(), start.copy())[:4]",
            "gold_call": "_oracle_fit_upsilon_mass_spectrum(counts_gold, edges, masses, start)[:4]",
            "tol": 1e-3,
        },
        # --- Invalid: model undefined at the starting point (sigma1 <= 0) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
counts = np.ones(75)
bad = np.array([10.0, 10.0, 10.0, 10.0, 9.45, -0.06, 0.0, 0.0])
def run_model():
    try:
        fit_upsilon_mass_spectrum(counts, edges.copy(), masses.copy(), bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fit_upsilon_mass_spectrum(counts, edges, masses, bad)
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
