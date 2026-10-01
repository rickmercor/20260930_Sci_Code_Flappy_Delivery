"""
Compute the expected per-bin signal counts of the three Upsilon(nS) resonances in one $(p_{\mathrm{T}}, |y|)$ bin, using the signal model of the CMS 13.6 TeV measurement.



Each resonance is described by the paper's per-state line shape, built from Crystal Ball functions (see $crystal_ball_bin_fractions$), and the three resonances are tied together by the paper's inter-state constraints, so that the only free signal parameters are the three yields and the mean $\mu_1$ and width $\sigma_1$ of the Upsilon(1S) line shape. The shape constants that the paper keeps fixed in the fit are stated in the background; none of them is an input. All shapes are normalized to unit integral on the histogram window $[e_0, e_K]$ (`edges[0]` to `edges[-1]`), and counts are bin integrals.

In the CMS 13.6 TeV Upsilon analysis each resonance is described by a sum of Crystal Ball functions sharing a common mean, which captures the $p_{\mathrm{T}}$- and $\eta$-dependent dimuon mass resolution together with the final-state-radiation tail. The three states have nearly identical decay kinematics, so their line shapes are tied together: a common momentum-scale offset moves all peaks together relative to the world-average masses, and the mass resolution, dominated by the relative muon-momentum resolution, scales with the resonance mass. As a result only the three yields and two shape parameters of the Upsilon(1S) float in the fit; all other shape constants are fixed from simulation studies documented in the paper. Section 4 of the paper fixes them as follows. Each state's line shape is



$$f\,\mathrm{CB}_1+(1-f)\,\mathrm{CB}_2,$$



with both components sharing the state's mean, $\mathrm{CB}_1$ the narrower component with weight $f=0.4$, the wide-component width of each state equal to 1.55 times its own narrow-component width ($\sigma_2=1.55\,\sigma_1$), $\alpha_1=\alpha_2=2$, $n_1=1$ and $n_2=2$. State $s$ has mean and narrow-component width



$$\mu_s=\mu_1+(M_s-M_{1S}),\qquad \sigma_{1,s}=\sigma_1\,\dfrac{M_s}{M_{1S}},$$



where $M_s$ are the supplied world-average masses. In this task each component is normalized to unit integral on the window before the weights are applied.

Returns
-------
A float array of shape $(3,K)$ gives the expected counts per bin for Upsilon(1S), Upsilon(2S) and Upsilon(3S), one row per state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def upsilon_signal_expected_counts(edges: np.ndarray, yields: np.ndarray, mu1: float,
                                   sigma1: float, masses: np.ndarray) -> np.ndarray:
    r'''Expected per-bin counts of the three $\Upsilon(nS)$ peaks in the CMS fit model.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing bin edges (GeV).
    yields : np.ndarray
        Shape $(3,)$, yields $N_{1S}, N_{2S}, N_{3S}$ of $\Upsilon(1S)$,
        $\Upsilon(2S)$ and $\Upsilon(3S)$, each $\geq0$.
    mu1 : float
        Mean $\mu_1$ of the $\Upsilon(1S)$ line shape (GeV).
    sigma1 : float
        Core width $\sigma_1>0$ of the narrowest Crystal Ball component of
        the $\Upsilon(1S)$ line shape (GeV).
    masses : np.ndarray
        Shape $(3,)$, increasing positive world-average masses
        $M_{1S}, M_{2S}, M_{3S}$ (GeV).

    Returns
    -------
    counts : np.ndarray
        Shape $(3,K)$. Row $s$ is $N_s$ times the window-normalized bin
        fractions of the paper's line shape for state $s$, with the paper's
        fixed shape constants and inter-state constraints.

    Raises
    ------
    ValueError
        If yields or masses do not have shape $(3,)$ or contain non-finite
        values, if any yield is negative, if masses are not positive and
        strictly increasing, or under any condition for which
        crystal_ball_bin_fractions raises (invalid edges, non-finite mu1,
        $\sigma_1\leq0$).
    '''
    return counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_CMS_F_NARROW = 0.4       # weight of the narrower Crystal Ball
_CMS_WIDTH_RATIO = 1.55   # sigma2 / sigma1
_CMS_ALPHA1, _CMS_N1 = 2.0, 1.0
_CMS_ALPHA2, _CMS_N2 = 2.0, 2.0


def _double_cb_bin_fractions(edges, mu, sigma1):
    r"""$f\,\mathrm{CB}_1+(1-f)\,\mathrm{CB}_2$, each component window-normalized, common mean."""
    frac1 = _oracle_crystal_ball_bin_fractions(edges, mu, sigma1, _CMS_ALPHA1, _CMS_N1)
    frac2 = _oracle_crystal_ball_bin_fractions(edges, mu, _CMS_WIDTH_RATIO * sigma1,
                                               _CMS_ALPHA2, _CMS_N2)
    return _CMS_F_NARROW * frac1 + (1.0 - _CMS_F_NARROW) * frac2


def _oracle_upsilon_signal_expected_counts(edges: np.ndarray, yields: np.ndarray, mu1: float,
                                           sigma1: float, masses: np.ndarray) -> np.ndarray:
    yields = np.asarray(yields, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if yields.shape != (3,) or masses.shape != (3,):
        raise ValueError("yields and masses must have shape (3,)")
    if not (np.all(np.isfinite(yields)) and np.all(np.isfinite(masses))):
        raise ValueError("yields and masses must be finite")
    if np.any(yields < 0):
        raise ValueError("yields must be non-negative")
    if np.any(masses <= 0) or not np.all(np.diff(masses) > 0):
        raise ValueError("masses must be positive and strictly increasing")
    rows = []
    for s in range(3):
        mu_s = mu1 + (masses[s] - masses[0])        # world-average mass differences (paper, Sec. 4)
        sig_s = sigma1 * masses[s] / masses[0]      # widths scale with mass
        rows.append(yields[s] * _double_cb_bin_fractions(edges, mu_s, sig_s))
    return np.vstack(rows).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: fit-like configuration on the task histogram ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
yields = np.array([2925.5, 1339.3, 947.5])
""",
            "call": "upsilon_signal_expected_counts(edges.copy(), yields.copy(), 9.4507, 0.0553, masses.copy())",
            "gold_call": "_oracle_upsilon_signal_expected_counts(edges, yields, 9.4507, 0.0553, masses)",
        },
        # --- Normal: only the Upsilon(3S) populated, coarser binning ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 31)
masses = np.array([9.46040, 10.0234, 10.3551])
""",
            "call": "upsilon_signal_expected_counts(edges.copy(), np.array([0.0, 0.0, 1000.0]), 9.45, 0.07, masses.copy())",
            "gold_call": "_oracle_upsilon_signal_expected_counts(edges, np.array([0.0, 0.0, 1000.0]), 9.45, 0.07, masses)",
        },
        # --- Boundary: zero yields give an all-zero (3, K) array ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
""",
            "call": "upsilon_signal_expected_counts(edges.copy(), np.zeros(3), 9.45, 0.06, masses.copy())",
            "gold_call": "_oracle_upsilon_signal_expected_counts(edges, np.zeros(3), 9.45, 0.06, masses)",
        },
        # --- Edge: mu1 shifted so the 3S peak sits near the upper window edge ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 31)
masses = np.array([9.46040, 10.0234, 10.3551])
""",
            "call": "upsilon_signal_expected_counts(edges.copy(), np.array([100.0, 50.0, 25.0]), 10.55, 0.09, masses.copy())",
            "gold_call": "_oracle_upsilon_signal_expected_counts(edges, np.array([100.0, 50.0, 25.0]), 10.55, 0.09, masses)",
        },
        # --- Edge: artificial test masses (constraints must use the supplied masses) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.0, 12.0, 41)
masses = np.array([9.0, 10.0, 11.0])
""",
            "call": "upsilon_signal_expected_counts(edges.copy(), np.array([300.0, 200.0, 100.0]), 9.1, 0.05, masses.copy())",
            "gold_call": "_oracle_upsilon_signal_expected_counts(edges, np.array([300.0, 200.0, 100.0]), 9.1, 0.05, masses)",
        },
        # --- Invalid: negative yield ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
masses = np.array([9.46040, 10.0234, 10.3551])
def run_model():
    try:
        upsilon_signal_expected_counts(edges.copy(), np.array([10.0, -1.0, 5.0]), 9.45, 0.06, masses.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_upsilon_signal_expected_counts(edges, np.array([10.0, -1.0, 5.0]), 9.45, 0.06, masses)
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
