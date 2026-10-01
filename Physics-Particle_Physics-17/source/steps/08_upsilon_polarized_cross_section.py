"""
ORCHESTRATOR. End-to-end reproduction of the task:



1. Edges: 75 bins of 40 MeV on $[8.5, 11.5]$ GeV; masses are the world-average values given in the task, $[9.46040, 10.0234, 10.3551]$ GeV.

2. Pseudo-data: $generate_pseudo_data$ with yields $[2800, 1300, 900]$, means equal to the masses minus 0.012 GeV, widths $[0.072, 0.077, 0.080]$ GeV, $N_{\mathrm{bkg}}=7000$, exponential slope $c=0.4\ \mathrm{GeV^{-1}}$, and the given seed.

3. Fit: $fit_upsilon_mass_spectrum$ (signal from $upsilon_signal_expected_counts$, background from $quadratic_background_bin_fractions$) from the start point $[2500, 1200, 800, 7000, 9.45, 0.06, 0.0, 0.0]$.

4. Cross section of the requested state with $cross_section_times_bf$: $\mathcal{L}=37.4\ \mathrm{fb^{-1}}$, $70<p_{\mathrm{T}}<100$ GeV, $|y|<0.6$, single-muon efficiencies 0.92 and 0.90, $\rho_{\mathrm{pair}}=0.95$, $A=0.38$.

5. Rescale to $\lambda_\theta$ with $polarization_scale_factor$, using the paper's Table 1 factors for this $(p_{\mathrm{T}}, |y|)$ bin, $k_+=1.12$ ($\lambda_\theta=+1$) and $k_-=0.83$ ($\lambda_\theta=-1$), which are not inputs of this function.



The task answer uses $state_index = 2$, $\lambda_\theta=-0.5$ and $seed = 13600$.

The orchestrator reproduces the analysis chain for a single $(p_{\mathrm{T}}, |y|)$ bin: data histogram, constrained fit to extract the Upsilon(nS) yield, conversion to $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ with luminosity, bin widths, efficiency and acceptance, and final re-interpretation under a chosen polarization hypothesis. This mirrors how phenomenologists re-use the published unpolarized results in NRQCD global fits that assume a specific polarization.

Returns
-------
A native Python float gives the polarized $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ of the chosen Upsilon state in pb/GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def upsilon_polarized_cross_section(lambda_theta: float = -0.5, seed: int = 13600,
                                    state_index: int = 2) -> float:
    r'''Full pipeline: pseudo-data, constrained fit, cross section, polarization.

    Parameters
    ----------
    lambda_theta : float
        Target helicity-frame polar anisotropy $\lambda_\theta\in[-1,1]$.
    seed : int
        Seed for the pseudo-data Poisson draw.
    state_index : int
        0, 1 or 2 for $\Upsilon(1S)$, $\Upsilon(2S)$ or $\Upsilon(3S)$.

    Returns
    -------
    xsec : float
        $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ of the requested state in
        pb/GeV for $70<p_{\mathrm{T}}<100$ GeV and $|y|<0.6$, under the
        requested polarization, as a native Python float.

    Raises
    ------
    ValueError
        If state_index is not one of 0, 1, 2, or if $\lambda_\theta$ is not
        finite or lies outside $[-1,1]$.
    '''
    return xsec  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_TABLE1_K_PLUS_70_100_Y06 = 1.12    # CMS Table 1, 70-100 GeV, |y| < 0.6, lambda = +1
_TABLE1_K_MINUS_70_100_Y06 = 0.83   # CMS Table 1, 70-100 GeV, |y| < 0.6, lambda = -1


def _oracle_upsilon_polarized_cross_section(lambda_theta: float = -0.5, seed: int = 13600,
                                            state_index: int = 2) -> float:
    if state_index not in (0, 1, 2):
        raise ValueError("state_index must be 0, 1 or 2")
    k = _oracle_polarization_scale_factor(lambda_theta, _TABLE1_K_PLUS_70_100_Y06,
                                          _TABLE1_K_MINUS_70_100_Y06)
    edges = np.linspace(8.5, 11.5, 76)
    masses = np.array([9.46040, 10.0234, 10.3551])
    counts = _oracle_generate_pseudo_data(edges, np.array([2800.0, 1300.0, 900.0]),
                                          masses - 0.012, np.array([0.072, 0.077, 0.080]),
                                          7000.0, 0.4, seed)
    start = np.array([2500.0, 1200.0, 800.0, 7000.0, 9.45, 0.06, 0.0, 0.0])
    p = _oracle_fit_upsilon_mass_spectrum(counts, edges, masses, start)
    xsec0 = _oracle_cross_section_times_bf(p[state_index], 37.4, 70.0, 100.0, 0.0, 0.6,
                                           0.92, 0.90, 0.95, 0.38)
    return float(k * xsec0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task answer ---
        {
            "setup": "",
            "call": "upsilon_polarized_cross_section(-0.5, 13600, 2)",
            "gold_call": "_oracle_upsilon_polarized_cross_section(-0.5, 13600, 2)",
            "tol": 2e-6,
        },
        # --- Boundary: fully transverse Upsilon(1S) ---
        {
            "setup": "",
            "call": "upsilon_polarized_cross_section(1.0, 13600, 0)",
            "gold_call": "_oracle_upsilon_polarized_cross_section(1.0, 13600, 0)",
            "tol": 2e-6,
        },
        # --- Edge: fully longitudinal Upsilon(2S), different seed ---
        {
            "setup": "",
            "call": "upsilon_polarized_cross_section(-1.0, 7, 1)",
            "gold_call": "_oracle_upsilon_polarized_cross_section(-1.0, 7, 1)",
            "tol": 2e-6,
        },
        # --- Invalid: bad state index ---
        {
            "setup": """def run_model():
    try:
        upsilon_polarized_cross_section(-0.5, 13600, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_upsilon_polarized_cross_section(-0.5, 13600, 3)
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
