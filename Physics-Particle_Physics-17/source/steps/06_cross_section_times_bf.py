"""
Convert a fitted Upsilon signal yield in one $(p_{\mathrm{T}}, |y|)$ bin into the quantity reported by the CMS 13.6 TeV analysis: the $p_{\mathrm{T}}$-differential cross section times dimuon branching fraction, per unit rapidity, $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$, in pb/GeV, as defined by eq. (1.1) of the paper. The luminosity units, the rapidity interval associated with an $|y|$ bin, and the way the single-muon efficiencies and the muon-pair correction factor enter the dimuon efficiency follow the paper and are stated in the background.

CMS reports the $p_{\mathrm{T}}$-differential cross section times branching fraction per unit rapidity. The fitted yield is divided by the integrated luminosity, the widths of the transverse-momentum and rapidity intervals, the dimuon detection efficiency and the geometric acceptance (eq. (1.1) of the paper). Eq. (1.1) of the paper reads



$$\mathcal{B}\,\dfrac{d^2\sigma}{dy\,dp_{\mathrm{T}}}=\dfrac{N}{\mathcal{L}\,\Delta y\,\Delta p_{\mathrm{T}}\,\epsilon\,A},$$



with $\mathcal{L}$ in $\mathrm{pb^{-1}}$ ($1\ \mathrm{fb^{-1}}=1000\ \mathrm{pb^{-1}}$) so that the result is in pb/GeV. The paper takes $\Delta y=1.2$ for both of its $|y|$ bins ($|y|<0.6$ and $0.6<|y|<1.2$): an absolute-rapidity bin covers both signs of $y$, so $\Delta y=2\,(|y|_{\max}-|y|_{\min})$. Section 3 computes the dimuon efficiency as the product of the two single-muon efficiencies times the muon-pair correction factor, $\epsilon=\epsilon_{\mu1}\,\epsilon_{\mu2}\,\rho_{\mathrm{pair}}$.

Returns
-------
A native Python float gives $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ in pb/GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cross_section_times_bf(n_signal: float, lumi_fb_inv: float, pt_min: float, pt_max: float,
                           abs_y_min: float, abs_y_max: float, eff_mu1: float, eff_mu2: float,
                           rho_pair: float, acceptance: float) -> float:
    r'''$\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ in pb/GeV for one $(p_{\mathrm{T}}, |y|)$ bin.

    Parameters
    ----------
    n_signal : float
        Fitted signal yield $N\geq0$.
    lumi_fb_inv : float
        Integrated luminosity $\mathcal{L}>0$ in $\mathrm{fb^{-1}}$.
    pt_min, pt_max : float
        $p_{\mathrm{T}}$ bin edges (GeV), with $0\leq$ pt_min $<$ pt_max.
    abs_y_min, abs_y_max : float
        Edges of the $|y|$ (absolute rapidity) bin, with
        $0\leq$ abs_y_min $<$ abs_y_max.
    eff_mu1, eff_mu2 : float
        Single-muon efficiencies $\epsilon_{\mu1}$ and $\epsilon_{\mu2}$ of the
        two muons, each in $(0,1]$.
    rho_pair : float
        Muon-pair (close-pair trigger and vertex) correction factor
        $\rho_{\mathrm{pair}}$, in $(0,1]$.
    acceptance : float
        Dimuon acceptance $A$, in $(0,1]$.

    Returns
    -------
    xsec : float
        $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ as defined by the
        paper's eq. (1.1), native Python float in pb/GeV.

    Raises
    ------
    ValueError
        If any input is not finite, if $N<0$, $\mathcal{L}\leq0$,
        pt_min $<0$, pt_max $\leq$ pt_min, abs_y_min $<0$,
        abs_y_max $\leq$ abs_y_min, or if eff_mu1, eff_mu2, rho_pair or
        acceptance is outside $(0,1]$.
    '''
    return xsec  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cross_section_times_bf(n_signal: float, lumi_fb_inv: float, pt_min: float, pt_max: float,
                                   abs_y_min: float, abs_y_max: float, eff_mu1: float, eff_mu2: float,
                                   rho_pair: float, acceptance: float) -> float:
    vals = [n_signal, lumi_fb_inv, pt_min, pt_max, abs_y_min, abs_y_max,
            eff_mu1, eff_mu2, rho_pair, acceptance]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all inputs must be finite")
    if n_signal < 0 or lumi_fb_inv <= 0:
        raise ValueError("n_signal must be >= 0 and luminosity > 0")
    if pt_min < 0 or pt_max <= pt_min or abs_y_min < 0 or abs_y_max <= abs_y_min:
        raise ValueError("invalid pT or |y| interval")
    for v in (eff_mu1, eff_mu2, rho_pair, acceptance):
        if not (0.0 < v <= 1.0):
            raise ValueError("efficiencies and acceptance must lie in (0, 1]")
    lumi_pb = 1000.0 * lumi_fb_inv
    delta_y = 2.0 * (abs_y_max - abs_y_min)
    delta_pt = pt_max - pt_min
    eps = eff_mu1 * eff_mu2 * rho_pair
    return float(n_signal / (lumi_pb * delta_y * delta_pt * eps * acceptance))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: task bin 70-100 GeV, |y| < 0.6 ---
        {
            "setup": "",
            "call": "cross_section_times_bf(947.5257, 37.4, 70.0, 100.0, 0.0, 0.6, 0.92, 0.90, 0.95, 0.38)",
            "gold_call": "_oracle_cross_section_times_bf(947.5257, 37.4, 70.0, 100.0, 0.0, 0.6, 0.92, 0.90, 0.95, 0.38)",
        },
        # --- Normal: forward |y| interval 0.6-1.2 ---
        {
            "setup": "",
            "call": "cross_section_times_bf(947.5257, 37.4, 70.0, 100.0, 0.6, 1.2, 0.92, 0.90, 0.95, 0.38)",
            "gold_call": "_oracle_cross_section_times_bf(947.5257, 37.4, 70.0, 100.0, 0.6, 1.2, 0.92, 0.90, 0.95, 0.38)",
        },
        # --- Boundary: zero yield, all efficiencies exactly 1 ---
        {
            "setup": "",
            "call": "cross_section_times_bf(0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0)",
            "gold_call": "_oracle_cross_section_times_bf(0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0)",
        },
        # --- Edge: very narrow bin and tiny acceptance ---
        {
            "setup": "",
            "call": "cross_section_times_bf(3.0, 0.001, 199.9, 200.0, 1.19, 1.2, 0.5, 0.5, 0.1, 1e-4)",
            "gold_call": "_oracle_cross_section_times_bf(3.0, 0.001, 199.9, 200.0, 1.19, 1.2, 0.5, 0.5, 0.1, 1e-4)",
        },
        # --- Invalid: acceptance > 1 ---
        {
            "setup": """def run_model():
    try:
        cross_section_times_bf(10.0, 37.4, 70.0, 100.0, 0.0, 0.6, 0.9, 0.9, 0.95, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cross_section_times_bf(10.0, 37.4, 70.0, 100.0, 0.0, 0.6, 0.9, 0.9, 0.95, 1.5)
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
