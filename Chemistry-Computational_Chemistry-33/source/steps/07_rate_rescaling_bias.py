"""
Final orchestrator. It chains the earlier steps to answer one question about an accelerated simulation of a driven, channelled catalytic surface: if the fast in-channel diffusion is treated as infinitely fast, which is what a scheme that rescales the rate constants of quasi-equilibrated events does once its equilibration test passes, by what fraction is the reaction rate it reports wrong?

The reaction propensity of a configuration is k_rxn times its doubly occupied bonds along a channel plus k_rxn_cross times those across channels. With o_0, ..., o_K the order-by-order coefficients of the window-averaged reaction rate, its k-th order truncation is <R>^(k) = sum over l <= k of eps^l o_l. Its zeroth order is the quasi-equilibrium prediction, the one an infinitely fast diffusion assumption produces, and the requested order is the corrected answer at this separation; the benchmark asks for the third. The number returned is the fractional bias of the quasi-equilibrium prediction,

    Delta = (<R>^(K) - <R>^(0)) / <R>^(0),

a dimensionless ratio, positive when the quasi-equilibrium picture understates the rate. It is referred to the quasi-equilibrium prediction because that is the number an accelerated run actually reports.

The initial reduced distribution is the column of Q that belongs to the configuration initial_config. For the empty surface that column selects a class holding one configuration, and the higher-order reduced terms therefore start at zero.

Final orchestrator. It chains the earlier steps to answer one question about an accelerated simulation of a driven, channelled catalytic surface: if the fast in-channel diffusion is treated as infinitely fast, which is what a scheme that rescales the rate constants of quasi-equilibrated events does once its equilibration test passes, by what fraction is the reaction rate it reports wrong?

Returns
-------
float: (<R>^(K) - <R>^(0)) / <R>^(0), the fractional bias of the quasi-equilibrium prediction of the window-averaged reaction rate at the requested order (third for the benchmark), as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rate_rescaling_bias(nrow: int = 3, ncol: int = 5, k_ads: float = 8.5, k_des: float = 4.0, k_rxn: float = 10.0, k_rxn_cross: float = 4.0, kappa: float = 1.0, drive_bias: float = 0.5, kappa_cross: float = 0.4, interaction: float = 1.5, eps: float = 0.02, t_start: float = 0.02, t_end: float = 0.08, initial_config: int = 0, order: int = 3) -> float: """Returns the fractional finite-rate bias of the quasi-equilibrium reaction rate. Raises: ValueError if nrow, ncol, initial_config or order is not an integer; if the lattice does not have between 1 and 15 sites; if initial_config does not index a configuration of the lattice; if order is below 1; if eps is not positive and finite; if a rate constant or hopping prefactor is negative or non-finite, or drive_bias lies outside [-1, 1]; if t_start is not finite and non-negative or t_end is not finite and greater than t_start; or if the quasi-equilibrium rate vanishes, which leaves the fractional bias undefined."""; return bias

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rate_rescaling_bias(nrow: int = 3, ncol: int = 5, k_ads: float = 8.5,
                                k_des: float = 4.0, k_rxn: float = 10.0,
                                k_rxn_cross: float = 4.0, kappa: float = 1.0,
                                drive_bias: float = 0.5, kappa_cross: float = 0.4,
                                interaction: float = 1.5, eps: float = 0.02,
                                t_start: float = 0.02, t_end: float = 0.08,
                                initial_config: int = 0, order: int = 3) -> float:
    for name, v in (("nrow", nrow), ("ncol", ncol), ("initial_config", initial_config),
                    ("order", order)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    nrow, ncol, initial_config, order = int(nrow), int(ncol), int(initial_config), int(order)
    if nrow < 1 or ncol < 1 or nrow * ncol > 15:
        raise ValueError("lattice must have between 1 and 15 sites")
    if initial_config < 0 or initial_config >= (1 << (nrow * ncol)):
        raise ValueError("initial_config must index a configuration of the lattice")
    if order < 1:
        raise ValueError("order must be >= 1")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite number")
    t_start, t_end = float(t_start), float(t_end)
    if not np.isfinite(t_start) or t_start < 0.0:
        raise ValueError("t_start must be a finite non-negative number")
    if not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_end must be finite and greater than t_start")
    F, S, event_rate = _oracle_lattice_generators(
        nrow, ncol, k_ads, k_des, k_rxn, k_rxn_cross, kappa, drive_bias,
        kappa_cross, interaction)
    event_rate = np.asarray(event_rate, dtype=float)
    Q = np.asarray(_oracle_fast_projector(F), dtype=float)
    K0 = _oracle_driven_steady_state(F, Q)
    resp, L = _oracle_hierarchy_closure(F, S, Q, K0, order)
    ptilde0 = Q[:, initial_config].copy()
    coeffs = _oracle_window_averaged_state(L, ptilde0, t_start, t_end)
    orders = np.asarray(_oracle_turnover_orders(event_rate, K0, resp, coeffs), dtype=float)
    if orders[0] == 0.0:
        raise ValueError("the quasi-equilibrium rate vanishes, so the bias is undefined")
    correction = sum(eps ** k * orders[k] for k in range(1, order + 1))
    return float(correction / orders[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = "import numpy as np\n"
    def _raises(kw):
        return setup_common + """
def run_model():
    try:
        rate_rescaling_bias(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_rate_rescaling_bias(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (kw, kw)
    def _case(args):
        return {"setup": setup_common,
                "call": "float(rate_rescaling_bias(%s))" % args,
                "gold_call": "float(_oracle_rate_rescaling_bias(%s))" % args}
    return [
        # --- Normal: the task's own 3 x 5 configuration at third order ---
        _case(""),
        # --- Normal: slower and less reactive cross-channel pairs on the task
        # lattice ---
        _case("k_rxn_cross=2.0, kappa_cross=0.1"),
        # --- Normal: the corrected prediction at second order ---
        _case("order=2"),
        # --- Normal: attractive interaction on a smaller driven lattice ---
        _case("2, 3, 3.0, 1.5, 6.0, 2.4, 1.0, 0.7, 0.4, -0.8, 0.05, 0.05, 0.2"),
        # --- Normal: channels four sites long, with a window opening at t = 0 ---
        _case("3, 4, t_start=0.0, t_end=0.05"),
        # --- Boundary: an undriven surface of the task's size ---
        _case("drive_bias=0.0"),
        # --- Boundary: a single column ---
        _case("4, 1, 2.0, 1.0, 3.0, 5.0, 1.0, 0.5, 0.7, 1.2, 0.05, 0.1, 0.4"),
        # --- Edge: no lateral interaction on the task lattice ---
        _case("interaction=0.0"),
        # --- Edge: a surface prepared with one adsorbate already on site 0 ---
        _case("2, 3, 5.0, 2.0, 8.0, 3.2, 1.0, 0.5, 0.4, 1.0, 0.02, 0.03, 0.09, 1"),
        # --- Edge: a fifth-order prediction on a smaller cell ---
        _case("2, 3, 5.0, 2.0, 8.0, 3.2, 1.0, 0.5, 0.4, 1.0, 0.03, 0.03, 0.09, 0, 5"),
        # --- Invalid: a non-positive separation parameter ---
        {"setup": _raises("eps=0.0"), "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: an inert surface leaves the fractional bias undefined ---
        {"setup": _raises("3, 3, k_rxn=0.0, k_rxn_cross=0.0"), "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: an order below one ---
        {"setup": _raises("order=0"), "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a lattice of sixteen sites ---
        {"setup": _raises("4, 4"), "call": "run_model()", "gold_call": "run_oracle()"},
    ]
