"""
Implement find_mixed_modes, which returns every unperturbed (non-rotating)
quadrupolar mixed-mode frequency of the asymptotic model inside a frequency
window.

The mixed modes are the solutions of the p-g coupling relation of the
asymptotic description used in infer_coupling_factor (uniform acoustic comb
of spacing delta_nu anchored at nu_p, pure gravity modes of periods
(n + 1/2 + eps_g) * delta_pi, coupling factor q). Every solution in the
window must be returned.

Returns
-------
np.ndarray, one-dimensional, sorted mixed-mode frequencies in microhertz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_mixed_modes(nu_lo: float, nu_hi: float, q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    '''All unperturbed l = 2 mixed-mode frequencies in [nu_lo, nu_hi].

    Parameters
    ----------
    nu_lo : float
        Lower edge of the window, in microhertz, strictly positive.
    nu_hi : float
        Upper edge of the window, in microhertz, strictly greater than nu_lo.
    q : float
        Coupling factor in (0, 1).
    eps_g : float
        Gravity phase (any real value; only its fractional part matters).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    nu_modes : np.ndarray
        One-dimensional float array of the mixed-mode frequencies in the
        window, in microhertz, sorted in increasing order, each accurate to
        better than 1e-9 microhertz. Empty if the window holds no mode.

    Raises
    ------
    ValueError
        If nu_lo is not strictly positive, if nu_hi <= nu_lo, if q is not in
        (0, 1), or if delta_nu or delta_pi is not strictly positive.
    '''
    return nu_modes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _coupling_residual(nu: "np.ndarray", q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    """Pole-free residual sin(theta_p) cos(theta_g) - q cos(theta_p) sin(theta_g), nu in microhertz."""
    nu = np.asarray(nu, dtype=float)
    theta_p = np.pi * (nu - nu_p) / delta_nu
    theta_g = np.pi / (delta_pi * nu * 1e-6) - np.pi * eps_g
    return np.sin(theta_p) * np.cos(theta_g) - q * np.cos(theta_p) * np.sin(theta_g)


def _oracle_find_mixed_modes(nu_lo: float, nu_hi: float, q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    """Reference implementation: one Brent solve per interval between consecutive poles."""
    nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi))
    if nu_lo <= 0.0 or nu_hi <= nu_lo:
        raise ValueError("the window must satisfy 0 < nu_lo < nu_hi")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    if delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("delta_nu and delta_pi must be strictly positive")
    # tan(theta_p) - q tan(theta_g) increases monotonically from -inf to +inf between
    # consecutive poles (nu_p + (k + 1/2) delta_nu for theta_p, the pure gravity modes
    # 1e6 / (delta_pi (n + 1/2 + eps_g)) for theta_g), so every such interval holds
    # exactly one mixed mode; bracketing them removes any dependence on a scan step.
    k_lo = np.floor((nu_lo - nu_p) / delta_nu - 0.5) - 1
    k_hi = np.ceil((nu_hi - nu_p) / delta_nu - 0.5) + 1
    p_poles = nu_p + (np.arange(k_lo, k_hi + 1) + 0.5) * delta_nu
    n_hi = np.ceil(1e6 / (delta_pi * nu_lo) - 0.5 - eps_g) + 1
    n_lo = np.floor(1e6 / (delta_pi * nu_hi) - 0.5 - eps_g) - 1
    n_vals = np.arange(n_lo, n_hi + 1)
    n_vals = n_vals[n_vals + 0.5 + eps_g > 0.0]
    g_poles = 1e6 / (delta_pi * (n_vals + 0.5 + eps_g))
    poles = np.sort(np.concatenate([p_poles, g_poles]))
    roots = []
    for left, right in zip(poles[:-1], poles[1:]):
        if right <= left or right < nu_lo or left > nu_hi:
            continue
        margin = 1e-6 * (right - left)
        lo, hi = left + margin, right - margin
        if hi <= lo:
            continue
        f_lo = _coupling_residual(lo, q, eps_g, delta_nu, nu_p, delta_pi)
        f_hi = _coupling_residual(hi, q, eps_g, delta_nu, nu_p, delta_pi)
        if f_lo == 0.0:
            root = lo
        elif f_hi == 0.0:
            root = hi
        elif f_lo * f_hi < 0.0:
            root = brentq(_coupling_residual, lo, hi, args=(q, eps_g, delta_nu, nu_p, delta_pi),
                          xtol=1e-13, rtol=4 * np.finfo(float).eps, maxiter=500)
        else:
            continue
        if nu_lo <= root <= nu_hi:
            roots.append(float(root))
    return np.array(sorted(set(roots)), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = "q, eps_g = 0.0360054005, 0.2299280975\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n"
    invalid_setup = """
def run_model():
    try:
        find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's window of three large separations around the radial mode ---
        {
            "setup": task + "nu_lo, nu_hi = 371.204 - 1.5 * 26.50, 371.204 + 1.5 * 26.50\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: strong coupling, dense low-frequency window ---
        {
            "setup": "q, eps_g = 0.35, 0.31\ndelta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422\nnu_lo, nu_hi = 150.0, 260.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: very dense window (gravity modes 0.1-0.8 microhertz apart, several hundred solutions) ---
        {
            "setup": "q, eps_g = 0.003, 0.4\ndelta_nu, nu_p, delta_pi = 12.0, 100.0, 40.0\nnu_lo, nu_hi = 60.0, 140.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: a luminous giant (small Delta nu, low frequencies) whose gravity modes are
        #     0.02-0.1 microhertz apart across the window ---
        {
            "setup": "q, eps_g = 0.02, 0.15\ndelta_nu, nu_p, delta_pi = 4.0, 30.0, 60.0\nnu_lo, nu_hi = 20.0, 40.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Boundary: narrow window containing exactly the two observed modes ---
        {
            "setup": task + "nu_lo, nu_hi = 366.0, 369.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: very weak coupling, modes collapse onto the pure p and g positions ---
        {
            "setup": "q, eps_g = 1e-4, 0.5\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\nnu_lo, nu_hi = 355.0, 385.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: a pure gravity mode 0.0005 microhertz from an acoustic anti-node with q = 0.002,
        #     so that one solution lies within 1e-6 microhertz of that gravity mode ---
        {
            "setup": "q, eps_g = 0.002, 0.5938303180452777\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\nnu_lo, nu_hi = 360.0, 400.0\n",
            "call": "find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)",
        },
        # --- Invalid: empty window (nu_hi <= nu_lo) ---
        {
            "setup": task + "nu_lo, nu_hi = 380.0, 380.0\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
