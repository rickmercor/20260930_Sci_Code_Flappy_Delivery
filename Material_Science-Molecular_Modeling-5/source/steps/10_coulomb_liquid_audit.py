"""
Assemble the audit of the self-consistent scheme at coupling Gamma and density rho. Row 0: the variational splitting length, then the three excess-energy values of step 9 at that splitting length. Row 1: the reduced specific heat of the first-order scheme, which is the excess energy minus Gamma times its derivative with respect to Gamma at fixed density, with the splitting length re-solved at every coupling and the derivative converged to 1e-8 by Richardson extrapolation of central differences; the Debye-Hueckel specific heat (Gamma/4) kappa0 K1(kappa0); the opposite-charge structure factor at q = 2; and the charge structure factor at q = 0.5. Row 2: the coupling gamma0 in the bracket [gamma_lo, gamma_hi] at which the first-order excess energy changes sign at this density, converged to 1e-10 with a bracketing root finder over the full chain; the variational splitting length at gamma0; the total pair potential vs + Gl at contact for the input state; and T++ at contact for the input state. Row 3: the screened kernel Gl at u = 0; the variational residual at the trial splitting length 0.7; and the opposite-charge and like-charge Mayer functions at contact, all for the input state. Every entry must be produced by the corresponding earlier step. Reject invalid parameters, a bracket that is not ordered and positive, and a bracket on which the energy does not change sign.

The audit ties the chain together where it is most sensitive: the specific heat is a coupling derivative that magnifies every inaccuracy of the correlation corrections, and the sign-change coupling of the excess energy is a root of the entire scheme, in which the variational splitting length, the correlation corrections with their disk geometry and the energy integral all enter at once. The source reports that at reduced density 0.15 its excess energy changes sign near a coupling of six.

Returns
-------
A (4, 4) float64 array laid out as described, with the sign-change coupling in row 2, column 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coulomb_liquid_audit(Gamma: float, rho: float, gamma_lo: float, gamma_hi: float) -> "np.ndarray":
    """Assemble the audit of the self-consistent scheme at coupling Gamma and density rho. Row 0:
    the variational splitting length, then the three excess-energy values of step 9 at that
    splitting length. Row 1: the reduced specific heat of the first-order scheme, which is the
    excess energy minus Gamma times its derivative with respect to Gamma at fixed density, with
    the splitting length re-solved at every coupling and the derivative converged to 1e-8 by
    Richardson extrapolation of central differences; the Debye-Hueckel specific heat (Gamma/4)
    kappa0 K1(kappa0); the opposite-charge structure factor at q = 2; and the charge structure
    factor at q = 0.5. Row 2: the coupling gamma0 in the bracket [gamma_lo, gamma_hi] at which
    the first-order excess energy changes sign at this density, converged to 1e-10 with a
    bracketing root finder over the full chain; the variational splitting length at gamma0; the
    total pair potential vs + Gl at contact for the input state; and T++ at contact for the
    input state. Row 3: the screened kernel Gl at u = 0; the variational residual at the trial
    splitting length 0.7; and the opposite-charge and like-charge Mayer functions at contact,
    all for the input state. Every entry must be produced by the corresponding earlier step.
    Reject invalid parameters, a bracket that is not ordered and positive, and a bracket on
    which the energy does not change sign.

    Args:
        Gamma: positive finite float, the Coulomb coupling of the input state.
        rho: positive finite float, the total reduced ion density rho a^2 of the input state.
        gamma_lo: positive finite float, the lower end of the coupling bracket for the sign change of the
            first-order excess energy at density rho.
        gamma_hi: finite float greater than gamma_lo, the upper end of that bracket.

    Returns:
        A numpy float64 array of shape (4, 4). Row 0: the variational splitting length, the uncorrected
        excess energy, the first-order excess energy and the Debye-Hueckel energy at the input state. Row 1:
        the first-order reduced specific heat CV = E - Gamma dE/dGamma at fixed density (splitting length
        re-solved at every coupling, derivative converged to 1e-8), the Debye-Hueckel specific heat
        (Gamma/4) kappa0 K1(kappa0), S+-(q = 2) and SZZ(q = 0.5). Row 2: the sign-change coupling gamma0 of
        the first-order energy in [gamma_lo, gamma_hi] (converged to 1e-10), the variational splitting length
        at gamma0, the total pair potential vs + Gl at contact and T++ at contact for the input state. Row 3:
        Gl at u = 0, the variational residual at the trial splitting length 0.7, and the opposite-charge and
        like-charge Mayer functions at contact, all for the input state.

    Raises:
        ValueError: if Gamma or rho is not a positive finite number; or if the bracket does not satisfy
            0 < gamma_lo < gamma_hi with both ends finite.
        RuntimeError: if the first-order energy does not change sign on the bracket.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _energy_full(Gamma, rho):
    s = _oracle_splitting_length(Gamma, rho)
    E = _oracle_excess_energy(Gamma, rho, s)
    return float(E[1]), float(E[0]), s


def _oracle_coulomb_liquid_audit(Gamma: float, rho: float, gamma_lo: float, gamma_hi: float) -> "np.ndarray":
    Gamma, rho = _check_state(Gamma, rho)
    gamma_lo = float(gamma_lo); gamma_hi = float(gamma_hi)
    if not (0.0 < gamma_lo < gamma_hi) or not np.isfinite(gamma_hi):
        raise ValueError("bracket must satisfy 0 < gamma_lo < gamma_hi")
    s = _oracle_splitting_length(Gamma, rho)
    K = _oracle_filter_kernels(np.array([1e-8, 2.0]), s, Gamma, rho)
    if abs(K[2, 0] * rho - 1.0) > 1e-6:
        raise RuntimeError("screened kernel limit violated")
    gl0 = float(_oracle_long_range_kernel(np.array([0.0]), s, Gamma, rho)[0])
    w1 = float(_oracle_short_range_potential(np.array([1.0]), s, Gamma)[0] + _oracle_long_range_kernel(np.array([1.0]), s, Gamma, rho)[0])
    hc = _oracle_mayer_functions(np.array([1.0]), s, Gamma, rho)
    res07 = _oracle_variational_residual(0.7, Gamma, rho)
    E = _oracle_excess_energy(Gamma, rho, s)
    h = 0.02 * Gamma
    d1 = (_energy_full(Gamma + h, rho)[0] - _energy_full(Gamma - h, rho)[0]) / (2.0 * h)
    d2 = (_energy_full(Gamma + 0.5 * h, rho)[0] - _energy_full(Gamma - 0.5 * h, rho)[0]) / h
    dE = (4.0 * d2 - d1) / 3.0
    cv = E[1] - Gamma * dE
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    cv_dh = 0.25 * Gamma * kap * k1(kap)
    Sq = _oracle_structure_factors(np.array([0.5, 2.0]), s, Gamma, rho)
    T1 = _oracle_pair_convolutions(np.array([1.0]), s, Gamma, rho)
    f = lambda g: _energy_full(g, rho)[0]
    flo, fhi = f(gamma_lo), f(gamma_hi)
    if flo * fhi > 0.0:
        raise RuntimeError("no sign change on the bracket")
    g0 = brentq(f, gamma_lo, gamma_hi, xtol=1e-10, rtol=1e-12, maxiter=100)
    s0 = _oracle_splitting_length(g0, rho)
    audit = np.array([[s, E[0], E[1], E[2]],
                      [cv, cv_dh, Sq[1, 1], Sq[2, 0]],
                      [g0, s0, w1, T1[0, 0]],
                      [gl0, res07, hc[0, 0], hc[1, 0]]])
    return audit

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\ngamma_lo, gamma_hi = 8.0, 9.2\n',
         'call': 'coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'gold_call': '_oracle_coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'tol': 2e-06},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\ngamma_lo, gamma_hi = 7.4, 8.4\n',
         'call': 'coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'gold_call': '_oracle_coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'tol': 2e-06},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\ngamma_lo, gamma_hi = 7.2, 8.2\n# boundary: the strongest coupling and highest density of the test set\n',
         'call': 'coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'gold_call': '_oracle_coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'tol': 2e-06},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\ngamma_lo, gamma_hi = 10.2, 11.2\n# edge: the most dilute state, whose sign change lies above Gamma = 10\n',
         'call': 'coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'gold_call': '_oracle_coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi)',
         'tol': 2e-06},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\ngamma_lo, gamma_hi = 9.2, 8.0\n# invalid input: a reversed bracket must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi))',
         'gold_call': '_catches_value_error(lambda: _oracle_coulomb_liquid_audit(Gamma, rho, gamma_lo, gamma_hi))'},
    ]
