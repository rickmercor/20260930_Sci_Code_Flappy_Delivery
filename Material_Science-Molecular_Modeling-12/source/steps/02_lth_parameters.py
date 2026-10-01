"""
Determine the first-estimate parameters of the mesoparticle local thermodynamic (LTh) model from the reduced macroscopic state (reduced temperature T*, pressure P*, macroscopic heat capacity per mesoparticle C_V*, thermal expansion coefficient alpha*, isothermal compressibility kappa* and bulk mesoparticle density c*; k_B = 1 in reduced units). The model gives every mesoparticle a pressure pi(theta, n) = pi00 + (alpha / kappa_T)(theta - theta0) + (1 / kappa_T) ln(n / n00) as a function of its dressed temperature theta and its density n, with reference values theta0, n00 and pi00 = pi(n00, theta0), a mesoscopic compressibility kappa_T and a mesoscopic expansion coefficient alpha, and an internal energy u = C_V theta + V(n) that is linear in the temperature with a mesoscopic heat capacity C_V. Fix theta0 = T* and n00 = c*, and determine pi00, kappa_T, alpha and C_V by requiring that the ensemble of mesoparticles reproduce the macroscopic pressure, isothermal compressibility, thermal expansion coefficient and isochoric heat capacity in the structureless limit in which density fluctuations are neglected and the pair distribution function is one everywhere, so that every particle sits at the bulk density c* and at the reservoir temperature: the macroscopic pressure is then the ideal mesoparticle gas term c* T* plus the particle pressure pi(T*, c*), and the internal energy per mesoparticle is the translational (3/2) T* plus the internal C_V T* plus V(c*). Derive the four relations from these two equations of state and their temperature and density derivatives (the macroscopic compressibility and expansion coefficient are defined from the total pressure at constant temperature and at constant pressure respectively) and return the parameters. Raise ValueError if T*, kappa* or c* is not positive, if the ideal-gas contribution kappa* c* T* reaches one, or if the resulting C_V is not positive.

The mesoscopic parameters are not the macroscopic ones: the mesoparticles' own translational degrees of freedom already supply an ideal-gas pressure c* k_B T* and a heat capacity (3/2) k_B, so the particle pressure, compressibility, expansion coefficient and heat capacity have to absorb only the remainder. For liquid argon at the reference state of the source (T* = 0.0111, P* = 0.127) the corrections are of the order of one percent for the compressibility and of the expansion coefficient and 3/2 for the heat capacity, which the source tabulates as kappa_T* = 1.01, alpha* = 29.35 and C_V* = 12.51 (the macroscopic value) with pi00* = 0.1161.

Returns
-------
A (6,) float64 array [theta0, n00, pi00, kappa_T, alpha, C_V] in reduced units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar):
    """Determine the first-estimate parameters of the mesoparticle local thermodynamic (LTh)
    model from the reduced macroscopic state (reduced temperature T*, pressure P*,
    macroscopic heat capacity per mesoparticle C_V*, thermal expansion coefficient alpha*,
    isothermal compressibility kappa* and bulk mesoparticle density c*; k_B = 1 in reduced
    units). A (6,) float64 array [theta0, n00, pi00, kappa_T, alpha, C_V] in reduced units."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar):
    if t_star <= 0 or kappa_bar <= 0 or cbar <= 0:
        raise ValueError("temperature, compressibility and density must be positive")
    if kappa_bar*cbar*t_star >= 1.0:
        raise ValueError("ideal-gas compressibility exceeds the macroscopic one")
    pi00 = p_star - cbar*t_star
    kappa = kappa_bar/(1.0 - kappa_bar*cbar*t_star)
    alpha = alpha_bar*(1.0 + kappa*cbar*t_star) - kappa*cbar
    cv = cv_bar - 1.5
    if cv <= 0:
        raise ValueError("internal heat capacity must be positive")
    return np.array([t_star, cbar, pi00, kappa, alpha, cv])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: reduced liquid-argon state\nt_star = 0.01105407273373585\np_star = 0.1271119\ncv_bar = 12.508324924443508\nalpha_bar = 30.02042848761392\nkappa_bar = 1.0\ncbar = 1.0\n',
         'call': 'lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)',
         'gold_call': '_oracle_lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)'},
        {'setup': 'import numpy as np\n# boundary: heat capacity just above the C_V = 3/2 positivity floor\nt_star = 0.02\np_star = 0.10\ncv_bar = 1.500001\nalpha_bar = 1.0\nkappa_bar = 1.0\ncbar = 1.0\n',
         'call': 'lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)',
         'gold_call': '_oracle_lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)'},
        {'setup': 'import numpy as np\n# edge: near the ideal-gas compressibility threshold kappa*c*T = 1\nt_star = 0.49\np_star = 0.80\ncv_bar = 3.0\nalpha_bar = 2.0\nkappa_bar = 2.0\ncbar = 1.0\n',
         'call': 'lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)',
         'gold_call': '_oracle_lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)'},
    ]
