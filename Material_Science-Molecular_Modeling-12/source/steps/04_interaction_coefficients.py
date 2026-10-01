"""
Evaluate the mean-field interaction coefficients of the LTh model at the reservoir temperature T for one or more values of the mean primitive density nb; params is the parameter vector [theta0, n00, pi00, kappa_T, alpha, C_V] of the previous steps. Integrating the particle entropy out of the canonical partition function leaves a configurational weight exp(-sum_i W(T, n_i) / k_B T) with the effective one-particle potential W(T, n) = V(n) + k_B T ln psi(n), where V(n) is the density-only part of the internal energy and ln psi(n) = -[C_V + alpha / (n kappa_T)] / k_B is the density-dependent part of the dressed entropy; the density derivative of W is the particle pressure at the reservoir temperature divided by n^2, dW/dn = pi(T, n) / n^2. Obtain V(n) from the model: the particle Helmholtz free energy f(theta, n) follows by integrating pi = n^2 df/dn at fixed theta from the pressure equation of state of the previous step, up to a function of theta alone that only feeds the temperature-dependent part C_V theta of u = f - theta df/dtheta; V(n) is the theta-independent remainder of u. Expanding W(T, n_i(nb_i)) around the mean field nb to first order in the deviation of the primitive density gives a one-body term plus the pair term [W_n] sum_{j != i} w(r_ij), with [W_n] = dW/dnb evaluated at nb through the chain rule with zeta = dn/dnb of the previous step, and the next coefficient [W_nn] = d[W_n]/dnb is needed for the fluctuation corrections. Return, for each nb, the corrected density n, the particle pressure pi(T, n), V(n), [W_n] and [W_nn]. Raise ValueError if T, kappa_T or n00 is not positive.

The temperature dependence of the potential is what makes non-isothermal GenDPDE consistent: W depends on the reservoir temperature through the entropic factor psi, and the force between two mesoparticles at the mean field is minus the derivative of the pair term, so it is proportional to pi zeta / n^2 times the kernel derivative. All quantities are in reduced units with k_B = 1. For liquid argon at the reference state and R_cut* = 2.1564, f_cut = 1.33 the coefficients are of order 0.1 to 0.7.

Returns
-------
An array of shape nb.shape + (5,) holding [n, pi, V(n), W_n, W_nn] for each nb; a scalar nb gives shape (5,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def interaction_coefficients(nb, temperature, rcut, fcut, params):
    """Evaluate the mean-field interaction coefficients of the LTh model at the reservoir
    temperature T for one or more values of the mean primitive density nb; params is the
    parameter vector [theta0, n00, pi00, kappa_T, alpha, C_V] of the previous steps. An
    array of shape nb.shape + (5,) holding [n, pi, V(n), W_n, W_nn] for each nb; a scalar nb
    gives shape (5,)."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interaction_coefficients(nb, temperature, rcut, fcut, params):
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    if temperature <= 0 or kappa <= 0 or n00 <= 0:
        raise ValueError("temperature, compressibility and reference density must be positive")
    pv = _oracle_particle_volume(nb, rcut, fcut)
    n, zeta, zeta_n = pv[..., 0], pv[..., 1], pv[..., 2]
    pi = pi00 + alpha/kappa*(temperature - theta0) + np.log(n/n00)/kappa
    vpot = -pi00/n + alpha*theta0/(n*kappa) - (np.log(n/n00) + 1.0)/(n*kappa)
    wn = pi*zeta/n**2
    wnn = zeta**2/(kappa*n**3) + pi*zeta_n/n**2 - 2.0*pi*zeta**2/n**3
    return np.stack([n, pi, vpot, wn, wnn], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid state and scalar mean density\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nnb = 0.8583615\nrcut = 2.1564\nfcut = 1.33\n',
         'call': 'interaction_coefficients(nb, temperature, rcut, fcut, params)',
         'gold_call': '_oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted fcut = 1\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nnb = 1.0\nrcut = 1.6839\nfcut = 1.0\n',
         'call': 'interaction_coefficients(nb, temperature, rcut, fcut, params)',
         'gold_call': '_oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)'},
        {'setup': 'import numpy as np\n# edge: vectorized densities at the supercritical reference state\nparams = np.array([0.08276268387913284, 1.0, 0.4999046161208672, 1.0902303934047828, 9.77794449766948, 7.063391679042095])\ntemperature = 0.08276268387913284\nnb = np.array([0.30, 0.80, 1.40])\nrcut = 1.6839\nfcut = 1.35\n',
         'call': 'interaction_coefficients(nb, temperature, rcut, fcut, params)',
         'gold_call': '_oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)'},
    ]
