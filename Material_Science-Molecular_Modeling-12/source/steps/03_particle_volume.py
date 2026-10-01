"""
Evaluate the corrected mesoparticle density of the source's partition-of-unity volume construction, for one or more values of the primitive local density nb. The primitive density of particle i is nb_i = sum over j != i of w(r_ij) with the normalised quadratic kernel w(r) = 15 / (2 pi R_cut^3) (1 - r / R_cut)^2 for r < R_cut and zero beyond (its volume integral is one). The particle volume is its share of space under a mean-field partition of unity in which every other particle contributes a uniform background equal to nb_i: V_i(nb_i) = 4 pi integral from 0 to R_tilde of r^2 w_tilde(r) / (w_tilde(r) + nb_i) dr, where w_tilde is the same normalised quadratic kernel but of range R_tilde = R_cut / f_cut, f_cut >= 1 being a scaling factor tuned so that the corrected density reproduces the bulk density in simulations. The corrected density is n_i = 1 / V_i. Return, for each nb, n, zeta = dn / dnb and zeta_n = d zeta / dnb; evaluate the integral and both derivatives exactly (in closed form, or by differentiating under the integral sign and integrating to at least 1e-12), not by finite differences. Raise ValueError if nb is not positive, if R_cut is not positive or if f_cut is below one.

Estimating a particle volume directly as the inverse of the kernel-smoothed density is what produces the pairing instability of density-dependent potentials in dense liquids; the partition of unity phi_i(r) = w(|r - r_i|) / sum_j w(|r - r_j|) instead assigns every point of space to the particles around it, so that the volumes add up to the total volume. With the uniform-background approximation the integral is elementary in the variable k = (2 pi / 15) R_tilde^3 nb and involves arctan(1 / sqrt k) and ln((k + 1) / k). The corrected density enters the equations of motion only through n_i(nb_i) and its derivative zeta_i, which multiplies the pairwise forces, so both must be exact.

Returns
-------
An array of shape nb.shape + (3,) holding [n, zeta, zeta_n] for each nb; a scalar nb gives shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def particle_volume(nb, rcut, fcut):
    """Evaluate the corrected mesoparticle density of the source's partition-of-unity volume
    construction, for one or more values of the primitive local density nb. An array of
    shape nb.shape + (3,) holding [n, zeta, zeta_n] for each nb; a scalar nb gives shape
    (3,)."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_particle_volume(nb, rcut, fcut):
    nb = np.asarray(nb, dtype=float)
    if rcut <= 0 or fcut < 1.0:
        raise ValueError("rcut must be positive and fcut at least one")
    if np.any(nb <= 0):
        raise ValueError("primitive density must be positive")
    rt3 = (rcut/fcut)**3
    kp = 2.0*np.pi/15.0*rt3                       # dk/dnb
    k = kp*nb
    s = np.sqrt(k)
    at = np.arctan(1.0/s)
    lg = np.log((1.0 + k)/k)
    vol = 4.0*np.pi*rt3*(1.0/3.0 - s*(1.0 - k)*at - k*(1.0 - lg))
    v1 = -4.0*np.pi*rt3*(1.5 + (1.0 - 3.0*k)/(2.0*s)*at - lg)
    v2 = 4.0*np.pi*rt3*((1.0 + 3.0*k)/(4.0*k*s)*at - 3.0/(4.0*k))
    n = 1.0/vol
    zeta = -v1*kp/vol**2
    zeta_n = (2.0*v1*v1/vol**3 - v2/vol**2)*kp*kp
    return np.stack([n, zeta, zeta_n], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: scalar mean density at the largest benchmark cutoff\nnb = 0.8583615\nrcut = 2.1564\nfcut = 1.33\n',
         'call': 'particle_volume(nb, rcut, fcut)',
         'gold_call': '_oracle_particle_volume(nb, rcut, fcut)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted fcut = 1\nnb = 1.0\nrcut = 1.6839\nfcut = 1.0\n',
         'call': 'particle_volume(nb, rcut, fcut)',
         'gold_call': '_oracle_particle_volume(nb, rcut, fcut)'},
        {'setup': 'import numpy as np\n# edge: vectorized primitive densities spanning dilute to dense values\nnb = np.array([0.05, 0.35, 1.0, 2.0])\nrcut = 1.3365\nfcut = 1.41\n',
         'call': 'particle_volume(nb, rcut, fcut)',
         'gold_call': '_oracle_particle_volume(nb, rcut, fcut)'},
    ]
