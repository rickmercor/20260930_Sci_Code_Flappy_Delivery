"""
Evaluate the macroscopic equations of state of the mesoparticle fluid from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .. N, N = len(g)) at bulk density rho and reservoir temperature T. First recover the mean field nb = rho integral w g (rectangle rule) and the coefficients of the interaction step at that mean field. The pressure follows from the volume derivative of the configurational free energy: P = rho k_B T - (1 / (3 V)) < sum_i sum_{j != i} [W_n]_{nb_i} r_ij w'(r_ij) >, where [W_n]_{nb_i} is the coefficient evaluated at the instantaneous primitive density nb_i = sum_{k != i} w(r_ik) of particle i and w' is the kernel derivative. Expand [W_n]_{nb_i} to first order about the mean field with the coefficient [W_nn] of the interaction step, express the resulting averages of a homogeneous fluid through the pair distribution function and the triplet distribution function, and reduce the genuine triplet average with the Kirkwood superposition approximation g3(r_i, r_j, r_k) = g(r_ij) g(r_ik) g(r_jk). Work out the reduction yourself; the outcome is the ideal term plus three excess contributions of distinct origin, which are returned separately: the mean-field contribution, linear in [W_n]; the contribution in [W_nn] that involves only pair correlations; and the contribution in [W_nn] that involves triplet correlations, which reduces to a radial integral over a convolution of two radial functions and is to be evaluated with the discrete transform pair of the HNC step (product of the two forward transforms inverted back to the grid) followed by the rectangle-rule radial integral, every other radial integral also by the rectangle rule. The internal energy per particle is U/N = (3/2) k_B T + C_V T + V(n), the translational and internal kinetic parts plus the density part of the internal energy evaluated at the mean corrected density n = n(nb) (its fluctuation average vanishes to first order because the mean field is defined from the same g). Return [nb, n, P, mean-field contribution, pair-correlation fluctuation contribution, triplet-correlation fluctuation contribution, U/N]. Raise ValueError if g is not a one-dimensional table with at least eight nodes or if T, rho or dr is not positive.

Because the potential is many-body, the energetic route to the pressure carries three-body correlations even at the lowest order of the expansion; the source evaluates them with the Kirkwood superposition, which is adequate for a simple fluid at moderate density and without directional interactions, and finds that the fluctuation corrections grow as the cutoff shrinks. The convolution structure of the triplet term is what makes it computable on the grid at the cost of two forward and one inverse transform. Reduced units, k_B = 1.

Returns
-------
A (7,) float64 array [nb, n, P, mean-field pair term, fluctuation pair term, fluctuation triplet term, U/N] in reduced units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def eos_state(g, temperature, rho, rcut, fcut, params, dr):
    """Evaluate the macroscopic equations of state of the mesoparticle fluid from a pair
    distribution function g tabulated on the grid r_i = i dr (i = 1 . A (7,) float64 array
    [nb, n, P, mean-field pair term, fluctuation pair term, fluctuation triplet term, U/N]
    in reduced units."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _fourier_pair(n_grid, dr):
    r, k = _grid(n_grid, dr)
    S = np.sin(np.outer(k, r))
    dk = k[0]
    fwd = lambda f: 4.0*np.pi*dr/k*(S @ (r*f))
    inv = lambda F: dk/(2.0*np.pi**2*r)*(S @ (k*F))
    return r, k, fwd, inv

def _oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or g.size < 8:
        raise ValueError("g must be a one-dimensional table with at least eight nodes")
    if temperature <= 0 or rho <= 0 or dr <= 0:
        raise ValueError("temperature, density and spacing must be positive")
    n_grid = g.size
    r, k, fwd, inv = _fourier_pair(n_grid, dr)
    w, wp = _kernel(r, rcut)
    vol = 4.0*np.pi*dr*r*r
    nb = rho*float(np.sum(vol*w*g))
    coef = _oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)
    n, pi, vpot, wn, wnn = [float(x) for x in coef]
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    h = g - 1.0
    i1 = float(np.sum(vol*r*wp*g))
    i2 = float(np.sum(vol*r*w*wp*g))
    conv = inv(fwd(w*g)*fwd(h))
    i3 = float(np.sum(vol*r*wp*g*conv))
    t1 = -wn*rho**2*i1/3.0
    t2 = -wnn*rho**2*i2/3.0
    t3 = -wnn*rho**3*i3/3.0
    p = rho*temperature + t1 + t2 + t3
    u = 1.5*temperature + vpot + cv*temperature
    return np.array([nb, n, p, t1, t2, t3, u])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: liquid-like correlation hole and first peak on a resolved grid\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nrho = 1.0\nrcut = 2.1564\nfcut = 1.33\ndr = 0.05\nr = dr * np.arange(1, 65)\ng = 1.0 - 0.8 * np.exp(-(r / 0.7)**2) + 0.12 * np.exp(-((r - 1.2) / 0.25)**2)\n',
         'call': 'eos_state(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted g table length of eight nodes\nparams = np.array([0.08276268387913284, 1.0, 0.4999046161208672, 1.0902303934047828, 9.77794449766948, 7.063391679042095])\ntemperature = 0.08276268387913284\nrho = 0.5\nrcut = 1.0\nfcut = 1.0\ndr = 0.15\ng = np.linspace(0.7, 1.05, 8)\n',
         'call': 'eos_state(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr)'},
        {'setup': 'import numpy as np\n# edge: structureless g(r)=1, which removes the h=g-1 correlation contribution\nparams = np.array([0.08276268387913284, 1.0, 0.4999046161208672, 1.0902303934047828, 9.77794449766948, 7.063391679042095])\ntemperature = 0.08276268387913284\nrho = 0.6\nrcut = 1.2\nfcut = 1.0\ndr = 0.1\ng = np.ones(16)\n',
         'call': 'eos_state(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr)'},
    ]
