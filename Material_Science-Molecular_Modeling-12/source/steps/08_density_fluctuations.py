"""
Quantify the density fluctuations that the first-order expansion of the previous steps neglects, from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .. N) at bulk density rho and reservoir temperature T. Recover the mean field nb = rho integral w g (rectangle rule) and the corrected density n, zeta and dzeta/dnb at that mean field. The variance of the primitive density of a particle about the mean field, <(nb_i - nb)^2> with nb_i = sum_{k != i} w(r_ik), is an average over the same homogeneous ensemble: express it through the pair and triplet distribution functions, separate the self term of the double sum from the genuine triplet average, reduce the latter with the Kirkwood superposition approximation, and evaluate the convolution that results with the discrete transform pair of the HNC step (product of the two forward transforms inverted back to the grid) and every radial integral with the rectangle rule. Then evaluate the next term of the expansion of the density part of the internal energy that the energy equation of state truncates: expanding V(n(nb_i)) to second order in nb_i - nb about the mean field and averaging gives V(n) plus one half of the second derivative [V_nn] = d^2 V / dnb^2 at the mean field (chain rule through n(nb) with zeta and dzeta/dnb, V(n) being the density part of the internal energy of the interaction step) times the variance. Return [variance, relative fluctuation sqrt(variance) / nb, [V_nn], second-order energy correction (1/2) [V_nn] variance, and the internal energy per particle U/N of the energy equation of state plus that correction]. Raise ValueError if g is not a one-dimensional table with at least eight nodes or if T, rho or dr is not positive.

The expansion of the many-body potential about the mean field is controlled by the size of the density fluctuations a particle sees, which the source estimates with Poisson statistics as a relative variance proportional to 1 / (c R_cut^3): the forces become harsher as the kernel shrinks and vanish when it spans the whole volume. With the structure in hand the variance can be evaluated exactly at the superposition level, and its size decides how far the first-order energy equation of state and the second-order pressure can be trusted at each kernel range. For liquid argon the relative fluctuation is of order 0.1 at the largest kernel range and 0.4 at the smallest, and the neglected energy term is a few percent of the internal energy.

Returns
-------
A (5,) float64 array [variance of the primitive density, relative fluctuation, [V_nn], second-order energy correction, corrected U/N] in reduced units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def density_fluctuations(g, temperature, rho, rcut, fcut, params, dr):
    """Quantify the density fluctuations that the first-order expansion of the previous steps
    neglects, from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .
    A (5,) float64 array [variance of the primitive density, relative fluctuation, [V_nn],
    second-order energy correction, corrected U/N] in reduced units."""
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

def _oracle_density_fluctuations(g, temperature, rho, rcut, fcut, params, dr):
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
    pv = _oracle_particle_volume(nb, rcut, fcut)
    n, zeta, zeta_n = [float(x) for x in pv]
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    h = g - 1.0
    conv = inv(fwd(w*g)*fwd(h))
    var = rho*float(np.sum(vol*w*w*g)) + rho**2*float(np.sum(vol*w*g*conv))
    rel = np.sqrt(var)/nb
    pi_zero = pi00 - alpha*theta0/kappa + np.log(n/n00)/kappa          # particle pressure at zero temperature
    v1 = pi_zero/n**2
    v2 = 1.0/(kappa*n**3) - 2.0*pi_zero/n**3
    vnn = v2*zeta**2 + v1*zeta_n
    du2 = 0.5*vnn*var
    vpot = -pi00/n + alpha*theta0/(n*kappa) - (np.log(n/n00) + 1.0)/(n*kappa)
    u1 = 1.5*temperature + vpot + cv*temperature
    return np.array([var, rel, vnn, du2, u1 + du2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: liquid-like correlated structure\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nrho = 1.0\nrcut = 2.1564\nfcut = 1.33\ndr = 0.05\nr = dr * np.arange(1, 65)\ng = 1.0 - 0.8 * np.exp(-(r / 0.7)**2) + 0.12 * np.exp(-((r - 1.2) / 0.25)**2)\n',
         'call': 'density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted g table length of eight nodes\nparams = np.array([0.08276268387913284, 1.0, 0.4999046161208672, 1.0902303934047828, 9.77794449766948, 7.063391679042095])\ntemperature = 0.08276268387913284\nrho = 0.5\nrcut = 1.0\nfcut = 1.0\ndr = 0.15\ng = np.linspace(0.7, 1.05, 8)\n',
         'call': 'density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)'},
        {'setup': 'import numpy as np\n# edge: structureless g(r)=1, where the correlation part of the convolution vanishes\nparams = np.array([0.08276268387913284, 1.0, 0.4999046161208672, 1.0902303934047828, 9.77794449766948, 7.063391679042095])\ntemperature = 0.08276268387913284\nrho = 0.6\nrcut = 1.2\nfcut = 1.0\ndr = 0.1\ng = np.ones(16)\n',
         'call': 'density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)',
         'gold_call': '_oracle_density_fluctuations(g, temperature, rho, rcut, fcut, params, dr)'},
    ]
