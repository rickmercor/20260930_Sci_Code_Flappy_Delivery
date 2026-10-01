"""
Predict the pair distribution function of the mesoparticle fluid of bulk density rho at reservoir temperature T without simulation, on the grid r_i = i dr, i = 1 .. n_grid, by closing the mean-field expansion self-consistently with the HNC structure. The mean field around a particle is the average primitive density it sees, nb = rho integral d^3r w(r) g(r), evaluated on the grid by the rectangle rule 4 pi dr sum_i r_i^2 w(r_i) g(r_i) (the same rule is used for every radial integral of this task). The pair potential that enters the HNC equation is the pairwise interaction of the expanded configurational energy sum_i W(T, n_i): with the expansion of the previous step every pair (i, j) appears once in the sum over i and once in the sum over j, and the pair potential is the coefficient of w(r_ij) in the total energy of the pair, which you must work out; it is evaluated with [W_n] at the current mean field and divided by T to give beta u on the grid. Start from g = 1 (so nb = rho times the kernel integral on the grid), alternate the HNC solution of the previous step and the update of nb until two successive values of nb agree to 1e-13, and return the HNC g of the converged mean field. Raise ValueError if T, rho, n_grid or dr is not positive, or if the grid does not extend beyond R_cut.

The expansion of the potential in density fluctuations is made around a reference pair distribution function that is in principle arbitrary; the source closes the problem by identifying it with the HNC solution itself, so the mean field, the interaction coefficients and the structure are determined together. For liquid argon at the reference state the correlation hole is deep at the smallest cutoff (g is essentially zero below r* = 0.5 for R_cut* = 1.3365) and shallow at the largest (g(0) of order 0.1 for R_cut* = 2.1564), and the first peak of g moves out with the cutoff, from about r* = 1.2 to 1.9.

Returns
-------
A (n_grid,) float64 array g(r_i) of the self-consistent structure.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr):
    """Predict the pair distribution function of the mesoparticle fluid of bulk density rho at
    reservoir temperature T without simulation, on the grid r_i = i dr, i = 1 .. n_grid, by
    closing the mean-field expansion self-consistently with the HNC structure. A (n_grid,)
    float64 array g(r_i) of the self-consistent structure."""
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

def _oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr):
    if temperature <= 0 or rho <= 0 or n_grid < 8 or dr <= 0:
        raise ValueError("temperature, density, grid size and spacing must be positive")
    if n_grid*dr < rcut:
        raise ValueError("the grid must extend beyond the cutoff")
    r, k = _grid(n_grid, dr)
    w, wp = _kernel(r, rcut)
    vol = 4.0*np.pi*dr*r*r
    nb = rho*float(np.sum(vol*w))                        # g = 1 start
    g = np.ones(n_grid)
    for outer in range(500):
        coef = _oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)
        wn = float(coef[3])
        betau = 2.0*wn*w/temperature
        g = _oracle_hnc_structure(betau, rho, dr)
        nb_new = rho*float(np.sum(vol*w*g))
        done = abs(nb_new - nb) < 1.0e-13
        nb = nb_new
        if done:
            break
    else:
        raise RuntimeError("mean-field density did not converge")
    coef = _oracle_interaction_coefficients(nb, temperature, rcut, fcut, params)
    betau = 2.0*float(coef[3])*w/temperature
    return _oracle_hnc_structure(betau, rho, dr)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid state\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nrho = 1.0\nrcut = 2.1564\nfcut = 1.33\nn_grid = 100\ndr = 0.025\n',
         'call': 'self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)',
         'gold_call': '_oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted HNC grid length n_grid = 8, with the grid just beyond the cutoff\nparams = np.array([0.1, 1.0, 0.0, 1000.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 0.5\nfcut = 1.0\nn_grid = 8\ndr = 0.06666666666666667\n',
         'call': 'self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)',
         'gold_call': '_oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)'},
        {'setup': 'import numpy as np\n# edge: dilute, strongly structured valid state with a large volume-rescaling factor\nparams = np.array([0.1, 1.0, 0.0, 10.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 1.0\nfcut = 2.0\nn_grid = 24\ndr = 0.05\n',
         'call': 'self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)',
         'gold_call': '_oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)'},
    ]
