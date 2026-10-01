"""
Return the ordinary mixed logarithmic-coupling entropy response.

The final response connects the transition model, cavity ring amplitudes, photon reduction and entropy series. Taylor coefficients are converted to ordinary derivatives once, at the end.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_cavity(gaps, factors, dipoles, frequencies, couplings, orders, renyi=2.0):
    """Return the ordinary mixed logarithmic-coupling entropy response.

    Parameters
    ----------
    gaps, factors : real arrays, shapes (n,), (n,r)
        Positive gaps, 1<=n,r<=8. D=diag(gaps), J=factors@factors.T,
        Ae=D+J, Be=J. No orbital relaxation or extra spin factor.
    dipoles, frequencies, couplings : real arrays, shapes (n,m), (m,), (m,)
        1<=m<=3, frequencies positive. Mode k uses ell_k=couplings[k]*exp(x_k).
        Construct the A and B matrices specified in cavity_jets. Electronic
        coordinates precede photons. Energies are hartree; dipoles and factors
        have sqrt(hartree) units; x_k and couplings are dimensionless.
    orders : integer sequence, length m
        Each entry is 0..3. One logarithmic parameter per photon mode.
    renyi : float, default 2.0
        Exactly 2.0 or 1.5, with the entropy_jets domain. Signed and zero base
        couplings are allowed for 2.0; 1.5 requires a strictly mixed photon base.
    
    Returns
    -------
    response : float
        Ordinary derivative d**sum(orders) S_renyi / product_k dx_k**orders[k]
        at all x_k=0. Multiply the selected Taylor coefficient by
        product_k factorial(orders[k]) exactly once. All-zero orders return
        the base entropy. Call the earlier sub-problem chain.
    
    Raises
    ------
    ValueError
        Propagate the stability condition from stable_ring and the covariance
        and Renyi-domain conditions from entropy_jets.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return response

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_cavity(gaps, factors, dipoles, frequencies, couplings, orders, renyi=2.0):
    from math import factorial, prod
    ea, eb = _oracle_transition_blocks(gaps,factors)
    a, b = _oracle_cavity_jets(ea,eb,dipoles,frequencies,couplings,orders)
    zero = (0,)*len(orders)
    omega, t0 = _oracle_stable_ring(a[zero],b[zero])
    t = _oracle_ring_jets(a,b,t0)
    covariance = _oracle_photon_covariance_jets(t,len(gaps))
    entropy = _oracle_entropy_jets(covariance,renyi)
    index = tuple(map(int,orders))
    return float(prod(factorial(k) for k in index)*entropy[index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'name': 'target',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'name': 'base_entropy',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "model['orders']=[0,0]\n",
      'tol': 2e-08},
     {'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'name': 'one_mode_off',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "model['couplings'][0]=0.\n",
      'tol': 2e-08},
     {'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'name': 'different_order',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "model['orders']=[1,2]\n"
               "model['couplings']=[-.2,.7]\n",
      'tol': 2e-08},
     {'name': 'three_mode_fractional_benchmark',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n",
      'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'tol': 2e-08},
     {'name': 'fractional_mixed_order',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n"
               '\n'
               "model['orders']=[1,2,1]",
      'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'tol': 2e-08},
     {'name': 'three_modes_renyi2',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n"
               '\n'
               "model['renyi']=2.0",
      'call': 'solve_cavity(**model)',
      'gold_call': '_oracle_solve_cavity(**model)',
      'tol': 2e-08}]
