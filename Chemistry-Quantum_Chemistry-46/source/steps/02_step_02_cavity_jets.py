"""
Expand the cavity RPA matrices in independently varied logarithmic couplings.

Each cavity mode contributes dipole self-energy and both rotating and counter-rotating couplings. Independent logarithmic parameters generate pure-axis matrix coefficients and mixed amplitude responses.

Returns
-------
return a, b
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_jets(electronic_a, electronic_b, dipoles, frequencies, couplings, orders):
    """Expand the cavity RPA matrices in independently varied logarithmic couplings.

    Parameters
    ----------
    electronic_a, electronic_b : real symmetric arrays, shape (n,n)
        electronic_a-electronic_b is positive definite.
    dipoles : real array, shape (n,m)
    frequencies : positive real array, shape (m,)
    couplings : real array, shape (m,)
        Signed or zero base couplings are allowed; 1 <= m <= 3.
    orders : integer sequence, length m
        Each order lies between 0 and 3. Parameter x_k changes only mode k:
        ell_k=couplings[k]*exp(x_k).
    
    Returns
    -------
    a, b : real arrays, shape (*(orders+1),n+m,n+m)
        Taylor coefficients of A=[[electronic_a+Delta,g],[g.T,diag(frequencies)]]
        and B=[[electronic_b+Delta,g],[g.T,zeros((m,m))]], where
        Delta=sum_k ell_k**2*outer(dipoles[:,k],dipoles[:,k]) and
        g[:,k]=-sqrt(frequencies[k]/2)*ell_k*dipoles[:,k].
        Electronic coordinates precede photons. Coefficients involving positive
        powers of more than one parameter vanish in a and b.
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
    return a, b

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cavity_jets(electronic_a, electronic_b, dipoles, frequencies, couplings, orders):
    import numpy as np
    from math import factorial
    ea, eb = np.asarray(electronic_a, float), np.asarray(electronic_b, float)
    d, w, lam = np.asarray(dipoles, float), np.asarray(frequencies, float), np.asarray(couplings, float)
    orders = tuple(map(int, orders))
    ne, nm = d.shape
    zero = (0,)*len(orders)
    a = np.zeros(tuple(k+1 for k in orders)+(ne+nm, ne+nm))
    b = np.zeros_like(a)
    a[zero][:ne,:ne], b[zero][:ne,:ne] = ea, eb
    a[zero][ne:,ne:] = np.diag(w)
    for mode in range(nm):
        for degree in range(orders[mode]+1):
            index = tuple(degree if axis == mode else 0 for axis in range(nm))
            dse = lam[mode]**2*2.0**degree/factorial(degree)*np.outer(d[:,mode],d[:,mode])
            g = -np.sqrt(w[mode]/2)*lam[mode]/factorial(degree)*d[:,mode]
            for tensor in (a,b):
                tensor[index][:ne,:ne] += dse
                tensor[index][:ne,ne+mode] += g
                tensor[index][ne+mode,:ne] += g
    return a, b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': "cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[3,3])",
      'gold_call': "_oracle_cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[3,3])",
      'name': 'mixed_rectangle',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n",
      'tol': 2e-08},
     {'call': "cavity_jets(ea,eb,model['dipoles'],model['frequencies'],[0,-.4],[2,1])",
      'gold_call': "_oracle_cavity_jets(ea,eb,model['dipoles'],model['frequencies'],[0,-.4],[2,1])",
      'name': 'one_decoupled',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n",
      'tol': 2e-08},
     {'call': "cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[0,0])",
      'gold_call': "_oracle_cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[0,0])",
      'name': 'base_only',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n",
      'tol': 2e-08},
     {'call': 'cavity_jets([[2]],[[1]],[[.6,-.2]],[.5,1.1],[-.8,.3],[0,3])',
      'gold_call': '_oracle_cavity_jets([[2]],[[1]],[[.6,-.2]],[.5,1.1],[-.8,.3],[0,3])',
      'name': 'asymmetric_orders',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'name': 'three_axis_rectangle',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n"
               '\n'
               'import numpy as np\n'
               "ea,eb=transition_blocks(model['gaps'],model['factors'])",
      'call': "cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[3,1,2])",
      'gold_call': "_oracle_cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],[3,1,2])",
      'tol': 2e-08},
     {'name': 'one_photon_axis',
      'setup': 'import numpy as np',
      'call': 'cavity_jets([[1.1]],[[.13]],[[.6]],[.57],[-.4],[3])',
      'gold_call': '_oracle_cavity_jets([[1.1]],[[.13]],[[.6]],[.57],[-.4],[3])',
      'tol': 2e-08}]
