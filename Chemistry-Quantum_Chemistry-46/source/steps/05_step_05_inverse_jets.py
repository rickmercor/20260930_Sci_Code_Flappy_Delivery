"""
Invert an ordered multivariate matrix Taylor series.

Matrix inversion preserves the order of products. A nonsymmetric base and noncommuting perturbations cannot be replaced by scalar reciprocal rules

Returns
-------
return inverse
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def inverse_jets(matrix):
    """Invert an ordered multivariate matrix Taylor series.

    Parameters
    ----------
    matrix : real array, shape (*grid,n,n)
        Its base matrix is invertible; symmetry is not required.
    
    Returns
    -------
    inverse : real array, same shape
        The ordered convolution matrix@inverse is I at the all-zero multi-index
        and zero at every other retained coefficient. A singular base lies outside
        this function's input domain. Integer-valued input arrays are valid.
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
    return inverse

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_inverse_jets(matrix):
    import numpy as np
    matrix = np.asarray(matrix, float)
    shape = matrix.shape[:-2]
    zero = (0,)*len(shape)
    out = np.zeros_like(matrix)
    out[zero] = np.linalg.inv(matrix[zero])
    for alpha in sorted(np.ndindex(shape), key=sum)[1:]:
        residual = np.zeros_like(matrix[zero])
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            if beta == zero:
                continue
            remainder = tuple(x-y for x,y in zip(alpha,beta))
            residual += matrix[beta]@out[remainder]
        out[alpha] = -out[zero]@residual
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'inverse_jets(m)',
      'gold_call': '_oracle_inverse_jets(m)',
      'name': 'noncommuting',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'm=np.zeros((4,4,2,2));m[0,0]=[[2,.3],[-.2,1]];m[1,0]=[[.2,-.4],[.1,.3]];m[0,1]=[[.1,.6],[-.2,.2]];m[1,1]=[[.03,.04],[.02,-.01]]\n',
      'tol': 2e-08},
     {'call': 'inverse_jets(m)',
      'gold_call': '_oracle_inverse_jets(m)',
      'name': 'integer_series',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'm=np.zeros((3,2,2,2),dtype=int);m[0,0]=[[2,1],[0,3]];m[1,0]=[[0,1],[1,0]];m[0,1]=[[1,0],[0,-1]]\n',
      'tol': 2e-08},
     {'call': 'inverse_jets(np.array([[[[2.,.5],[.5,1.]]]]))',
      'gold_call': '_oracle_inverse_jets(np.array([[[[2.,.5],[.5,1.]]]]))',
      'name': 'constant',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'inverse_jets(m)',
      'gold_call': '_oracle_inverse_jets(m)',
      'name': 'one_axis',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'm=np.zeros((1,4,1,1));m[0,0,0,0]=2;m[0,1,0,0]=-.5;m[0,2,0,0]=.12\n',
      'tol': 2e-08},
     {'name': 'three_noncommuting_axes',
      'setup': 'import numpy as np\n'
               'm=np.zeros((3,2,4,2,2));m[0,0,0]=[[1.2,.3],[-.2,.8]]\n'
               'm[1,0,0]=[[.2,-.1],[.3,.05]];m[0,1,0]=[[.1,.2],[-.15,.07]]\n'
               'm[0,0,1]=[[-.1,.13],[.17,.2]];m[1,1,1]=[[.03,-.02],[.01,.05]]\n',
      'call': 'inverse_jets(m)',
      'gold_call': '_oracle_inverse_jets(m)',
      'tol': 2e-08}]
