"""
Obtain the stabilizing ring amplitude from the positive RPA subspace.

The positive-frequency invariant subspace fixes the stable amplitude. Its ratio is well defined when positive frequencies are degenerate; individual eigenvectors need not be unique.

Returns
-------
return omega, t
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stable_ring(a, b):
    """Obtain the stabilizing ring amplitude from the positive RPA subspace.

    Parameters
    ----------
    a, b : symmetric real arrays, shape (n,n)
        Both a-b and a+b must have minimum eigenvalue strictly above 1e-12.
    
    Returns
    -------
    omega : real array, shape (n,)
        Positive frequencies of [[a,b],[-b,-a]] in ascending order.
    t : real symmetric array, shape (n,n)
        The stable ratio Y@inv(X), where [X;Y] spans the positive invariant
        subspace. It satisfies b+a@t+t@a+t@b@t=0 and a+b@t has positive spectrum.
        Degenerate positive frequencies are allowed. Do not return the other branch.
    
    Raises
    ------
    ValueError
        If either a-b or a+b has minimum eigenvalue at most 1e-12.
    
    Inputs are finite real array-like objects. Do not mutate inputs. Import
    dependencies inside the function. NumPy and SciPy are available. Numeric
    tests use absolute and relative tolerances of 2e-8.
    """
    return omega, t

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_stable_ring(a, b):
    import numpy as np
    from scipy.linalg import eig
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(np.linalg.eigvalsh(a-b).min(), np.linalg.eigvalsh(a+b).min()) <= 1e-12:
        raise ValueError('The RPA quadratic form is not strictly positive.')
    n = len(a)
    rpa = np.block([[a, b], [-b, -a]])
    values, vectors = eig(rpa)
    idx = np.where(values.real > 0)[0]
    idx = idx[np.argsort(values[idx].real)]
    omega = values[idx].real
    x, y = vectors[:n, idx].real, vectors[n:, idx].real
    t = np.linalg.solve(x.T, y.T).T
    t = (t+t.T)/2
    return omega, t

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'stable_ring(a[0,0],b[0,0])',
      'gold_call': '_oracle_stable_ring(a[0,0],b[0,0])',
      'name': 'full_model',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n",
      'tol': 2e-08},
     {'call': 'stable_ring(np.eye(3),np.zeros((3,3)))',
      'gold_call': '_oracle_stable_ring(np.eye(3),np.zeros((3,3)))',
      'name': 'degenerate',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'stable_ring([[1.2,.13],[.13,.82]],[[.14,-.07],[-.07,.23]])',
      'gold_call': '_oracle_stable_ring([[1.2,.13],[.13,.82]],[[.14,-.07],[-.07,.23]])',
      'name': 'noncommuting',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'expect_value_error(stable_ring,[[1.]],[[1.2]])',
      'gold_call': 'expect_value_error(_oracle_stable_ring,[[1.]],[[1.2]])',
      'name': 'unstable',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'def expect_value_error(fn, *args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n",
      'tol': 2e-08},
     {'call': 'stable_ring([[.7]],[[.65]])',
      'gold_call': '_oracle_stable_ring([[.7]],[[.65]])',
      'name': 'wrong_branch_guard',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08}]
