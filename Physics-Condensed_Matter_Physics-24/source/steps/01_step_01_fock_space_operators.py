"""
Construct the fermionic creation and annihilation matrices for a finite Fock space. The input fixes the number of modes and the returned arrays hold the real matrix elements of every creation operator together with its corresponding annihilation operator. Invalid input raises ValueError: n_modes must be an integer between 1 and 16.

The occupation-number basis for n_modes fermionic modes has dimension 2**n_modes, and bit i of each integer basis label records whether mode i is occupied. Fermions are not free to be represented by bare occupation flips, because exchanging two of them must cost a sign. The Jordan-Wigner construction supplies that sign: filling an empty mode i multiplies the matrix element by (-1)**nu, where nu counts the occupied modes with indices below i. That string of signs is what makes the resulting matrices obey canonical anticommutation, and every energy and expectation value computed later inherits its correctness from this step.

Returns
-------
tuple of numpy.ndarray, the real creation and annihilation operator matrices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fock_space_operators(n_modes):
    """Build Jordan-Wigner operators in the occupation-number basis.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes, from 1 through 16.

    Returns
    -------
    c_dag : numpy.ndarray
        Real creation-operator array with shape
        ``(n_modes, 2**n_modes, 2**n_modes)``.
    c : numpy.ndarray
        Real annihilation-operator array with shape
        ``(n_modes, 2**n_modes, 2**n_modes)``.
    
    Raises
    ------
    ValueError
        If ``n_modes`` is not an integer, or lies outside 1 through 16.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fock_space_operators(n_modes):
    if isinstance(n_modes, (bool, np.bool_)) or not isinstance(n_modes, (int, np.integer)):
        raise ValueError("n_modes must be an integer")
    if n_modes < 1 or n_modes > 16:
        raise ValueError("n_modes must be between 1 and 16")
    dim = 2 ** n_modes
    c_dag = np.zeros((n_modes, dim, dim))
    c = np.zeros((n_modes, dim, dim))
    for i in range(n_modes):
        for s in range(dim):
            if not (s >> i) & 1:
                new_s = s | (1 << i)
                sign = 0
                for j in range(i):
                    if (s >> j) & 1:
                        sign += 1
                phase = (-1) ** sign
                c_dag[i, new_s, s] = phase
                c[i, s, new_s] = phase
    return c_dag, c

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n',
      'call': '(lambda ops: float(np.max(np.abs(ops[1][1] @ ops[0][1] + ops[0][1] @ ops[1][1] - '
              'np.eye(8)))))(fock_space_operators(3))',
      'gold_call': '(lambda ops: float(np.max(np.abs(ops[1][1] @ ops[0][1] + ops[0][1] @ ops[1][1] - '
                   'np.eye(8)))))(_oracle_fock_space_operators(3))'},
     {'setup': 'import numpy as np\n',
      'call': '(lambda ops: np.diag(ops[0][0] @ ops[1][0]).tolist())(fock_space_operators(1))',
      'gold_call': '(lambda ops: np.diag(ops[0][0] @ ops[1][0]).tolist())(_oracle_fock_space_operators(1))'},
     {'setup': '',
      'call': 'float(fock_space_operators(3)[0][2, 5, 1])',
      'gold_call': 'float(_oracle_fock_space_operators(3)[0][2, 5, 1])'},
     {'setup': '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(fock_space_operators, 0)',
      'gold_call': '_exception_code(_oracle_fock_space_operators, 0)'}]
