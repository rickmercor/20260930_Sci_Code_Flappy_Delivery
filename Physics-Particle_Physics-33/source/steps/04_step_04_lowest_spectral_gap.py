"""
Lowest spectral gap of a finite real square matrix.

Return the difference between the second-lowest and the lowest eigenvalue of the supplied finite real square matrix, computed from a dense Hermitian diagonalization of the symmetrized array (H + H.T)/2. Degenerate lowest or second-lowest eigenvalues are allowed: sort the full spectrum in ascending order and subtract the first entry from the second. The matrix is square with at least two rows. A one-by-one matrix is outside the domain. Return the gap as a native Python float. Accept NumPy arrays and nested lists or tuples. The matrix array representation must have an integer or floating dtype; boolean, complex, object, datetime and timedelta dtypes are outside the domain. Interpret matrix entries as float64 values; their conversions must be finite. Paper provenance: Section IV.A.1 and IV.A.3 define the finite T1 and T3 Hamiltonians. The gap definition above specifies the spectral comparison used here. Include every import the implementation needs inside the function body.

Returns
-------
gap : float Second-lowest eigenvalue minus the lowest eigenvalue.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lowest_spectral_gap(hamiltonian: object) -> float:
    """Lowest spectral gap of a real square matrix after symmetrization.

    Parameters
    ----------
    hamiltonian : array-like of shape (n, n)
        Finite real square matrix with n at least 2. It is symmetrized before
        diagonalization.

    Returns
    -------
    gap : float
        Second-lowest eigenvalue minus the lowest eigenvalue.

    Raises
    ------
    ValueError
        If hamiltonian is not a finite real square matrix with at least two rows,
        or a finite spectral gap cannot be computed.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_lowest_spectral_gap(hamiltonian: object) -> float:
    import numpy as np

    try:
        raw = np.asarray(hamiltonian)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 2:
        raise ValueError('hamiltonian must be a square matrix with at least two rows')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        same_sign = np.signbit(H) == np.signbit(H.T)
        symmetric = np.where(same_sign, H + 0.5 * (H.T - H),
                             0.5 * (H + H.T))
    if not np.all(np.isfinite(symmetric)):
        raise ValueError('symmetrized hamiltonian entries must be finite')
    try:
        ev = np.sort(np.linalg.eigvalsh(symmetric))
    except np.linalg.LinAlgError as exc:
        raise ValueError('hamiltonian could not be diagonalized') from exc
    with np.errstate(over='ignore', invalid='ignore'):
        gap = ev[1] - ev[0]
    if not np.isfinite(gap):
        raise ValueError('hamiltonian must produce a finite spectral gap')
    return float(gap)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\nimport numpy as np\nM = np.diag([4.0, 1.0, 7.0, 2.5])\n',
  'call': 'lowest_spectral_gap(M)',
  'gold_call': '_oracle_lowest_spectral_gap(M)'},
 {'setup': '# case: normal\nimport numpy as np\nD = np.diag([1.0, 5.0, 2.0, 9.0])\n',
  'call': 'lowest_spectral_gap(D)',
  'gold_call': '_oracle_lowest_spectral_gap(D)'},
 {'setup': '# case: edge\nimport numpy as np\nG = np.diag([2.0, 2.0, 5.0])\n',
  'call': 'lowest_spectral_gap(G)',
  'gold_call': '_oracle_lowest_spectral_gap(G)'},
 {'setup': '# case: boundary\nimport numpy as np\nP = np.array([[2.0, 1.0], [1.0, 2.0]])\n',
  'call': 'lowest_spectral_gap(P)',
  'gold_call': '_oracle_lowest_spectral_gap(P)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'A = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 5.0]])\n',
  'call': 'lowest_spectral_gap(A)',
  'gold_call': '_oracle_lowest_spectral_gap(A)'},
 {'setup': 'H = _oracle_assemble_t1_hamiltonian(4, 3, 1.0, 1.0)\n',
  'call': 'lowest_spectral_gap(H)',
  'gold_call': '_oracle_lowest_spectral_gap(H)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n'
           'import numpy as np\n'
           'H=np.ones((1,1))\n',
  'call': 'capture_value_error(lambda: lowest_spectral_gap(H))',
  'gold_call': 'capture_value_error(lambda: _oracle_lowest_spectral_gap(H))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n'
           'import numpy as np\n'
           'H=np.ones((2,3))\n',
  'call': 'capture_value_error(lambda: lowest_spectral_gap(H))',
  'gold_call': 'capture_value_error(lambda: _oracle_lowest_spectral_gap(H))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n'
           'import numpy as np\n'
           'H=np.array([[0.0,np.inf],[0.0,1.0]])\n',
  'call': 'capture_value_error(lambda: lowest_spectral_gap(H))',
  'gold_call': 'capture_value_error(lambda: _oracle_lowest_spectral_gap(H))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n'
           'import numpy as np\n'
           'H=np.eye(2,dtype=complex)\n',
  'call': 'capture_value_error(lambda: lowest_spectral_gap(H))',
  'gold_call': 'capture_value_error(lambda: _oracle_lowest_spectral_gap(H))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(type(value) is float)\n',
  'call': '_has_return_type(lowest_spectral_gap(np.array([[2., .5], [.5, 3.]])))',
  'gold_call': '_has_return_type(_oracle_lowest_spectral_gap(np.array([[2., .5], [.5, 3.]])))'}]
