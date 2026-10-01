"""
Basis states reachable from the free vacuum.

A truncated Kogut-Susskind Hamiltonian written in a product link basis decomposes into invariant coordinate blocks: its off-diagonal entries connect only those configurations that a single hop relates, so a simulation started in one configuration never leaves that configuration's block. Index 0 of the product basis is the free vacuum, the configuration in which every link carries the trivial irrep. Given a square Hamiltonian matrix and a magnitude threshold, return the sorted indices of every basis state joined to index 0 by a chain of off-diagonal entries whose magnitude exceeds the threshold. Diagonal entries never join two states, and an entry joins its row and column in either direction, so the matrix need not be symmetric. tolerance is a finite non-negative float; an entry is a link when its absolute value is strictly greater than tolerance. Accept NumPy arrays and nested lists or tuples. The matrix array representation must have an integer or floating dtype; boolean, complex, object, datetime and timedelta dtypes are outside the domain. tolerance accepts Python or NumPy real scalars, excluding boolean and temporal values. Interpret matrix entries and tolerance as float64 values; their conversions must be finite. Paper provenance: Section III.A, Eq. (4), generates allowed configurations from the free vacuum by a bounded number of hops. For the T1 and T3 Hamiltonians of Section IV.A that Krylov-truncated set is the invariant coordinate block reached from the all-singlet configuration through the retained hopping transitions, which is the block returned here. Include every import the implementation needs inside the function body.

Returns
-------
indices : np.ndarray, shape (n_connected,), int Strictly increasing basis indices reachable from index 0. Always contains 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vacuum_connected_indices(hamiltonian: object, tolerance: float) -> "np.ndarray":
    """Sorted basis indices in the vacuum block of a Hamiltonian.

    Parameters
    ----------
    hamiltonian : array-like, shape (dim, dim)
        Square Hamiltonian in a product computational basis whose index 0 is the vacuum.
        Finite entries. dim is at least 1.
    tolerance : float
        Magnitude threshold on an off-diagonal entry. Finite and at least 0. An entry
        links two indices when its absolute value is strictly greater than tolerance.

    Returns
    -------
    indices : np.ndarray, shape (n_connected,), int
        Strictly increasing basis indices reachable from index 0. Always contains 0.

    Raises
    ------
    ValueError
        If the Hamiltonian is not a non-empty finite real square matrix, or if
        tolerance is not a finite non-negative real scalar.
    """
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_vacuum_connected_indices(hamiltonian: object,
                                     tolerance: float) -> "np.ndarray":
    import numpy as np

    try:
        raw = np.asarray(hamiltonian)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('hamiltonian must be square')
    if H.shape[0] < 1:
        raise ValueError('hamiltonian must have at least one row')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    if (isinstance(tolerance, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))):
        raise ValueError('tolerance must be a finite real scalar')
    try:
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('tolerance must be a finite real scalar') from exc
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError('tolerance must be finite and at least 0')
    adjacency = np.abs(H) > tol
    np.fill_diagonal(adjacency, False)
    adjacency = adjacency | adjacency.T
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(adjacency[i])[0]:
            j = int(j)
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return np.array(sorted(seen), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: boundary\n',
  'call': 'vacuum_connected_indices([[0.0, 0.3, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 1.0]], 0.0)',
  'gold_call': '_oracle_vacuum_connected_indices([[0.0, 0.3, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, '
               '1.0]], 0.0)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'C = np.zeros((5, 5))\n'
           'C[0, 1] = 0.7\n'
           'C[2, 1] = 0.4\n'
           'C[3, 4] = C[4, 3] = 0.9\n',
  'call': 'vacuum_connected_indices(C, 0.0)',
  'gold_call': '_oracle_vacuum_connected_indices(C, 0.0)'},
 {'setup': '# case: normal\n'
           'import numpy as np\n'
           'A = np.zeros((6, 6))\n'
           'A[0, 1] = A[1, 0] = 0.5\n'
           'A[1, 2] = A[2, 1] = -0.25\n'
           'A[3, 4] = A[4, 3] = 0.75\n'
           'A[5, 5] = 9.0\n'
           'A[0, 0] = -1.0\n',
  'call': 'vacuum_connected_indices(A, 0.0)',
  'gold_call': '_oracle_vacuum_connected_indices(A, 0.0)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'B = np.zeros((4, 4))\n'
           'B[0, 1] = B[1, 0] = 1e-10\n'
           'B[1, 2] = B[2, 1] = 0.4\n'
           'B[3, 3] = 2.0\n',
  'call': 'vacuum_connected_indices(B, 1e-8)',
  'gold_call': '_oracle_vacuum_connected_indices(B, 1e-8)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'B = np.zeros((4, 4))\n'
           'B[0, 1] = B[1, 0] = 1e-10\n'
           'B[1, 2] = B[2, 1] = 0.4\n'
           'B[3, 3] = 2.0\n',
  'call': 'vacuum_connected_indices(B, 0.0)',
  'gold_call': '_oracle_vacuum_connected_indices(B, 0.0)'},
 {'setup': 'H = _oracle_assemble_t3_hamiltonian(4, 3, 1.0, 1.0)\n',
  'call': 'vacuum_connected_indices(H, 1e-12)',
  'gold_call': '_oracle_vacuum_connected_indices(H, 1e-12)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'E = np.zeros((4, 4))\n'
           'E[0, 1] = E[1, 0] = 0.25\n'
           'E[1, 2] = E[2, 1] = 0.5\n',
  'call': 'vacuum_connected_indices(E, 0.25)',
  'gold_call': '_oracle_vacuum_connected_indices(E, 0.25)'},
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
           'F=np.eye(2)\n'
           'F[0,1]=np.nan\n',
  'call': 'capture_value_error(lambda: vacuum_connected_indices(F, 0.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_vacuum_connected_indices(F, 0.0))'},
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
           'F=np.eye(2, dtype=complex)\n',
  'call': 'capture_value_error(lambda: vacuum_connected_indices(F, 0.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_vacuum_connected_indices(F, 0.0))'},
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
           'F=np.eye(2)\n',
  'call': 'capture_value_error(lambda: vacuum_connected_indices(F, -1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_vacuum_connected_indices(F, -1.0))'},
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
           'F=np.eye(2)\n',
  'call': 'capture_value_error(lambda: vacuum_connected_indices(F, float("nan")))',
  'gold_call': 'capture_value_error(lambda: _oracle_vacuum_connected_indices(F, float("nan")))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(isinstance(value, np.ndarray) and value.dtype.kind in "iu")\n',
  'call': '_has_return_type(vacuum_connected_indices(np.array([[1., .5, 0.], [.5, 2., 0.], [0., '
          '0., 3.]]), 1e-12))',
  'gold_call': '_has_return_type(_oracle_vacuum_connected_indices(np.array([[1., .5, 0.], [.5, 2., '
               '0.], [0., 0., 3.]]), 1e-12))'}]
