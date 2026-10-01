"""
Lowest spectral gap of a Hamiltonian restricted to a block.

Restricting a Hamiltonian to a set of basis indices means keeping exactly the rows and the columns carrying those indices, in increasing index order, and discarding everything else. Return the lowest spectral gap of that restricted matrix, defined exactly as in the previous step: eigenvalues counted with multiplicity, the matrix symmetrized before diagonalization, so a degenerate ground state gives zero. The index array must be strictly increasing, must hold at least two entries, and every entry must be a valid row of the Hamiltonian. Accept NumPy arrays and nested lists or tuples. The matrix array representation must have an integer or floating dtype; boolean, complex, object, datetime and timedelta dtypes are outside the domain. Index arrays must have an integer dtype. Interpret matrix entries as float64 values; their conversions must be finite. Paper provenance: Section III.A motivates restriction from the initial vacuum, and Section IV.A supplies the finite Hamiltonians. This step applies the preceding spectral-gap operation to the coordinate block selected by the given indices. Include every import the implementation needs inside the function body.

Returns
-------
gap : float Second smallest minus smallest eigenvalue of the restricted, symmetrized block, with multiplicity. Native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def subspace_gap(hamiltonian: object, indices: object) -> float:
    """Lowest gap of a Hamiltonian restricted to selected basis indices.

    Parameters
    ----------
    hamiltonian : array-like, shape (dim, dim)
        Square Hamiltonian with finite entries.
    indices : array-like, shape (n_kept,), int
        Strictly increasing basis indices to keep. At least two entries, each in
        range(dim).

    Returns
    -------
    gap : float
        Second smallest minus smallest eigenvalue of the restricted, symmetrized block,
        with multiplicity. Native Python float.

    Raises
    ------
    ValueError
        If hamiltonian is not a finite real square matrix, or if indices is not a
        strictly increasing one-dimensional integer array of at least two valid rows,
        or a finite spectral gap cannot be computed.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_subspace_gap(hamiltonian: object, indices: object) -> float:
    import numpy as np

    try:
        raw_h = np.asarray(hamiltonian)
        if raw_h.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw_h.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('hamiltonian must be square')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    try:
        idx = np.asarray(indices)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('indices must be a one-dimensional integer array') from exc
    if idx.ndim != 1:
        raise ValueError('indices must be one dimensional')
    if idx.dtype.kind not in 'iu':
        raise ValueError('indices must contain integers')
    if idx.size < 2:
        raise ValueError('indices must hold at least two entries')
    if np.any(idx[1:] <= idx[:-1]):
        raise ValueError('indices must be strictly increasing')
    if idx.min() < 0 or idx.max() >= H.shape[0]:
        raise ValueError('indices must be valid rows of the hamiltonian')
    idx = idx.astype(np.intp, copy=False)
    block = H[np.ix_(idx, idx)]
    return float(_oracle_lowest_spectral_gap(block))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: boundary\n',
  'call': 'subspace_gap([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]], [0, 1])',
  'gold_call': '_oracle_subspace_gap([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]], [0, 1])'},
 {'setup': '# case: normal\n'
           'import numpy as np\n'
           'M = np.diag([5.0, 1.0, 9.0, 3.0, 7.0])\n'
           'keep = np.array([0, 1, 3, 4])\n'
           'keep2 = np.array([0, 2])\n',
  'call': 'subspace_gap(M, keep)',
  'gold_call': '_oracle_subspace_gap(M, keep)'},
 {'setup': '# case: normal\n'
           'import numpy as np\n'
           'M = np.diag([5.0, 1.0, 9.0, 3.0, 7.0])\n'
           'keep = np.array([0, 1, 3, 4])\n'
           'keep2 = np.array([0, 2])\n',
  'call': 'subspace_gap(M, keep2)',
  'gold_call': '_oracle_subspace_gap(M, keep2)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'N = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 4.0]])\n'
           'sel = np.array([0, 1, 2])\n',
  'call': 'subspace_gap(N, sel)',
  'gold_call': '_oracle_subspace_gap(N, sel)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'Q = np.diag([3.0, 7.0, 3.0, 11.0])\n'
           'deg = np.array([0, 2, 3])\n',
  'call': 'subspace_gap(Q, deg)',
  'gold_call': '_oracle_subspace_gap(Q, deg)'},
 {'setup': 'H = _oracle_assemble_t3_hamiltonian(4, 3, 1.0, 1.0)\n'
           'blk = _oracle_vacuum_connected_indices(H, 1e-12)\n',
  'call': 'subspace_gap(H, blk)',
  'gold_call': '_oracle_subspace_gap(H, blk)'},
 {'setup': 'G = _oracle_assemble_t1_hamiltonian(4, 3, 1.0, 1.0)\n'
           'blk1 = _oracle_vacuum_connected_indices(G, 1e-12)\n',
  'call': 'subspace_gap(G, blk1)',
  'gold_call': '_oracle_subspace_gap(G, blk1)'},
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
           'H=np.eye(3)\n'
           'idx=np.array([0.0,1.0])\n',
  'call': 'capture_value_error(lambda: subspace_gap(H, idx))',
  'gold_call': 'capture_value_error(lambda: _oracle_subspace_gap(H, idx))'},
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
           'H=np.eye(3)\n'
           'idx=np.array([0,3])\n',
  'call': 'capture_value_error(lambda: subspace_gap(H, idx))',
  'gold_call': 'capture_value_error(lambda: _oracle_subspace_gap(H, idx))'},
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
           'H=np.eye(3)\n'
           'H[1,1]=np.nan\n'
           'idx=np.array([0,1])\n',
  'call': 'capture_value_error(lambda: subspace_gap(H, idx))',
  'gold_call': 'capture_value_error(lambda: _oracle_subspace_gap(H, idx))'},
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
           'H=np.eye(3)\n'
           'idx=np.array([1,0])\n',
  'call': 'capture_value_error(lambda: subspace_gap(H, idx))',
  'gold_call': 'capture_value_error(lambda: _oracle_subspace_gap(H, idx))'},
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
           'H=np.eye(2)\n'
           'indices=np.array(0)\n',
  'call': 'capture_value_error(lambda: subspace_gap(H, indices))',
  'gold_call': 'capture_value_error(lambda: _oracle_subspace_gap(H, indices))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(type(value) is float)\n',
  'call': '_has_return_type(subspace_gap(np.diag([1., 2., 4.]), np.array([0, 2], dtype=int)))',
  'gold_call': '_has_return_type(_oracle_subspace_gap(np.diag([1., 2., 4.]), np.array([0, 2], '
               'dtype=int)))'}]
