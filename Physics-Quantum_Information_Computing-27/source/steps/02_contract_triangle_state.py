"""
Contract three bipartite resource states with the local triangle-network unitaries.

The resource ordering is $A_0B_0$, $B_1C_0$, $C_1A_1$, while the output ordering is $A_0A_1B_0B_1C_0C_1$. A local four-dimensional unitary acts on each party after the independent resources are combined. In particular, the last resource is ordered $C_1,A_1$, not $A_1,C_1$. Tensor products, leg permutations and complex amplitudes must preserve this wiring without a conjugation on any ket leg.

Returns
-------
np.ndarray, normalized complex state vector of shape (64,) in A0,A1,B0,B1,C0,C1 order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contract_triangle_state(links: "np.ndarray", unitaries: "np.ndarray") -> "np.ndarray":
    """Contract three bipartite resource states with the local triangle-network unitaries.

    Parameters
    ----------
    links : np.ndarray
        Complex array (3,4); rows are normalized kets for AB, BC and CA.
        The two-qubit order within each row is 00,01,10,11.
    unitaries : np.ndarray
        Complex array (3,4,4) of unitary gates for A, B and C, with
        output basis on rows and input basis on columns.

    Returns
    -------
    state : np.ndarray
        Complex ket of shape (64,), ordered A0,A1,B0,B1,C0,C1.

    Raises
    ------
    ValueError
        If shapes or finite-value requirements fail, a link norm squared differs from one by more than 1e-10, or a gate fails unitarity with entrywise absolute tolerance 1e-10.

    Notes
    -----
    Preserve both input arrays. Return the contracted ket itself; do not choose a new overall phase. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_contract_triangle_state(links: "np.ndarray", unitaries: "np.ndarray") -> "np.ndarray":
    import numpy as np

    s = np.asarray(links, dtype=complex)
    u = np.asarray(unitaries, dtype=complex)
    if (s.shape != (3, 4) or u.shape != (3, 4, 4)
            or not np.all(np.isfinite(s)) or not np.all(np.isfinite(u))):
        raise ValueError("Use finite link vectors (3,4) and local matrices (3,4,4).")
    if (not np.allclose(np.sum(abs(s)**2, axis=1), 1.0, atol=1e-10, rtol=0)
            or not np.allclose(u.conj().transpose(0, 2, 1) @ u, np.eye(4), atol=1e-10, rtol=0)):
        raise ValueError("Links must be normalized and local matrices unitary.")
    tensor = np.einsum("ab,cd,ef->afbcde", s[0].reshape(2, 2),
                       s[1].reshape(2, 2), s[2].reshape(2, 2)).reshape(4, 4, 4)
    return np.einsum("ia,jb,kc,abc->ijk", u[0], u[1], u[2], tensor).reshape(64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerical checks for this operation."""
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X = np.array([[0, 1], [1, 0]], dtype=complex)\n'
               'Y = np.array([[0, -1j], [1j, 0]], dtype=complex)\n'
               'Z = np.diag([1., -1.])\n'
               'P = np.kron(X, Y)\n'
               'Q = np.kron(Z, np.eye(2))\n'
               'k = 2\n'
               'links = np.array([[1, (k+1+e)/10+1j*(e+1)/7,\n'
               '                   (2*e-k)/9-1j/5, .8-1j*(k+e)/13] for e in range(3)])\n'
               'links /= np.linalg.norm(links, axis=1)[:, None]\n'
               'unitaries = np.empty((3, 4, 4), dtype=complex)\n'
               'for a in range(3):\n'
               '    alpha, eta = (k+a+1)*np.pi/13, (2*k-a+2)*np.pi/17\n'
               '    unitaries[a] = ((np.cos(alpha)*np.eye(4)-1j*np.sin(alpha)*P)\n'
               '                   @ (np.cos(eta)*np.eye(4)-1j*np.sin(eta)*Q))\n'
               'args = (links, unitaries)\n',
      'call': '_invoke(contract_triangle_state)',
      'gold_call': '_invoke(_oracle_contract_triangle_state)',
      'tol': 2e-10},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'links = np.eye(4, dtype=complex)[[1, 2, 3]]\n'
               'args = (links, unitaries)\n',
      'call': '_invoke(contract_triangle_state)',
      'gold_call': '_invoke(_oracle_contract_triangle_state)',
      'tol': 2e-10},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'links[2] = np.array([1, 0, 0, 1j])/np.sqrt(2)\n'
               'args = (links, unitaries)\n',
      'call': '_invoke(contract_triangle_state)',
      'gold_call': '_invoke(_oracle_contract_triangle_state)',
      'tol': 2e-10},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X = np.array([[0, 1], [1, 0]], dtype=complex)\n'
               'Y = np.array([[0, -1j], [1j, 0]], dtype=complex)\n'
               'Z = np.diag([1., -1.])\n'
               'P = np.kron(X, Y)\n'
               'Q = np.kron(Z, np.eye(2))\n'
               'k = 2\n'
               'links = np.array([[1, (k+1+e)/10+1j*(e+1)/7,\n'
               '                   (2*e-k)/9-1j/5, .8-1j*(k+e)/13] for e in range(3)])\n'
               'links /= np.linalg.norm(links, axis=1)[:, None]\n'
               'unitaries = np.empty((3, 4, 4), dtype=complex)\n'
               'for a in range(3):\n'
               '    alpha, eta = (k+a+1)*np.pi/13, (2*k-a+2)*np.pi/17\n'
               '    unitaries[a] = ((np.cos(alpha)*np.eye(4)-1j*np.sin(alpha)*P)\n'
               '                   @ (np.cos(eta)*np.eye(4)-1j*np.sin(eta)*Q))\n'
               'links[0] *= 1j\n'
               'unitaries[1] = unitaries[1].conj().T\n'
               'args = (links, unitaries)\n',
      'call': '_invoke(contract_triangle_state)',
      'gold_call': '_invoke(_oracle_contract_triangle_state)',
      'tol': 2e-10},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'args = (links, unitaries)\n',
      'call': '_invoke(contract_triangle_state)',
      'gold_call': '_invoke(_oracle_contract_triangle_state)',
      'tol': 2e-10},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'links[0] *= 2\n'
               'args = (links, unitaries)\n',
      'call': '_invalid(contract_triangle_state)',
      'gold_call': '_invalid(_oracle_contract_triangle_state)',
      'tol': 0.0},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'unitaries[0,0,0] = 0\n'
               'args = (links, unitaries)\n',
      'call': '_invalid(contract_triangle_state)',
      'gold_call': '_invalid(_oracle_contract_triangle_state)',
      'tol': 0.0},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def _invoke(function):\n'
               '    local = deepcopy(args)\n'
               '    before = deepcopy(local)\n'
               '    value = function(*local)\n'
               '    for original, after in zip(before, local):\n'
               '        if isinstance(original, np.ndarray):\n'
               '            assert np.array_equal(original, after, equal_nan=True), "Input arrays were '
               'modified."\n'
               '    return value\n'
               '\n'
               'def _invalid(function):\n'
               '    try:\n'
               '        _invoke(function)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'links = np.tile([1.+0j, 0, 0, 0], (3, 1))\n'
               'unitaries = np.tile(np.eye(4, dtype=complex), (3, 1, 1))\n'
               'args = (links[:, :3], unitaries)\n',
      'call': '_invalid(contract_triangle_state)',
      'gold_call': '_invalid(_oracle_contract_triangle_state)',
      'tol': 0.0}]
