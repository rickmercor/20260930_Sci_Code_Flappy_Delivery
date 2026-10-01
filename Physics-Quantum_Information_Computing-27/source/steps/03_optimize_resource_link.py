"""
Optimize one resource ket while the other links and all local unitaries remain fixed.

The support objective is the expectation of a frozen Hermitian residual in the current pure network state. Holding every factor except one resource fixed gives a four-dimensional variational problem: the contracted environment defines an effective Hermitian operator on that resource. Use the network contraction from contract_triangle_state, retain complex coherences, and select the maximizing spectral subspace according to the stated degeneracy convention.

Returns
-------
np.ndarray, normalized complex resource ket of shape (4,) with the specified spectral-subspace and phase conventions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimize_resource_link(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", edge: int) -> "np.ndarray":
    """Optimize one resource ket while the other links and all local unitaries remain fixed.

    Parameters
    ----------
    residual : np.ndarray
        Finite Hermitian (64,64) support operator, absolute Hermiticity tolerance 1e-10.
    links : np.ndarray
        Normalized complex resource kets (3,4), as in contract_triangle_state.
    unitaries : np.ndarray
        Local unitary gates (3,4,4), as in contract_triangle_state.
    edge : int
        Resource to replace: 0 for A0B0, 1 for B1C0, 2 for C1A1.

    Returns
    -------
    updated_link : np.ndarray
        Complex normalized ket of shape (4,) in the selected top eigenspace.
        Eigenvalues within 1e-12*max(1,max(abs(eigenvalues))) of the maximum
        belong to this subspace. Project the old link into it and normalize;
        if the projection norm is at most 1e-12, project basis vectors
        00,01,10,11 in order and use the first projection with norm above 1e-12.
        Fix phase so the first component of magnitude above 1e-12 is positive real.

    Raises
    ------
    ValueError
        If the residual is not finite Hermitian with the stated shape, edge is not an integer from zero through two, or link/gate validation in contract_triangle_state fails. Booleans are not indices.

    Notes
    -----
    Preserve all input arrays. Earlier public function contract_triangle_state is available. The defined near-degenerate spectral cluster is part of this finite numerical protocol. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return updated_link

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_optimize_resource_link(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", edge: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    r = np.asarray(residual, dtype=complex)
    if (r.shape != (64, 64) or not np.all(np.isfinite(r))
            or not np.allclose(r, r.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("Residual must be a finite Hermitian (64,64) matrix.")
    if not isinstance(edge, Integral) or isinstance(edge, (bool, np.bool_)) or not 0 <= edge < 3:
        raise ValueError("Edge must be an integer in {0,1,2}.")
    _oracle_contract_triangle_state(links, unitaries)
    s = np.asarray(links, dtype=complex)
    embedding = np.empty((64, 4), dtype=complex)
    for component in range(4):
        trial = s.copy()
        trial[edge] = np.eye(4)[component]
        embedding[:, component] = _oracle_contract_triangle_state(trial, unitaries)
    effective = embedding.conj().T @ r @ embedding
    effective = (effective + effective.conj().T) / 2
    values, vectors = np.linalg.eigh(effective)
    cluster = values >= values[-1] - 1e-12 * max(1.0, float(np.max(abs(values))))
    space = vectors[:, cluster]
    updated = space @ (space.conj().T @ s[edge])
    norm = np.linalg.norm(updated)
    if norm <= 1e-12:
        for component in range(4):
            updated = space @ space[component].conj()
            norm = np.linalg.norm(updated)
            if norm > 1e-12:
                break
    updated /= norm
    first = int(np.flatnonzero(abs(updated) > 1e-12)[0])
    updated *= np.exp(-1j * np.angle(updated[first]))
    return updated

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
               'rng = np.random.default_rng(82)\n'
               'a = rng.normal(size=(64,64)) + 1j*rng.normal(size=(64,64))\n'
               'r = (a+a.conj().T)/20\n'
               'args = (r, links, unitaries, 0)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 5e-09},
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
               'rng = np.random.default_rng(37)\n'
               'a = rng.normal(size=(64,64)) + 1j*rng.normal(size=(64,64))\n'
               'r = (a+a.conj().T)/20\n'
               'args = (r, links, unitaries, 2)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 5e-09},
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
               'args = (np.eye(64), links, unitaries, 1)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 2e-09},
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
               'links[0] = np.array([1, 1j, 2, -1j])/np.sqrt(7)\n'
               'd = np.zeros(64)\n'
               'd[[0, 8]] = 1\n'
               'args = (np.diag(d), links, unitaries, 0)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 2e-09},
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
               'd = np.zeros(64)\n'
               'd[[8, 32]] = 1\n'
               'args = (np.diag(d), links, unitaries, 0)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 2e-09},
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
               'links[0] = np.array([1, 1j, 2, -1j])/np.sqrt(7)\n'
               'd = np.zeros(64)\n'
               'd[0], d[8] = 1, 1-4e-13\n'
               'args = (np.diag(d), links, unitaries, 0)\n',
      'call': '_invoke(optimize_resource_link)',
      'gold_call': '_invoke(_oracle_optimize_resource_link)',
      'tol': 2e-09},
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
               'args = (np.eye(64), links, unitaries, 3)\n',
      'call': '_invalid(optimize_resource_link)',
      'gold_call': '_invalid(_oracle_optimize_resource_link)',
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
               'r = np.eye(64, dtype=complex)\n'
               'r[0,1] = 1j\n'
               'args = (r, links, unitaries, 0)\n',
      'call': '_invalid(optimize_resource_link)',
      'gold_call': '_invalid(_oracle_optimize_resource_link)',
      'tol': 0.0}]
