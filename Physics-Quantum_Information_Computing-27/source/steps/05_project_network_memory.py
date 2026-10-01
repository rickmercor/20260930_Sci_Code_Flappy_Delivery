"""
Find the closest density matrix in the convex hull of the current anchor and retained network states.

The memory correction minimizes Hilbert-Schmidt distance over nonnegative normalized mixture weights. Include the current mixed anchor as a vertex in addition to every retained atom, so eviction does not erase the accumulated mixture. Complex Hermitian matrices form a real inner-product space for this optimization. Duplicate or linearly dependent vertices may make weights nonunique, but the closest density matrix is unique; return that density matrix rather than a particular weight vector.

Returns
-------
np.ndarray, complex closest convex-mixture density matrix of shape (D,D); mixture coefficients are not returned.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_network_memory(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray") -> "np.ndarray":
    """Find the closest density matrix in the convex hull of the current anchor and retained network states.

    Parameters
    ----------
    target : np.ndarray
        Density matrix (D,D), D >= 2.
    anchor : np.ndarray
        Current feasible density matrix of the same shape.
    memory : np.ndarray
        Zero through five retained density matrices, shape (m,D,D), in age order.
        Their network membership is a caller-established precondition.

    Returns
    -------
    projected_state : np.ndarray
        Complex matrix (D,D), the unique closest point in the real convex hull
        of anchor and all memory entries, under Frobenius distance. Return anchor
        for an empty memory. Entrywise numerical accuracy 2e-9 is sufficient.

    Raises
    ------
    ValueError
        If shapes disagree, more than five memory entries are supplied, or a matrix is nonfinite, non-Hermitian, non-unit-trace or nonpositive within absolute tolerance 1e-10.

    Notes
    -----
    Preserve all input arrays. Rank-deficient and nearly affinely dependent memory states are valid. Equivalent convex optimization methods are accepted; no particular mixture coefficients are required. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return projected_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_project_network_memory(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from itertools import combinations

    rho = np.asarray(target, dtype=complex)
    current = np.asarray(anchor, dtype=complex)
    atoms = np.asarray(memory, dtype=complex)
    if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2
            or current.shape != rho.shape or atoms.ndim != 3
            or atoms.shape[1:] != rho.shape or len(atoms) > 5):
        raise ValueError("Use same-sized square states and zero through five memory states.")
    vertices = np.concatenate((current[None], atoms), axis=0)
    for matrix in [rho, *vertices]:
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    if len(atoms) == 0:
        return current.copy()
    flat = vertices.reshape(len(vertices), -1)
    design = np.concatenate((flat.real, flat.imag), axis=1).T
    goal = np.concatenate((rho.ravel().real, rho.ravel().imag))
    best_error = np.inf
    best_state = current.copy()
    for count in range(1, len(vertices) + 1):
        for face in combinations(range(len(vertices)), count):
            base = design[:, face[0]]
            if count == 1:
                weights = np.ones(1)
            else:
                differences = design[:, face[1:]] - base[:, None]
                coefficients = np.linalg.lstsq(differences, goal - base, rcond=1e-13)[0]
                weights = np.r_[1 - coefficients.sum(), coefficients]
            if np.min(weights) < -1e-11:
                continue
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            displacement = design[:, face] @ weights - goal
            error = float(np.dot(displacement, displacement))
            if error < best_error:
                best_error = error
                best_state = np.tensordot(weights, vertices[list(face)], axes=1)
    return best_state

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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (rho, anchor, np.array([zero, atom, one]))\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (rho, anchor, empty)\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (zero, anchor, np.array([zero, one]))\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (rho, anchor, np.array([zero, atom, zero, atom, anchor]))\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'v = np.array([1., 1e-5j]); v /= np.linalg.norm(v)\n'
               'near = np.outer(v, v.conj())\n'
               'args = (.37*zero+.63*near, anchor, np.array([zero, near, zero]))\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (rho, zero, np.array([one]))\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.diag([.05,.2,.75]).astype(complex)\n'
               'anchor = np.eye(3)/3\n'
               'bank = np.array([np.diag([1.,0,0]),np.diag([0,1.,0])])\n'
               'args = (rho,anchor,bank)\n',
      'call': '_invoke(project_network_memory)',
      'gold_call': '_invoke(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (2*rho, anchor, np.array([zero]))\n',
      'call': '_invalid(project_network_memory)',
      'gold_call': '_invalid(_oracle_project_network_memory)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (np.diag([1.1,-.1]), anchor, np.array([zero]))\n',
      'call': '_invalid(project_network_memory)',
      'gold_call': '_invalid(_oracle_project_network_memory)',
      'tol': 0.0}]
