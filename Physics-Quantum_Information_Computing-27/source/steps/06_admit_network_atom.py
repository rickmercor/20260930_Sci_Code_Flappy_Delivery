"""
Admit a descent-producing network candidate into an oldest-first bounded memory.

A candidate is admitted only when the infinitesimal segment from the current anchor toward it has strictly negative directional derivative of the squared Hilbert-Schmidt distance to the target. A zero directional derivative is rejection. Admission appends the candidate and discards the oldest stored entries only when capacity is exceeded; rejection leaves the memory unchanged. This operation does not perform the convex projection, which is handled by project_network_memory.

Returns
-------
np.ndarray, complex retained-memory array of shape (m_new,D,D), oldest first, after strict descent admission and eviction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def admit_network_atom(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray", candidate: "np.ndarray", capacity: int) -> "np.ndarray":
    """Admit a descent-producing network candidate into an oldest-first bounded memory.

    Parameters
    ----------
    target : np.ndarray
        Density matrix (D,D), D >= 2.
    anchor : np.ndarray
        Current density matrix of the same shape.
    memory : np.ndarray
        Retained states (m,D,D), oldest first, with 0 <= m <= capacity.
    candidate : np.ndarray
        Proposed density matrix (D,D), with feasible network membership established
        by the caller. The main protocol supplies pure-state projectors.
    capacity : int
        Maximum retained atom count, from 1 through 5; booleans are invalid.

    Returns
    -------
    updated_memory : np.ndarray
        Complex array (m_new,D,D), preserving age order. This is a copy of the
        input bank on rejection, otherwise the appended bank truncated from its
        oldest end. The current anchor is not itself stored in this bank.

    Raises
    ------
    ValueError
        If capacity is outside its integer range, array shapes disagree, the bank already exceeds capacity, or density-matrix validation fails at absolute tolerance 1e-10.

    Notes
    -----
    Preserve every input array. Apply the strict sign comparison without a tolerance band; the supplied positive and negative cases have resolved signs. An exactly zero derivative is rejected. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return updated_memory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_admit_network_atom(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray", candidate: "np.ndarray", capacity: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    rho, current, atom = (np.asarray(v, dtype=complex) for v in (target, anchor, candidate))
    bank = np.asarray(memory, dtype=complex)
    if not isinstance(capacity, Integral) or isinstance(capacity, (bool, np.bool_)) or not 1 <= capacity <= 5:
        raise ValueError("Capacity must be an integer from 1 through 5.")
    if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2
            or current.shape != rho.shape or atom.shape != rho.shape
            or bank.ndim != 3 or bank.shape[1:] != rho.shape or len(bank) > capacity):
        raise ValueError("Density-matrix and memory shapes must agree.")
    for matrix in [rho, current, atom, *bank]:
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    score = float(np.vdot(rho - current, atom - current).real)
    if score <= 0:
        return bank.copy()
    return np.concatenate((bank, atom[None]), axis=0)[-capacity:].copy()

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
               'args = (zero, anchor, empty, zero, 3)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (zero, anchor, np.array([zero]), one, 3)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (anchor, anchor, np.array([zero]), one, 1)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (atom, anchor, np.array([zero, one]), atom, 2)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (anchor, zero, empty, one, 1)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (anchor+1e-8*(zero-anchor), anchor, np.array([one]), zero, 1)\n',
      'call': '_invoke(admit_network_atom)',
      'gold_call': '_invoke(_oracle_admit_network_atom)',
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
               'rho = np.array([[.63, .11+.17j], [.11-.17j, .37]])\n'
               'anchor = np.eye(2, dtype=complex)/2\n'
               'v = np.array([1, 1j])/np.sqrt(2)\n'
               'atom = np.outer(v, v.conj())\n'
               'zero = np.diag([1., 0.]).astype(complex)\n'
               'one = np.diag([0., 1.]).astype(complex)\n'
               'empty = np.empty((0, 2, 2), dtype=complex)\n'
               'args = (zero, anchor, empty, zero, 0)\n',
      'call': '_invalid(admit_network_atom)',
      'gold_call': '_invalid(_oracle_admit_network_atom)',
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
               'args = (zero, anchor, np.array([zero,one]), zero, 1)\n',
      'call': '_invalid(admit_network_atom)',
      'gold_call': '_invalid(_oracle_admit_network_atom)',
      'tol': 0.0}]
