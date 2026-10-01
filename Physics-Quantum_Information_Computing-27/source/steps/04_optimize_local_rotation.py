"""
Optimize a one-parameter local unitary rotation for the frozen network support objective.

Replace the gate of the chosen party by $e^{-i\theta G}U$ with $G=G^\dagger$ and $G^2=I$. The expectation in the contracted pure state is a sinusoid of twice the rotation angle; its phase depends on the complex transition matrix element between the current state and the state with $G$ applied at that party. Derive the maximizing rotation from that expectation rather than from a real-only overlap. The gate acts on the left of the existing unitary.

Returns
-------
np.ndarray, complex updated local gate of shape (4,4), with the maximizing angle in [-pi/2, pi/2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimize_local_rotation(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", party: int, generator: "np.ndarray") -> "np.ndarray":
    """Optimize a one-parameter local unitary rotation for the frozen network support objective.

    Parameters
    ----------
    residual : np.ndarray
        Finite Hermitian support operator (64,64), tolerance 1e-10.
    links : np.ndarray
        Normalized complex resource states (3,4).
    unitaries : np.ndarray
        Local unitary gates (3,4,4).
    party : int
        Index 0, 1 or 2 for A, B or C; booleans are invalid.
    generator : np.ndarray
        Finite Hermitian involution (4,4); both identities hold within 1e-10.

    Returns
    -------
    updated_unitary : np.ndarray
        Complex gate of shape (4,4) maximizing the support objective over
        theta in [-pi/2, pi/2). With F(theta) denoting that objective,
        use theta=0 when its second-harmonic amplitude is at most
        1e-14*max(1,abs(F(0)),abs(F(pi/2))); otherwise use the maximizing
        representative in that half-open interval.

    Raises
    ------
    ValueError
        If the residual, party, generator or network inputs violate their stated shape, finite-value, index, normalization, Hermiticity or unitarity contracts.

    Notes
    -----
    Preserve all input arrays. Use the earlier public contraction operation. Return the gate, not the angle or an independently phase-adjusted gate. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return updated_unitary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_optimize_local_rotation(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", party: int, generator: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    r = np.asarray(residual, dtype=complex)
    g = np.asarray(generator, dtype=complex)
    if (r.shape != (64, 64) or not np.all(np.isfinite(r))
            or not np.allclose(r, r.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("Residual must be a finite Hermitian (64,64) matrix.")
    if (g.shape != (4, 4) or not np.all(np.isfinite(g))
            or not np.allclose(g, g.conj().T, atol=1e-10, rtol=0)
            or not np.allclose(g @ g, np.eye(4), atol=1e-10, rtol=0)):
        raise ValueError("Generator must be a finite Hermitian involution of shape (4,4).")
    if not isinstance(party, Integral) or isinstance(party, (bool, np.bool_)) or not 0 <= party < 3:
        raise ValueError("Party must be an integer in {0,1,2}.")
    psi = _oracle_contract_triangle_state(links, unitaries)
    u = np.asarray(unitaries, dtype=complex)
    trial = u.copy()
    trial[party] = g @ u[party]
    phi = _oracle_contract_triangle_state(links, trial)
    a = float(np.vdot(psi, r @ psi).real)
    b = float(np.vdot(phi, r @ phi).real)
    coherence = np.vdot(psi, r @ phi)
    cosine_coefficient = (a - b) / 2
    sine_coefficient = float(coherence.imag)
    if np.hypot(cosine_coefficient, sine_coefficient) <= 1e-14 * max(1.0, abs(a), abs(b)):
        theta = 0.0
    else:
        theta = float(0.5 * np.arctan2(sine_coefficient, cosine_coefficient))
        if theta >= np.pi / 2:
            theta -= np.pi
    return (np.cos(theta) * np.eye(4) - 1j * np.sin(theta) * g) @ u[party]

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
               'rng = np.random.default_rng(33)\n'
               'a = rng.normal(size=(64,64)) + 1j*rng.normal(size=(64,64))\n'
               'args = ((a+a.conj().T)/20, links, unitaries, 0, P)\n',
      'call': '_invoke(optimize_local_rotation)',
      'gold_call': '_invoke(_oracle_optimize_local_rotation)',
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
               'r = np.diag(np.linspace(-2, 1, 64))\n'
               'args = (r, links, unitaries, 2, Q)\n',
      'call': '_invoke(optimize_local_rotation)',
      'gold_call': '_invoke(_oracle_optimize_local_rotation)',
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
               'args = (np.eye(64), links, unitaries, 1, P)\n',
      'call': '_invoke(optimize_local_rotation)',
      'gold_call': '_invoke(_oracle_optimize_local_rotation)',
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
               'args = (np.diag(np.arange(64)), links, unitaries, 0, np.eye(4))\n',
      'call': '_invoke(optimize_local_rotation)',
      'gold_call': '_invoke(_oracle_optimize_local_rotation)',
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
               'r = np.zeros((64,64))\n'
               'r[32,32] = 1\n'
               'g = np.kron(np.array([[0,1],[1,0]]), np.eye(2))\n'
               'args = (r, links, unitaries, 0, g)\n',
      'call': '_invoke(optimize_local_rotation)',
      'gold_call': '_invoke(_oracle_optimize_local_rotation)',
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
               'args = (np.eye(64), links, unitaries, 0, 2*P)\n',
      'call': '_invalid(optimize_local_rotation)',
      'gold_call': '_invalid(_oracle_optimize_local_rotation)',
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
               'args = (np.eye(64), links, unitaries, -1, P)\n',
      'call': '_invalid(optimize_local_rotation)',
      'gold_call': '_invalid(_oracle_optimize_local_rotation)',
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
               'g = P.copy()\n'
               'g[0,1] += .5j\n'
               'args = (np.eye(64), links, unitaries, 0, g)\n',
      'call': '_invalid(optimize_local_rotation)',
      'gold_call': '_invalid(_oracle_optimize_local_rotation)',
      'tol': 0.0}]
