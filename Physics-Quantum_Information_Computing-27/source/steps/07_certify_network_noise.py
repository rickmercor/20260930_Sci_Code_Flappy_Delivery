"""
Convert a feasible network approximation into a sufficient white-noise fraction.

Use the constructive inscribed-ball certificate for the full convex set of network states. The Hilbert-Schmidt ball centered at the maximally mixed state is separable across each bipartition and therefore lies in that network set. Join a known feasible approximant to the appropriate boundary point of this ball, and intersect this segment with the white-noise line through the target. The resulting noise fraction is sufficient even if the approximant was not globally optimal, and it need not be an exact entanglement threshold.

Returns
-------
float, unrounded dimensionless sufficient white-noise fraction from the constructive network-state ball certificate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certify_network_noise(target: "np.ndarray", approximant: "np.ndarray") -> float:
    """Convert a feasible network approximation into a sufficient white-noise fraction.

    Parameters
    ----------
    target : np.ndarray
        Density matrix (D,D), D >= 2.
    approximant : np.ndarray
        Density matrix of the same shape, whose membership in the full network
        set has already been established by its construction.

    Returns
    -------
    noise_fraction : float
        Native finite Python float giving the inscribed-ball certificate for
        (1-epsilon)*target + epsilon*I/D. Use Hilbert-Schmidt distance and the
        dimension-dependent bipartite-separability ball for the full network set;
        a zero residual gives zero noise. Return the unrounded value.

    Raises
    ------
    ValueError
        If shapes disagree or either input fails finite density-matrix validation at absolute tolerance 1e-10. The function does not independently decide network membership.

    Notes
    -----
    Preserve both arrays. The output is a sufficient noise level, not the exact minimal noise over all network constructions. Use the physical Hilbert-space dimension, not its squared operator-space dimension. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return noise_fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_certify_network_noise(target: "np.ndarray", approximant: "np.ndarray") -> float:
    import numpy as np

    rho = np.asarray(target, dtype=complex)
    sigma = np.asarray(approximant, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2 or sigma.shape != rho.shape:
        raise ValueError("Use square same-sized density matrices with dimension at least two.")
    for matrix in (rho, sigma):
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    delta = float(np.linalg.norm(rho - sigma))
    dimension = rho.shape[0]
    radius = 1.0 / np.sqrt(float(dimension) * (dimension - 1))
    return float(delta / (delta + radius))

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
               'args = (rho, anchor)\n',
      'call': '_invoke(certify_network_noise)',
      'gold_call': '_invoke(_oracle_certify_network_noise)',
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
               'args = (rho, rho.copy())\n',
      'call': '_invoke(certify_network_noise)',
      'gold_call': '_invoke(_oracle_certify_network_noise)',
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
               'args = (zero, one)\n',
      'call': '_invoke(certify_network_noise)',
      'gold_call': '_invoke(_oracle_certify_network_noise)',
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
               'a = np.eye(64, dtype=complex)/64\n'
               'r = .8*a; r[0,0] += .2\n'
               'args = (r,a)\n',
      'call': '_invoke(certify_network_noise)',
      'gold_call': '_invoke(_oracle_certify_network_noise)',
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
               'args = (anchor+1e-8*(zero-anchor), anchor)\n',
      'call': '_invoke(certify_network_noise)',
      'gold_call': '_invoke(_oracle_certify_network_noise)',
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
               'args = (rho, np.eye(3)/3)\n',
      'call': '_invalid(certify_network_noise)',
      'gold_call': '_invalid(_oracle_certify_network_noise)',
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
               'args = (np.diag([1.2,-.2]), anchor)\n',
      'call': '_invalid(certify_network_noise)',
      'gold_call': '_invalid(_oracle_certify_network_noise)',
      'tol': 0.0}]
