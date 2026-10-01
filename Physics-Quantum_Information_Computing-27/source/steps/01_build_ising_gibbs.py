"""
Construct the Gibbs density matrix of a periodic transverse-field Ising ring.

The thermal target is a normalized matrix exponential of a local spin Hamiltonian. Use $H=-\sum_{j=0}^{n-1}J_j Z_jZ_{(j+1)\bmod n}-\sum_{j=0}^{n-1}h_jX_j$, with site 0 the most significant computational-basis bit. Every listed bond contributes once, including both directed labels when $n=2$. A low-temperature calculation must retain normalization when unshifted Boltzmann factors overflow; a zero-temperature limit with degenerate ground states retains their equal thermal weights.

Returns
-------
np.ndarray, complex Gibbs density matrix of shape (2**n, 2**n), normalized to unit trace.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_ising_gibbs(couplings: "np.ndarray", fields: "np.ndarray", beta: float) -> "np.ndarray":
    """Construct the Gibbs density matrix of a periodic transverse-field Ising ring.

    Parameters
    ----------
    couplings : np.ndarray
        Finite real vector of n bond strengths, with 2 <= n <= 8.
    fields : np.ndarray
        Finite real vector of n transverse fields; negative entries are allowed.
    beta : float
        Finite nonnegative inverse temperature in reciprocal energy units.

    Returns
    -------
    density_matrix : np.ndarray
        Complex array of shape (2**n, 2**n), with unit trace and site-0-first ordering.

    Raises
    ------
    ValueError
        If vector shapes disagree, their length is outside 2 through 8, their entries are complex or nonfinite, or beta is not a finite nonnegative real scalar.

    Notes
    -----
    Preserve every input array. At beta=0 return the maximally mixed state. Numerical work must remain finite in the low-temperature cases. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return density_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_ising_gibbs(couplings: "np.ndarray", fields: "np.ndarray", beta: float) -> "np.ndarray":
    import numpy as np
    from numbers import Real

    j = np.asarray(couplings)
    h = np.asarray(fields)
    if (j.ndim != 1 or h.shape != j.shape or not 2 <= j.size <= 8
            or np.iscomplexobj(j) or np.iscomplexobj(h)):
        raise ValueError("Use equally sized real vectors with 2 through 8 entries.")
    j, h = j.astype(float), h.astype(float)
    if (not np.all(np.isfinite(j)) or not np.all(np.isfinite(h))
            or not isinstance(beta, Real) or not np.isfinite(beta) or beta < 0):
        raise ValueError("Couplings and fields must be finite; beta must be finite and nonnegative.")
    n = j.size
    d = 1 << n
    if beta == 0:
        return np.eye(d, dtype=complex) / d
    basis = np.arange(d)
    spins = 1 - 2 * ((basis[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    diagonal = -np.sum(j * spins * np.roll(spins, -1, axis=1), axis=1)
    hamiltonian = np.diag(diagonal)
    for site in range(n):
        hamiltonian[basis, basis ^ (1 << (n - 1 - site))] -= h[site]
    energies, vectors = np.linalg.eigh(hamiltonian)
    with np.errstate(over="ignore"):
        weights = np.exp(-float(beta) * (energies - energies[0]))
    weights /= weights.sum()
    return np.asarray((vectors * weights) @ vectors.T, dtype=complex)

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
               'args = (np.array([1., .7, -1.2]), np.array([.2, .9, .4]), .8)\n',
      'call': '_invoke(build_ising_gibbs)',
      'gold_call': '_invoke(_oracle_build_ising_gibbs)',
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
               'args = (np.array([1., -.5]), np.array([.6, .2]), 0.)\n',
      'call': '_invoke(build_ising_gibbs)',
      'gold_call': '_invoke(_oracle_build_ising_gibbs)',
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
               'args = (np.array([.4, 1.2, .7, .3]), np.zeros(4), 2.1)\n',
      'call': '_invoke(build_ising_gibbs)',
      'gold_call': '_invoke(_oracle_build_ising_gibbs)',
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
               'args = (np.ones(3), np.zeros(3), 1e4)\n',
      'call': '_invoke(build_ising_gibbs)',
      'gold_call': '_invoke(_oracle_build_ising_gibbs)',
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
               'args = (np.array([.5, 1.3]), np.array([.4, -.8]), 1.7)\n',
      'call': '_invoke(build_ising_gibbs)',
      'gold_call': '_invoke(_oracle_build_ising_gibbs)',
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
               'args = (np.ones(3), np.ones(2), 1.)\n',
      'call': '_invalid(build_ising_gibbs)',
      'gold_call': '_invalid(_oracle_build_ising_gibbs)',
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
               'args = (np.ones(3), np.ones(3), -1.)\n',
      'call': '_invalid(build_ising_gibbs)',
      'gold_call': '_invalid(_oracle_build_ising_gibbs)',
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
               'args = (np.ones(3), np.array([1., np.nan, 0.]), 1.)\n',
      'call': '_invalid(build_ising_gibbs)',
      'gold_call': '_invalid(_oracle_build_ising_gibbs)',
      'tol': 0.0}]
