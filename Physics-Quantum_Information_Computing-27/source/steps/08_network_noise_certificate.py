"""
Compute the finite-budget triangle-network white-noise certificate for a six-spin thermal state.

A feasible mixed network state is refined by deterministic local support searches followed by bounded-memory convex corrections. The inner resource and gate optimizations are sequential, while the support residual stays frozen during a trial. Accepted candidates immediately alter the corrected mixture used by the next trial. The endpoint is a constructive upper bound on sufficient white noise, with no claim of convergence to the full network boundary.

Returns
-------
float, unrounded dimensionless sufficient noise fraction after the complete ordered triangle-network calculation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def network_noise_certificate(couplings: "np.ndarray", fields: "np.ndarray", beta: float, seed_ids: "np.ndarray", sweep_counts: "np.ndarray", capacity: int) -> float:
    """Compute the finite-budget triangle-network white-noise certificate for a six-spin thermal state.

    Parameters
    ----------
    couplings : np.ndarray
        Six finite real nearest-neighbor bond strengths, in periodic site order.
    fields : np.ndarray
        Six finite real transverse fields, with site 0 the most significant bit.
    beta : float
        Finite nonnegative inverse temperature.
    seed_ids : np.ndarray
        One-dimensional integer array of seed labels in [0,1000], in trial order.
        Repetition is permitted. An empty integer array specifies no trials.
    sweep_counts : np.ndarray
        Integer array matching seed_ids, with each entry in [0,20].
    capacity : int
        Retained pure-state memory capacity from 1 through 5; booleans are invalid.

    Returns
    -------
    noise_fraction : float
        Native unrounded Python float for the sufficient white-noise fraction
        after precisely the requested finite sequence of trials.

    Raises
    ------
    ValueError
        If the six-spin Hamiltonian, integer schedules, capacity or upstream numerical contracts are violated.

    Notes
    -----
    Use P=X tensor Y and Q=Z tensor I. For seed k and edge e=0,1,2, normalize
    (1, (k+1+e)/10+i*(e+1)/7, (2*e-k)/9-i/5, 0.8-i*(k+e)/13).
    Initialize U_a=exp[-i*(k+a+1)*pi*P/13] exp[-i*(2*k-a+2)*pi*Q/17].
    The right factor acts first. Start the anchor at I/64 with empty memory.
    Within each trial freeze target-anchor; perform the stated number of sweeps.
    A sweep updates edges 0,1,2, then parties 0,1,2 with P then Q at each party.
    Apply every resource/gate replacement immediately. Contract the candidate,
    admit it to memory with the strict descent rule, and project using the
    current anchor and returned memory. Recompute the residual for the next trial.
    A rejected candidate leaves the state unchanged under exact projection.
    Return the final constructive noise certificate without rounding.

    Compose the earlier public functions build_ising_gibbs,
    contract_triangle_state, optimize_resource_link, optimize_local_rotation,
    project_network_memory, admit_network_atom and certify_network_noise.
    Preserve all caller-owned input arrays. Every trial reinitializes its
    resources and gates from its own seed; it does not continue the previous
    trial's local circuit. Import needed packages inside the completed function; no numerical module aliases are prebound.
    """
    return noise_fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_network_noise_certificate(couplings: "np.ndarray", fields: "np.ndarray", beta: float, seed_ids: "np.ndarray", sweep_counts: "np.ndarray", capacity: int) -> float:
    import numpy as np
    from numbers import Integral

    j, h = np.asarray(couplings), np.asarray(fields)
    ids, sweeps = np.asarray(seed_ids), np.asarray(sweep_counts)
    if j.shape != (6,) or h.shape != (6,):
        raise ValueError("The triangle benchmark requires six couplings and six fields.")
    if (ids.ndim != 1 or sweeps.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(sweeps.dtype, np.integer) or np.any(ids < 0) or np.any(sweeps < 0)
            or np.any(ids > 1000) or np.any(sweeps > 20)):
        raise ValueError("Use integer seed IDs in [0,1000] and matching sweep counts in [0,20].")
    if not isinstance(capacity, Integral) or isinstance(capacity, (bool, np.bool_)) or not 1 <= capacity <= 5:
        raise ValueError("Capacity must be an integer from 1 through 5.")
    rho = _oracle_build_ising_gibbs(j, h, beta)
    current = np.eye(64, dtype=complex) / 64
    memory = np.empty((0, 64, 64), dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    p = np.kron(x, y)
    q = np.kron(z, np.eye(2))
    for k, count in zip(ids, sweeps):
        residual = rho - current
        links = np.array([[1, (k + 1 + e) / 10 + 1j * (e + 1) / 7,
                           (2 * e - k) / 9 - 1j / 5, 0.8 - 1j * (k + e) / 13]
                          for e in range(3)], dtype=complex)
        links /= np.linalg.norm(links, axis=1)[:, None]
        unitaries = np.empty((3, 4, 4), dtype=complex)
        for party in range(3):
            alpha = (k + party + 1) * np.pi / 13
            eta = (2 * k - party + 2) * np.pi / 17
            unitaries[party] = ((np.cos(alpha) * np.eye(4) - 1j * np.sin(alpha) * p)
                                @ (np.cos(eta) * np.eye(4) - 1j * np.sin(eta) * q))
        for _ in range(int(count)):
            for edge in range(3):
                links[edge] = _oracle_optimize_resource_link(residual, links, unitaries, edge)
            for party in range(3):
                for generator in (p, q):
                    unitaries[party] = _oracle_optimize_local_rotation(
                        residual, links, unitaries, party, generator)
        psi = _oracle_contract_triangle_state(links, unitaries)
        candidate = np.outer(psi, psi.conj())
        memory = _oracle_admit_network_atom(rho, current, memory, candidate, capacity)
        current = _oracle_project_network_memory(rho, current, memory)
    return _oracle_certify_network_noise(rho, current)

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
               'args = (np.array([1.,.8,1.1,.6,1.2,.9]), np.array([.6,.9,1.,.7,.8,1.1]), .6, '
               'np.arange(4), np.array([1,0,2,1]), 2)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.ones(6), np.full(6,.7), .9, np.array([],dtype=int), np.array([],dtype=int), '
               '4)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.ones(6), np.ones(6), .7, np.array([3]), np.array([2]), 1)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.array([1.,-.4,.8,.6,1.1,.7]), np.full(6,.8), .3, np.array([2,0,2]), '
               'np.array([1,2,0]), 1)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.ones(6), np.full(6,.6), 50., np.array([0,4,7]), np.array([1,1,1]), 2)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.ones(6), np.ones(6), 0., np.array([4,0]), np.array([1,2]), 2)\n',
      'call': '_invoke(network_noise_certificate)',
      'gold_call': '_invoke(_oracle_network_noise_certificate)',
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
               'args = (np.ones(6), np.ones(6), .5, np.array([0,1]), np.array([1]), 2)\n',
      'call': '_invalid(network_noise_certificate)',
      'gold_call': '_invalid(_oracle_network_noise_certificate)',
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
               'args = (np.ones(5), np.ones(5), .5, np.array([0]), np.array([1]), 2)\n',
      'call': '_invalid(network_noise_certificate)',
      'gold_call': '_invalid(_oracle_network_noise_certificate)',
      'tol': 0.0}]
