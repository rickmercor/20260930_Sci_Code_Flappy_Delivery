"""
Compute a prepared-state hole estimate by composing the complete layer-projected workflow.

This end-to-end observable calculation combines interacting operator propagation, layer-aware truncation, and an exact Gaussian preparation pullback. The result is a deterministic finite-projection estimate, not the exact many-body quench probability and not a renormalized retained-string approximation.

Returns
-------
float, the unrounded dimensionless hole-projector estimate after the specified projected quench and exact Gaussian preparation pullback
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_prepared_hole_probability(n_sites: int, edges: "np.ndarray", hopping: "np.ndarray", interaction: float, dt: float, n_steps: int, cap: int, epsilon: float, prep_h: "np.ndarray", prep_time: float, occupation: "np.ndarray", target: int) -> float:
    r"""Compute the layer-projected hole estimate for a Gaussian-prepared quench.

    Parameters
    ----------
    n_sites : int
        Number of spinful sites, 1 to 3.
    edges : numpy.ndarray
        Shape (E, 2) ordered edge list satisfying compile_hubbard_layer.
    hopping : numpy.ndarray
        Real hopping energies of shape (E,).
    interaction : float
        Uniform real on-site Hubbard interaction energy.
    dt : float
        Nonnegative duration of one symmetric layer, in inverse energy units.
    n_steps : int
        Nonnegative number of repeated layers; zero means preparation only.
    cap : int
        Nonnegative endpoint unpaired-Majorana cap; the layer routine supplies its temporary allowance.
    epsilon : float
        Nonnegative coefficient cutoff after each elementary rotation; equality is retained.
    prep_h : numpy.ndarray
        Hermitian one-body preparation Hamiltonian of shape (2*n_sites, 2*n_sites), in energy units. Its matrix exponential defines the preparation without diagonalization conventions.
    prep_time : float
        Real preparation duration; V=exp(-i*prep_time*prep_h) and the many-body preparation is exp(-i*prep_time*sum_ij prep_h[i,j] f_i-dagger f_j).
    occupation : numpy.ndarray
        Binary initial Fock occupations of shape (2*n_sites,).
    target : int
        Zero-based site at which to evaluate the hole projector (1-n_up)(1-n_down).

    Returns
    -------
    hole_estimate : float
        Native Python float, the dimensionless hole-projector expectation defined by the finite projected computation. Return the unrounded estimate without clipping it to [0, 1].

    Raises
    ------
    ValueError
        If n_steps is negative or target is outside the site range; documented ValueError conditions of invoked preceding functions are also propagated (edge order or length mismatch, negative cutoff or cap, nonunitary preparation, or nonbinary occupation).

    Notes
    -----
    Construct the physical hole observable in the same Hermitian Majorana convention, compile the symmetric layer with compile_hubbard_layer, and repeatedly use propagate_trotter_layer. Pull the resulting observable back through the preparation with pull_back_gaussian_state, with no new truncation, and contract using contract_fock_expansion. Elementary propagation uses rotate_and_project_expansion and majorana_product_table through these dependencies. Do not replace the projected calculation by exact state evolution. All arrays are unchanged and no randomness is used.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm

def _oracle_compute_prepared_hole_probability(n_sites: int, edges: 'np.ndarray', hopping: 'np.ndarray', interaction: float, dt: float, n_steps: int, cap: int, epsilon: float, prep_h: 'np.ndarray', prep_time: float, occupation: 'np.ndarray', target: int) -> float:
    """Evaluate the specified deterministic reference operation."""
    if n_steps < 0 or not 0 <= target < n_sites:
        raise ValueError('The layer count or target site is invalid.')
    a = 3 << 4 * target
    b = 3 << 4 * target + 2
    observable = np.array([[0, 0.25], [a, -0.25], [b, -0.25], [a | b, -0.25]], dtype=float)
    gates = _oracle_compile_hubbard_layer(n_sites, edges, hopping, interaction, dt)
    for _ in range(n_steps):
        observable = _oracle_propagate_trotter_layer(observable, gates, 2 * n_sites, epsilon, cap)
    one_body = expm(-1j * prep_time * np.asarray(prep_h))
    pulled = _oracle_pull_back_gaussian_state(observable, one_body)
    return _oracle_contract_fock_expansion(pulled, occupation)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'prompt_benchmark',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 3.7, '
              '0.21, 3, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'gold_call': '_preserving_call(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 3.7, 0.21, 3, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'tol': 5e-10},
     {'name': 'different_quench',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 2.1, '
              '0.17, 2, 2, 1e-4, prep_h.copy(), 0.11, occupation.copy(), 0)',
      'gold_call': '_preserving_call(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 2.1, 0.17, 2, 2, 1e-4, prep_h.copy(), 0.11, occupation.copy(), 0)',
      'tol': 5e-10},
     {'name': 'preparation_only',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 3.7, '
              '0.21, 0, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'gold_call': '_preserving_call(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 3.7, 0.21, 0, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'tol': 5e-10},
     {'name': 'fully_paired_endpoints',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 3.7, '
              '0.21, 3, 0, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'gold_call': '_preserving_call(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 3.7, 0.21, 3, 0, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'tol': 5e-10},
     {'name': 'negative_layer_count',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 3.7, 0.21, '
              '-1, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'gold_call': '_value_error(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 3.7, 0.21, -1, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 1)',
      'tol': 1e-10},
     {'name': 'invalid_target_site',
      'setup': 'import numpy as np\n'
               'edges = np.array([[0, 1], [0, 2]], dtype=int)\n'
               'hopping = np.array([1.0, 0.73])\n'
               'B = np.array([[0.4, 1.0, 0.2j], [1.0, -0.3, 0.6], [-0.2j, 0.6, 0.2]], dtype=complex)\n'
               'prep_h = np.kron(B, np.eye(2))\n'
               'occupation = np.array([1, 0, 0, 1, 1, 0], dtype=int)\n'
               '\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(compute_prepared_hole_probability, 3, edges.copy(), hopping.copy(), 3.7, 0.21, '
              '3, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 3)',
      'gold_call': '_value_error(_oracle_compute_prepared_hole_probability, 3, edges.copy(), '
                   'hopping.copy(), 3.7, 0.21, 3, 2, 2e-4, prep_h.copy(), 0.23, occupation.copy(), 3)',
      'tol': 1e-10}]
