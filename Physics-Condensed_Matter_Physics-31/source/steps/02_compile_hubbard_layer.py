"""
Compile a spinful Hubbard layer into a fixed chronological sequence of elementary rotations.

Fermionic hopping and on-site repulsion have different Majorana decompositions. A symmetric composition must preserve their coefficients, Hermitian-gauge signs, and microgate ordering because intermediate projections can make otherwise equivalent factorizations numerically inequivalent.

Returns
-------
numpy.ndarray of shape (8E + 3L, 2), the integer-valued generator masks and dimensionless rotation angles in state-chronological order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compile_hubbard_layer(n_sites: int, edges: "np.ndarray", hopping: "np.ndarray", interaction: float, dt: float) -> "np.ndarray":
    r"""Compile a symmetric Hubbard layer into chronological Majorana rotations.

    Parameters
    ----------
    n_sites : int
        Number of spinful sites, from 1 to 3; mode m = 2*site + spin, with up=0 and down=1.
    edges : numpy.ndarray
        Integer array of shape (E, 2), with 0 <= i < j < n_sites. Edge rows are chronological and need not be lexicographically sorted. An empty (0, 2) array is allowed.
    hopping : numpy.ndarray
        Real hopping energies of shape (E,), paired with the edge rows; the Hamiltonian uses minus hopping times the Hermitian hopping operator.
    interaction : float
        Real on-site Hubbard interaction energy, uniform over sites; either sign and zero are supported.
    dt : float
        Nonnegative layer duration, in inverse energy units with hbar=1.

    Returns
    -------
    gates : numpy.ndarray
        Float64 array of shape (8*E + 3*n_sites, 2), with columns [generator mask, rotation angle]. Row r denotes exp(-i*angle*mu(mask)/2); row 0 acts first on a state. Zero-angle rows remain present.

    Raises
    ------
    ValueError
        If an edge violates i < j or the number of hopping energies differs from the number of edge rows.

    Notes
    -----
    The first hopping half-layer visits edges in supplied order, then up and down spins, and for each mode pair i<j visits supports (2*i,2*j+1) and (2*i+1,2*j), in that order. The middle interaction block visits sites in increasing order and supports (up pair), (down pair), (up pair plus down pair), in that order. The last hopping half-layer reverses the entire first microgate list. Each hopping half-layer lasts dt/2; the interaction block lasts dt. Constant Hamiltonian terms are omitted. Use the Hermitian gauge defined by majorana_product_table. Inputs remain unchanged; there is no RNG.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compile_hubbard_layer(n_sites: int, edges: 'np.ndarray', hopping: 'np.ndarray', interaction: float, dt: float) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    edges = np.asarray(edges)
    hopping = np.asarray(hopping, dtype=float)
    if len(edges) != len(hopping) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError('Edge order or hopping length is invalid.')
    half = []
    for (x, y), t in zip(edges, hopping):
        for spin in range(2):
            i = 2 * int(x) + spin
            j = 2 * int(y) + spin
            half.extend([[1 << 2 * i | 1 << 2 * j + 1, -t * dt / 2], [1 << 2 * i + 1 | 1 << 2 * j, t * dt / 2]])
    center = []
    for site in range(n_sites):
        a = 3 << 4 * site
        b = 3 << 4 * site + 2
        center.extend([[a, interaction * dt / 2], [b, interaction * dt / 2], [a | b, -interaction * dt / 2]])
    return np.array(half + center + half[::-1], float).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'corner_cluster',
      'setup': 'import numpy as np\n'
               'e=np.array([[0,1],[0,2]],dtype=int)\n'
               't=np.array([1.,.73])\n'
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
      'call': '_preserving_call(compile_hubbard_layer, 3, e.copy(), t.copy(), 3.7, 0.21)',
      'gold_call': '_preserving_call(_oracle_compile_hubbard_layer, 3, e.copy(), t.copy(), 3.7, 0.21)',
      'tol': 1e-10},
     {'name': 'zero_duration',
      'setup': 'import numpy as np\n'
               'e=np.array([[0,1],[0,2]],dtype=int)\n'
               't=np.array([1.,.73])\n'
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
      'call': '_preserving_call(compile_hubbard_layer, 3, e.copy(), t.copy(), 3.7, 0.0)',
      'gold_call': '_preserving_call(_oracle_compile_hubbard_layer, 3, e.copy(), t.copy(), 3.7, 0.0)',
      'tol': 1e-10},
     {'name': 'isolated_attractive_site',
      'setup': 'import numpy as np\n'
               'e=np.empty((0,2),int)\n'
               't=np.empty(0)\n'
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
      'call': '_preserving_call(compile_hubbard_layer, 1, e.copy(), t.copy(), -2.3, 0.4)',
      'gold_call': '_preserving_call(_oracle_compile_hubbard_layer, 1, e.copy(), t.copy(), -2.3, 0.4)',
      'tol': 1e-10},
     {'name': 'signed_hopping_order',
      'setup': 'import numpy as np\n'
               'e=np.array([[1,2],[0,1]])\n'
               't=np.array([-.8,.6])\n'
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
      'call': '_preserving_call(compile_hubbard_layer, 3, e.copy(), t.copy(), 0.0, 0.13)',
      'gold_call': '_preserving_call(_oracle_compile_hubbard_layer, 3, e.copy(), t.copy(), 0.0, 0.13)',
      'tol': 1e-10},
     {'name': 'reversed_edge',
      'setup': 'import numpy as np\n'
               'e=np.array([[1,0]])\n'
               't=np.array([1.])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(compile_hubbard_layer, 2, e.copy(), t.copy(), 1.0, 0.1)',
      'gold_call': '_value_error(_oracle_compile_hubbard_layer, 2, e.copy(), t.copy(), 1.0, 0.1)',
      'tol': 1e-10},
     {'name': 'mismatched_hoppings',
      'setup': 'import numpy as np\n'
               'e=np.array([[0,1],[0,2]],dtype=int)\n'
               't=np.array([1.,.73])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(compile_hubbard_layer, 3, e.copy(), np.array([1.]), 1.0, 0.1)',
      'gold_call': '_value_error(_oracle_compile_hubbard_layer, 3, e.copy(), np.array([1.]), 1.0, 0.1)',
      'tol': 1e-10}]
