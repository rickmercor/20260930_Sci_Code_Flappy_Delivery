"""
Extract the rooted charge-to-link flow map from the reduction atlas.

Figure 1 of the source paper roots the maximal tree at the origin n0, but the paper writes down no orientation, descendant set or charge-to-link map. Orienting each tree link away from that root, and representing the resulting reconstruction by descendant indicators, are task conventions derived from that tree. Step 01 stores the corresponding descendant indicators S in the final V columns of its atlas, in ordered n3/n2/n1 family order. For `V=L**3`, read S from `atlas[:,6+V:6+2*V]` and preserve the atlas row order. Every row has value one on the complete descendant subtree below that oriented link and zero elsewhere. Return `F=a*S` without centring its rows, changing its orientation, or replacing it by an incidence pseudoinverse. If B=atlas[:,6:6+V] is the dimensionless incidence block and D=B/a, the rooted orientation gives the exact full-space identity `D.T@F = I - e_origin 1.T.` The root term vanishes on globally neutral site charges, so D.T@F@q=q whenever sum(q)=0. This sign and uncentred convention also fixes the tree-flow energy used in Step 08.

Returns
-------
flow_map : np.ndarray, shape (L**3 - 1, L**3), float Ordered map whose rows are +a on each oriented link's descendant subtree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_ordered_tree_charge_flow_map(
    side_length, lattice_spacing, tree_atlas
) -> np.ndarray:
    '''Rooted charge-to-link reconstruction map for the ordered maximal tree.

    Parameters
    ----------
    side_length : int
        Number L of sites per open-lattice direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    flow_map : np.ndarray, shape (L**3 - 1, L**3), float
        Ordered map whose rows are +a on each oriented link's descendant subtree.

    Raises
    ------
    ValueError
        If a lattice argument is invalid, or the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.zeros((int(side_length) ** 3 - 1, int(side_length) ** 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_ordered_tree_charge_flow_map(
    side_length, lattice_spacing, tree_atlas
):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    V = L ** 3
    try:
        raw_atlas = np.asarray(tree_atlas)
        if np.iscomplexobj(raw_atlas):
            if not np.all(np.isfinite(raw_atlas)) or np.any(raw_atlas.imag != 0.0):
                raise ValueError("tree atlas must be finite and real")
            raw_atlas = raw_atlas.real
        atlas = np.asarray(raw_atlas, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")
    return a * atlas[:, 6 + V : 6 + 2 * V]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\nL=3\na=0.41\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_ordered_tree_charge_flow_map(L,a,T)',
  'gold_call': '_oracle_build_ordered_tree_charge_flow_map(L,a,T)'},
 {'setup': '# case: boundary\nL=2\na=1.0\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_ordered_tree_charge_flow_map(L,a,T)',
  'gold_call': '_oracle_build_ordered_tree_charge_flow_map(L,a,T)'},
 {'setup': '# case: edge\nL=5\na=0.19\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_ordered_tree_charge_flow_map(L,a,T)',
  'gold_call': '_oracle_build_ordered_tree_charge_flow_map(L,a,T)'},
 {'setup': '# case: normal\nL=4\na=1.7\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_ordered_tree_charge_flow_map(L,a,T)',
  'gold_call': '_oracle_build_ordered_tree_charge_flow_map(L,a,T)'},
 {'setup': '# case: edge\nL=5\na=0.73\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_ordered_tree_charge_flow_map(L,a,T)',
  'gold_call': '_oracle_build_ordered_tree_charge_flow_map(L,a,T)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'L=3\n'
           'T=build_maximal_tree_reduction_atlas(L)\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except Exception: return 2\n',
  'call': 'tuple((status(fn) for fn in (lambda: build_ordered_tree_charge_flow_map(True, 0.4, T), lambda: '
          'build_ordered_tree_charge_flow_map(1, 0.4, T), lambda: build_ordered_tree_charge_flow_map(6, 0.4, '
          'T), lambda: build_ordered_tree_charge_flow_map(3, -0.1, T), lambda: '
          'build_ordered_tree_charge_flow_map(3, np.nan, T), lambda: build_ordered_tree_charge_flow_map(3, '
          '0.4, T[:1]))))',
  'gold_call': 'tuple((status(fn) for fn in (lambda: _oracle_build_ordered_tree_charge_flow_map(True, 0.4, '
               'T), lambda: _oracle_build_ordered_tree_charge_flow_map(1, 0.4, T), lambda: '
               '_oracle_build_ordered_tree_charge_flow_map(6, 0.4, T), lambda: '
               '_oracle_build_ordered_tree_charge_flow_map(3, -0.1, T), lambda: '
               '_oracle_build_ordered_tree_charge_flow_map(3, np.nan, T), lambda: '
               '_oracle_build_ordered_tree_charge_flow_map(3, 0.4, T[:1]))))'}]
