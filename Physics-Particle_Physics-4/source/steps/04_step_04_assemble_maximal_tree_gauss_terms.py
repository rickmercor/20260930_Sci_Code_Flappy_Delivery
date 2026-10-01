"""
Assemble the three ordered Gauss-kernel terms from the tree atlas.

The source's hierarchical gauge is A_3=0 throughout the volume, A_2=0 on the n3=0 plane, and A_1=0 on the n2=n3=0 line. Its section on the Hamiltonian of Yang-Mills theory and gauge fixing gives the resulting Gauss kernel, which contains, in this order, -`partial_3**2` in the volume, -delta_(n3,0) `partial_2**2` on the boundary plane, and -delta_(n2,0)delta_(n3,0) `partial_1**2` on the boundary line. The preceding atlas stores the oriented dimensionless incidence matrix B in columns 6:6+V and family codes 3,2,1 in column 0. For family i, construct `K_i=B_i.T@B_i/a**2`. Return the three K_i in the order 3,2,1, without summing. This incidence Gram construction implements the one-sided endpoint rule of the source's endnote on the definition of derivatives and Gauss's law: active path endpoints have diagonal `1/a**2`, while the interior diagonal is `2/a**2`. Keep the constant global zero mode and n1-fastest site ordering.

Returns
-------
terms : np.ndarray, shape (3, L**3, L**3), float The three positive source-kernel terms, ordered n3, n2, n1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_maximal_tree_gauss_terms(
    side_length, lattice_spacing, tree_atlas
) -> np.ndarray:
    '''Ordered derivative terms in the maximal-tree-gauge Gauss kernel.

    Parameters
    ----------
    side_length : int
        Number L of sites on each side of the open cubic lattice, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    terms : np.ndarray, shape (3, L**3, L**3), float
        The three positive source-kernel terms, ordered n3, n2, n1.

    Raises
    ------
    ValueError
        If a lattice argument is invalid, or the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.zeros((3, int(side_length) ** 3, int(side_length) ** 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_maximal_tree_gauss_terms(
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

    incidence = atlas[:, 6 : 6 + V]
    terms = np.empty((3, V, V), dtype=float)
    for output_index, family in enumerate((3, 2, 1)):
        block = incidence[np.rint(atlas[:, 0]).astype(int) == family]
        derivative = block / a
        terms[output_index] = derivative.T @ derivative
    return terms

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\nL=3\na=0.41\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'assemble_maximal_tree_gauss_terms(L,a,T)',
  'gold_call': '_oracle_assemble_maximal_tree_gauss_terms(L,a,T)'},
 {'setup': '# case: boundary\nL=2\na=1.0\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'assemble_maximal_tree_gauss_terms(L,a,T)',
  'gold_call': '_oracle_assemble_maximal_tree_gauss_terms(L,a,T)'},
 {'setup': '# case: edge\nL=5\na=0.19\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'assemble_maximal_tree_gauss_terms(L,a,T)',
  'gold_call': '_oracle_assemble_maximal_tree_gauss_terms(L,a,T)'},
 {'setup': '# case: normal\nL=4\na=1.7\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'assemble_maximal_tree_gauss_terms(L,a,T)',
  'gold_call': '_oracle_assemble_maximal_tree_gauss_terms(L,a,T)'},
 {'setup': '# case: edge\nL=5\na=0.73\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'assemble_maximal_tree_gauss_terms(L,a,T)',
  'gold_call': '_oracle_assemble_maximal_tree_gauss_terms(L,a,T)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'L=3\n'
           'T=build_maximal_tree_reduction_atlas(L)\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except Exception: return 2\n',
  'call': 'tuple((status(fn) for fn in (lambda: assemble_maximal_tree_gauss_terms(True, 0.4, T), lambda: '
          'assemble_maximal_tree_gauss_terms(1, 0.4, T), lambda: assemble_maximal_tree_gauss_terms(6, 0.4, '
          'T), lambda: assemble_maximal_tree_gauss_terms(3, 0.0, T), lambda: '
          'assemble_maximal_tree_gauss_terms(3, np.nan, T), lambda: assemble_maximal_tree_gauss_terms(3, '
          '0.4, T[:1]))))',
  'gold_call': 'tuple((status(fn) for fn in (lambda: _oracle_assemble_maximal_tree_gauss_terms(True, 0.4, '
               'T), lambda: _oracle_assemble_maximal_tree_gauss_terms(1, 0.4, T), lambda: '
               '_oracle_assemble_maximal_tree_gauss_terms(6, 0.4, T), lambda: '
               '_oracle_assemble_maximal_tree_gauss_terms(3, 0.0, T), lambda: '
               '_oracle_assemble_maximal_tree_gauss_terms(3, np.nan, T), lambda: '
               '_oracle_assemble_maximal_tree_gauss_terms(3, 0.4, T[:1]))))'}]
