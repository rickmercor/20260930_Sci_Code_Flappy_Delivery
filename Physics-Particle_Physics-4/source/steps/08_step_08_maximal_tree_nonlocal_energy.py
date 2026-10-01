"""
Evaluate the gauge-fixed nonlocal Yang-Mills interaction.

Compose the seven preceding constructions at gauge coupling g=0.91 and residual global frame index 2: the rooted reduction atlas, the QSVT singlet Green and field-amplitude digitization of Equations (9) and (24)-(25), the reduced Gauss response and pulled-back interaction coefficients, the three ordered kernel terms and their sum, the singlet Green matrix, the rooted tree charge-flow map, and the temporal field. Form the fixed momentum source from `(M0+g*M1)@p`, reshape its entries in `b*V+n` order to (8,V), and subtract each row's site mean. For this mean-zero array Qraw, use the single positive scale `s=sqrt(sum_b (1+0.1*b)**2)/norm(Qraw, 'fro')`, b=0,...,7, and set `Q=s*Qraw`. The momentum route is `s**2*(p.T@(C0+g*C1+g**2*C2)@p)`: C0,C1,C2 already contain the factor one half. Apply the scale twice in this quadratic form. Evaluate Equation (8) through the `g**0`, `g**1`, and `g**2` physical-momentum coefficients. Before returning the unrounded scalar, check that value against the independent routes. Write D for the atlas incidence block divided by a, F for the Step-06 flow map, G for the singlet Green matrix, Q for the normalized eight-row source, A0 for the Step-07 temporal field, and flows for Q@F.T. Exactly six identities are checked, each entrywise as `abs(left-right) <= 1e-5 + 1e-5*abs(right)`, in this order: the QSVT Green at degree 5 and the field-amplitude operators at four qubits and unit cutoff obey the singlet and digitization identities (the Green annihilates the constant vector, A endpoints are +-a_max, and A equals the Pauli-Z LCU); every ordered kernel term equals the Gram matrix of its own block of D rows; every colour row of Q sums to zero; flows@D equals Q; then A0 is obtained from `solve_temporal_gauge_field` and must equal -Q@G.T; then the energy equals `0.5*trace(Q@G@Q.T)`, minus one half the sum of Q times A0, and one half the squared norm of the tree flows. Call the temporal solver only after the colour-row sums have been verified to vanish. A source that fails neutrality must raise RuntimeError here; it must not propagate ValueError from Step 07. A nonfinite or zero source norm counts as inconsistent as well. Every one of those failures raises RuntimeError, never ValueError; the input checks named in the Raises section raise ValueError, never RuntimeError. The tolerance accommodates accumulated floating-point error in the composed routes.

Returns
-------
energy : float Unrounded positive total 0.5*sum_b q_b.T@K_plus@q_b.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def maximal_tree_nonlocal_energy(side_length, lattice_spacing) -> float:
    '''Nonlocal interaction energy for the fixed eight-colour maximal-tree source.

    Parameters
    ----------
    side_length : int
        Number L of sites in each open-lattice direction, 2 <= L <= 5.
    lattice_spacing : float
        Finite positive lattice spacing a.

    Returns
    -------
    energy : float
        Unrounded positive total 0.5*sum_b q_b.T@K_plus@q_b.

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded), or lattice_spacing is not finite and positive.
    RuntimeError
        If any of the six independently represented reconstruction,
        tree, kernel, flow, field, and energy identities disagree beyond
        the stated 1e-5 tolerance, or if the reduced response gives a
        nonfinite or zero source norm.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_maximal_tree_nonlocal_energy(side_length, lattice_spacing):
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
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")

    V = L ** 3
    atlas = _oracle_build_maximal_tree_reduction_atlas(L)
    qsvt_green, z_weights, operators = (
        _oracle_assemble_qsvt_field_amplitude_objects(L, a, 5, 4, 1.0, atlas)
    )
    ones = np.ones(V, dtype=float)
    labels = np.arange(16)
    pauli_a = np.zeros((16, 16), dtype=float)
    for bit in range(4):
        pauli_z = np.diag(1.0 - 2.0 * ((labels >> bit) & 1).astype(float))
        pauli_a = pauli_a + z_weights[bit] * pauli_z
    if (
        not np.allclose(qsvt_green @ ones, 0.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0, 0, 0], -1.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0, 15, 15], 1.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0], pauli_a, rtol=1e-5, atol=1e-5)
    ):
        raise RuntimeError("QSVT Green or field-amplitude digitization is inconsistent")
    incidence = atlas[:, 6 : 6 + V]
    D = incidence / a
    terms = _oracle_assemble_maximal_tree_gauss_terms(L, a, atlas)
    K = np.sum(terms, axis=0)
    cuts = (0, L * L * (L - 1), L ** 3 - L)
    stops = (cuts[1], cuts[2], L ** 3 - 1)
    for term, lo, hi in zip(terms, cuts, stops):
        block = D[lo:hi]
        if not np.allclose(term, block.T @ block, rtol=1e-5, atol=1e-5):
            raise RuntimeError("tree-incidence family and Gauss-kernel term disagree")

    F = _oracle_build_ordered_tree_charge_flow_map(L, a, atlas)
    G = _oracle_invert_gauss_kernel_on_singlet(K)
    response, coefficients, momenta = (
        _oracle_build_reduced_nonabelian_gauss_response(L, a, 2, atlas)
    )
    coupling = 0.91
    source_flat = (response[0] + coupling * response[1]) @ momenta
    Q = source_flat.reshape(8, V)
    Q -= np.mean(Q, axis=1, keepdims=True)
    source_amplitude = float(np.max(np.abs(Q)))
    source_norm = (
        source_amplitude * np.linalg.norm(Q / source_amplitude)
        if np.isfinite(source_amplitude) and source_amplitude > 0.0
        else source_amplitude
    )
    target_norm = np.linalg.norm(1.0 + 0.1 * np.arange(8, dtype=float))
    if not np.isfinite(source_norm) or source_norm == 0.0:
        raise RuntimeError("the reduced response produced a degenerate source")
    source_scale = target_norm / source_norm
    Q *= source_scale
    if not np.allclose(np.sum(Q, axis=1), 0.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("source violates the residual global site singlet")
    flows = Q @ F.T
    if not np.allclose(flows @ D, Q, rtol=1e-5, atol=1e-5):
        raise RuntimeError("rooted tree flows do not reproduce neutral charge")

    A0 = _oracle_solve_temporal_gauge_field(K, Q)
    if not np.allclose(A0, -Q @ G.T, rtol=1e-5, atol=1e-5):
        raise RuntimeError("independent temporal-field routes disagree")
    coefficient_matrix = (
        coefficients[0]
        + coupling * coefficients[1]
        + coupling * coupling * coefficients[2]
    )
    # Scaling both factors applies s**2 before a potentially overflowing contraction.
    scaled_momenta = source_scale * momenta
    energy = float(scaled_momenta @ coefficient_matrix @ scaled_momenta)
    green_energy = float(np.einsum("bi,ij,bj->", 0.5 * Q, G, Q))
    temporal_energy = float(np.sum((-0.5 * Q) * A0))
    if not np.allclose(
        energy, green_energy, rtol=1e-5, atol=1e-5
    ) or not np.allclose(
        energy, temporal_energy, rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError("coefficient, Green, and temporal energies disagree")
    if not np.allclose(
        energy, np.sum((0.5 * flows) * flows), rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError("Green and rooted-flow energies disagree")
    return energy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: boundary\nL=2\na=1.0\n',
  'call': 'maximal_tree_nonlocal_energy(L,a)',
  'gold_call': '_oracle_maximal_tree_nonlocal_energy(L,a)'},
 {'setup': 'L=3\na=0.21\n',
  'call': 'maximal_tree_nonlocal_energy(L,a)',
  'gold_call': '_oracle_maximal_tree_nonlocal_energy(L,a)'},
 {'setup': 'L=4\na=0.83\n',
  'call': 'maximal_tree_nonlocal_energy(L,a)',
  'gold_call': '_oracle_maximal_tree_nonlocal_energy(L,a)'},
 {'setup': 'L=5\na=0.38\n',
  'call': 'maximal_tree_nonlocal_energy(L,a)',
  'gold_call': '_oracle_maximal_tree_nonlocal_energy(L,a)'},
 {'setup': 'L=5\na=0.55\n',
  'call': 'maximal_tree_nonlocal_energy(L,a)',
  'gold_call': '_oracle_maximal_tree_nonlocal_energy(L,a)'},
 {'setup': 'import numpy as np\n'
           'def status(fn):\n'
           '    try:\n'
           '        fn(); return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': '(status(lambda: maximal_tree_nonlocal_energy(True, 0.37)), status(lambda: '
          'maximal_tree_nonlocal_energy(1, 0.37)), status(lambda: maximal_tree_nonlocal_energy(6, '
          '0.37)), status(lambda: maximal_tree_nonlocal_energy(5, np.nan)))',
  'gold_call': '(status(lambda: _oracle_maximal_tree_nonlocal_energy(True, 0.37)), status(lambda: '
               '_oracle_maximal_tree_nonlocal_energy(1, 0.37)), status(lambda: '
               '_oracle_maximal_tree_nonlocal_energy(6, 0.37)), status(lambda: '
               '_oracle_maximal_tree_nonlocal_energy(5, np.nan)))'}]
