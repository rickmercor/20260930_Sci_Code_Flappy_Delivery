"""
Evaluate a subsystem selected-space total energy including the molecular scalar offset.

Combine the effective one-body and two-body terms with both scalar contributions in the prescribed selected determinant space.

Returns
-------
native Python float: the lowest selected-space eigenvalue plus the downfolded scalar offset.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selected_energy(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", determinants: "np.ndarray") -> float:
    """Evaluate a complete downfolded subsystem total energy.

    Parameters
    ----------
    v, eps, occupied, active : np.ndarray
        Reference inputs: chemists' tensor, fixed orbital energies,
        doubly occupied reference indices and ordered active indices.
    determinants : np.ndarray
        Distinct packed determinants in the LOCAL active orbital order.
        Each has 2*len(intersection(occupied,active)) electrons.

    Use screened_interaction, corrected_one_body, reference_offset,
    and correlation_offset. Assemble the selected Hamiltonian in the
    supplied determinant order and return its lowest eigenvalue plus
    the two scalar contributions, each included once. Packed spin
    orbitals put all local alpha orbitals before all local beta orbitals.
    Do not enlarge the determinant space or change reference energies.

    Returns
    -------
    native Python float: the lowest selected-space eigenvalue plus the downfolded scalar offset.

    Raises
    ------
    ValueError
    If v or eps has complex dtype (even with zero imaginary parts),
    or cannot be converted to real finite numeric arrays;
    if eps is not a nonempty vector or v does not have shape (n,n,n,n),
    where n=len(eps); if v lacks p/q exchange, r/s exchange, or
    pair-exchange symmetry (absolute tolerance 1e-12, relative
    tolerance zero); if occupied is not a one-dimensional vector of
    distinct integer indices in [0,n) (an empty vector is allowed);
    if any occupied-to-virtual gap is nonpositive; or if a computed
    reference quantity is nonfinite.
    Also if active is not a nonempty one-dimensional vector of distinct
    integer indices in [0,n); if the full, constrained, or screened
    active-space RPA problem violates the RPA stability/finite
    result conditions stated below; or if a downfolded result is nonfinite.
    RPA stability means every eigenvalue of D^(1/2)(D+4K)D^(1/2) is positive;
    all RPA couplings must be finite and symmetric within atol=1e-12,
    rtol=0, and their eigendecompositions must succeed.
    For the selected-Hamiltonian assembly, let m=len(active), and let
    t_eff and W denote the downfolded one- and two-electron arrays.
    Raise if m is outside 1<=m<=8; if t_eff or W is not real and finite;
    if t_eff is not symmetric with shape (m,m); if W does not have
    shape (m,m,m,m) or lacks p/q, r/s, or pair-exchange symmetry
    (all symmetry checks use atol=1e-12, rtol=0); if determinants is
    not a nonempty vector of distinct integers in [0,2**(2*m)); if
    their total electron counts differ; or if the computed matrix
    contains a nonfinite entry.
    Different spin projections with the same total electron count
    are allowed, as is the vacuum determinant 0.
    Also if any determinant has an electron count other than twice
    the number of occupied indices included in active, if the final
    eigensolver fails, or if the final energy is nonfinite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_selected_energy(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", determinants: "np.ndarray") -> float:
    import numpy as np
    w = _oracle_screened_interaction(v, eps, occupied, active)
    t = _oracle_corrected_one_body(v, eps, occupied, active, w)
    e_ref = _oracle_reference_offset(v, eps, occupied, active, w)
    e_corr = _oracle_correlation_offset(v, eps, occupied, active, w)
    ham = _qc38_selected_hamiltonian(t, w, determinants)
    required = 2*len(set(int(i) for i in occupied) & set(int(i) for i in active))
    if any(int(d).bit_count() != required for d in determinants):
        raise ValueError('Incorrect active electron count.')
    try:
        value = float(np.linalg.eigvalsh(ham)[0] + (e_ref + e_corr))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Selected eigenproblem failed.') from exc
    if not np.isfinite(value):
        raise ValueError('Nonfinite energy.')
    return value

def _qc38_selected_hamiltonian(h: "np.ndarray", v: "np.ndarray", determinants: "np.ndarray") -> "np.ndarray":
    import numpy as np
    try:
        if np.iscomplexobj(h) or np.iscomplexobj(v):
            raise ValueError('Real inputs required.')
        h = np.asarray(h, dtype=float)
        v = np.asarray(v, dtype=float)
        ds = np.asarray(determinants)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric input.') from exc
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= h.shape[0] <= 8:
        raise ValueError('h must be square with 1 to 8 orbitals.')
    n = len(h)
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, atol=1e-12, rtol=0):
        raise ValueError('h must be finite and symmetric.')
    if v.shape != (n, n, n, n) or not np.all(np.isfinite(v)):
        raise ValueError('Invalid interaction tensor.')
    if any(not np.allclose(v, w, atol=1e-12, rtol=0) for w in
           (v.swapaxes(0, 1), v.swapaxes(2, 3), v.transpose(2, 3, 0, 1))):
        raise ValueError('Invalid Coulomb symmetries.')
    if ds.ndim != 1 or ds.size == 0 or ds.dtype.kind not in 'iu':
        raise ValueError('Determinants must be a nonempty integer vector.')
    if np.any(ds < 0) or np.any(ds >= 1 << (2*n)) or len(np.unique(ds)) != len(ds):
        raise ValueError('Invalid or repeated determinants.')
    if len({int(d).bit_count() for d in ds}) != 1:
        raise ValueError('Determinants must have the same total electron count.')
    index = {int(d): i for i, d in enumerate(ds)}
    out = np.zeros((len(ds), len(ds)))
    for col, det in enumerate(ds):
        for p, q in np.ndindex(n, n):
            for spin in (0, 1):
                bits, sign = int(det), 1
                for pos, create in ((q+spin*n, False), (p+spin*n, True)):
                    if bool((bits >> pos) & 1) == create:
                        sign = 0; break
                    sign *= -1 if (bits & ((1 << pos)-1)).bit_count() % 2 else 1
                    bits ^= 1 << pos
                if sign and bits in index:
                    out[index[bits], col] += sign*h[p, q]
        for p, q, r, s in np.ndindex(n, n, n, n):
            for spin in (0, 1):
                for tau in (0, 1):
                    bits, sign = int(det), 1
                    for pos, create in ((q+spin*n, False), (s+tau*n, False),
                                        (r+tau*n, True), (p+spin*n, True)):
                        if bool((bits >> pos) & 1) == create:
                            sign = 0; break
                        sign *= -1 if (bits & ((1 << pos)-1)).bit_count() % 2 else 1
                        bits ^= 1 << pos
                    if sign and bits in index:
                        out[index[bits], col] += 0.5*sign*v[p, q, r, s]
    if not np.all(np.isfinite(out)):
        raise ValueError('Nonfinite Hamiltonian.')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base0 = """
import numpy as np
M = np.array([[0.5, 0.16, 0.3, 0.12], [0.16, 0.36, 0.09, 0.14], [0.3, 0.09, 0.3, 0.11], [0.12, 0.14, 0.11, 0.42]])
N = np.array([[0.2, -0.08, 0.22, 0.09], [-0.08, 0.28, 0.07, -0.1], [0.22, 0.07, 0.24, 0.13], [0.09, -0.1, 0.13, 0.32]])
R = np.diag([0.12, 0.08, 0.05, 0.04])
S = 0.03 * np.outer([1, -1, 2, 1], [2, 1, -1, 1])
factors = np.array([np.block([[M, R], [R.T, 0.9 * M]]), np.block([[N, S], [S.T, 1.1 * N]])])
eps = np.array([-0.8, -0.55, -0.35, 1.1, -0.75, -0.48, -0.29, 1.25])
v = np.einsum('lpq,lrs->pqrs', factors, factors)
occupied = np.array([0, 4], dtype=int)
active = np.array([0, 1, 2, 4, 5, 6], dtype=int)
samples = np.array([[9, 9], [9, 10], [9, 17], [9, 18], [12, 12], [33, 33], [9, 3], [9, 24]], dtype=int)
"""
    base1 = """
import numpy as np
factors = np.array([[[0.4, 0.12], [0.12, 0.3]], [[0.1, -0.04], [-0.04, 0.2]]])
v = np.einsum('lpq,lrs->pqrs', factors, factors)
eps = np.array([-0.7, 0.2])
occupied = np.array([0], dtype=int)
active = np.array([0, 1], dtype=int)
"""
    cases = []
    cases.append({
        "setup": base0 + """

v = v[:4, :4, :4, :4]
eps = eps[:4]
occupied = np.array([0])
active = np.array([0, 1, 2])
dets = np.array([9, 36, 17, 10, 3, 24])
""",
        'call': 'selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
        'gold_call': '_oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
    })
    cases.append({
        "setup": base1 + """

dets = np.array([5, 6, 9, 10])
""",
        'call': 'selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
        'gold_call': '_oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
    })
    cases.append({
        "setup": base0 + """

active = np.array([4, 5, 6])
dets = np.array([9, 36, 17, 10, 3, 24])
""",
        'call': 'selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
        'gold_call': '_oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
    })
    cases.append({
        "setup": base1 + """

active = np.array([1])
dets = np.array([0])
""",
        'call': 'selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
        'gold_call': '_oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
    })
    cases.append({
        "setup": base1 + """

v = np.zeros_like(v)
dets = np.array([5, 10])
""",
        'call': 'selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
        'gold_call': '_oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())',
    })
    cases.append({
        "setup": base1 + """

dets = np.array([1, 2])

def run_model():
    try:
        selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    cases.append({
        "setup": base1 + """

dets = np.array([], dtype=int)

def run_model():
    try:
        selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_selected_energy(v.copy(), eps.copy(), occupied.copy(), active.copy(), dets.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    return cases
