"""
Orchestrate the two-source screened, size-consistent selected-CI interaction energy.

The dimer and two fragments use independent reference Hamiltonians built from their restricted bare integrals. Determinant spaces are derived jointly from fixed dimer samples. The final observable is E_AB-E_A-E_B, including every scalar offset.

Returns
-------
native Python float: E_AB-E_A-E_B, with no rounding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interaction_energy(factors: "np.ndarray", eps: "np.ndarray", split: int, occupied: "np.ndarray", active: "np.ndarray", samples: "np.ndarray") -> float:
    """Compute the complete screened selected-CI interaction energy.

    Parameters
    ----------
    factors : np.ndarray
        Real symmetric Coulomb factors (r,n,n); v[p,q,r,s] is their
        Gram contraction sum_l factors[l,p,q]*factors[l,r,s].
    eps : np.ndarray
        Prescribed canonical energies (n,).
    split : int
        Full spatial labels [0,split) belong to A; [split,n) to B.
    occupied : np.ndarray
        Global doubly occupied reference labels, with no duplicates.
    active : np.ndarray
        Ordered active labels, A entries followed by B entries;
        internal fragment ordering need not be ascending.
    samples : np.ndarray
        Fixed raw (alpha_mask,beta_mask) rows whose bit positions refer
        to positions in active, NOT global orbital labels.

    Use selected_spaces at the active reference fragment electron
    counts. Restrict the BARE tensor and eps independently to A, B,
    and AB before each downfolding; each subsystem constructs its own
    reference h. Remap occupied/active labels into each subsystem's
    local full-orbital numbering. Solve each prescribed selected space
    with selected_energy, and return E_AB-E_A-E_B. Nuclear energies
    are zero. Do not extract fragment energies from an already
    screened dimer tensor, impose fragment Ms=0, or substitute full CI.

    Returns
    -------
    native Python float: E_AB-E_A-E_B, with no rounding.

    Raises
    ------
    ValueError
    If factors or eps has complex dtype (even with zero imaginary
    parts). If factors is not a real finite array (r,n,n) with 2<=n<=12
    and symmetric factors (atol=1e-12, rtol=0); r=0 is permitted.
    If eps is not a real finite vector of length n; if split is not
    an integer in [1,n) (booleans invalid); if occupied is not a vector
    of distinct integer indices in [0,n), or a reference occupied-to-
    virtual gap is nonpositive. If active is not a vector of 2 to 8
    distinct integer indices in [0,n), omits a fragment, or fails to
    put all A entries before all B entries. If samples is not a
    nonempty integer (k,2) array of in-range active masks; if any row
    does not have Nalpha=Nbeta equal to the number of active occupied
    reference orbitals; or if spin completion produces no neutral
    configuration at the active reference fragment electron counts.
    If any required full/constrained/active RPA problem is not strictly
    stable, an eigendecomposition fails, a required Coulomb or matrix
    symmetry fails at atol=1e-12, rtol=0, or any computed intermediate
    or energy is nonfinite. Strict RPA stability means positive
    eigenvalues of D^(1/2)(D+4K)D^(1/2) in the spatial convention.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_interaction_energy(factors: "np.ndarray", eps: "np.ndarray", split: int, occupied: "np.ndarray", active: "np.ndarray", samples: "np.ndarray") -> float:
    import numpy as np
    try:
        if np.iscomplexobj(factors):
            raise ValueError('Real factors required.')
        factors = np.asarray(factors, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid factors.') from exc
    if factors.ndim != 3 or factors.shape[1] != factors.shape[2] or not 2 <= factors.shape[1] <= 12:
        raise ValueError('Factors require shape (r,n,n), 2<=n<=12.')
    if not np.all(np.isfinite(factors)) or not np.allclose(factors, factors.swapaxes(1, 2), atol=1e-12, rtol=0):
        raise ValueError('Factors must be finite and symmetric.')
    n = factors.shape[1]
    if isinstance(split, (bool, np.bool_)) or not isinstance(split, (int, np.integer)) or not 1 <= split < n:
        raise ValueError('Invalid fragment split.')
    v = np.einsum('lpq,lrs->pqrs', factors, factors)
    _qc38_reference_data(v, eps, occupied)
    try:
        ax = np.asarray(active)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid active indices.') from exc
    if ax.ndim != 1 or not 2 <= ax.size <= 8 or ax.dtype.kind not in 'iu':
        raise ValueError('Active indices require 2 to 8 integer entries.')
    if np.any(ax < 0) or np.any(ax >= n) or len(np.unique(ax)) != len(ax):
        raise ValueError('Invalid active indices.')
    aa = [int(i) for i in ax if i < split]
    ab = [int(i) for i in ax if i >= split]
    if not aa or not ab or list(ax) != aa+ab:
        raise ValueError('Active A orbitals must precede active B orbitals.')
    occ = [int(i) for i in occupied]
    n_a_e = 2*len(set(aa) & set(occ))
    n_active_occ = len(set(int(i) for i in ax) & set(occ))
    sa, sb, sab = _oracle_selected_spaces(len(aa), len(ab), n_a_e, samples)
    for alpha, beta in np.asarray(samples):
        if int(alpha).bit_count() != n_active_occ or int(beta).bit_count() != n_active_occ:
            raise ValueError('Samples disagree with active reference spin counts.')
    energies = []
    for inds, act, dets in ((list(range(split)), aa, sa),
                            (list(range(split, n)), ab, sb),
                            (list(range(n)), list(ax), sab)):
        local_occ = np.array([inds.index(i) for i in occ if i in inds], dtype=int)
        local_act = np.array([inds.index(i) for i in act], dtype=int)
        energies.append(_oracle_selected_energy(v[np.ix_(inds, inds, inds, inds)],
                        np.asarray(eps)[inds], local_occ, local_act, dets))
    return float(energies[2]-energies[0]-energies[1])

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

""",
        'call': 'interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
        'gold_call': '_oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
    })
    cases.append({
        "setup": base0 + """

fa = np.zeros((4, 8, 8))
fa[:2, :4, :4] = factors[:, :4, :4]
fa[2:, 4:, 4:] = factors[:, 4:, 4:]
factors = fa
""",
        'call': 'interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
        'gold_call': '_oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
    })
    cases.append({
        "setup": base0 + """

perm = np.array([2, 0, 1, 5, 3, 4])
active = active[perm]
samples = np.array([[sum(((int(mask) >> int(old) & 1) << new for new, old in enumerate(perm))) for mask in row] for row in samples], dtype=int)
""",
        'call': 'interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
        'gold_call': '_oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
    })
    cases.append({
        "setup": base0 + """

samples = np.vstack([samples[::-1], samples[:3]])
""",
        'call': 'interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
        'gold_call': '_oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
    })
    cases.append({
        "setup": base0 + """

factors = np.zeros((0, 8, 8))
""",
        'call': 'interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
        'gold_call': '_oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())',
    })
    cases.append({
        "setup": base0 + """

samples = samples[:1].copy()
samples[0, 0] = 1

def run_model():
    try:
        interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    cases.append({
        "setup": base0 + """

active = np.array([0, 4, 1, 5, 2, 6])

def run_model():
    try:
        interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_interaction_energy(factors.copy(), eps.copy(), 4, occupied.copy(), active.copy(), samples.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    return cases
