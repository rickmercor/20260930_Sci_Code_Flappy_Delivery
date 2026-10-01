"""
Evaluate every determinant-one orbital frame through the seven preceding functions and return the frame with the lowest one-step variational energy.

The final step composes the preceding operations. It builds both particle sectors and the electronic Hamiltonian once, then transforms each candidate, expands the row-zero hole states, forms the singular pencil, and compares the projected generalized minima.

Returns
-------
One-based integer index of the candidate with the smallest updated variational energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_orbital_update(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    determinant_orbitals: np.ndarray,
    mixing_candidates: np.ndarray,
    nuclear_energy: float,
    metric_cutoff: float,
) -> int:
    """Select a determinant-one frame for one exact orbital update.

    Parameters
    ----------
    one_body : np.ndarray
        Symmetric spin-orbital one-electron matrix.
    antisym_two_body : np.ndarray
        Antisymmetrized two-electron tensor v[p,q,r,s].
    determinant_orbitals : np.ndarray
        Orbital rows with shape (n_det, n_electrons, n_orbitals).
    mixing_candidates : np.ndarray
        Candidate SL(n) frames with shape
        (n_candidates, n_det, n_electrons, n_electrons).
    nuclear_energy : float
        Scalar additive nuclear energy.
    metric_cutoff : float
        Relative positive-metric eigenvalue cutoff.

    Returns
    -------
    int
        One-based index of the lowest-energy candidate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_orbital_update(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    determinant_orbitals: np.ndarray,
    mixing_candidates: np.ndarray,
    nuclear_energy: float,
    metric_cutoff: float,
) -> int:
    """Compose the seven orbital-update steps and rank candidate frames."""
    import numpy as np

    orbitals = np.asarray(determinant_orbitals, dtype=float)
    candidates = np.asarray(mixing_candidates, dtype=float)

    if (
        orbitals.ndim != 3
        or orbitals.shape[0] < 1
        or orbitals.shape[1] < 1
    ):
        raise ValueError(
            "determinant_orbitals must have shape "
            "(n_det,n_electrons,n_orbitals)"
        )

    n_det, n_electrons, n_orbitals = orbitals.shape
    expected = (n_det, n_electrons, n_electrons)

    if (
        candidates.ndim != 4
        or candidates.shape[0] < 1
        or candidates.shape[1:] != expected
    ):
        raise ValueError(
            "mixing_candidates has an invalid shape"
        )

    hole_basis = _oracle_enumerate_fock_states(
        n_orbitals,
        n_electrons - 1,
    )
    electron_basis = _oracle_enumerate_fock_states(
        n_orbitals,
        n_electrons,
    )

    hamiltonian = _oracle_build_active_space_hamiltonian(
        one_body,
        antisym_two_body,
        electron_basis,
        nuclear_energy,
    )

    ranking = []

    for index, mixing in enumerate(candidates):
        transformed = _oracle_transform_orbital_frames(
            orbitals,
            mixing,
        )

        hole_amplitudes = np.stack(
            [
                _oracle_expand_slater_determinant(
                    transformed[i, 1:, :],
                    hole_basis,
                )
                for i in range(n_det)
            ]
        )

        lift = _oracle_assemble_hole_lift(
            hole_amplitudes,
            hole_basis,
            electron_basis,
            n_orbitals,
        )

        pencil = _oracle_form_eidos_pencil(
            lift,
            hamiltonian,
        )

        solution = _oracle_solve_eidos_update(
            pencil,
            metric_cutoff,
        )

        ranking.append(
            (float(solution[0]), index)
        )

    return int(min(ranking)[1] + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_setup = """import numpy as np
rng = np.random.default_rng(26082026)
m, n, n_det = 8, 4, 3

raw_h = rng.normal(size=(m,m))
one_body = 0.11*(raw_h+raw_h.T) + np.diag(np.linspace(-1.35,1.10,m))

pairs = [(p,q) for p in range(m) for q in range(p+1,m)]
raw_pairs = rng.normal(size=(len(pairs),len(pairs)))
pair_matrix = 0.035*(raw_pairs+raw_pairs.T) + np.diag(np.linspace(0.22,0.58,len(pairs)))
antisym_two_body = np.zeros((m,m,m,m))
for a,(p,q) in enumerate(pairs):
    for b,(r,s) in enumerate(pairs):
        value = pair_matrix[a,b]
        antisym_two_body[p,q,r,s] = value
        antisym_two_body[q,p,r,s] = -value
        antisym_two_body[p,q,s,r] = -value
        antisym_two_body[q,p,s,r] = value

q0,_ = np.linalg.qr(rng.normal(size=(m,n)))
q1,_ = np.linalg.qr(q0 + 0.018*rng.normal(size=(m,n)))
q2,_ = np.linalg.qr(rng.normal(size=(m,n)))
determinant_orbitals = np.stack([q0.T,q1.T,q2.T])

def _frame(candidate_id):
    frame_rng = np.random.default_rng(48151623 + candidate_id)
    scale = 10.0**frame_rng.uniform(-1.2,0.25)
    blocks = []
    for _ in range(n_det):
        block = np.eye(n)
        for _ in range(7):
            i,j = frame_rng.choice(n,size=2,replace=False)
            shear = np.eye(n)
            shear[i,j] = frame_rng.normal(scale=scale)
            block = shear @ block
        blocks.append(block)
    return np.stack(blocks)

candidate_ids = [402,1895,1839,1009,3817,3383,4790,3572,1958,1503]
mixing_candidates = np.stack([_frame(i) for i in candidate_ids])
close_shear = np.eye(n)
close_shear[3,0] = 0.23
mixing_candidates[4,2] = close_shear @ mixing_candidates[4,2]
nuclear_energy = 1.70
metric_cutoff = 1e-13
"""

    call = (
        "select_orbital_update(one_body, antisym_two_body, "
        "determinant_orbitals, mixing_candidates, "
        "nuclear_energy, metric_cutoff)"
    )

    gold_call = (
        "_oracle_select_orbital_update(one_body, antisym_two_body, "
        "determinant_orbitals, mixing_candidates, "
        "nuclear_energy, metric_cutoff)"
    )

    reordered = base_setup + """
order = np.array([5,0,9,4,3,1,6,8,2,7])
mixing_candidates = mixing_candidates[order]
"""

    subset = base_setup + """
keep = np.array([0,1,2,3,4,6,7,8,9])
mixing_candidates = mixing_candidates[keep]
"""

    return [
        {
            "setup": base_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": reordered,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": subset,
            "call": call,
            "gold_call": gold_call,
        },
    ]
