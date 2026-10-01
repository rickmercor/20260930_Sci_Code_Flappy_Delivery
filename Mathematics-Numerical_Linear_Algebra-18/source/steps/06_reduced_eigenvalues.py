"""
Compute the eigenvalues of the reduced system obtained by projecting the augmented system back to the dimension of its unaugmented part.

Kp and Mp are the partitioned matrices, T is the augmented transformation, and nCB is the number of leading columns of T that constitute its unaugmented part.

The augmented pencil is formed by congruence with T. Its leading nCB eigenvectors are retained and used to project it, and the resulting pencil of dimension nCB is solved. Eigenvalues that are not strictly positive are unphysical and are discarded before the ascending ordering is applied.

The function returns the retained eigenvalues as a one-dimensional float array in ascending order.

Raises ValueError if T is not two dimensional with one row per equation of the system; or if nCB is below one or exceeds the number of columns of T.

Appending vectors to a reduction basis enlarges the subspace and therefore improves the reduced model, but it also enlarges the reduced model, which defeats the purpose if the comparison of interest is against a model of the original size. A secondary projection resolves this. The enlarged pencil is solved, its lowest eigenvectors are retained, and the pencil is projected onto them. The result has the smaller dimension while carrying the information from the enlarged basis, and the eigenvalues of the retained part are preserved exactly by the operation.

Projections of this kind inherit a variational structure. Each basis considered here spans a subspace of the next, so the reduced eigenvalues decrease as the basis is enlarged and are bounded below by the exact eigenvalues of the unreduced system. Every eigenvalue that the procedure can legitimately produce is therefore positive.

That bound is also a diagnostic. Enlarging a basis with vectors of very different character can make the projected mass matrix severely ill conditioned, and an eigenvalue solver applied to such a pencil may return values with no physical meaning, including negative ones. Because the ordering theorem forbids them, they can be identified and discarded rather than being carried into the ascending ordering, where they would displace every subsequent index.

Returns
-------
np.ndarray, one dimensional, dtype float, ascending
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduced_eigenvalues(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray',
                        nCB: int) -> 'np.ndarray':
    '''Eigenvalues of the secondary-projected reduced system.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    T : np.ndarray
        (n, m) augmented transformation.
    nCB : int
        Number of leading columns of T forming its unaugmented part.

    Returns
    -------
    np.ndarray
        One-dimensional float array of retained eigenvalues, ascending.
    '''
    return eigenvalues  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reduced_eigenvalues(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray',
                                nCB: int) -> 'np.ndarray':
    import numpy as np
    from scipy.linalg import eigh
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != Kp.shape[0]:
        raise ValueError("T must have one row per equation of the system")
    if not 1 <= nCB <= T.shape[1]:
        raise ValueError("nCB out of range")
    Ka = T.T @ Kp @ T
    Ma = T.T @ Mp @ T
    w, V = eigh(Ka, Ma, subset_by_index=[0, nCB-1])
    Ts = V[:, :nCB]
    lb, _ = eigh(Ts.T @ Ka @ Ts, Ts.T @ Ma @ Ts)
    return np.sort(lb[lb > 0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    valid = """import numpy as np
def cand(m, ncb):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += constraint_modes(Kp, counts, k)
        res += residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    return reduced_eigenvalues(Kp, Mp, T[:, :m], ncb)
def gold(m, ncb):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = _oracle_assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = _oracle_fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += _oracle_constraint_modes(Kp, counts, k)
        res += _oracle_residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    return _oracle_reduced_eigenvalues(Kp, Mp, T[:, :m], ncb)
"""
    invalid = """import numpy as np
def run_model(rows, ncb):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += constraint_modes(Kp, counts, k)
        res += residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    try:
        reduced_eigenvalues(Kp, Mp, T[:rows, :], ncb)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(rows, ncb):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = _oracle_assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = _oracle_fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += _oracle_constraint_modes(Kp, counts, k)
        res += _oracle_residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    try:
        _oracle_reduced_eigenvalues(Kp, Mp, T[:rows, :], ncb)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: the surrogate augmented basis ---
        {
            "setup": valid,
            "call": "cand(None, 62)",
            "gold_call": "gold(None, 62)",
            "tol": 1e-7,
        },
        # --- boundary: unaugmented basis only ---
        {
            "setup": valid,
            "call": "cand(62, 62)",
            "gold_call": "gold(62, 62)",
            "tol": 1e-7,
        },
        # --- edge: projection dimension above the unaugmented size ---
        {
            "setup": valid,
            "call": "cand(None, 80)",
            "gold_call": "gold(None, 80)",
            "tol": 1e-7,
        },
        # --- invalid: projection dimension of zero ---
        {
            "setup": invalid,
            "call": "run_model(None, 0)",
            "gold_call": "run_gold(None, 0)",
        },
        # --- invalid: basis of the wrong height ---
        {
            "setup": invalid,
            "call": "run_model(10, 62)",
            "gold_call": "run_gold(10, 62)",
        },
    ]
