"""
Extract the part of one reduced mode that is carried by the unaugmented columns of the basis.

Kp, Mp, T and nCB are as in the preceding step. mode_index is the one-based position of the requested mode in the ascending ordering of the retained reduced eigenvalues.

The mode is expanded to the coordinates of the full partitioned system, and only the contribution of the first nCB columns of T is kept. This is not the unaugmented part of T applied to the reduced mode, because the secondary projection mixes the unaugmented and augmented coordinates; the contribution must be taken after that mixing. The result is signed so that its entry of largest magnitude is positive.

The function returns a one-dimensional float array with one entry per equation of the full partitioned system.

Raises ValueError if T is not two dimensional with one row per equation of the system; if nCB is below one or exceeds the number of columns of T; or if mode_index is below one or exceeds the number of retained eigenvalues.

A reduced mode is a vector of generalised coordinates. To interpret it physically it must be expanded through the basis back to the coordinates of the original system, whereupon it becomes an approximation to a mode shape of the unreduced structure.

When the basis is a concatenation of distinguishable groups of columns, the expanded vector decomposes into the contributions of those groups, and the decomposition is informative: it separates what the original basis could represent from what the appended vectors added. Comparing a quantity computed from one part alone against the same quantity computed from the whole isolates the effect of the enlargement.

The decomposition has to respect the secondary projection. That projection is a change of coordinates in the enlarged space, and it does not preserve the separation between the groups: a reduced coordinate after projection is a combination of both. Splitting the basis first and applying the projected mode to one part only is therefore not the same operation as applying the whole basis and then splitting, and the two give different vectors.

Returns
-------
np.ndarray of shape (n,), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def base_contribution(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray', nCB: int,
                      mode_index: int) -> 'np.ndarray':
    '''Unaugmented part of one expanded reduced mode.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    T : np.ndarray
        (n, m) augmented transformation.
    nCB : int
        Number of leading columns of T forming its unaugmented part.
    mode_index : int
        One-based index into the ascending retained eigenvalues.

    Returns
    -------
    np.ndarray
        (n,) float array.
    '''
    return base_vector  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_base_contribution(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray', nCB: int,
                              mode_index: int) -> 'np.ndarray':
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
    lb, Vb = eigh(Ts.T @ Ka @ Ts, Ts.T @ Ma @ Ts)
    keep = lb > 0
    lb = lb[keep]; Vb = Vb[:, keep]
    if not 1 <= mode_index <= lb.size:
        raise ValueError("mode_index out of range")
    q = Ts @ Vb[:, mode_index-1]
    u0 = T[:, :nCB] @ q[:nCB]
    if u0[np.argmax(np.abs(u0))] < 0:
        u0 = -u0
    return u0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    valid = """import numpy as np
def cand(mode):
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
    return base_contribution(Kp, Mp, T, 62, mode)
def gold(mode):
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
    return _oracle_base_contribution(Kp, Mp, T, 62, mode)
"""
    invalid = """import numpy as np
def run_model(mode):
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
        base_contribution(Kp, Mp, T, 62, mode)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(mode):
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
        _oracle_base_contribution(Kp, Mp, T, 62, mode)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: the fifth mode of the surrogate ---
        {
            "setup": valid,
            "call": "cand(5)",
            "gold_call": "gold(5)",
            "tol": 1e-7,
        },
        # --- boundary: the lowest mode ---
        {
            "setup": valid,
            "call": "cand(1)",
            "gold_call": "gold(1)",
            "tol": 1e-7,
        },
        # --- edge: a high mode of the surrogate ---
        {
            "setup": valid,
            "call": "cand(41)",
            "gold_call": "gold(41)",
            "tol": 1e-7,
        },
        # --- invalid: mode index of zero ---
        {
            "setup": invalid,
            "call": "run_model(0)",
            "gold_call": "run_gold(0)",
        },
        # --- invalid: mode index beyond the retained set ---
        {
            "setup": invalid,
            "call": "run_model(100000)",
            "gold_call": "run_gold(100000)",
        },
    ]
