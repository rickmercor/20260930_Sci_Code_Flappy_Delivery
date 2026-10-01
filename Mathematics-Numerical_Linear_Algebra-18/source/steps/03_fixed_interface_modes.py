"""
Compute the retained normal modes of substructure k with its entire interface held fixed.

Kp and Mp are the partitioned stiffness and mass matrices. counts holds the four block sizes of the partitioned ordering, in the order established by the assembly step. k selects the substructure and nd is the number of modes to retain.

Modes are taken in ascending order of eigenvalue and normalised so that the interior mass block reduces to the identity over them. Each column is signed so that its entry of largest magnitude is positive, which removes the arbitrary sign of an eigenvector.

The function returns a float array with one row per equation of the full partitioned system and nd columns. Rows outside substructure k are zero, so blocks belonging to different substructures can be added or concatenated without overlapping.

Raises ValueError if counts does not have four entries; if counts does not sum to the number of equations in Kp; if k is not 0, 1 or 2; or if nd is below one or exceeds the size of substructure k.

Holding the interface of a substructure fixed and solving the resulting eigenproblem yields a set of vectors describing how that part vibrates in isolation. Because the interface is restrained, these modes vanish on it and describe purely internal motion. They are orthogonal with respect to both the stiffness and the mass of the interior block, which is why a normalisation convention can reduce one of the two to the identity.

Only a small number of these modes is kept. Which ones are kept is not arbitrary: the lower modes carry the deformation patterns that dominate the global dynamic response, so retaining them in ascending order of eigenvalue extracts the most accuracy from a given number of vectors. The discarded modes are not negligible, but their influence on the lowest global frequencies is small, and recovering part of it is the purpose of the later steps.

Eigenvectors are determined only up to scale and sign where the corresponding eigenvalue is simple, and up to an arbitrary rotation within any set of coincident eigenvalues. Scale is fixed by the normalisation. The sign convention below is a reproducibility aid for this implementation rather than a property a caller can rely on, because the largest-magnitude entry of a mode of a symmetric structure is generally attained at several degrees of freedom at once. No quantity computed from a congruence transformation depends on either choice.

Returns
-------
np.ndarray of shape (n, nd), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fixed_interface_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray',
                          k: int, nd: int) -> 'np.ndarray':
    '''Retained fixed-interface normal modes of one substructure.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    counts : np.ndarray
        (4,) integer block sizes of the partitioned ordering.
    k : int
        Substructure index, 0, 1 or 2.
    nd : int
        Number of modes to retain.

    Returns
    -------
    np.ndarray
        (n, nd) float array, zero outside the rows of substructure k.
    '''
    return retained_modes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fixed_interface_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray',
                                  k: int, nd: int) -> 'np.ndarray':
    import numpy as np
    from scipy.linalg import eigh
    counts = np.asarray(counts, dtype=int)
    if counts.size != 4:
        raise ValueError("counts must have four entries")
    if int(counts.sum()) != Kp.shape[0]:
        raise ValueError("counts do not sum to the system size")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    o = np.concatenate([[0], np.cumsum(counts)])
    if not 1 <= nd <= counts[k]:
        raise ValueError("nd out of range")
    ii = slice(o[k], o[k+1])
    w, V = eigh(Kp[ii, ii], Mp[ii, ii], subset_by_index=[0, nd-1])
    for c in range(nd):
        if V[np.argmax(np.abs(V[:, c])), c] < 0:
            V[:, c] *= -1.0
    out = np.zeros((Kp.shape[0], nd))
    out[ii, :] = V
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    valid = """import numpy as np
def cand(k, nd):
    counts = np.array([24, 24, 48, 48])
    S = assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    P = fixed_interface_modes(S[0], S[1], counts, k, nd)
    return P @ P.T
def gold(k, nd):
    counts = np.array([24, 24, 48, 48])
    S = _oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    P = _oracle_fixed_interface_modes(S[0], S[1], counts, k, nd)
    return P @ P.T
"""
    invalid = """import numpy as np
def run_model(blocks, k, nd):
    counts = np.array(blocks)
    S = assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    try:
        fixed_interface_modes(S[0], S[1], counts, k, nd)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(blocks, k, nd):
    counts = np.array(blocks)
    S = _oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    try:
        _oracle_fixed_interface_modes(S[0], S[1], counts, k, nd)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: first substructure, three retained modes ---
        {
            "setup": valid,
            "call": "cand(0, 3)",
            "gold_call": "gold(0, 3)",
            "tol": 1e-7,
        },
        # --- boundary: third substructure, a single retained mode ---
        {
            "setup": valid,
            "call": "cand(2, 1)",
            "gold_call": "gold(2, 1)",
            "tol": 1e-7,
        },
        # --- edge: every mode of the substructure retained ---
        {
            "setup": valid,
            "call": "cand(0, 24)",
            "gold_call": "gold(0, 24)",
            "tol": 1e-7,
        },
        # --- invalid: substructure index out of range ---
        {
            "setup": invalid,
            "call": "run_model((24, 24, 48, 48), 3, 2)",
            "gold_call": "run_gold((24, 24, 48, 48), 3, 2)",
        },
        # --- invalid: zero retained modes ---
        {
            "setup": invalid,
            "call": "run_model((24, 24, 48, 48), 0, 0)",
            "gold_call": "run_gold((24, 24, 48, 48), 0, 0)",
        },
        # --- invalid: block sizes do not sum to the system size ---
        {
            "setup": invalid,
            "call": "run_model((24, 24, 48, 47), 0, 2)",
            "gold_call": "run_gold((24, 24, 48, 47), 0, 2)",
        },
    ]
