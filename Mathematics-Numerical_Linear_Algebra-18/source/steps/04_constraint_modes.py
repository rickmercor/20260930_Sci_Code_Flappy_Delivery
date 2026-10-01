"""
Compute the static response of the interior of substructure k to unit displacement at each interface degree of freedom in turn, the remaining interface degrees of freedom being held at zero.

Kp is the partitioned stiffness matrix and counts holds the four block sizes of the partitioned ordering. k selects the substructure.

The function returns a float array with one row per equation of the full partitioned system and one column per interface degree of freedom. Rows outside the interior of substructure k are zero. In particular the interface rows of the returned array are zero: the identity belonging on those rows is supplied once when the three substructure blocks are combined, not once per substructure.

Raises ValueError if counts does not have four entries; if counts does not sum to the number of equations in Kp; or if k is not 0, 1 or 2.

The fixed-interface modes describe how a substructure moves when its interface does not. They cannot by themselves represent motion of the assembled structure, because in the assembled structure the interface does move. A second set of vectors supplies exactly that missing part: the static deformation of the interior produced by displacing the interface.

These vectors are obtained by solving the interior equilibrium equations with a prescribed interface displacement and no interior loading. One vector is generated per interface degree of freedom. Because the response is static, no approximation is committed in this part of the representation: any interface motion is reproduced exactly, together with the interior deformation it induces, and rigid body motion of the whole substructure is contained in their span.

Taken together with the retained normal modes, these vectors span the subspace onto which the substructure is projected. The division of labour is clean, the static part being exact and the modal part truncated, which is why the accuracy of the reduction is governed entirely by the truncation of the latter.

Returns
-------
np.ndarray of shape (n, nb), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constraint_modes(Kp: 'np.ndarray', counts: 'np.ndarray', k: int) -> 'np.ndarray':
    '''Static interior response to unit interface displacement.

    Parameters
    ----------
    Kp : np.ndarray
        (n, n) partitioned stiffness matrix.
    counts : np.ndarray
        (4,) integer block sizes of the partitioned ordering.
    k : int
        Substructure index, 0, 1 or 2.

    Returns
    -------
    np.ndarray
        (n, nb) float array with nb the interface size, zero outside the interior
        rows of substructure k and zero on the interface rows.
    '''
    return static_response  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_constraint_modes(Kp: 'np.ndarray', counts: 'np.ndarray', k: int) -> 'np.ndarray':
    import numpy as np
    counts = np.asarray(counts, dtype=int)
    if counts.size != 4:
        raise ValueError("counts must have four entries")
    if int(counts.sum()) != Kp.shape[0]:
        raise ValueError("counts do not sum to the system size")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    o = np.concatenate([[0], np.cumsum(counts)])
    ii = slice(o[k], o[k+1])
    bb = slice(o[3], o[4])
    out = np.zeros((Kp.shape[0], counts[3]))
    out[ii, :] = -np.linalg.solve(Kp[ii, ii], Kp[ii, bb])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    valid = """import numpy as np
def cand(k):
    counts = np.array([24, 24, 48, 48])
    S = assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    return constraint_modes(S[0], counts, k)
def gold(k):
    counts = np.array([24, 24, 48, 48])
    S = _oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    return _oracle_constraint_modes(S[0], counts, k)
"""
    invalid = """import numpy as np
def run_model(blocks, k):
    counts = np.array(blocks)
    S = assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    try:
        constraint_modes(S[0], counts, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(blocks, k):
    counts = np.array(blocks)
    S = _oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    try:
        _oracle_constraint_modes(S[0], counts, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: first substructure ---
        {
            "setup": valid,
            "call": "cand(0)",
            "gold_call": "gold(0)",
        },
        # --- boundary: third substructure, the one carrying the free end ---
        {
            "setup": valid,
            "call": "cand(2)",
            "gold_call": "gold(2)",
        },
        # --- edge: middle substructure, which touches both interface layers ---
        {
            "setup": valid,
            "call": "cand(1)",
            "gold_call": "gold(1)",
        },
        # --- invalid: negative substructure index ---
        {
            "setup": invalid,
            "call": "run_model((24, 24, 48, 48), -1)",
            "gold_call": "run_gold((24, 24, 48, 48), -1)",
        },
        # --- invalid: three block sizes instead of four ---
        {
            "setup": invalid,
            "call": "run_model((24, 24, 96), 0)",
            "gold_call": "run_gold((24, 24, 96), 0)",
        },
    ]
