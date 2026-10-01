"""
Convert the potential of the joint run spanned by two adjacent vertices into the normalised conditional distribution of the second vertex phasing given the first.

Two adjacent vertices of the SNP line graph share exactly one SNP, so together they span one longer run whose phasings are the only objects that can hold both of their phasings at once. Marginalising the joint potential over the runs compatible with a given pair of endpoint phasings, then normalising over the second endpoint, turns a factor into a transition.

Returns
-------
np.ndarray of shape (Mp, Mc), float: the row-normalised conditional distribution of the child phasing given the parent phasing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_transition_matrix(parent_positions: np.ndarray,
                            parent_phasings: np.ndarray,
                            child_positions: np.ndarray,
                            child_phasings: np.ndarray,
                            joint_positions: np.ndarray,
                            joint_phasings: np.ndarray,
                            joint_potentials: np.ndarray) -> np.ndarray:
    """Build the conditional distribution of a child phasing given a parent one.

    A joint phasing supports the pair (i, j) when the columns it holds at the
    parent positions reproduce parent phasing i as a multiset of haplotype rows
    and, at the same time, the columns it holds at the child positions
    reproduce child phasing j as a multiset of haplotype rows. Rows of the
    result whose total support vanishes are set to the uniform distribution.

    Parameters
    ----------
    parent_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the parent vertex.
    parent_phasings : np.ndarray
        Integer array of shape (Mp, K, Pp) of canonical parent phasings, whose
        rows are sorted in ascending lexicographic order.
    child_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the child vertex.
    child_phasings : np.ndarray
        Integer array of shape (Mc, K, Pc) of canonical child phasings, whose
        rows are sorted in ascending lexicographic order.
    joint_positions : np.ndarray
        One-dimensional integer array of the SNP indices of the joint run; it
        contains every parent and every child position.
    joint_phasings : np.ndarray
        Integer array of shape (Mj, K, Pj) of canonical phasings of the joint
        run, in the same column order as ``joint_positions``.
    joint_potentials : np.ndarray
        Array of shape (Mj,) of non-negative floats, the potential of each
        joint phasing.

    Returns
    -------
    transition : np.ndarray
        Array of shape (Mp, Mc) of non-negative floats whose rows each sum to
        one; entry (i, j) is the probability of child phasing j given parent
        phasing i.

    Raises
    ------
    ValueError
        If any phasing array is not a three-dimensional integer-valued array
        with entries in {0, 1}, if the three ploidies disagree, if any position
        array is not one-dimensional with distinct entries matching the width
        of its phasings, if the parent or child positions are not all contained
        in ``joint_positions``, or if ``joint_potentials`` is not a
        one-dimensional array of finite non-negative entries of length Mj.
    """
    return transition  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_transition_matrix(parent_positions: np.ndarray,
                                    parent_phasings: np.ndarray,
                                    child_positions: np.ndarray,
                                    child_phasings: np.ndarray,
                                    joint_positions: np.ndarray,
                                    joint_phasings: np.ndarray,
                                    joint_potentials: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _positions(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.allclose(array, np.round(array), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} entries must be integer valued")
        array = np.round(array).astype(int)
        if np.unique(array).size != array.size:
            raise ValueError(f"{name} entries must be distinct")
        return array

    def _phasings(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 3 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty three-dimensional array")
        if not np.all(np.isin(array, (0.0, 1.0))):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(int)

    def _canonical(matrix):
        return matrix[np.lexsort(matrix[:, ::-1].T)]

    parent_cols = _positions(parent_positions, "parent_positions")
    child_cols = _positions(child_positions, "child_positions")
    joint_cols = _positions(joint_positions, "joint_positions")
    parents = _phasings(parent_phasings, "parent_phasings")
    children = _phasings(child_phasings, "child_phasings")
    joints = _phasings(joint_phasings, "joint_phasings")

    if not (parents.shape[1] == children.shape[1] == joints.shape[1]):
        raise ValueError("the three phasing arrays must share the same ploidy")
    if parents.shape[2] != parent_cols.size or children.shape[2] != child_cols.size:
        raise ValueError("each phasing width must match its position array")
    if joints.shape[2] != joint_cols.size:
        raise ValueError("the joint phasing width must match joint_positions")
    if not (set(parent_cols.tolist()) <= set(joint_cols.tolist())
            and set(child_cols.tolist()) <= set(joint_cols.tolist())):
        raise ValueError("parent and child positions must be contained in joint_positions")

    weights = np.asarray(joint_potentials, dtype=float)
    if weights.ndim != 1 or weights.size != joints.shape[0]:
        raise ValueError("joint_potentials must be one-dimensional of length Mj")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("joint_potentials entries must be finite and non-negative")

    order = joint_cols.tolist()
    parent_index = [order.index(int(c)) for c in parent_cols]
    child_index = [order.index(int(c)) for c in child_cols]

    parent_key = {_canonical(p).tobytes(): i for i, p in enumerate(parents)}
    child_key = {_canonical(c).tobytes(): j for j, c in enumerate(children)}

    support = np.zeros((parents.shape[0], children.shape[0]), dtype=float)
    for m, joint in enumerate(joints):
        i = parent_key.get(_canonical(joint[:, parent_index]).tobytes())
        j = child_key.get(_canonical(joint[:, child_index]).tobytes())
        if i is not None and j is not None:
            support[i, j] += weights[m]

    row_sums = support.sum(axis=1)
    transition = np.empty_like(support)
    uniform = np.full(children.shape[0], 1.0 / float(children.shape[0]))
    for i in range(parents.shape[0]):
        transition[i] = support[i] / row_sums[i] if row_sums[i] > 0.0 else uniform

    return transition

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a tetraploid pair of adjacent vertices sharing SNP 1, with
        #     read-derived joint potentials (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                            [[0, 0], [0, 1], [1, 0], [1, 1]],
                            [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                           [[0, 0], [0, 1], [1, 0], [1, 1]],
                           [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
                           [[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]],
                           [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([0.5, 1.5, 2.5, 0.25, 3.0, 0.75, 1.0, 2.0, 0.125])
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Valid: the same geometry with a flat joint potential, the case of a
        #     joint run that no fragment covers completely ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                            [[0, 0], [0, 1], [1, 0], [1, 1]],
                            [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                           [[0, 0], [0, 1], [1, 0], [1, 1]],
                           [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
                           [[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]],
                           [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=int)
joint_potentials = np.ones(9)
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Valid: vertices sharing their lower SNP rather than the middle one,
        #     so the shared column sits at a different place in the joint run ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([0, 1])
child_positions = np.array([0, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                            [[0, 0], [0, 1], [1, 0], [1, 1]],
                            [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                           [[0, 0], [0, 1], [1, 0], [1, 1]],
                           [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
                           [[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]],
                           [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([0.5, 1.5, 2.5, 0.25, 3.0, 0.75, 1.0, 2.0, 0.125])
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Boundary: a joint potential that vanishes on every run supporting
        #     one parent phasing, whose row must fall back to the uniform law ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                            [[0, 0], [0, 1], [1, 0], [1, 1]],
                            [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                           [[0, 0], [0, 1], [1, 0], [1, 1]],
                           [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
                           [[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]],
                           [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Edge: a parent occupying the upper two columns of the joint run, so
        #     that most joint phasings project onto it out of canonical row order
        #     and the projections have to be reordered before they are matched ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([1, 2])
child_positions = np.array([0, 1])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                            [[0, 0], [0, 1], [1, 0], [1, 1]],
                            [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]],
                           [[0, 0], [0, 1], [1, 0], [1, 1]],
                           [[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                           [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
                           [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]],
                           [[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]],
                           [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([0.5, 1.5, 2.5, 0.25, 3.0, 0.75, 1.0, 2.0,
                             0.125, 4.0, 0.625, 1.75])
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Edge: a diploid pair, where each vertex carries a single phasing
        #     and the transition collapses to a one-by-one matrix ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 1, 0], [1, 0, 1]],
                           [[0, 1, 1], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([2.0, 5.0])
""",
            "call": "sig(build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
            "gold_call": "sig(_oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials), 1.0)",
        },
        # --- Invalid: a child position that the joint run does not contain ---
        {
            "setup": """import numpy as np
parent_positions = np.array([0, 1])
child_positions = np.array([1, 7])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=int)
joint_potentials = np.array([1.0])
def run_model():
    try:
        build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative joint potential ---
        {
            "setup": """import numpy as np
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 1, 0], [1, 0, 1]],
                           [[0, 1, 1], [1, 0, 0]]], dtype=int)
joint_potentials = np.array([1.0, -0.5])
def run_model():
    try:
        build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: phasing arrays that disagree on the ploidy ---
        {
            "setup": """import numpy as np
parent_positions = np.array([0, 1])
child_positions = np.array([1, 2])
joint_positions = np.array([0, 1, 2])
parent_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
child_phasings = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
joint_phasings = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=int)
joint_potentials = np.array([1.0])
def run_model():
    try:
        build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_transition_matrix(parent_positions, parent_phasings, child_positions, child_phasings, joint_positions, joint_phasings, joint_potentials)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
