"""
A compatible node can be reused because its reach is contained in the target carrier set. An exact reach receives the mutation without a topology change. Otherwise, compatible nodes are considered by decreasing reach size and retained only when their reaches are disjoint from those already selected; any carriers left uncovered remain direct leaf children of a new mutation node.

Inputs

------

candidate_lists: Compatible internal node identifiers for every update.

descendant_lists: Reach D(n) for every snapshot node.

target_carriers: Binary target carrier matrix.

n_samples: Number of sample leaves.

Returns

-------

attachment_plan: Exact matches, selected candidates, uncovered leaves, target lists, and creation flags.

Returns
-------
dict of native integer lists describing each exact match or greedy disjoint cover
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def plan_greedy_attachments(
    candidate_lists: list,
    descendant_lists: list,
    target_carriers: np.ndarray,
    n_samples: int,
) -> dict:
    """Plan deterministic reuse-aware attachments on a fixed snapshot.

    Parameters
    ----------
    candidate_lists : list
        Compatible internal node identifiers for each update.
    descendant_lists : list
        Reach list for each node in the discovery snapshot.
    target_carriers : np.ndarray
        Binary carrier matrix with one row per update.
    n_samples : int
        Number of sample leaves.

    Raises
    ------
    ValueError
        If `target_carriers` has the wrong shape, is empty, or is not binary;
        if `n_samples` is invalid; if the candidate count differs from the
        update count; if a target carrier set is empty; if a candidate
        identifier is out of range; or if a candidate reach is not contained
        in its target.

    Returns
    -------
    attachment_plan : dict
        Native numeric lists describing exact reuse or new-node children.
    """
    return attachment_plan  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_plan_greedy_attachments(
    candidate_lists: list,
    descendant_lists: list,
    target_carriers: np.ndarray,
    n_samples: int,
) -> dict:
    """Reference implementation."""
    targets = np.asarray(target_carriers)
    if targets.ndim != 2 or targets.shape[1] != n_samples or targets.shape[0] == 0:
        raise ValueError("target_carriers has the wrong shape")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("target_carriers must be binary")
    if len(candidate_lists) != targets.shape[0]:
        raise ValueError("candidate_lists must have one entry per update")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be positive")

    exact_nodes = []
    selected_all = []
    uncovered_all = []
    target_lists = []
    creates_node = []
    for update, raw_candidates in enumerate(candidate_lists):
        target = set(np.flatnonzero(targets[update]).astype(int).tolist())
        if not target:
            raise ValueError("target carrier sets must be nonempty")
        candidates = sorted(set(int(node) for node in raw_candidates))
        if any(node < n_samples or node >= len(descendant_lists) for node in candidates):
            raise ValueError("candidate identifiers must name snapshot internal nodes")
        reaches = {node: set(int(sample) for sample in descendant_lists[node]) for node in candidates}
        if any(not reach.issubset(target) for reach in reaches.values()):
            raise ValueError("every candidate reach must be contained in its target")

        exact = [node for node in candidates if reaches[node] == target]
        target_lists.append(sorted(target))
        if exact:
            exact_nodes.append(min(exact))
            selected_all.append([])
            uncovered_all.append([])
            creates_node.append(0)
            continue

        covered = set()
        selected = []
        for node in sorted(candidates, key=lambda value: (-len(reaches[value]), value)):
            if covered.isdisjoint(reaches[node]):
                selected.append(node)
                covered.update(reaches[node])
        exact_nodes.append(-1)
        selected_all.append(selected)
        uncovered_all.append(sorted(target.difference(covered)))
        creates_node.append(1)

    return {
        "exact_nodes": exact_nodes,
        "selected_candidates": selected_all,
        "uncovered_samples": uncovered_all,
        "target_lists": target_lists,
        "creates_node": creates_node,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = '''
def _pack_lol(rows):
    out = [float(len(rows))]
    for row in rows:
        out.append(float(len(row)))
        out.extend(float(v) for v in row)
    return out
def _pack_plan(p):
    out = [float(len(p["exact_nodes"]))]
    out.extend(float(v) for v in p["exact_nodes"])
    for key in ("selected_candidates", "uncovered_samples", "target_lists"):
        out.extend(_pack_lol(p[key]))
    out.extend(float(v) for v in p["creates_node"])
    return out
'''
    return [
        {
            "setup": pack + """import numpy as np
candidates = [[4]]
descendants = [[0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]]
targets = np.array([[1, 1, 1, 0]], dtype=np.uint8)
n_samples = 4
""",
            "call": "_pack_plan(plan_greedy_attachments(candidates, descendants, targets, n_samples))",
            "gold_call": "_pack_plan(_oracle_plan_greedy_attachments(candidates, descendants, targets, n_samples))",
        },
        {
            "setup": pack + """import numpy as np
candidates = [[2]]
descendants = [[0], [1], [0, 1]]
targets = np.array([[1, 1]], dtype=np.uint8)
n_samples = 2
""",
            "call": "_pack_plan(plan_greedy_attachments(candidates, descendants, targets, n_samples))",
            "gold_call": "_pack_plan(_oracle_plan_greedy_attachments(candidates, descendants, targets, n_samples))",
        },
        {
            "setup": """import numpy as np
candidates = [[2]]
descendants = [[0], [1], [0, 1]]
targets = np.array([[1, 0]], dtype=np.uint8)
n_samples = 2
def run_model():
    try:
        plan_greedy_attachments(candidates, descendants, targets, n_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_plan_greedy_attachments(candidates, descendants, targets, n_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
