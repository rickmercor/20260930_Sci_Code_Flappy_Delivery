"""
Candidate discovery is separated from mutation application. Every update in one batch consumes a plan derived from the same read-only snapshot, so a new node created for an earlier update cannot become a candidate for a later update in that batch. Exact matches reuse a snapshot node, while other plans append one node in fixed update order.

Inputs

------

children: Ordered child lists for the discovery snapshot.

attachment_plan: Fixed-snapshot exact matches and new-node child plans.

n_samples: Number of sample leaves.

Returns

-------

updated_graph: Updated child lists, attachment nodes, new node identifiers, and edge count.

Returns
-------
dict of native integer lists plus the final directed edge count
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from typing import Any


def apply_snapshot_batch(children: list, attachment_plan: dict, n_samples: int) -> dict[str, Any]:
    """Apply a fixed-snapshot attachment plan in deterministic order.

    Parameters
    ----------
    children : list
        Ordered child lists for the snapshot graph.
    attachment_plan : dict
        Exact nodes, selected candidates, uncovered leaves, and creation flags.
    n_samples : int
        Number of sample leaves.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` is invalid, required plan fields are
        missing, plan-field lengths differ, or `creates_node` is not binary.
        Also raised for an inconsistent exact-match plan, an out-of-range
        selected candidate, an invalid uncovered sample, or empty or repeated
        children in a new-node plan.

    Returns
    -------
    updated_graph : dict
        Updated graph and per-update attachment identifiers.
    """
    return updated_graph  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_apply_snapshot_batch(children: list, attachment_plan: dict, n_samples: int) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    required = {"exact_nodes", "selected_candidates", "uncovered_samples", "creates_node"}
    if not isinstance(attachment_plan, dict) or not required.issubset(attachment_plan):
        raise ValueError("attachment_plan is missing required fields")
    lengths = [len(attachment_plan[key]) for key in required]
    if len(set(lengths)) != 1:
        raise ValueError("attachment plan fields must have matching lengths")

    snapshot_size = len(children)
    updated = [list(map(int, node_children)) for node_children in children]
    attachment_nodes = []
    new_node_ids = []
    for update in range(lengths[0]):
        exact = int(attachment_plan["exact_nodes"][update])
        selected = [int(node) for node in attachment_plan["selected_candidates"][update]]
        uncovered = [int(sample) for sample in attachment_plan["uncovered_samples"][update]]
        creates = int(attachment_plan["creates_node"][update])
        if creates not in (0, 1):
            raise ValueError("creates_node entries must be binary")
        if creates == 0:
            if exact < n_samples or exact >= snapshot_size or selected or uncovered:
                raise ValueError("an exact plan must reuse one snapshot internal node")
            attachment_nodes.append(exact)
            new_node_ids.append(-1)
            continue
        if exact != -1:
            raise ValueError("a new-node plan cannot also specify an exact node")
        if any(node < n_samples or node >= snapshot_size for node in selected):
            raise ValueError("selected candidates must come from the fixed snapshot")
        if any(sample < 0 or sample >= n_samples for sample in uncovered):
            raise ValueError("uncovered sample identifiers are invalid")
        node_children = selected + uncovered
        if not node_children or len(node_children) != len(set(node_children)):
            raise ValueError("a new node needs distinct snapshot children")
        new_node = len(updated)
        updated.append(node_children)
        attachment_nodes.append(new_node)
        new_node_ids.append(new_node)

    return {
        "children": updated,
        "attachment_nodes": attachment_nodes,
        "new_node_ids": new_node_ids,
        "edge_count": sum(len(node_children) for node_children in updated),
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
def _pack_batch(b):
    out = _pack_lol(b["children"])
    out.append(float(len(b["attachment_nodes"])))
    out.extend(float(v) for v in b["attachment_nodes"])
    out.append(float(len(b["new_node_ids"])))
    out.extend(float(v) for v in b["new_node_ids"])
    out.append(float(b["edge_count"]))
    return out
'''
    return [
        {
            "setup": pack + """children = [[], [], [], [], [0, 1], [2, 3]]
plan = {"exact_nodes": [-1, -1], "selected_candidates": [[4], [4]], "uncovered_samples": [[2], [3]], "creates_node": [1, 1]}
n_samples = 4
""",
            "call": "_pack_batch(apply_snapshot_batch(children, plan, n_samples))",
            "gold_call": "_pack_batch(_oracle_apply_snapshot_batch(children, plan, n_samples))",
        },
        {
            "setup": pack + """children = [[], [], [0, 1]]
plan = {"exact_nodes": [2], "selected_candidates": [[]], "uncovered_samples": [[]], "creates_node": [0]}
n_samples = 2
""",
            "call": "_pack_batch(apply_snapshot_batch(children, plan, n_samples))",
            "gold_call": "_pack_batch(_oracle_apply_snapshot_batch(children, plan, n_samples))",
        },
        {
            "setup": """children = [[], [], [0, 1]]
plan = {"exact_nodes": [-1], "selected_candidates": [[3]], "uncovered_samples": [[]], "creates_node": [1]}
n_samples = 2
def run_model():
    try:
        apply_snapshot_batch(children, plan, n_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_apply_snapshot_batch(children, plan, n_samples)
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
