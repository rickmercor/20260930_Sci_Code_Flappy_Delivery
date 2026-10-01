"""
A genotype representation graph stores the reach D(n) of node n as the union of the disjoint reaches of its children, with D(s) = {s} for a sample leaf. A reach of density |D(n)| / N at least tau is represented by a width-N bitset; a lower-density reach remains a sparse list. Ordered node identifiers place every child before its parent.

Inputs

------

children: Child node identifiers for every graph node.

n_samples: Number of leaf nodes at the beginning of the node array.

dense_threshold: Carrier-set density at which bitset storage is used.

Returns

-------

encoding: Native numeric descendant lists, representation flags, storage payloads, and cardinalities.

Returns
-------
dict of native integer lists describing each node's reach and adaptive representation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from typing import Any


def encode_adaptive_descendants(children: list, n_samples: int, dense_threshold: float) -> dict[str, Any]:
    """Encode every node reach with adaptive sparse or dense storage.

    Parameters
    ----------
    children : list
        List whose entry n contains the child identifiers of node n.
    n_samples : int
        Number of sample leaves, identified by 0 through n_samples - 1.
    dense_threshold : float
        Inclusive density threshold for dense bitset storage.

    Raises
    ------
    ValueError
        If `children` is empty, `n_samples` does not define a nonempty leaf
        prefix, or `dense_threshold` is outside [0, 1]. Also raised if a leaf
        has children, an internal node has no children, a child list has the
        wrong type or repeats an identifier, a child does not precede its
        parent, or sibling child reaches overlap.

    Returns
    -------
    encoding : dict
        Descendant lists, dense flags, encoded payloads, and cardinalities.
    """
    return encoding  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_encode_adaptive_descendants(children: list, n_samples: int, dense_threshold: float) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify a nonempty leaf prefix")
    if not isinstance(dense_threshold, (int, float)) or not 0.0 <= float(dense_threshold) <= 1.0:
        raise ValueError("dense_threshold must be in [0, 1]")
    if any(list(children[s]) for s in range(n_samples)):
        raise ValueError("sample leaves must have no children")

    descendants = []
    dense_flags = []
    storage = []
    cardinalities = []
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("each children entry must be a list or tuple")
        node_children = [int(child) for child in raw_children]
        if len(node_children) != len(set(node_children)):
            raise ValueError("a node cannot repeat a child")
        if any(child < 0 or child >= node for child in node_children):
            raise ValueError("every child identifier must precede its parent")
        if node < n_samples:
            reach = [node]
        else:
            if not node_children:
                raise ValueError("an internal node must have at least one child")
            seen = set()
            for child in node_children:
                child_reach = set(descendants[child])
                if seen.intersection(child_reach):
                    raise ValueError("child reaches must be disjoint")
                seen.update(child_reach)
            reach = sorted(seen)

        is_dense = int((len(reach) / n_samples) >= float(dense_threshold))
        if is_dense:
            n_words = (n_samples + 63) // 64
            payload = [0] * n_words
            for sample in reach:
                payload[sample // 64] |= 1 << (sample % 64)
        else:
            payload = list(reach)
        descendants.append(reach)
        dense_flags.append(is_dense)
        storage.append(payload)
        cardinalities.append(len(reach))

    return {
        "descendant_lists": descendants,
        "dense_flags": dense_flags,
        "storage": storage,
        "cardinalities": cardinalities,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack_setup = '''
def _pack_encoding(enc):
    out = []
    for key in ("descendant_lists", "storage"):
        rows = enc[key]
        out.append(float(len(rows)))
        for row in rows:
            out.append(float(len(row)))
            out.extend(float(v) for v in row)
    out.extend(float(v) for v in enc["dense_flags"])
    out.extend(float(v) for v in enc["cardinalities"])
    return out
'''
    return [
        {
            "setup": pack_setup + """children = [[], [], [], [], [0, 1], [2, 3], [4, 5]]
n_samples = 4
dense_threshold = 0.75
""",
            "call": "_pack_encoding(encode_adaptive_descendants(children, n_samples, dense_threshold))",
            "gold_call": "_pack_encoding(_oracle_encode_adaptive_descendants(children, n_samples, dense_threshold))",
        },
        {
            "setup": pack_setup + """children = [[]]
n_samples = 1
dense_threshold = 1.0
""",
            "call": "_pack_encoding(encode_adaptive_descendants(children, n_samples, dense_threshold))",
            "gold_call": "_pack_encoding(_oracle_encode_adaptive_descendants(children, n_samples, dense_threshold))",
        },
        {
            "setup": """children = [[], [], [0, 1], [2, 0]]
n_samples = 2
dense_threshold = 0.5
def run_model():
    try:
        encode_adaptive_descendants(children, n_samples, dense_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_encode_adaptive_descendants(children, n_samples, dense_threshold)
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
