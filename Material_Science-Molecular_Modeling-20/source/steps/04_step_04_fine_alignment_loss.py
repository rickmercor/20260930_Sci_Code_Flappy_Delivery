"""
Calculate the fine-alignment loss of the supplied connected atom groups using Eq. (3).

For each connected edge, the paper takes the minimum squared distance over all atom pairs across that edge. The edge contributions are then summed. The supplied groups are already oriented.

Returns
-------
A float containing the total fine-alignment loss in Å².
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fine_alignment_loss(node_groups: list, linker_groups: list) -> float:
    """Evaluate the fine-alignment objective.

    Parameters
    ----------
    node_groups, linker_groups : list
        Equally long lists of nonempty atom groups.
        Each atom is a finite Cartesian 3-vector in Å.
        Entry e on each side forms one connected edge.
        Coordinates are in a common aligned frame.

    Returns
    -------
    float
        Sum of one minimum squared atom-pair distance per edge, in Å^2.
        Zero edges return 0.0.

    Raises
    ------
    ValueError
        Unequal edge counts, empty group, or malformed/nonfinite coordinates.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fine_alignment_loss(node_groups: list, linker_groups: list) -> float:
    import math
    if len(node_groups) != len(linker_groups):
        raise ValueError('edge count mismatch')
    for group in node_groups + linker_groups:
        if not group or any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in group)):
            raise ValueError('invalid atom group')
    total = 0.0
    for node_atoms, linker_atoms in zip(node_groups, linker_groups):
        minimum = min((sum(((a[k] - b[k]) ** 2 for k in range(3))) for a in node_atoms for b in linker_atoms))
        total += minimum
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'fine_alignment_loss([[[0,0,0],[100,0,0]]], [[[1,0,0],[50,0,0]]])', 'gold_call': '_oracle_fine_alignment_loss([[[0,0,0],[100,0,0]]], [[[1,0,0],[50,0,0]]])'}, {'setup': '', 'call': 'fine_alignment_loss([[[0,0,0]],[[0,0,0]]], [[[0,0,2]],[[0,3,0]]])', 'gold_call': '_oracle_fine_alignment_loss([[[0,0,0]],[[0,0,0]]], [[[0,0,2]],[[0,3,0]]])'}, {'setup': '', 'call': 'fine_alignment_loss([], [])', 'gold_call': '_oracle_fine_alignment_loss([], [])'}, {'setup': '', 'call': 'fine_alignment_loss([[[0,0,0]]], [[[0,0,0]]])', 'gold_call': '_oracle_fine_alignment_loss([[[0,0,0]]], [[[0,0,0]]])'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fine_alignment_loss([], [[[0,0,0]]]))', 'gold_call': '_check_error(lambda: _oracle_fine_alignment_loss([], [[[0,0,0]]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fine_alignment_loss([[]], [[[0,0,0]]]))', 'gold_call': '_check_error(lambda: _oracle_fine_alignment_loss([[]], [[[0,0,0]]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fine_alignment_loss([[[0,0]]], [[[0,0,0]]]))', 'gold_call': '_check_error(lambda: _oracle_fine_alignment_loss([[[0,0]]], [[[0,0,0]]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fine_alignment_loss([[[0,0,0]]], [[[float("nan"),0,0]]]))', 'gold_call': '_check_error(lambda: _oracle_fine_alignment_loss([[[0,0,0]]], [[[float("nan"),0,0]]]))'}]
