"""
Determine whether a parent colour and its direct child colours satisfy the local proper-colouring limits.

Use black = 0, white = 1 and grey = 2; parent = -1 represents the exterior root. For an ordinary pair, the local vector contains the inverse of the parent colour and the colours of its direct paired children. A valid vector contains at most one black, at most one white and at most two grey entries.

Returns
-------
return valid  # int: 1 when valid, otherwise 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def check_local_coloring(parent_color: int, child_colors: tuple[int, ...]) -> int:
    """Return a numeric validity indicator."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_check_local_coloring(parent_color: int, child_colors: tuple[int, ...]) -> int:
    if parent_color not in {-1, 0, 1, 2}:
        raise ValueError("parent_color must be -1, 0, 1, or 2")
    if any(color not in {0, 1, 2} for color in child_colors):
        raise ValueError("Each child colour must be 0, 1, or 2")
    observed = list(child_colors)
    if parent_color != -1:
        observed.append({0: 1, 1: 0, 2: 2}[parent_color])
    return int(observed.count(0) <= 1 and observed.count(1) <= 1 and observed.count(2) <= 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {"setup": "parent_color = 0; child_colors = (0,)", "call": "check_local_coloring(parent_color, child_colors)", "gold_call": "_oracle_check_local_coloring(parent_color, child_colors)"},
        {"setup": "parent_color = -1; child_colors = ()", "call": "check_local_coloring(parent_color, child_colors)", "gold_call": "_oracle_check_local_coloring(parent_color, child_colors)"},
        {"setup": "parent_color = 0; child_colors = (1,)", "call": "check_local_coloring(parent_color, child_colors)", "gold_call": "_oracle_check_local_coloring(parent_color, child_colors)"},
    ]
