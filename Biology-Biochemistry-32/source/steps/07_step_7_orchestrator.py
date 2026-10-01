"""
Orchestrate the six preceding RNA inverse-folding steps and return the final GC percentage.

The final function must call the preceding public functions in dependency order rather than reimplementing them. For the fixed 33-nucleotide target and m = 2, the correct pipeline returns 48.4848.

Returns
-------
return gc_percentage  # float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_pipeline(
    structure: str = "((((((...)))(((...)))(((...))))))",
    m: int = 2,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> float:
    """Return the final numeric result from the complete public pipeline."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_pipeline(
    structure: str = "((((((...)))(((...)))(((...))))))",
    m: int = 2,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> float:
    _oracle_parse_target(structure)
    _, _, m5_count, m3_count = _oracle_analyse_target(structure)
    if m5_count or m3_count:
        return -1.0
    coloring = _oracle_find_separated_coloring(structure, m)
    if coloring is None:
        return -1.0
    sequence = _oracle_construct_sequence(structure, coloring)
    evaluation = _oracle_evaluate_design(sequence, structure, minimum_span, gc_decimals)
    return evaluation[7]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return main-target, boundary, and edge integration cases."""
    return [
        {"setup": 'structure = "((((((...)))(((...)))(((...))))))"; m = 2; minimum_span = 0; gc_decimals = 4', "call": "run_pipeline(structure, m, minimum_span, gc_decimals)", "gold_call": "_oracle_run_pipeline(structure, m, minimum_span, gc_decimals)"},
        {"setup": 'structure = "..."; m = 2; minimum_span = 0; gc_decimals = 4', "call": "run_pipeline(structure, m, minimum_span, gc_decimals)", "gold_call": "_oracle_run_pipeline(structure, m, minimum_span, gc_decimals)"},
        {"setup": 'structure = "(((...)))"; m = 2; minimum_span = 0; gc_decimals = 4', "call": "run_pipeline(structure, m, minimum_span, gc_decimals)", "gold_call": "_oracle_run_pipeline(structure, m, minimum_span, gc_decimals)"},
        {"setup": 'structure = "((...))((...))((...))((...))((...))"; m = 2; minimum_span = 0; gc_decimals = 4', "call": "run_pipeline(structure, m, minimum_span, gc_decimals)", "gold_call": "_oracle_run_pipeline(structure, m, minimum_span, gc_decimals)"},
        {"setup": 'structure = "(.)."; m = 2; minimum_span = 0; gc_decimals = 4', "call": "run_pipeline(structure, m, minimum_span, gc_decimals)", "gold_call": "_oracle_run_pipeline(structure, m, minimum_span, gc_decimals)"},
    ]
