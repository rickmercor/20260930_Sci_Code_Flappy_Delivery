"""
Parse a dot-bracket RNA target into ordered zero-based base-pair coordinates and unpaired coordinates.

Matching parentheses represent paired nucleotides and dots represent unpaired positions. A stack matches each closing parenthesis to the most recent unmatched opening parenthesis. Ordering pairs by their left coordinate gives a deterministic numeric representation for later steps.

Returns
-------
return (pairs, unpaired)  # tuple[tuple[tuple[int, int], ...], tuple[int, ...]]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parse_target(structure: str) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Return ordered base-pair coordinates and unpaired coordinates."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_parse_target(
    structure: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    if not isinstance(structure, str) or not structure:
        raise ValueError("Supply a nonempty dot-bracket string")
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    unpaired: list[int] = []
    for position, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unmatched ')' at position {position}")
            pairs.append((stack.pop(), position))
        elif symbol == ".":
            unpaired.append(position)
        else:
            raise ValueError(f"Invalid symbol {symbol!r} at position {position}")
    if stack:
        raise ValueError(f"Unmatched '(' at positions {stack}")
    return tuple(sorted(pairs)), tuple(unpaired)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    pin = (
        "import math\n"
        "def _pin(result):\n"
        "    pairs, unpaired = result\n"
        "    flat = [float(v) for pair in pairs for v in pair] + [float(v) + 1000.0 for v in unpaired]\n"
        "    return (sum(math.sin(0.31 * (i + 1)) * v for i, v in enumerate(flat))\n"
        "            + sum(math.cos(0.13 * (i + 1)) * v * v for i, v in enumerate(flat))\n"
        "            + 10000.0 * len(pairs) + len(unpaired))\n"
    )
    return [
        {"setup": pin + 'structure = "(((...)))"',
         "call": "_pin(parse_target(structure))",
         "gold_call": "_pin(_oracle_parse_target(structure))"},
        {"setup": pin + 'structure = "."',
         "call": "_pin(parse_target(structure))",
         "gold_call": "_pin(_oracle_parse_target(structure))"},
        {"setup": pin + 'structure = "()()"',
         "call": "_pin(parse_target(structure))",
         "gold_call": "_pin(_oracle_parse_target(structure))"},
    ]
