"""
Enumerate, for every nonempty subset of cell types, the subsets reached by toggling one cell type, marking the toggles that leave the admissible set.

The analytic significance of a maximum over a discrete family is not obtained by integrating the maximum directly. The device that makes it tractable is the discrete local maxima decomposition: the event that the field exceeds a high threshold somewhere is decomposed over the sites at which the field attains a local maximum, and for a threshold well into the tail the contribution of sites that are not local maxima is negligible. What turns this into a formula is a definition of locality, that is, a neighbourhood structure on the index set of the field.




For an all-subset scan the index set is the collection of nonempty subsets of the cell types, and the natural neighbourhood is the one induced by the search itself. Two subsets are adjacent when they differ by one cell type, so a subset with C cell types available has at most C neighbours, obtained by adding a cell type that is absent or dropping one that is present. In the binary encoding, where a subset is an integer whose set bits are its members, adjacency is a single bit flip and the neighbour table is immediate.




One toggle is inadmissible. Dropping the only member of a singleton subset produces the empty set, which carries no statistic because the meta-analysis of no studies is undefined; the scan never visits it. Singletons therefore have one neighbour fewer than every other subset, and the correction must be told so rather than silently treating the empty set as a site with a statistic of zero, which would multiply every singleton's contribution by a spurious factor and bias the analytic significance downward. Marking the inadmissible toggle explicitly, rather than dropping it from a ragged list, keeps the table rectangular and keeps the neighbour count of each subset recoverable.




The size of the neighbourhood is what makes the decomposition useful. It grows linearly in the number of cell types while the index set grows exponentially, so the correction costs several pairwise conditional probabilities proportional to C 2^C rather than the full pairwise structure of the field, and the threshold exceedance probability it produces accounts for the dependence between overlapping subsets that a Bonferroni bound throws away.

Returns
-------
np.ndarray of shape (2**n_cell_types - 1, n_cell_types), int: the index of each one-toggle neighbour, or -1 where the toggle is inadmissible.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_neighbour_table(n_cell_types: int) -> np.ndarray:
    """Return the one-cell-type-toggle neighbour table of the nonempty subsets.

    Subsets are enumerated in binary order: row m - 1 corresponds to the subset
    containing cell type c whenever bit c of the integer m is set, for
    m = 1, ..., 2**n_cell_types - 1.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.

    Returns
    -------
    neighbours : np.ndarray
        Integer array of shape (2**n_cell_types - 1, n_cell_types) whose entry
        (a, c) is the index of the subset obtained from subset a by toggling
        cell type c, or -1 when that toggle is inadmissible.
    """
    return neighbours  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_neighbour_table(n_cell_types: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_cell_types, bool) or not isinstance(n_cell_types, (int, np.integer)):
        raise ValueError("n_cell_types must be an integer")
    count = int(n_cell_types)
    if not 1 <= count <= 14:
        raise ValueError("n_cell_types must lie between 1 and 14")

    table = np.full((2 ** count - 1, count), -1, dtype=int)
    for mask in range(1, 2 ** count):
        for cell_type in range(count):
            toggled = mask ^ (1 << cell_type)
            # The empty set carries no statistic and is never visited.
            table[mask - 1, cell_type] = toggled - 1 if toggled >= 1 else -1
    return table

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark number of cell types (normal scenario) ---
        {
            "setup": """import numpy as np
n_cell_types = 7
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_neighbour_table(n_cell_types))",
            "gold_call": "digest(_oracle_build_neighbour_table(n_cell_types))",
        },
        # --- Valid: three cell types, small enough to check by hand ---
        {
            "setup": """import numpy as np
n_cell_types = 3
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_neighbour_table(n_cell_types))",
            "gold_call": "digest(_oracle_build_neighbour_table(n_cell_types))",
        },
        # --- Boundary: a single cell type, whose only toggle is inadmissible ---
        {
            "setup": """import numpy as np
n_cell_types = 1
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_neighbour_table(n_cell_types))",
            "gold_call": "digest(_oracle_build_neighbour_table(n_cell_types))",
        },
        # --- Edge: the largest admissible index set ---
        {
            "setup": """import numpy as np
n_cell_types = 14
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_neighbour_table(n_cell_types))",
            "gold_call": "digest(_oracle_build_neighbour_table(n_cell_types))",
        },
        # --- Invalid: zero cell types ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_neighbour_table(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_neighbour_table(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-integer count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_neighbour_table(3.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_neighbour_table(3.5)
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
