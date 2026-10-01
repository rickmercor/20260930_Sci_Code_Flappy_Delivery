"""
A band gap is a property of a dispersion sampled over a locus of wave vectors, not of any one wave vector, so once the branches have been collected into a table with one row per sampled wave vector and one column per branch, the gap between two consecutive branches is read from the extremes of two columns. The upper edge of the lower branch is the largest value that column takes over the locus and the lower edge of the upper branch is the smallest value the next column takes, and the signed width is the second minus the first. Signed, because the two branches may overlap: a positive width is a frequency interval in which no branch of the sampled set propagates, and a negative width records by how much the branches cross in frequency without there being any wave vector at which both are present.

Reading a maximum and a minimum off a sampled locus is only meaningful if the extremes are resolved, and the cheap indicator is where they fall. If either sits strictly inside the locus, the width plainly depends on how finely the locus was sampled. If both sit at an endpoint, and the endpoints are the high-symmetry points that any sampling of the locus must contain, then the width did not move under the samplings tried; that is evidence and not proof, since a finer sampling can still find an interior extremum that a coarser one stepped over. This step therefore reports the positions of both extremes alongside the width, and a flag recording whether both fell on endpoints, and leaves the strength of that evidence to the caller.

One more reading comes free from the same table and is worth having, because the branch pair a lattice is designed around is not always the one an implementation reaches first. Scanning every consecutive pair and keeping the widest signed gap says which pair carries the largest frequency interval free of any sampled branch, and how wide it is. On a lattice whose gap has been closed by immersion that widest value is negative, and the pair that carries it is the least overlapped rather than the most open.

The table is required to be ordered: at each wave vector the branches are listed in ascending frequency, which is what makes a column a branch rather than an arbitrary selection. A row that is not ascending means the branches were collected without sorting, or that two families were mixed, and the width read from such a table means nothing, so it is rejected rather than reported.

Returns
-------
dict holding the float signed_width in hertz, negative when the branches overlap; the floats lower_branch_top and upper_branch_bottom, the two edges in hertz; the native ints lower_extreme_row and upper_extreme_row, the rows at which those extremes occur; the native int edges_on_endpoints, one when both extremes fall on the first or last row of the table and zero otherwise; the native int widest_pair_lower_branch, the one-based lower index of the consecutive pair carrying the largest signed gap anywhere in the table; and the float widest_pair_width, that gap in hertz.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_gap_edges(branch_table: np.ndarray, lower_branch: int) -> dict:
    """Read the signed gap width between two consecutive branches and locate its edges.

    Parameters
    ----------
    branch_table : np.ndarray
        Branch frequencies in hertz, shape (n_point, n_branch), each row ascending.
    lower_branch : int
        One-based index of the lower of the two branches bounding the gap.

    Returns
    -------
    dict
        Under the keys signed_width, lower_branch_top, upper_branch_bottom, lower_extreme_row, upper_extreme_row, edges_on_endpoints, widest_pair_lower_branch and widest_pair_width.

    Raises
    ------
    ValueError
        When branch_table is not a two-dimensional array of finite values with at least two rows and two columns, when any row is not in ascending order, or when lower_branch is not an integer of one or more that leaves a column above it.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ordered_table(values):
    """Return a branch table as a float array once its rows are known to be ascending."""
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("branch_table wants at least two rows and two columns")
    if not np.isfinite(table).all():
        raise ValueError("branch_table holds an entry that is not finite")
    if bool((np.diff(table, axis=1) < 0.0).any()):
        raise ValueError("every row of branch_table must list the branches in ascending order")
    return table


def _oracle_locate_gap_edges(branch_table: np.ndarray, lower_branch: int) -> dict:
    """Reference implementation."""
    table = _ordered_table(branch_table)
    if not isinstance(lower_branch, (int, np.integer)) or int(lower_branch) < 1:
        raise ValueError("lower_branch wants an integer of one or more")
    lower = int(lower_branch)
    if lower >= table.shape[1]:
        raise ValueError("branch_table needs a column above lower_branch")
    below = table[:, lower - 1]
    above = table[:, lower]
    lower_row = int(np.argmax(below))
    upper_row = int(np.argmin(above))
    last = table.shape[0] - 1
    on_endpoints = int(lower_row in (0, last) and upper_row in (0, last))
    every = table[:, 1:].min(axis=0) - table[:, :-1].max(axis=0)
    widest = int(np.argmax(every))
    return {
        "widest_pair_lower_branch": widest + 1,
        "widest_pair_width": float(every[widest]),
        "signed_width": float(above[upper_row] - below[lower_row]),
        "lower_branch_top": float(below[lower_row]),
        "upper_branch_bottom": float(above[upper_row]),
        "lower_extreme_row": lower_row,
        "upper_extreme_row": upper_row,
        "edges_on_endpoints": on_endpoints,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [{'setup': 'import numpy as np\n'
           '# an open gap whose edges both fall at the ends of the locus, which is the shape the\n'
           '# graded configuration produces\n'
           'TABLE = np.array([[10.0, 40.0, 90.0],\n'
           '                  [11.0, 42.0, 88.0],\n'
           '                  [12.0, 45.0, 86.0],\n'
           '                  [13.0, 47.0, 85.0],\n'
           '                  [14.0, 50.0, 84.0]])\n'
           'def digest(out):\n'
           '    return (round(out["signed_width"], 9), round(out["lower_branch_top"], 9),\n'
           '            round(out["upper_branch_bottom"], 9), out["lower_extreme_row"],\n'
           '            out["upper_extreme_row"], out["edges_on_endpoints"],\n'
           '            out["widest_pair_lower_branch"], round(out["widest_pair_width"], 9))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(locate_gap_edges)(TABLE, 2))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_locate_gap_edges)(TABLE, 2))'},
 {'setup': 'import numpy as np\n'
           '# an overlap, with the maximum of the lower branch strictly inside the locus, so the\n'
           '# width is negative and the endpoint flag is cleared, and the widest pair in the table\n'
           '# is not the pair asked for\n'
           'TABLE = np.array([[10.0, 40.0, 60.0],\n'
           '                  [11.0, 70.0, 75.0],\n'
           '                  [12.0, 55.0, 58.0],\n'
           '                  [13.0, 50.0, 80.0]])\n'
           'def digest(out):\n'
           '    return (round(out["signed_width"], 9), round(out["lower_branch_top"], 9),\n'
           '            round(out["upper_branch_bottom"], 9), out["lower_extreme_row"],\n'
           '            out["upper_extreme_row"], out["edges_on_endpoints"],\n'
           '            out["widest_pair_lower_branch"], round(out["widest_pair_width"], 9))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(locate_gap_edges)(TABLE, 2))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_locate_gap_edges)(TABLE, 2))'},
 {'setup': 'import numpy as np\n'
           '# boundary case: two rows and two columns, the smallest admissible table, and a gap of\n'
           '# exactly zero where the branches touch without crossing\n'
           'TABLE = np.array([[1.0, 3.0], [2.0, 2.0]])\n'
           'def digest(out):\n'
           '    return (round(out["signed_width"], 9), round(out["lower_branch_top"], 9),\n'
           '            round(out["upper_branch_bottom"], 9), out["lower_extreme_row"],\n'
           '            out["upper_extreme_row"], out["edges_on_endpoints"],\n'
           '            out["widest_pair_lower_branch"], round(out["widest_pair_width"], 9))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(locate_gap_edges)(TABLE, 1))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_locate_gap_edges)(TABLE, 1))'},
 {'setup': 'import numpy as np\n'
           'GOOD = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [1.5, 3.5, 5.5]])\n'
           'UNSORTED = np.array([[1.0, 3.0, 5.0], [4.0, 2.0, 6.0], [1.5, 3.5, 5.5]])\n'
           'def sentinel(fn):\n'
           '    codes = []\n'
           '    trials = [(GOOD, 0), (GOOD, 3), (GOOD, 2.0), (UNSORTED, 1),\n'
           '              (GOOD[:1], 1), (np.array([1.0, 2.0, 3.0]), 1),\n'
           '              (np.array([[1.0, np.nan], [2.0, 3.0]]), 1)]\n'
           '    for args in trials:\n'
           '        try:\n'
           '            fn(*args)\n'
           '            codes.append(0)\n'
           '        except ValueError:\n'
           '            codes.append(1)\n'
           '        except Exception:\n'
           '            codes.append(2)\n'
           '    return tuple(codes)\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'sentinel(_scicode_fresh_inputs(locate_gap_edges))',
  'gold_call': 'sentinel(_scicode_fresh_inputs(_oracle_locate_gap_edges))'}]
