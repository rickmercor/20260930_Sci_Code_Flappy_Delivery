"""
Build batched periodic neighbor lists for the validated crystals.

Only the crystals that survived validation are passed in. Neighbors

are every intermolecular pair between the asymmetric unit and a

neighboring image (each axis translation in {−1, 0, 1}, excluding the

zero translation) inside the printed 5.0 Å cutoff. Each unordered image

pair appears exactly once: for atoms i < j use all 26 translations, and

for i = j use only the 13 translations whose (tx, ty, tz) is

lexicographically positive, so that (i, j, t) and its equivalent

(j, i, −t) are never both stored. The intramolecular bond is not a neighbor; it is handled

later by the bonded term of the Lennard-Jones energy model. Store each

crystal as integer tuples (i, j, tx, ty, tz) in a padded tensor.

Unused slots are -1. neighbor_counts is the number of valid tuples

per crystal. Later energy evaluations must consume this same

representation.

Returns
-------
return neighbor_pairs, neighbor_counts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_batched_pbc_neighbor_lists(
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Return padded periodic neighbor lists.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).

    Returns
    -------
    neighbor_pairs : np.ndarray
        Integer tuples (i, j, tx, ty, tz), shape (n_crystals, max_pairs, 5).
        Unused slots are -1.
    neighbor_counts : np.ndarray
        Number of valid pairs per crystal, shape (n_crystals,).

    Raises
    ------
    ValueError
        If cells and coordinates are misaligned or non-finite.
    """
    return neighbor_pairs, neighbor_counts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_batched_pbc_neighbor_lists(
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    cells = np.asarray(cells, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have 3 columns")
    if cells.shape[0] == 0:
        raise ValueError("cells is empty")
    if not np.all(np.isfinite(cells)):
        raise ValueError("cells contains non-finite values")
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape (n_crystals, n_atoms, 3)")
    if cells.shape[0] != coords.shape[0]:
        raise ValueError("cells and coords must describe the same crystals")
    if not np.all(np.isfinite(coords)) or np.any(cells <= 0):
        raise ValueError("cells and coords must be finite and physical")

    bag = []
    for m in range(cells.shape[0]):
        cr = coords[m]
        cell = cells[m]
        n = cr.shape[0]
        pairs = []
        for i in range(n):
            for j in range(n):
                for tx in (-1, 0, 1):
                    for ty in (-1, 0, 1):
                        for tz in (-1, 0, 1):
                            intramolecular = (tx, ty, tz) == (0, 0, 0) and i != j
                            self_null = (tx, ty, tz) == (0, 0, 0) and i == j
                            if intramolecular or self_null:
                                continue
                            if i > j:
                                continue
                            if i == j and (
                                tx < 0
                                or (tx == 0 and ty < 0)
                                or (tx == 0 and ty == 0 and tz < 0)
                            ):
                                continue
                            shift = np.array(
                                [tx * cell[0], ty * cell[1], tz * cell[2]],
                                dtype=float,
                            )
                            d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                            if d < 5.0:
                                pairs.append((i, j, tx, ty, tz))
        bag.append(pairs)
    max_p = max((len(p) for p in bag), default=0)
    if max_p == 0:
        raise ValueError("every neighbor list is empty")
    padded = -np.ones((cells.shape[0], max_p, 5), dtype=int)
    counts = np.zeros(cells.shape[0], dtype=int)
    for m, pairs in enumerate(bag):
        counts[m] = len(pairs)
        if pairs:
            padded[m, : len(pairs)] = np.asarray(pairs, dtype=int)
    return padded, counts

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Independent six-row batch. Not the production Sobol/experimental cells.
cells = np.array([
    [4.25, 4.25, 4.25],
    [4.35, 4.35, 4.35],
    [4.60, 4.60, 4.60],
    [4.80, 4.80, 4.80],
    [5.10, 5.10, 5.10],
    [4.70, 4.70, 4.70],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[2.175, 2.175, 1.625], [2.175, 2.175, 2.725]],
    [[2.30, 2.30, 1.75], [2.30, 2.30, 2.85]],
    [[2.40, 2.40, 1.85], [2.40, 2.40, 2.95]],
    [[2.55, 2.55, 2.00], [2.55, 2.55, 3.10]],
    [[2.35, 2.35, 1.80], [2.35, 2.35, 2.90]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    _, counts = build_batched_pbc_neighbor_lists(cells, coords)
    return counts

def run_gold():
    _fresh_inputs()
    _, counts = _oracle_build_batched_pbc_neighbor_lists(cells, coords)
    return counts
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([[4.25, 4.25, 4.25]], dtype=float)
coords = np.array([[[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]]], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def _contact_profile(pairs, counts, cells_ref, coords_ref):
    rows = pairs[0, :counts[0]]
    d = [float(np.linalg.norm(coords_ref[0][j] + np.array([tx, ty, tz]) * cells_ref[0] - coords_ref[0][i]))
         for i, j, tx, ty, tz in rows]
    pad_ok = float(np.all(pairs[0, counts[0]:] == -1)) if pairs.shape[1] > counts[0] else 1.0
    head = [float(counts[0]), pad_ok, float(np.max(pairs[0, :counts[0], 2:]))]
    return np.concatenate([head, np.sort(np.asarray(d, dtype=float))])

def run_model():
    _fresh_inputs()
    c0, x0 = cells.copy(), coords.copy()
    pairs, counts = build_batched_pbc_neighbor_lists(cells, coords)
    return _contact_profile(pairs, counts, c0, x0)

def run_gold():
    _fresh_inputs()
    c0, x0 = cells.copy(), coords.copy()
    pairs, counts = _oracle_build_batched_pbc_neighbor_lists(cells, coords)
    return _contact_profile(pairs, counts, c0, x0)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

# A longer c axis changes the periodic-image neighbor count.
cells = np.array([
    [4.25, 4.25, 4.25],
    [4.25, 4.25, 9.20],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[2.125, 2.125, 4.05], [2.125, 2.125, 5.15]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    _, counts = build_batched_pbc_neighbor_lists(cells, coords)
    return counts

def run_gold():
    _fresh_inputs()
    _, counts = _oracle_build_batched_pbc_neighbor_lists(cells, coords)
    return counts
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([[4.25, 4.25, -1.0]], dtype=float)
coords = np.array([[[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]]], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        build_batched_pbc_neighbor_lists(cells, coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_build_batched_pbc_neighbor_lists(cells, coords)
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
