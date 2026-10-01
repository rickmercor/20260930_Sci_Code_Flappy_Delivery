"""
Apply triple-radius validation to trial crystals.

The batch already includes the generated rows and the flagged

experimental row. The covalent and van der Waals tables are printed

in the problem statement. Use the geometric keep/reject scheme

from the source and write a 0/1 mask of the same length as the

batch. On this small-Z cell the framework part of that scheme is the

27-image intermolecular contact graph named in the prompt: two sites

are in contact when their paper framework spheres overlap. Use

from the source paper how that graph is used. Do not replace the

full scheme by one contact cutoff or by a density number.

Returns
-------
return keep_mask
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_triple_radius_validation(
    atomic_numbers: "np.ndarray",
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "np.ndarray":
    """
    Return a 0/1 keep mask.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).

    Returns
    -------
    np.ndarray
        Integer keep mask of shape (n_crystals,).

    Raises
    ------
    ValueError
        If cells or coordinates are misaligned, or numerical values
        are invalid.
    """
    return keep_mask

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_triple_radius_validation(
    atomic_numbers: "np.ndarray",
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "np.ndarray":
    covalent = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
    vdw = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52}
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    cells = np.asarray(cells, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if z.size == 0:
        raise ValueError("atomic_numbers is empty")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have 3 columns")
    if cells.shape[0] == 0:
        raise ValueError("cells is empty")
    if not np.all(np.isfinite(cells)):
        raise ValueError("cells contains non-finite values")
    if coords.ndim != 3 or coords.shape[1:] != (z.size, 3):
        raise ValueError("coords must have shape (n_crystals, n_atoms, 3)")
    if cells.shape[0] != coords.shape[0]:
        raise ValueError("cells and coords must describe the same crystals")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contain non-finite values")
    if not np.all(np.isin(z, np.array([1, 6, 7, 8], dtype=int))):
        raise ValueError("atomic numbers must belong to the fixture element set")
    if np.any(cells <= 0):
        raise ValueError("cell lengths must be positive")

    keep = np.zeros(cells.shape[0], dtype=int)
    rmax = max(vdw[int(zi)] for zi in z)
    framework_cutoff = 2.0 * 1.50 * rmax
    integrity_limit = 1.15 * (covalent[int(z[0])] + covalent[int(z[1])])
    for m in range(cells.shape[0]):
        cr = coords[m]
        cell = cells[m]
        bond = float(np.linalg.norm(cr[1] - cr[0]))
        if bond > integrity_limit:
            continue
        clash = False
        n = cr.shape[0]
        for i in range(n):
            for j in range(n):
                for tx in (-1, 0, 1):
                    for ty in (-1, 0, 1):
                        for tz in (-1, 0, 1):
                            if (tx, ty, tz) == (0, 0, 0):
                                continue
                            shift = np.array(
                                [tx * cell[0], ty * cell[1], tz * cell[2]],
                                dtype=float,
                            )
                            d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                            limit = 0.90 * (vdw[int(z[i])] + vdw[int(z[j])])
                            if d < limit:
                                clash = True
        if clash:
            continue
        first_offset = {0: np.zeros(3, dtype=int)}
        queue = [0]
        cycles = []
        while queue:
            i = queue.pop(0)
            oi = first_offset[i]
            for j in range(n):
                for tx in (-1, 0, 1):
                    for ty in (-1, 0, 1):
                        for tz in (-1, 0, 1):
                            if j == i and (tx, ty, tz) == (0, 0, 0):
                                continue
                            shift = np.array(
                                [tx * cell[0], ty * cell[1], tz * cell[2]],
                                dtype=float,
                            )
                            d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                            if d > framework_cutoff:
                                continue
                            proposed = oi + np.array([tx, ty, tz], dtype=int)
                            if j not in first_offset:
                                first_offset[j] = proposed
                                queue.append(j)
                            else:
                                delta = proposed - first_offset[j]
                                if np.any(delta != 0):
                                    cycles.append(delta)
        if len(first_offset) < n or not cycles:
            continue
        rank = np.linalg.matrix_rank(np.asarray(cycles, dtype=float))
        if int(rank) < 3:
            continue
        keep[m] = 1
    return keep

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
# Independent five-row batch. Not the production Sobol/experimental cells.
cells = np.array([
    [4.25, 4.25, 4.25],
    [3.40, 3.40, 3.40],
    [4.80, 4.80, 4.80],
    [4.25, 4.25, 9.20],
    [4.25, 4.25, 4.25],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[1.70, 1.70, 1.15], [1.70, 1.70, 2.25]],
    [[2.40, 2.40, 1.85], [2.40, 2.40, 2.95]],
    [[2.125, 2.125, 4.05], [2.125, 2.125, 5.15]],
    [[2.125, 2.125, 1.325], [2.125, 2.125, 3.125]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return apply_triple_radius_validation(atomic_numbers, cells, coords)

def run_gold():
    _fresh_inputs()
    return _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
# Two cubic cells with different intermolecular contacts.
cells = np.array([
    [4.25, 4.25, 4.25],
    [3.40, 3.40, 3.40],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[1.70, 1.70, 1.15], [1.70, 1.70, 2.25]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return apply_triple_radius_validation(atomic_numbers, cells, coords)

def run_gold():
    _fresh_inputs()
    return _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
# Two cells with different c-axis lengths.
cells = np.array([
    [4.80, 4.80, 4.80],
    [4.25, 4.25, 9.20],
], dtype=float)
coords = np.array([
    [[2.40, 2.40, 1.85], [2.40, 2.40, 2.95]],
    [[2.125, 2.125, 4.05], [2.125, 2.125, 5.15]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return apply_triple_radius_validation(atomic_numbers, cells, coords)

def run_gold():
    _fresh_inputs()
    return _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
# Two N2 packings with different intramolecular bond lengths.
cells = np.array([
    [4.25, 4.25, 4.25],
    [4.25, 4.25, 4.25],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[2.125, 2.125, 1.200], [2.125, 2.125, 3.050]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return apply_triple_radius_validation(atomic_numbers, cells, coords)

def run_gold():
    _fresh_inputs()
    return _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
cells = np.array([[4.25, 4.25, 4.25]], dtype=float)
coords = np.array([[[2.125, 2.125, np.nan], [2.125, 2.125, 2.675]]], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        apply_triple_radius_validation(atomic_numbers, cells, coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
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
