"""
The reduction this task rests on replaces the transmission problem across the boundary of D by a problem posed only outside it, so only the elements outside D are ever assembled. That splits the nodes into three classes, and getting the split right is the whole of this stage.

- Nodes strictly inside D touch no exterior element at all. They carry no equation and must be dropped; leaving them in produces a singular system with empty rows.
- Nodes lying on the boundary surface of D are where the Dirichlet data will later be imposed. A node is on that surface when it lies within the closed box and at least one of its three indices equals the low or the high face index in that direction.
- Every remaining node touched by an exterior element is free.

Return the exterior element list, the two node sets and the volume of D, which is the product of the three spans times h cubed. Return also the three edge lengths of the resonator in metre, which fix its volume and its shape. That volume enters the resonant frequency directly, so it is worth forming here rather than inferring it later. The element list is a list of element index triples and not an element-to-node connectivity table: row e holds the three integers (i, j, k) that locate element e on the grid, so the array has exactly three columns, and the node indices of its eight corners are formed later from those triples and the corner offsets.

Returns
-------
dict, holding h, volume_D, the exterior element list elements, the node index arrays surface_nodes and free_nodes, the counts n_surface and n_free, and the float array sides giving the three edge lengths of the resonator in metre.

Floquet-Bloch theory reduces the crystal to one cell, so the geometry that has to be built is a single cube of side L holding one resonator. Lay a uniform grid of n_side by n_side by n_side trilinear hexahedral elements over the cell, each of edge h equal to L over n_side, with nodes at the element corners and the node at integer triple (i, j, k) sitting at position (i, j, k) times h. Close the cell periodically, so that node index n_side in any direction is the same node as index zero; this leaves exactly n_side cubed distinct nodes.

The resonator D is an axis-aligned rectangular box spanning span_x, span_y and span_z whole elements, placed as centrally as the grid allows, with its low corner at element index (n_side minus span) integer-divided by two in each direction. Aligning the box with element boundaries matters: it makes every element either wholly inside D or wholly outside, so no element is ever cut and no element carries two materials.

Returns
-------
dict, holding h, volume_D, the exterior element list elements, the node index arrays surface_nodes and free_nodes, the counts n_surface and n_free, and the float array sides giving the three edge lengths of the resonator in metre.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def partition_unit_cell(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
) -> dict:
    """Grid the cell, place the box resonator and split the nodes into surface, free and discarded.

    Parameters
    ----------
    lattice_constant : float
        Edge of the cubic unit cell in metre, above zero.
    n_side : int
        Elements along each edge of the cell, at least four.
    spans : tuple
        Three integers giving the elements spanned by the resonator along each axis.

    Returns
    -------
    dict
        Under the keys h, volume_D, elements, surface_nodes, free_nodes, n_surface,
        n_free and sides. sides is the float array of shape (3,) holding the three edge
        lengths of the resonator in metre, so sides is spans times h and their product
        is volume_D.
        elements is an integer array of shape (n_exterior, 3) containing element
        index triples. surface_nodes and free_nodes are one-dimensional integer
        arrays of flat node indices, each sorted ascending. For the periodic
        node (i, j, k), the flat index is (i * n_side + j) * n_side + k,
        with each coordinate first reduced modulo n_side.

    Raises
    ------
    ValueError
        When lattice_constant is not finite and above zero, when n_side is not an integer of four or more, when spans does not hold exactly three integers, or when any span falls below one or exceeds n_side minus two.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _flat(idx, n_side):
    """Flat index of a node triple taken modulo the periodic cell."""
    return ((idx[..., 0] % n_side) * n_side * n_side
            + (idx[..., 1] % n_side) * n_side
            + (idx[..., 2] % n_side))


def _oracle_partition_unit_cell(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
) -> dict:
    """Reference implementation."""
    L = float(lattice_constant)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("lattice_constant must be finite and above zero")
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    span = tuple(spans)
    if len(span) != 3:
        raise ValueError("spans must hold exactly three integers")
    for s in span:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
            raise ValueError("every span must be an integer")
        if int(s) < 1 or int(s) > n_side - 2:
            raise ValueError("spans must lie in [1, n_side - 2]")
    span = np.array([int(s) for s in span], dtype=np.int64)

    lo = (n_side - span) // 2
    hi = lo + span
    grid = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(grid, grid, grid, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    elements = cells[~inside]

    rngs = [np.arange(lo[d], hi[d] + 1, dtype=np.int64) for d in range(3)]
    aa, bb, cc = np.meshgrid(rngs[0], rngs[1], rngs[2], indexing="ij")
    pts = np.stack([aa.ravel(), bb.ravel(), cc.ravel()], axis=1)
    on_face = np.any((pts == lo) | (pts == hi), axis=1)
    surface = np.unique(_flat(pts[on_face], n_side))

    offs = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)
    touched_idx = elements[:, None, :] + offs[None, :, :]
    touched = np.zeros(n_side ** 3, dtype=bool)
    touched[_flat(touched_idx, n_side).ravel()] = True
    every = np.arange(n_side ** 3, dtype=np.int64)
    free = every[touched & ~np.isin(every, surface)]

    h = L / n_side
    return {
        "h": h,
        "volume_D": float(np.prod(span)) * h ** 3,
        "elements": elements,
        "surface_nodes": surface,
        "free_nodes": free,
        "n_surface": int(surface.size),
        "n_free": int(free.size),
        "sides": span.astype(np.float64) * h,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def digest(out, n_side):
    el = out["elements"]
    return (round(float(out["h"]), 12), round(float(out["volume_D"]), 18), int(el.shape[0]),
            out["n_surface"], out["n_free"],
            # the column count, so a connectivity table returned in place of the index
            # triples is reported as the contract failure it is rather than read silently
            int(el.shape[1]),
            # per-column sums, since swapping two columns of the element list leaves the
            # total and the node sets alone but is a different geometry
            int(el[:, 0].sum()), int(el[:, 1].sum()), int(el[:, 2].sum()),
            int((el[:, 0] * 7 + el[:, 1] * 13 + el[:, 2] * 29).sum()),
            int(out["surface_nodes"][:5].sum()),
            int(out["free_nodes"][0]), int(out["free_nodes"][-1]),
            int(n_side ** 3 - out["n_surface"] - out["n_free"]))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(partition_unit_cell(0.02, 24, (12, 9, 6)), 24))",
            "gold_call": "flat(digest(_oracle_partition_unit_cell(0.02, 24, (12, 9, 6)), 24))",
        },
        {
            "setup": """import numpy as np
def digest(out, n_side):
    return (out["n_surface"], out["n_free"], int(out["elements"].shape[0]),
            round(float(out["volume_D"]), 18),
            int(n_side ** 3 - out["n_surface"] - out["n_free"]))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat((digest(partition_unit_cell(1.0, 8, (4, 4, 4)), 8), digest(partition_unit_cell(0.5, 10, (2, 6, 3)), 10)))",
            "gold_call": "flat((digest(_oracle_partition_unit_cell(1.0, 8, (4, 4, 4)), 8), digest(_oracle_partition_unit_cell(0.5, 10, (2, 6, 3)), 10)))",
        },
        {
            "setup": """
def verdict(fn, L=0.02, n=24, spans=(12, 9, 6)):
    try:
        fn(L, n, spans)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
def verdicts(fn):
    return flat((verdict(fn, L=0.0), verdict(fn, n=3), verdict(fn, spans=(12, 9)), verdict(fn, spans=(12, 9, 0)), verdict(fn, spans=(12, 9, 23)), verdict(fn, L=float('inf')), verdict(fn)))
""",
            'call': 'verdicts(partition_unit_cell)',
            'gold_call': 'verdicts(_oracle_partition_unit_cell)',
        },
    ]
