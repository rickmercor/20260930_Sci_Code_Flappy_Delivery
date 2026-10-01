"""
Relax a validated crystal batch with the two-step damped cell-length map.

Do not run MACE-OFF, SevenNet, DFT, or VASP. Energy and forces are the fixture-defined Lennard-Jones energy model: intermolecular 12-6 pairs from the neighbor list, together with the fixture-defined intramolecular harmonic bond.

The paper's batched structural-optimization workflow is reduced here to the fixture-defined two-step damped cell-length map, with the molecular Cartesian offset held fixed. Consume the incoming neighbor_pairs and neighbor_counts for the first enthalpy gradient. Rebuild the list after every cell-length update with the same 5.0 Å intermolecular rule. The cell-length gradient ∇_L H is a central finite difference of the fixture enthalpy at probe step h = 1e-4 Å: for each axis α, (∇_L H)_α = [H(L + h ê_α) − H(L − h ê_α)] / (2h). Using the source two-stage pressure schedule. Drop a structure if the fixture-defined energy model reports an instability.



Return the surviving cells, coordinates, zero-pressure energy-model energies, and their indices into the validated batch.

Returns
-------
return cells, coords, energies, keep_index
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_stage_batched_relax(
    cells: "np.ndarray",
    coords: "np.ndarray",
    neighbor_pairs: "np.ndarray",
    neighbor_counts: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Relax a validated crystal batch.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).
    neighbor_pairs : np.ndarray
        Padded neighbor tuples, shape (n_crystals, max_pairs, 5). Used for
        the first enthalpy gradient. Unused slots are -1.
    neighbor_counts : np.ndarray
        Valid pair counts, shape (n_crystals,). The first gradient of each
        crystal uses neighbor_pairs[m, :neighbor_counts[m]].

    Returns
    -------
    cells : np.ndarray
        Relaxed cell lengths.
    coords : np.ndarray
        Relaxed coordinates.
    energies : np.ndarray
        Fixture lattice energies, kJ/mol.
    keep_index : np.ndarray
        Indices retained after relaxation.

    Notes
    -----
    ∇_L H is a central finite difference of the fixture enthalpy at
    h = 1e-4 Å: (H(L + h ê_α) − H(L − h ê_α)) / (2h) on each cell axis.
    After every cell-length update the neighbor list is rebuilt with the
    same 5.0 Å intermolecular rule.

    Raises
    ------
    ValueError
        If the batch, neighbor lists, or fixture energies are invalid.
    """
    return cells, coords, energies, keep_index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_two_stage_batched_relax(
    cells: "np.ndarray",
    coords: "np.ndarray",
    neighbor_pairs: "np.ndarray",
    neighbor_counts: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    cells = np.asarray(cells, dtype=float)
    coords = np.asarray(coords, dtype=float)
    neighbor_pairs = np.asarray(neighbor_pairs, dtype=int)
    neighbor_counts = np.asarray(neighbor_counts, dtype=int).reshape(-1)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have 3 columns")
    if cells.shape[0] == 0:
        raise ValueError("cells is empty")
    if not np.all(np.isfinite(cells)):
        raise ValueError("cells contains non-finite values")
    if coords.ndim != 3 or coords.shape[0] != cells.shape[0]:
        raise ValueError("relaxed batch shape is inconsistent")
    if (
        neighbor_pairs.ndim != 3
        or neighbor_pairs.shape[0] != cells.shape[0]
        or neighbor_pairs.shape[2] != 5
    ):
        raise ValueError("neighbor_pairs must have shape (n_crystals, max_pairs, 5)")
    if neighbor_counts.size == 0 or neighbor_counts.size != cells.shape[0]:
        raise ValueError("neighbor_counts must have one entry per crystal")
    if np.any(neighbor_counts < 0) or np.any(neighbor_counts > neighbor_pairs.shape[1]):
        raise ValueError("neighbor_counts must lie in [0, max_pairs]")

    probe = 1e-4

    def _pairs_from_padded(pairs_m, count_m):
        return [tuple(int(x) for x in pairs_m[k]) for k in range(int(count_m))]

    def _rebuild_pairs(cell, cr):
        try:
            padded, counts = _oracle_build_batched_pbc_neighbor_lists(
                np.asarray(cell, dtype=float).reshape(1, 3),
                np.asarray(cr, dtype=float).reshape(1, -1, 3),
            )
        except ValueError:
            return []
        return _pairs_from_padded(padded[0], counts[0])

    out_cells = []
    out_coords = []
    out_energy = []
    keep_index = []
    for m in range(cells.shape[0]):
        cell = cells[m].copy()
        cr = coords[m].copy()
        pairs = _pairs_from_padded(neighbor_pairs[m], neighbor_counts[m])
        for pressure in (0.10, 0.0):
            for _step in range(2):
                grad = np.zeros(3, dtype=float)
                offsets = np.array([c - 0.5 * cell for c in cr], dtype=float)
                for axis in range(3):
                    energy_plus = []
                    for sign in (-1.0, 1.0):
                        cell_h = cell.copy()
                        cell_h[axis] += sign * probe
                        moved = np.vstack(
                            [0.5 * cell_h + offsets[0], 0.5 * cell_h + offsets[1]]
                        )
                        bond = float(np.linalg.norm(moved[1] - moved[0]))
                        if bond < 0.50:
                            e = -1.0e20
                        else:
                            e = 0.5 * 80.0 * (bond - 1.10) ** 2
                            for i, j, tx, ty, tz in pairs:
                                shift = np.array(
                                    [tx * cell_h[0], ty * cell_h[1], tz * cell_h[2]],
                                    dtype=float,
                                )
                                d = float(np.linalg.norm(moved[j] + shift - moved[i]))
                                if d < 0.50:
                                    e = -1.0e20
                                    break
                                sr = 3.00 / d
                                e += 4.0 * 1.8 * (sr ** 12 - sr ** 6)
                            if e > -1.0e19:
                                volume = float(cell_h[0] * cell_h[1] * cell_h[2])
                                e += pressure * 1.0 * volume
                        energy_plus.append(e)
                    if abs(energy_plus[0]) > 1.0e15 or abs(energy_plus[1]) > 1.0e15:
                        grad[axis] = 0.0
                    else:
                        grad[axis] = (energy_plus[1] - energy_plus[0]) / (2.0 * probe)
                new_cell = cell - 0.12 * np.clip(grad, -1.0, 1.0)
                new_cell = np.clip(new_cell, 2.2, 12.0)
                offsets = np.array([c - 0.5 * cell for c in cr], dtype=float)
                cr = np.vstack(
                    [0.5 * new_cell + offsets[0], 0.5 * new_cell + offsets[1]]
                )
                cell = new_cell
                pairs = _rebuild_pairs(cell, cr)
        bond = float(np.linalg.norm(cr[1] - cr[0]))
        if bond < 0.50:
            energy = -1.0e20
        else:
            energy = 0.5 * 80.0 * (bond - 1.10) ** 2
            collapsed = False
            for i, j, tx, ty, tz in pairs:
                shift = np.array(
                    [tx * cell[0], ty * cell[1], tz * cell[2]],
                    dtype=float,
                )
                d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                if d < 0.50:
                    energy = -1.0e20
                    collapsed = True
                    break
                sr = 3.00 / d
                energy += 4.0 * 1.8 * (sr ** 12 - sr ** 6)
            if not collapsed:
                energy = float(energy)
        if abs(energy) > 1.0e15:
            continue
        out_cells.append(cell)
        out_coords.append(cr)
        out_energy.append(energy)
        keep_index.append(m)
    if not out_cells:
        raise ValueError("every relaxed structure was removed by the collapse filter")
    return (
        np.asarray(out_cells, dtype=float),
        np.asarray(out_coords, dtype=float),
        np.asarray(out_energy, dtype=float),
        np.asarray(keep_index, dtype=int),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

cells = np.array([
    [4.25, 4.25, 4.25],
    [4.35, 4.35, 4.35],
    [4.80, 4.80, 4.80],
], dtype=float)
coords = np.array([
    [[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]],
    [[2.175, 2.175, 1.625], [2.175, 2.175, 2.725]],
    [[2.40, 2.40, 1.85], [2.40, 2.40, 2.95]],
], dtype=float)
bag = []
for m in range(cells.shape[0]):
    cr = coords[m]
    cell = cells[m]
    n = cr.shape[0]
    pairs_m = []
    for i in range(n):
        for j in range(n):
            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    for tz in (-1, 0, 1):
                        if (tx, ty, tz) == (0, 0, 0):
                            continue
                        if i > j:
                            continue
                        if i == j and (
                            tx < 0 or (tx == 0 and ty < 0) or (tx == 0 and ty == 0 and tz < 0)
                        ):
                            continue
                        shift = np.array([tx * cell[0], ty * cell[1], tz * cell[2]], dtype=float)
                        d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                        if d < 5.0:
                            pairs_m.append((i, j, tx, ty, tz))
    bag.append(pairs_m)
max_p = max(len(p) for p in bag)
pairs = -np.ones((cells.shape[0], max_p, 5), dtype=int)
counts = np.zeros(cells.shape[0], dtype=int)
for m, pairs_m in enumerate(bag):
    counts[m] = len(pairs_m)
    if pairs_m:
        pairs[m, : len(pairs_m)] = np.asarray(pairs_m, dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    _, _, energies, kept = two_stage_batched_relax(cells, coords, pairs, counts)
    return np.concatenate([energies, kept.astype(float)])

def run_gold():
    _fresh_inputs()
    _, _, energies, kept = _oracle_two_stage_batched_relax(cells, coords, pairs, counts)
    return np.concatenate([energies, kept.astype(float)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([[4.25, 4.25, 4.25]], dtype=float)
coords = np.array([[[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]]], dtype=float)
cr = coords[0]
cell = cells[0]
pairs_m = []
for i in range(2):
    for j in range(2):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if (tx, ty, tz) == (0, 0, 0):
                        continue
                    if i > j:
                        continue
                    if i == j and (
                        tx < 0 or (tx == 0 and ty < 0) or (tx == 0 and ty == 0 and tz < 0)
                    ):
                        continue
                    shift = np.array([tx * cell[0], ty * cell[1], tz * cell[2]], dtype=float)
                    d = float(np.linalg.norm(cr[j] + shift - cr[i]))
                    if d < 5.0:
                        pairs_m.append((i, j, tx, ty, tz))
pairs = np.asarray(pairs_m, dtype=int).reshape(1, -1, 5)
counts = np.array([len(pairs_m)], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    rel_cells, _, energy, _ = two_stage_batched_relax(cells, coords, pairs, counts)
    return np.concatenate([rel_cells.reshape(-1), energy])

def run_gold():
    _fresh_inputs()
    rel_cells, _, energy, _ = _oracle_two_stage_batched_relax(cells, coords, pairs, counts)
    return np.concatenate([rel_cells.reshape(-1), energy])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

# Nearly coincident atoms are an invalid oracle geometry.
cells = np.array([[4.25, 4.25, 4.25]], dtype=float)
coords = np.array([[[2.125, 2.125, 2.125], [2.125, 2.125, 2.275]]], dtype=float)
pairs = np.array([[[0, 0, 1, 0, 0]]], dtype=int)
counts = np.array([1], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        two_stage_batched_relax(cells, coords, pairs, counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_two_stage_batched_relax(cells, coords, pairs, counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([[4.25, 4.25, 4.25]], dtype=float)
coords = np.array([[[2.125, 2.125, 1.575], [2.125, 2.125, 2.675]]], dtype=float)
pairs = np.zeros((1, 1, 4), dtype=int)
counts = np.array([1], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        two_stage_batched_relax(cells, coords, pairs, counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_two_stage_batched_relax(cells, coords, pairs, counts)
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
