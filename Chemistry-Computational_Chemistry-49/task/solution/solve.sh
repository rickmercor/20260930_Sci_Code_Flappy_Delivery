#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def generate_sobol_trial_crystals(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    conf = np.asarray(conformer_coords, dtype=float)
    sobol = np.asarray(sobol_vectors, dtype=float)
    exp_cell = np.asarray(experimental_cell, dtype=float).reshape(-1)
    exp_coords = np.asarray(experimental_coords, dtype=float)
    if z.size == 0:
        raise ValueError("atomic_numbers is empty")
    if conf.ndim != 2 or conf.shape[1] != 3:
        raise ValueError("conformer_coords must have 3 columns")
    if sobol.ndim != 2 or sobol.shape[1] != 6:
        raise ValueError("sobol_vectors must have 6 columns")
    if exp_coords.ndim != 2 or exp_coords.shape[1] != 3:
        raise ValueError("experimental_coords must have 3 columns")
    if conf.shape[0] == 0 or sobol.shape[0] == 0 or exp_coords.shape[0] == 0:
        raise ValueError("coordinate blocks must be non-empty")
    if not np.all(np.isfinite(conf)) or not np.all(np.isfinite(sobol)):
        raise ValueError("conformer_coords contains non-finite values")
    if not np.all(np.isfinite(exp_coords)):
        raise ValueError("experimental_coords contains non-finite values")
    if conf.shape[0] != z.size or exp_coords.shape[0] != z.size:
        raise ValueError("coordinate blocks must match the atom count")
    if not np.all(np.isin(z, np.array([1, 6, 7, 8], dtype=int))):
        raise ValueError("atomic numbers must belong to the fixture element set")
    if z.size != 2:
        raise ValueError("this fixture uses a two-atom molecule")
    if exp_cell.size != 3 or np.any(exp_cell <= 0):
        raise ValueError("experimental cell must be three positive lengths")
    if not np.all(np.isfinite(exp_cell)):
        raise ValueError("experimental cell contains non-finite values")

    cells = []
    coords = []
    for s in sobol:
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size != 6:
            raise ValueError("each Sobol vector must have length 6")
        if np.any(s < 0.0) or np.any(s >= 1.0):
            raise ValueError("Sobol components must lie in [0, 1)")
        cell = 12.0 * (0.20 + 0.55 * s[:3])
        theta = 2.0 * np.pi * s[3]
        phi = np.pi * s[4]
        bond = 1.10 * (0.70 + 1.00 * s[5])
        axis = np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
            dtype=float,
        )
        if np.linalg.norm(axis) < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / np.linalg.norm(axis)
        centroid = 0.5 * np.asarray(cell, dtype=float)
        offset = 0.5 * float(bond) * axis
        coords.append(np.vstack([centroid - offset, centroid + offset]))
        cells.append(cell)
    cells.append(exp_cell.copy())
    coords.append(exp_coords.copy())
    is_experimental = np.zeros(len(cells), dtype=int)
    is_experimental[-1] = 1
    return (
        np.asarray(cells, dtype=float),
        np.asarray(coords, dtype=float),
        is_experimental,
    )

import numpy as np


def apply_triple_radius_validation(
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

import numpy as np


def build_batched_pbc_neighbor_lists(
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

import numpy as np


def two_stage_batched_relax(
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
            padded, counts = build_batched_pbc_neighbor_lists(
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

import numpy as np


def deduplicate_ccdc_packing(
    cells: "np.ndarray",
    coords: "np.ndarray",
    energies: "np.ndarray",
    is_experimental: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    cells = np.asarray(cells, dtype=float)
    coords = np.asarray(coords, dtype=float)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    flags = np.asarray(is_experimental, dtype=int).reshape(-1)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have 3 columns")
    if cells.shape[0] == 0:
        raise ValueError("cells is empty")
    if not np.all(np.isfinite(cells)):
        raise ValueError("cells contains non-finite values")
    if coords.ndim != 3 or coords.shape[0] != cells.shape[0]:
        raise ValueError("dedup batch shape is inconsistent")
    if energies.size != cells.shape[0] or flags.size != cells.shape[0]:
        raise ValueError("energies and flags must match the crystal count")
    if not np.all(np.isfinite(energies)):
        raise ValueError("energies contain non-finite values")

    order = np.argsort(energies, kind="stable")
    unique = []
    unique_flags = []
    for idx in order:
        matched = None
        for pos, j in enumerate(unique):
            cand = []
            for na in range(-2, 3):
                for nb in range(-2, 3):
                    for nc in range(-2, 3):
                        vec = np.array(
                            [na * cells[idx][0], nb * cells[idx][1], nc * cells[idx][2]],
                            dtype=float,
                        )
                        d = round(float(np.linalg.norm(vec)), 9)
                        cand.append((d, na, nb, nc))
            cand.sort()
            origin_a = 0.5 * np.asarray(cells[idx], dtype=float)
            da = {}
            for _, na, nb, nc in cand[:15]:
                da[(na, nb, nc)] = origin_a + np.array(
                    [na * cells[idx][0], nb * cells[idx][1], nc * cells[idx][2]]
                )
            cand = []
            for na in range(-2, 3):
                for nb in range(-2, 3):
                    for nc in range(-2, 3):
                        vec = np.array(
                            [na * cells[j][0], nb * cells[j][1], nc * cells[j][2]],
                            dtype=float,
                        )
                        d = round(float(np.linalg.norm(vec)), 9)
                        cand.append((d, na, nb, nc))
            cand.sort()
            origin_b = 0.5 * np.asarray(cells[j], dtype=float)
            db = {}
            for _, na, nb, nc in cand[:15]:
                db[(na, nb, nc)] = origin_b + np.array(
                    [na * cells[j][0], nb * cells[j][1], nc * cells[j][2]]
                )
            common = sorted(set(da) & set(db))
            if len(common) != 15:
                same = False
            else:
                pa = np.asarray([da[k] for k in common], dtype=float)
                pb = np.asarray([db[k] for k in common], dtype=float)
                pc = pa.mean(axis=0)
                qc = pb.mean(axis=0)
                x = pa - pc
                y = pb - qc
                u, _, vt = np.linalg.svd(x.T @ y)
                rot = vt.T @ u.T
                if np.linalg.det(rot) < 0.0:
                    vt = vt.copy()
                    vt[-1] *= -1.0
                    rot = vt.T @ u.T
                aligned = (pa - pc) @ rot + qc
                if float(np.max(np.linalg.norm(aligned - pb, axis=1))) > 0.20:
                    same = False
                else:
                    ua = coords[idx][1] - coords[idx][0]
                    ub = coords[j][1] - coords[j][0]
                    na = np.linalg.norm(ua)
                    nb = np.linalg.norm(ub)
                    if na < 1e-12 or nb < 1e-12:
                        angle = 180.0
                    else:
                        ua = ua / na
                        ub = ub / nb
                        angle = float(
                            np.degrees(
                                np.arccos(np.clip(abs(np.dot(ua, ub)), 0.0, 1.0))
                            )
                        )
                    same = angle <= 20.0
            if same:
                matched = pos
                break
        if matched is not None:
            if flags[idx] == 1:
                unique_flags[matched] = 1
            continue
        unique.append(int(idx))
        unique_flags.append(int(flags[idx]))
    unique = np.asarray(unique, dtype=int)
    unique_flags = np.asarray(unique_flags, dtype=int)
    return unique, cells[unique], coords[unique], energies[unique], unique_flags

import numpy as np


def relative_lattice_energy(
    unique_energies: "np.ndarray",
    unique_is_experimental: "np.ndarray",
    unique_cells: "np.ndarray",
    unique_coords: "np.ndarray",
    relaxed_cells: "np.ndarray",
    relaxed_coords: "np.ndarray",
    relaxed_is_experimental: "np.ndarray",
) -> "tuple[float, int]":
    energies = np.asarray(unique_energies, dtype=float).reshape(-1)
    flags = np.asarray(unique_is_experimental, dtype=int).reshape(-1)
    cells = np.asarray(unique_cells, dtype=float)
    coords = np.asarray(unique_coords, dtype=float)
    rel_cells = np.asarray(relaxed_cells, dtype=float)
    rel_coords = np.asarray(relaxed_coords, dtype=float)
    rel_flags = np.asarray(relaxed_is_experimental, dtype=int).reshape(-1)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("unique_cells must have 3 columns")
    if cells.shape[0] == 0:
        raise ValueError("unique_cells is empty")
    if not np.all(np.isfinite(cells)):
        raise ValueError("unique_cells contains non-finite values")
    if rel_cells.ndim != 2 or rel_cells.shape[1] != 3:
        raise ValueError("relaxed_cells must have 3 columns")
    if rel_cells.shape[0] == 0:
        raise ValueError("relaxed_cells is empty")
    if not np.all(np.isfinite(rel_cells)):
        raise ValueError("relaxed_cells contains non-finite values")
    if energies.size == 0:
        raise ValueError("unique list is empty")
    if flags.size != energies.size or cells.shape[0] != energies.size:
        raise ValueError("unique blocks must have the same length")
    if rel_flags.size != rel_cells.shape[0] or rel_coords.shape[0] != rel_cells.shape[0]:
        raise ValueError("relaxed blocks must have the same length")
    if np.count_nonzero(flags) != 1:
        raise ValueError("exactly one unique experimental structure is required")
    if np.count_nonzero(rel_flags) != 1:
        raise ValueError("exactly one relaxed experimental structure is required")
    exp_idx = int(np.flatnonzero(flags)[0])
    rel_exp = int(np.flatnonzero(rel_flags)[0])
    delta = float(energies[exp_idx] - np.min(energies))
    n_hits = 0
    for i in range(rel_flags.size):
        if rel_flags[i] == 1:
            continue
        cand = []
        for na in range(-2, 3):
            for nb in range(-2, 3):
                for nc in range(-2, 3):
                    vec = np.array(
                        [
                            na * rel_cells[i][0],
                            nb * rel_cells[i][1],
                            nc * rel_cells[i][2],
                        ],
                        dtype=float,
                    )
                    d = round(float(np.linalg.norm(vec)), 9)
                    cand.append((d, na, nb, nc))
        cand.sort()
        origin_a = 0.5 * np.asarray(rel_cells[i], dtype=float)
        da = {}
        for _, na, nb, nc in cand[:15]:
            da[(na, nb, nc)] = origin_a + np.array(
                [
                    na * rel_cells[i][0],
                    nb * rel_cells[i][1],
                    nc * rel_cells[i][2],
                ]
            )
        cand = []
        for na in range(-2, 3):
            for nb in range(-2, 3):
                for nc in range(-2, 3):
                    vec = np.array(
                        [
                            na * rel_cells[rel_exp][0],
                            nb * rel_cells[rel_exp][1],
                            nc * rel_cells[rel_exp][2],
                        ],
                        dtype=float,
                    )
                    d = round(float(np.linalg.norm(vec)), 9)
                    cand.append((d, na, nb, nc))
        cand.sort()
        origin_b = 0.5 * np.asarray(rel_cells[rel_exp], dtype=float)
        db = {}
        for _, na, nb, nc in cand[:15]:
            db[(na, nb, nc)] = origin_b + np.array(
                [
                    na * rel_cells[rel_exp][0],
                    nb * rel_cells[rel_exp][1],
                    nc * rel_cells[rel_exp][2],
                ]
            )
        common = sorted(set(da) & set(db))
        same = False
        if len(common) == 15:
            pa = np.asarray([da[k] for k in common], dtype=float)
            pb = np.asarray([db[k] for k in common], dtype=float)
            pc = pa.mean(axis=0)
            qc = pb.mean(axis=0)
            x = pa - pc
            y = pb - qc
            u, _, vt = np.linalg.svd(x.T @ y)
            rot = vt.T @ u.T
            if np.linalg.det(rot) < 0.0:
                vt = vt.copy()
                vt[-1] *= -1.0
                rot = vt.T @ u.T
            aligned = (pa - pc) @ rot + qc
            if float(np.max(np.linalg.norm(aligned - pb, axis=1))) <= 0.20:
                ua = rel_coords[i][1] - rel_coords[i][0]
                ub = rel_coords[rel_exp][1] - rel_coords[rel_exp][0]
                na = np.linalg.norm(ua)
                nb = np.linalg.norm(ub)
                if na < 1e-12 or nb < 1e-12:
                    angle = 180.0
                else:
                    ua = ua / na
                    ub = ub / nb
                    angle = float(
                        np.degrees(
                            np.arccos(np.clip(abs(np.dot(ua, ub)), 0.0, 1.0))
                        )
                    )
                same = angle <= 20.0
        if same:
            n_hits += 1
    return delta, int(n_hits)

import numpy as np


def run_bomlip_csp(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> float:
    cells, coords, is_exp = generate_sobol_trial_crystals(
        atomic_numbers,
        conformer_coords,
        sobol_vectors,
        experimental_cell,
        experimental_coords,
    )
    cells = np.asarray(cells, dtype=float)
    coords = np.asarray(coords, dtype=float)
    is_exp = np.asarray(is_exp, dtype=int).reshape(-1)
    if cells.shape[0] < 2:
        raise ValueError("the experimental packing must be appended to the trials")
    if int(is_exp[-1]) != 1 or int(np.sum(is_exp)) != 1:
        raise ValueError("exactly one appended experimental structure is required")

    keep = apply_triple_radius_validation(atomic_numbers, cells, coords)
    keep = np.asarray(keep, dtype=int).reshape(-1)
    if keep.size != cells.shape[0]:
        raise ValueError("keep mask must cover every trial crystal")
    valid = np.flatnonzero(keep)
    if valid.size == 0:
        raise ValueError("triple-radius validation removed every crystal")
    if int(keep[-1]) != 1:
        raise ValueError("the experimental packing must survive triple-radius validation")

    pairs, counts = build_batched_pbc_neighbor_lists(cells[valid], coords[valid])
    rel_cells, rel_coords, rel_e, kept = two_stage_batched_relax(
        cells[valid], coords[valid], pairs, counts
    )
    rel_cells = np.asarray(rel_cells, dtype=float)
    rel_coords = np.asarray(rel_coords, dtype=float)
    rel_e = np.asarray(rel_e, dtype=float).reshape(-1)
    kept = np.asarray(kept, dtype=int).reshape(-1)
    flags_r = is_exp[valid][kept]
    if rel_e.size != kept.size:
        raise ValueError("relaxed energies must align with keep_index")
    if int(np.sum(flags_r)) != 1:
        raise ValueError("the experimental packing must survive relaxation")

    unique_idx, u_cells, u_coords, u_e, u_flags = deduplicate_ccdc_packing(
        rel_cells, rel_coords, rel_e, flags_r
    )
    unique_idx = np.asarray(unique_idx, dtype=int).reshape(-1)
    if unique_idx.size < 1:
        raise ValueError("duplicate removal produced an empty unique list")

    delta, n_hits = relative_lattice_energy(
        u_e, u_flags, u_cells, u_coords, rel_cells, rel_coords, flags_r
    )
    delta = float(np.asarray(delta).reshape(-1)[0])
    n_hits = int(n_hits)
    if not np.isfinite(delta):
        raise ValueError("relative lattice energy must be finite")
    if n_hits < 0:
        raise ValueError("hit count cannot be negative")
    if delta + 1e-12 < 0.0:
        raise ValueError("relative lattice energy cannot be negative")
    return delta
SCICODE_GOLD_EOF
