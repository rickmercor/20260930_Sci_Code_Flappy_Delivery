"""
Remove duplicate packings with the paper's similarity rule.

The input is the relaxed batch, already scored by the energy model, with a

0/1 experimental flag on each row. Packing similarity is the source

test. The 0.2 Å paired-centroid residual

is the stated match cut. On this small-Z cell the paper's image

cluster has tied origin-distances. The instance keys that make that

cluster unique are: domain `{−2, −1, 0, 1, 2}³`; distance key =

origin-distance rounded to 9 decimals; remaining-tie key =

`(na, nb, nc)`; pairing key = shared `(na, nb, nc)`, not an

independent radius sort. A generated match family is scored by its

lowest fixture energy. The experimental packing is compared as a query

against each unique representative: a hit increments the hit count and

does not replace the experimental unique energy with that family's

energy. The experimental unique keeps the relaxed seed's own fixture

energy unless the seed itself matches an already-kept unique under the

printed pairing keys. Return the kept indices and the corresponding

unique cells, coordinates, energies, and flags.

Returns
-------
return unique_index, unique_cells, unique_coords, unique_energies, unique_flags
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def deduplicate_ccdc_packing(
    cells: "np.ndarray",
    coords: "np.ndarray",
    energies: "np.ndarray",
    is_experimental: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Deduplicate the relaxed packing list.

    Tied-image selection, index pairing, and lower-energy
    experimental inheritance are the instance keys in the module
    background. Packing similarity is from the source;
    the 0.2 Å residual is the stated match cut.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_relaxed, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_relaxed, n_atoms, 3).
    energies : np.ndarray
        Fixture energies, shape (n_relaxed,).
    is_experimental : np.ndarray
        0/1 flags, shape (n_relaxed,).

    Returns
    -------
    unique_index : np.ndarray
        Indices of the kept representatives.
    unique_cells, unique_coords, unique_energies, unique_flags
        Deduplicated blocks.

    Raises
    ------
    ValueError
        If geometries, energies, or experimental flags are misaligned.
    """
    return unique_index, unique_cells, unique_coords, unique_energies, unique_flags

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_deduplicate_ccdc_packing(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Independent relaxed-like geometries. Not the production 04 outputs.
cells = np.array([
    [5.20, 5.20, 5.60],
    [5.21, 5.21, 5.61],
    [5.80, 5.80, 6.10],
    [6.40, 6.40, 6.80],
    [5.40, 5.40, 5.80],
], dtype=float)
coords = np.array([
    [[2.60, 2.60, 2.25], [2.60, 2.60, 3.35]],
    [[2.605, 2.605, 2.255], [2.605, 2.605, 3.355]],
    [[2.90, 2.90, 2.50], [2.90, 2.90, 3.60]],
    [[3.20, 3.20, 2.85], [3.20, 3.20, 3.95]],
    [[2.70, 2.70, 2.35], [2.70, 2.70, 3.45]],
], dtype=float)
energies = np.array([-30.5, -22.0, -14.0, -6.0, -3.0], dtype=float)
flags = np.array([0, 0, 0, 0, 1], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    idx, _, _, out_e, out_f = deduplicate_ccdc_packing(
        cells, coords, energies, flags
    )
    return np.concatenate([idx.astype(float), out_e, out_f.astype(float)])

def run_gold():
    _fresh_inputs()
    idx, _, _, out_e, out_f = _oracle_deduplicate_ccdc_packing(
        cells, coords, energies, flags
    )
    return np.concatenate([idx.astype(float), out_e, out_f.astype(float)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([
    [5.20, 5.20, 5.60],
    [5.21, 5.21, 5.61],
], dtype=float)
coords = np.array([
    [[2.60, 2.60, 2.25], [2.60, 2.60, 3.35]],
    [[2.605, 2.605, 2.255], [2.605, 2.605, 3.355]],
], dtype=float)
energies = np.array([-10.0, -9.5], dtype=float)
flags = np.array([0, 1], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    idx, _, _, _, out_flags = deduplicate_ccdc_packing(cells, coords, energies, flags)
    return np.concatenate([np.array([idx.size], dtype=float), out_flags.astype(float)])

def run_gold():
    _fresh_inputs()
    idx, _, _, _, out_flags = _oracle_deduplicate_ccdc_packing(cells, coords, energies, flags)
    return np.concatenate([np.array([idx.size], dtype=float), out_flags.astype(float)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([
    [5.20, 5.20, 5.60],
    [6.40, 6.40, 6.80],
], dtype=float)
coords = np.array([
    [[2.60, 2.60, 2.25], [2.60, 2.60, 3.35]],
    [[3.20, 3.20, 2.85], [3.20, 3.20, 3.95]],
], dtype=float)
energies = np.array([-12.0, -8.0], dtype=float)
flags = np.array([1, 0], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    idx, _, _, out_e, out_f = deduplicate_ccdc_packing(cells, coords, energies, flags)
    return np.concatenate([idx.astype(float), out_e, out_f.astype(float)])

def run_gold():
    _fresh_inputs()
    idx, _, _, out_e, out_f = _oracle_deduplicate_ccdc_packing(cells, coords, energies, flags)
    return np.concatenate([idx.astype(float), out_e, out_f.astype(float)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

cells = np.array([[5.20, 5.20, 5.60]], dtype=float)
coords = np.array([[[2.60, 2.60, 2.25], [2.60, 2.60, 3.35]]], dtype=float)
energies = np.array([np.nan], dtype=float)
flags = np.array([1], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        deduplicate_ccdc_packing(cells, coords, energies, flags)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_deduplicate_ccdc_packing(cells, coords, energies, flags)
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
