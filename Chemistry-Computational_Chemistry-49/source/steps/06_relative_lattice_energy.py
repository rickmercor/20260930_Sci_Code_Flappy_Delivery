"""
Score the unique list with the paper's relative lattice energy.

Return the paper relative lattice energy of the unique experimental

packing and the paper hit count. The unique list already carries the experimental flag,

including any identity transferred during duplicate removal. n_hits

is counted on the pre-dedup relaxed batch, not by quoting a

literature success rate. Same-packing tests use the paper's

packing-similarity identity plus the printed 9-decimal /

index-pairing choice among tied lattice images. Return both numbers.

Returns
-------
return delta_E, n_hits
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_lattice_energy(
    unique_energies: "np.ndarray",
    unique_is_experimental: "np.ndarray",
    unique_cells: "np.ndarray",
    unique_coords: "np.ndarray",
    relaxed_cells: "np.ndarray",
    relaxed_coords: "np.ndarray",
    relaxed_is_experimental: "np.ndarray",
) -> "tuple[float, int]":
    """
    Relative lattice energy of the experimental unique packing.

    Parameters
    ----------
    unique_energies : np.ndarray
        Unique fixture energies, kJ/mol.
    unique_is_experimental : np.ndarray
        Unique 0/1 flags.
    unique_cells, unique_coords
        Unique geometries.
    relaxed_cells, relaxed_coords, relaxed_is_experimental
        Relaxed batch before duplicate removal.

    Returns
    -------
    delta_E : float
        Relative lattice energy, kJ/mol.
    n_hits : int
        Hit count.

    Raises
    ------
    ValueError
        If unique and relaxed blocks are misaligned or empty.
    """
    return delta_E, n_hits

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_relative_lattice_energy(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

unique_energies = np.array([-30.5, -22.0, -14.0, -6.0], dtype=float)
unique_flags = np.array([0, 1, 0, 0], dtype=int)
unique_cells = np.array([
    [7.10, 7.10, 7.50],
    [7.70, 7.70, 8.10],
    [8.30, 8.30, 8.70],
    [7.40, 7.40, 7.80],
], dtype=float)
unique_coords = np.array([
    [[3.55, 3.55, 3.20], [3.55, 3.55, 4.30]],
    [[3.85, 3.85, 3.50], [3.85, 3.85, 4.60]],
    [[4.15, 4.15, 3.80], [4.15, 4.15, 4.90]],
    [[3.70, 3.70, 3.35], [3.70, 3.70, 4.45]],
], dtype=float)
rel_cells = unique_cells.copy()
rel_coords = unique_coords.copy()
rel_flags = unique_flags.copy()

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    delta, n_hits = relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)

def run_gold():
    _fresh_inputs()
    delta, n_hits = _oracle_relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

unique_energies = np.array([-30.0, -22.0, -6.0], dtype=float)
unique_flags = np.array([0, 1, 0], dtype=int)
unique_cells = np.array([
    [7.10, 7.10, 7.50],
    [7.70, 7.70, 8.10],
    [8.30, 8.30, 8.70],
], dtype=float)
unique_coords = np.array([
    [[3.55, 3.55, 3.20], [3.55, 3.55, 4.30]],
    [[3.85, 3.85, 3.50], [3.85, 3.85, 4.60]],
    [[4.15, 4.15, 3.80], [4.15, 4.15, 4.90]],
], dtype=float)
rel_cells = unique_cells.copy()
rel_coords = unique_coords.copy()
rel_flags = unique_flags.copy()

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    delta, n_hits = relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)

def run_gold():
    _fresh_inputs()
    delta, n_hits = _oracle_relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

unique_energies = np.array([-22.0, -6.0], dtype=float)
unique_flags = np.array([0, 0], dtype=int)
unique_cells = np.array([
    [7.70, 7.70, 8.10],
    [8.30, 8.30, 8.70],
], dtype=float)
unique_coords = np.array([
    [[3.85, 3.85, 3.50], [3.85, 3.85, 4.60]],
    [[4.15, 4.15, 3.80], [4.15, 4.15, 4.90]],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        relative_lattice_energy(
            unique_energies, unique_flags, unique_cells, unique_coords,
            unique_cells, unique_coords, unique_flags,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_relative_lattice_energy(
            unique_energies, unique_flags, unique_cells, unique_coords,
            unique_cells, unique_coords, unique_flags,
        )
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

# Experimental unique is also the global minimum: relative energy is 0.
unique_energies = np.array([-22.0], dtype=float)
unique_flags = np.array([1], dtype=int)
unique_cells = np.array([[7.70, 7.70, 8.10]], dtype=float)
unique_coords = np.array([[[3.85, 3.85, 3.50], [3.85, 3.85, 4.60]]], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    delta, n_hits = relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        unique_cells, unique_coords, unique_flags,
    )
    return np.array([delta, n_hits], dtype=float)

def run_gold():
    _fresh_inputs()
    delta, n_hits = _oracle_relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        unique_cells, unique_coords, unique_flags,
    )
    return np.array([delta, n_hits], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

# Normal case with a non-zero hit count: two relaxed generated packings lie
# within the paired-centroid residual of the flagged experimental packing and
# one does not, so the count must be 2 and must exclude the flagged row itself.
unique_energies = np.array([-30.0, -22.0], dtype=float)
unique_flags = np.array([0, 1], dtype=int)
unique_cells = np.array([
    [7.10, 7.10, 7.50],
    [7.70, 7.70, 8.10],
], dtype=float)
unique_coords = np.array([
    [[3.55, 3.55, 3.20], [3.55, 3.55, 4.30]],
    [[3.85, 3.85, 3.50], [3.85, 3.85, 4.60]],
], dtype=float)
rel_cells = np.array([
    [7.10, 7.10, 7.50],
    [7.70, 7.70, 8.10],
    [7.75, 7.75, 8.15],
    [7.66, 7.66, 8.06],
], dtype=float)
rel_coords = np.array([
    [[3.550, 3.550, 3.200], [3.550, 3.550, 4.300]],
    [[3.850, 3.850, 3.500], [3.850, 3.850, 4.600]],
    [[3.875, 3.875, 3.525], [3.875, 3.875, 4.625]],
    [[3.830, 3.830, 3.480], [3.830, 3.830, 4.580]],
], dtype=float)
rel_flags = np.array([0, 1, 0, 0], dtype=int)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    delta, n_hits = relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)

def run_gold():
    _fresh_inputs()
    delta, n_hits = _oracle_relative_lattice_energy(
        unique_energies, unique_flags, unique_cells, unique_coords,
        rel_cells, rel_coords, rel_flags,
    )
    return np.array([delta, n_hits], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
