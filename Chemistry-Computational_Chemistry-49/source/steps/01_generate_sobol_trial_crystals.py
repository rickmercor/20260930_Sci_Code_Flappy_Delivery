"""
Generate Sobol trial crystals and append the experimental reference.

This instance replaces the paper's iterative cell shrink by the

closed-form map printed in the problem statement. After the generated

trials are placed, the supplied experimental packing is appended and

flagged. The fixture is one already-selected conformer of a two-atom

molecule.

Returns
-------
return cells, coords, is_experimental
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_sobol_trial_crystals(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Generate the trial-crystal batch.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    conformer_coords : np.ndarray
        Conformer coordinates, shape (n_atoms, 3).
    sobol_vectors : np.ndarray
        Sobol samples in [0, 1), shape (n_trials, 6).
    experimental_cell : np.ndarray
        Experimental cell lengths, shape (3,).
    experimental_coords : np.ndarray
        Experimental coordinates, shape (n_atoms, 3).

    Returns
    -------
    cells : np.ndarray
        Cell lengths, shape (n_trials + 1, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_trials + 1, n_atoms, 3).
    is_experimental : np.ndarray
        0/1 flags, shape (n_trials + 1,).

    Raises
    ------
    ValueError
        If Sobol vectors, conformer coordinates, or the experimental
        packing are invalid.
    """
    return cells, coords, is_experimental

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_generate_sobol_trial_crystals(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
s = np.zeros(6)
s[:3] = (np.array([4.25, 4.25, 4.25]) / 12.0 - 0.20) / 0.55
s[5] = (1.10 / 1.10 - 0.70) / 1.00
sobol_vectors = s.reshape(1, 6)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([
    [2.05, 2.05, 2.05 - 0.55],
    [2.05, 2.05, 2.05 + 0.55],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    cells, coords, flags = generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.concatenate([cells.reshape(-1), flags.astype(float), [coords[0, 1, 2] - coords[0, 0, 2]]])

def run_gold():
    _fresh_inputs()
    cells, coords, flags = _oracle_generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.concatenate([cells.reshape(-1), flags.astype(float), [coords[0, 1, 2] - coords[0, 0, 2]]])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
s = np.zeros(6)
s[:3] = (np.array([4.80, 5.10, 8.80]) / 12.0 - 0.20) / 0.55
s[3] = 0.37
s[4] = 0.41
s[5] = 0.30
sobol_vectors = s.reshape(1, 6)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([
    [2.05, 2.05, 1.50],
    [2.05, 2.05, 2.60],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    cells, coords, _ = generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.concatenate([cells[0], coords[0].reshape(-1)])

def run_gold():
    _fresh_inputs()
    cells, coords, _ = _oracle_generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.concatenate([cells[0], coords[0].reshape(-1)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
s = np.zeros(6)
s[:3] = (np.array([4.25, 4.25, 4.25]) / 12.0 - 0.20) / 0.55
s[5] = (1.85 / 1.10 - 0.70) / 1.00
sobol_vectors = s.reshape(1, 6)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([
    [2.05, 2.05, 1.50],
    [2.05, 2.05, 2.60],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    _, coords, _ = generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.linalg.norm(coords[0, 1] - coords[0, 0])

def run_gold():
    _fresh_inputs()
    _, coords, _ = _oracle_generate_sobol_trial_crystals(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
    return np.linalg.norm(coords[0, 1] - coords[0, 0])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
s = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.3, 0.0])
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([
    [2.05, 2.05, 1.50],
    [2.05, 2.05, 2.60],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        generate_sobol_trial_crystals(
            atomic_numbers, conformer_coords, s.reshape(1, -1),
            experimental_cell, experimental_coords,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_generate_sobol_trial_crystals(
            atomic_numbers, conformer_coords, s.reshape(1, -1),
            experimental_cell, experimental_coords,
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

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
s = np.zeros(6)
s[0] = 1.0
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([
    [2.05, 2.05, 1.50],
    [2.05, 2.05, 2.60],
], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        generate_sobol_trial_crystals(
            atomic_numbers, conformer_coords, s.reshape(1, 6),
            experimental_cell, experimental_coords,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_generate_sobol_trial_crystals(
            atomic_numbers, conformer_coords, s.reshape(1, 6),
            experimental_cell, experimental_coords,
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
    ]
