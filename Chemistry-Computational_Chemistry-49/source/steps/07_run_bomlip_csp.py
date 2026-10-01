"""
Orchestrate the complete BOMLIP-CSP ranking pipeline.

The inputs are the printed P1 two-atom fixture: nuclear charges, the

selected conformer, the eight Sobol rows, and the experimental cell

and coordinates. Run the earlier stages on those arrays and return

one relative lattice energy in kJ/mol. Do not run MACE-OFF, SevenNet,

DFT, or VASP. Do not replace the chain by a single energy evaluation

of the experimental cell, by a density-only generator, or by an

E/atom ranking. The tagged value is this fixture, not a published

average.

Returns
-------
return delta_E
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_bomlip_csp(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> float:
    """
    Execute the complete ranking pipeline.

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
    float
        Relative lattice energy, kJ/mol.

    Raises
    ------
    ValueError
        If the experimental packing is missing from the trial pool, or
        if any pipeline stage is invalid.
    """
    return delta_E

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_bomlip_csp(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> float:
    cells, coords, is_exp = _oracle_generate_sobol_trial_crystals(
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

    keep = _oracle_apply_triple_radius_validation(atomic_numbers, cells, coords)
    keep = np.asarray(keep, dtype=int).reshape(-1)
    if keep.size != cells.shape[0]:
        raise ValueError("keep mask must cover every trial crystal")
    valid = np.flatnonzero(keep)
    if valid.size == 0:
        raise ValueError("triple-radius validation removed every crystal")
    if int(keep[-1]) != 1:
        raise ValueError("the experimental packing must survive triple-radius validation")

    pairs, counts = _oracle_build_batched_pbc_neighbor_lists(cells[valid], coords[valid])
    rel_cells, rel_coords, rel_e, kept = _oracle_two_stage_batched_relax(
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

    unique_idx, u_cells, u_coords, u_e, u_flags = _oracle_deduplicate_ccdc_packing(
        rel_cells, rel_coords, rel_e, flags_r
    )
    unique_idx = np.asarray(unique_idx, dtype=int).reshape(-1)
    if unique_idx.size < 1:
        raise ValueError("duplicate removal produced an empty unique list")

    delta, n_hits = _oracle_relative_lattice_energy(
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
sobol_vectors = np.array([
    [0.2424242424242423, 0.2424242424242423, 0.2424242424242423, 0.00, 0.00, 0.30],
    [0.2484848484848484, 0.2484848484848484, 0.2484848484848484, 0.02, 0.00, 0.30],
    [0.2606060606060606, 0.2606060606060606, 0.2606060606060606, 0.00, 0.00, 0.30],
    [0.3181818181818181, 0.3181818181818181, 0.3181818181818181, 0.00, 0.05, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.9242424242424241, 0.00, 0.00, 0.30],
    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.00, 0.00, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.2575757575757575, 0.00, 0.00, 0.9363636363636363],
    [0.2303030303030303, 0.2303030303030303, 0.2303030303030303, 0.00, 0.00, 0.30],
], dtype=float)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([[2.05, 2.05, 1.50], [2.05, 2.05, 2.60]], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return run_bomlip_csp(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )

def run_gold():
    _fresh_inputs()
    return _oracle_run_bomlip_csp(
        atomic_numbers, conformer_coords, sobol_vectors,
        experimental_cell, experimental_coords,
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
sobol_vectors = np.array([
    [0.2424242424242423, 0.2424242424242423, 0.2424242424242423, 0.00, 0.00, 0.30],
    [0.2484848484848484, 0.2484848484848484, 0.2484848484848484, 0.02, 0.00, 0.30],
    [0.2606060606060606, 0.2606060606060606, 0.2606060606060606, 0.00, 0.00, 0.30],
    [0.3181818181818181, 0.3181818181818181, 0.3181818181818181, 0.00, 0.05, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.9242424242424241, 0.00, 0.00, 0.30],
    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.00, 0.00, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.2575757575757575, 0.00, 0.00, 0.9363636363636363],
    [0.2303030303030303, 0.2303030303030303, 0.2303030303030303, 0.00, 0.00, 0.30],
], dtype=float)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
experimental_coords = np.array([[2.05, 2.05, 1.50], [2.05, 2.05, 2.60]], dtype=float)
sobol = sobol_vectors.copy()
sobol[0, :3] = (np.array([4.80, 4.80, 4.80]) / 12.0 - 0.20) / 0.55

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    return run_bomlip_csp(
        atomic_numbers, conformer_coords, sobol,
        experimental_cell, experimental_coords,
    )

def run_gold():
    _fresh_inputs()
    return _oracle_run_bomlip_csp(
        atomic_numbers, conformer_coords, sobol,
        experimental_cell, experimental_coords,
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([7, 7], dtype=int)
conformer_coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]], dtype=float)
sobol_vectors = np.array([
    [0.2424242424242423, 0.2424242424242423, 0.2424242424242423, 0.00, 0.00, 0.30],
    [0.2484848484848484, 0.2484848484848484, 0.2484848484848484, 0.02, 0.00, 0.30],
    [0.2606060606060606, 0.2606060606060606, 0.2606060606060606, 0.00, 0.00, 0.30],
    [0.3181818181818181, 0.3181818181818181, 0.3181818181818181, 0.00, 0.05, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.9242424242424241, 0.00, 0.00, 0.30],
    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.00, 0.00, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.2575757575757575, 0.00, 0.00, 0.9363636363636363],
    [0.2303030303030303, 0.2303030303030303, 0.2303030303030303, 0.00, 0.00, 0.30],
], dtype=float)
experimental_coords = np.array([[2.05, 2.05, 1.50], [2.05, 2.05, 2.60]], dtype=float)
cell = np.array([9.20, 9.20, 9.20], dtype=float)

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        run_bomlip_csp(
            atomic_numbers, conformer_coords, sobol_vectors,
            cell, experimental_coords,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_run_bomlip_csp(
            atomic_numbers, conformer_coords, sobol_vectors,
            cell, experimental_coords,
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
sobol_vectors = np.array([
    [0.2424242424242423, 0.2424242424242423, 0.2424242424242423, 0.00, 0.00, 0.30],
    [0.2484848484848484, 0.2484848484848484, 0.2484848484848484, 0.02, 0.00, 0.30],
    [0.2606060606060606, 0.2606060606060606, 0.2606060606060606, 0.00, 0.00, 0.30],
    [0.3181818181818181, 0.3181818181818181, 0.3181818181818181, 0.00, 0.05, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.9242424242424241, 0.00, 0.00, 0.30],
    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.00, 0.00, 0.30],
    [0.2575757575757575, 0.2575757575757575, 0.2575757575757575, 0.00, 0.00, 0.9363636363636363],
    [0.2303030303030303, 0.2303030303030303, 0.2303030303030303, 0.00, 0.00, 0.30],
], dtype=float)
experimental_cell = np.array([4.10, 4.10, 4.10], dtype=float)
coords = np.array([[2.05, 2.05, 1.50], [2.05, 2.05, 2.60]], dtype=float)
coords[0, 0] = np.nan

_PRISTINE_ARRAYS = {k: v for k, v in list(globals().items()) if isinstance(v, np.ndarray)}
def _fresh_inputs():
    g = globals()
    for _k, _v in _PRISTINE_ARRAYS.items():
        g[_k] = _v.copy()
def run_model():
    _fresh_inputs()
    try:
        run_bomlip_csp(
            atomic_numbers, conformer_coords, sobol_vectors,
            experimental_cell, coords,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    _fresh_inputs()
    try:
        _oracle_run_bomlip_csp(
            atomic_numbers, conformer_coords, sobol_vectors,
            experimental_cell, coords,
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
