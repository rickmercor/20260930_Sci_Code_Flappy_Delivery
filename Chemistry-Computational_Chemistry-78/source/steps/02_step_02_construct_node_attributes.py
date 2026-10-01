"""
Build the per-atom node attributes of the source paper.

Each node is the eight-channel vector
(Z, covalent radius, Pauling electronegativity, n_s, n_p, n_d,
site_electronic, Step). f occupancy is omitted. site_electronic is
the work function on surface atoms and the HOMO-LUMO gap on
adsorbate atoms. Step is shared by every selected node.

After concatenation, apply the supplied training-set scales
channel-wise. Do not refit the scaler on this graph. Do not attach
bond distances.

Returns
-------
return node_features
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_node_attributes(
    atomic_numbers: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    """
    Construct the scaled node-feature matrix.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    bonded_adsorbate : np.ndarray
        Indices of the surface-bonded adsorbate atoms.
    is_surface : np.ndarray
        0/1 mask, 1 for clean-surface metal atoms, shape (n_atoms,).
    occupancies : np.ndarray
        Per-atom (s, p, d, f) occupancies, shape (n_atoms, 4).
    work_function : float
        Average work function of the clean surface.
    homo_lumo : float
        Gas-phase adsorbate HOMO-LUMO gap.
    step : float
        Facet encoding supplied with the fixture.
    feature_scales : np.ndarray
        Fitted channel scales, shape (8,).
    node_index : np.ndarray
        Local-site atom indices.

    Returns
    -------
    np.ndarray
        Scaled node features with shape (n_selected, 8).

    Raises
    ------
    ValueError
        If occupancies, masks, scales, or node indices are misaligned
        or invalid. A bonded adsorbate atom that is also marked
        surface is an invalid mask.
    """
    return node_features

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_node_attributes(
    atomic_numbers: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    n_features = 8
    covalent_radii = {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        13: 1.21,
        26: 1.32,
        28: 1.24,
        29: 1.32,
        30: 1.22,
        31: 1.22,
        47: 1.45,
        78: 1.36,
        79: 1.36,
    }
    pauling_en = {
        1: 2.20,
        6: 2.55,
        7: 3.04,
        8: 3.44,
        13: 1.61,
        26: 1.83,
        28: 1.91,
        29: 1.90,
        30: 1.65,
        31: 1.81,
        47: 1.93,
        78: 2.28,
        79: 2.54,
    }
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    bonded = np.asarray(bonded_adsorbate, dtype=int).reshape(-1)
    surface = np.asarray(is_surface, dtype=int).reshape(-1)
    occ = np.asarray(occupancies, dtype=float)
    node_index = np.asarray(node_index, dtype=int).reshape(-1)
    scales = np.asarray(feature_scales, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("at least one atom is required")
    if occ.ndim != 2 or occ.shape != (z.size, 4):
        raise ValueError("occupancies must have shape (n_atoms, 4)")
    if surface.size != z.size:
        raise ValueError("is_surface must have one entry per atom")
    if not np.all(np.isin(surface, (0, 1))):
        raise ValueError("is_surface must be a 0/1 mask")
    if node_index.size == 0:
        raise ValueError("node_index is empty")
    if np.any(node_index < 0) or np.any(node_index >= z.size):
        raise ValueError("node_index is out of range")
    if np.unique(node_index).size != node_index.size:
        raise ValueError("node_index must be unique")
    if not np.all(np.isin(bonded, node_index)):
        raise ValueError("bonded adsorbate atoms must remain in the local site")
    if np.any(surface[bonded] != 0):
        raise ValueError("bonded adsorbate atoms must not be surface atoms")
    if not np.isfinite(work_function) or not np.isfinite(homo_lumo) or not np.isfinite(step):
        raise ValueError("scalar surface/adsorbate descriptors must be finite")
    if scales.shape != (n_features,):
        raise ValueError("feature_scales must have length 8")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("feature_scales must be positive and finite")
    if not np.all(np.isfinite(occ)):
        raise ValueError("occupancies contain non-finite values")
    if np.any(occ[node_index] < 0):
        raise ValueError("orbital occupancies must be non-negative")
    for zi in z[node_index]:
        if int(zi) not in covalent_radii:
            raise ValueError("atomic numbers must belong to the paper element set")

    sel = node_index
    z_sel = z[sel]
    radius = np.array([covalent_radii[int(zi)] for zi in z_sel], dtype=float)
    chi = np.array([pauling_en[int(zi)] for zi in z_sel], dtype=float)
    site_electronic = np.where(surface[sel] == 1, float(work_function), float(homo_lumo))
    raw = np.column_stack(
        [
            z_sel.astype(float),
            radius,
            chi,
            occ[sel, 0],
            occ[sel, 1],
            occ[sel, 2],
            site_electronic,
            np.full(sel.size, float(step)),
        ]
    )
    return raw / scales

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
bonded_adsorbate = np.array([1])
is_surface = np.array([1, 0])
occupancies = np.array([
    [0.60, 0.20, 8.40, 0.00],
    [1.80, 4.20, 0.00, 0.00],
])
work_function = 5.14
homo_lumo = 8.37
step = 1.0
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
node_index = np.array([0, 1])

def run_model():
    return construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        work_function, homo_lumo, step, feature_scales.copy(), node_index.copy(),
    )

def run_gold():
    return _oracle_construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        work_function, homo_lumo, step, feature_scales.copy(), node_index.copy(),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([31, 6, 1])
bonded_adsorbate = np.array([1])
is_surface = np.array([1, 0, 0])
occupancies = np.array([
    [1.80, 1.00, 10.0, 0.0],
    [1.60, 2.40, 0.00, 0.0],
    [1.00, 0.00, 0.00, 0.0],
])
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
node_index = np.array([0, 1, 2])

def run_model():
    return construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )

def run_gold():
    return _oracle_construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
bonded_adsorbate = np.array([1])
is_surface = np.array([1, 0])
occupancies = np.array([
    [0.50, 0.10, 8.00, 0.00],
    [1.50, 4.00, 0.00, 0.00],
])
feature_scales = np.array([28.0, 1.24, 3.44, 2.00, 4.00, 8.00, 5.00, 2.00])
node_index = np.array([0, 1])

def run_model():
    return construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        4.00, 5.00, 2.0, feature_scales.copy(), node_index.copy(),
    )

def run_gold():
    return _oracle_construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        4.00, 5.00, 2.0, feature_scales.copy(), node_index.copy(),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
bonded_adsorbate = np.array([1])
is_surface = np.array([1, 1])
occupancies = np.array([
    [0.60, 0.20, 8.40, 0.00],
    [1.80, 4.20, 0.00, 0.00],
])
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
node_index = np.array([0, 1])

def run_model():
    try:
        construct_node_attributes(
            atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
            5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_construct_node_attributes(
            atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
            5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([31, 8])
bonded_adsorbate = np.array([1])
is_surface = np.array([1, 0])
occupancies = np.array([
    [1.80, 0.90, 10.0, 2.50],
    [1.80, 4.20, 0.00, 0.00],
])
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
node_index = np.array([0, 1])

def run_model():
    return construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )

def run_gold():
    return _oracle_construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28])
bonded_adsorbate = np.array([4, 6])
is_surface = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1])
occupancies = np.array([
    [0.62, 0.18, 8.41, 0.00],
    [0.58, 0.22, 8.38, 0.00],
    [1.82, 0.95, 10.00, 0.00],
    [0.71, 0.08, 8.55, 0.00],
    [1.84, 4.21, 0.00, 0.00],
    [1.58, 2.36, 0.00, 0.00],
    [1.79, 4.28, 0.00, 0.00],
    [0.98, 0.00, 0.00, 0.00],
    [0.65, 0.15, 8.40, 0.00],
])
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
node_index = np.arange(8)

def run_model():
    return construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )

def run_gold():
    return _oracle_construct_node_attributes(
        atomic_numbers.copy(), bonded_adsorbate.copy(), is_surface.copy(), occupancies.copy(),
        5.14, 8.37, 1.0, feature_scales.copy(), node_index.copy(),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
