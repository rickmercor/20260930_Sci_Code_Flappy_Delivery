"""
Orchestrate the complete sensitivity calculation and return its Frobenius norm. A passing implementation delegates to the sensitivity-matrix function and reduces every matrix entry exactly once; a failing result misses a pathway, response, or standardization and cannot reproduce the independent literal integration targets.

The Frobenius norm is a single deterministic measure of aggregate standardized response across the selected fate pools and source-mechanism parameters. It is a compact benchmark, not a universal ecological risk metric.

Returns
-------
float, the Frobenius norm of the standardized full-pathway sensitivity matrix as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def epic_fate_sensitivity_norm(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> float:
    """Return the aggregate standardized sensitivity norm.
 
    Returns
    -------
    norm : float
        Native Python float equal to the matrix Frobenius norm.
 
    Raises
    ------
    ValueError
        If any sensitivity-matrix or upstream simulation contract is invalid.
    """
    return norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_epic_fate_sensitivity_norm(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> float:
    """Reference end-to-end standardized sensitivity norm."""
    matrix = _oracle_pathway_sensitivity_matrix(
        forcing, profile, parameters, sensitive_keys, response_indices, relative_step
    )
    return float(np.linalg.norm(matrix, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    base = "import numpy as np\nforcing = {\n    'temperature_c': np.array([[18.0, 16.5, 15.0], [20.0, 18.2, 16.4], [22.0, 19.5, 17.0], [19.0, 17.5, 16.0], [24.0, 21.0, 18.5], [21.0, 19.0, 17.2], [17.0, 16.0, 15.2]], dtype=np.float64),\n    'moisture': np.array([[0.19, 0.24, 0.29], [0.2, 0.25, 0.3], [0.28, 0.3, 0.32], [0.24, 0.28, 0.31], [0.33, 0.34, 0.36], [0.27, 0.31, 0.34], [0.25, 0.29, 0.33]], dtype=np.float64),\n    'rainfall': np.array([0.0, 6.0, 18.0, 0.0, 32.0, 4.0, 15.0], dtype=np.float64),\n    'leaf_area_index': np.array([2.2, 2.18, 2.15, 2.12, 2.08, 2.04, 2.0], dtype=np.float64),\n    'ground_cover_fraction': np.array([0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66], dtype=np.float64),\n    'retention_current': np.array([35.0, 38.0, 52.0, 44.0, 68.0, 48.0, 57.0], dtype=np.float64),\n    'retention_maximum': np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0], dtype=np.float64)\n}\nprofile = {\n    'layer_thickness': np.array([1.0, 4.0, 10.0], dtype=np.float64),\n    'soil_mass': np.array([13.0, 52.0, 130.0], dtype=np.float64),\n    'field_capacity': np.array([0.31, 0.33, 0.35], dtype=np.float64),\n    'glyphosate_kf': np.array([38.37, 38.37, 38.37], dtype=np.float64),\n    'glyphosate_exponent': np.array([1.28, 1.28, 1.28], dtype=np.float64),\n    'ampa_kf': np.array([22.0, 26.0, 31.0], dtype=np.float64),\n    'ampa_exponent': np.array([1.16, 1.18, 1.2], dtype=np.float64)\n}\nparameters = {\n    'application_mass': 144.0,\n    'k_ref': 0.165,\n    'activation_energy': 54000.0,\n    'gas_constant': 8.314,\n    'reference_temperature_k': 293.15,\n    'moisture_exponent': 0.7,\n    'solubility_g_l': 12.0,\n    'foliar_half_life_days': 10.6,\n    'transformation_fraction': 0.3,\n    'movement_fraction': 0.5,\n    'adsorption_fraction': 0.08,\n    'interception_alpha': 0.25,\n    'glyphosate_kf_scale': 1.0,\n    'glyphosate_exponent_scale': 1.0,\n    'ampa_kf_scale': 1.0,\n    'ampa_exponent_scale': 1.0\n}\nsensitive_keys = ['k_ref', 'glyphosate_kf_scale', 'glyphosate_exponent_scale', 'ampa_kf_scale', 'ampa_exponent_scale', 'solubility_g_l', 'foliar_half_life_days', 'movement_fraction', 'adsorption_fraction', 'transformation_fraction']\nresponse_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)\nrelative_step = 0.0001\n"
    return [
        {"setup": base, "call": "epic_fate_sensitivity_norm(forcing,profile,parameters,sensitive_keys,response_indices,relative_step)", "gold_call": "_oracle_epic_fate_sensitivity_norm(forcing,profile,parameters,sensitive_keys,response_indices,relative_step)"},
        {"setup": base + "\nkeys=['k_ref']; idx=np.array([1],dtype=int)\n", "call": "epic_fate_sensitivity_norm(forcing,profile,parameters,keys,idx,5e-5)", "gold_call": "_oracle_epic_fate_sensitivity_norm(forcing,profile,parameters,keys,idx,5e-5)"},
        {"setup": base + "\nkeys=['transformation_fraction']; idx=np.array([4,5],dtype=int)\n", "call": "epic_fate_sensitivity_norm(forcing,profile,parameters,keys,idx,2e-4)", "gold_call": "_oracle_epic_fate_sensitivity_norm(forcing,profile,parameters,keys,idx,2e-4)"},
        {"setup": base + "\nkeys=['k_ref']; idx=np.array([1],dtype=int)\ndef run(fn):\n    try: fn(forcing,profile,parameters,keys,idx,1.0); return 0\n    except ValueError: return 1\n", "call": "run(epic_fate_sensitivity_norm)", "gold_call": "run(_oracle_epic_fate_sensitivity_norm)"},
        {"setup": base + "\nkeys=[]; idx=np.array([1],dtype=int)\ndef run(fn):\n    try: fn(forcing,profile,parameters,keys,idx,1e-4); return 0\n    except ValueError: return 1\n", "call": "run(epic_fate_sensitivity_norm)", "gold_call": "run(_oracle_epic_fate_sensitivity_norm)"},
    ]
