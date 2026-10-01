"""
Compute the response-by-parameter matrix from paired relative perturbations of the complete multi-day simulation. Passing behavior perturbs one positive parameter at a time, uses the unperturbed response for standardization, and rejects noninteger response indices; failure can look plausible while changing the final norm materially.

A standardized central sensitivity compares pathways with different units and magnitudes. Each perturbed run traverses all coupled mechanisms, so the matrix measures end-to-end response geometry rather than an isolated local derivative.

Returns
-------
np.ndarray with shape (R, P), the standardized central-difference sensitivities in requested response and parameter order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pathway_sensitivity_matrix(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> "np.ndarray":
    """Return the standardized full-pipeline response sensitivity matrix.
 
    sensitive_keys is a nonempty list of unique positive parameter names.
    response_indices is a nonempty 1D integer array of unique valid state
    indices whose baseline values are positive.
 
    Returns
    -------
    sensitivities : np.ndarray
        Shape (R,P), preserving response and parameter order.
 
    Raises
    ------
    ValueError
        If the step, keys, indices, baselines, fractions, or upstream
        simulation inputs violate their contracts. Float indices are invalid.
    """
    return sensitivities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_pathway_sensitivity_matrix(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> "np.ndarray":
    """Reference standardized central-difference sensitivity matrix."""
    if not np.isfinite(relative_step) or not 0.0 < relative_step < 1.0:
        raise ValueError("relative_step must be finite and lie in (0,1)")
    if not isinstance(sensitive_keys, list) or len(sensitive_keys) == 0 or len(set(sensitive_keys)) != len(sensitive_keys):
        raise ValueError("sensitive_keys must be a nonempty list of unique names")
    raw_indices = np.asarray(response_indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0 or raw_indices.dtype.kind not in "iu":
        raise ValueError("response_indices must be a nonempty integer array")
    baseline = _oracle_simulate_glyphosate_ampa_fate(forcing, profile, parameters)
    indices = raw_indices.astype(int, copy=False)
    if np.any(indices < 0) or np.any(indices >= baseline.size) or np.unique(indices).size != indices.size:
        raise ValueError("response_indices are out of range or duplicated")
    selected = baseline[indices]
    if np.any(selected <= 0.0):
        raise ValueError("selected baseline responses must be strictly positive")
    matrix = np.empty((indices.size, len(sensitive_keys)), dtype=float)
    for column, key in enumerate(sensitive_keys):
        if key not in parameters or not np.isfinite(parameters[key]) or parameters[key] <= 0.0:
            raise ValueError("every sensitive parameter must exist and be finite and positive")
        minus = dict(parameters)
        plus = dict(parameters)
        minus[key] = parameters[key] * (1.0 - relative_step)
        plus[key] = parameters[key] * (1.0 + relative_step)
        if key in {"movement_fraction", "adsorption_fraction", "transformation_fraction"} and (minus[key] < 0.0 or plus[key] > 1.0):
            raise ValueError("relative perturbation leaves a fraction outside [0,1]")
        y_minus = _oracle_simulate_glyphosate_ampa_fate(forcing, profile, minus)[indices]
        y_plus = _oracle_simulate_glyphosate_ampa_fate(forcing, profile, plus)[indices]
        matrix[:, column] = (y_plus - y_minus) / (2.0 * relative_step * selected)
    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    base = "import numpy as np\nforcing = {\n    'temperature_c': np.array([[18.0, 16.5, 15.0], [20.0, 18.2, 16.4], [22.0, 19.5, 17.0], [19.0, 17.5, 16.0], [24.0, 21.0, 18.5], [21.0, 19.0, 17.2], [17.0, 16.0, 15.2]], dtype=np.float64),\n    'moisture': np.array([[0.19, 0.24, 0.29], [0.2, 0.25, 0.3], [0.28, 0.3, 0.32], [0.24, 0.28, 0.31], [0.33, 0.34, 0.36], [0.27, 0.31, 0.34], [0.25, 0.29, 0.33]], dtype=np.float64),\n    'rainfall': np.array([0.0, 6.0, 18.0, 0.0, 32.0, 4.0, 15.0], dtype=np.float64),\n    'leaf_area_index': np.array([2.2, 2.18, 2.15, 2.12, 2.08, 2.04, 2.0], dtype=np.float64),\n    'ground_cover_fraction': np.array([0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66], dtype=np.float64),\n    'retention_current': np.array([35.0, 38.0, 52.0, 44.0, 68.0, 48.0, 57.0], dtype=np.float64),\n    'retention_maximum': np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0], dtype=np.float64)\n}\nprofile = {\n    'layer_thickness': np.array([1.0, 4.0, 10.0], dtype=np.float64),\n    'soil_mass': np.array([13.0, 52.0, 130.0], dtype=np.float64),\n    'field_capacity': np.array([0.31, 0.33, 0.35], dtype=np.float64),\n    'glyphosate_kf': np.array([38.37, 38.37, 38.37], dtype=np.float64),\n    'glyphosate_exponent': np.array([1.28, 1.28, 1.28], dtype=np.float64),\n    'ampa_kf': np.array([22.0, 26.0, 31.0], dtype=np.float64),\n    'ampa_exponent': np.array([1.16, 1.18, 1.2], dtype=np.float64)\n}\nparameters = {\n    'application_mass': 144.0,\n    'k_ref': 0.165,\n    'activation_energy': 54000.0,\n    'gas_constant': 8.314,\n    'reference_temperature_k': 293.15,\n    'moisture_exponent': 0.7,\n    'solubility_g_l': 12.0,\n    'foliar_half_life_days': 10.6,\n    'transformation_fraction': 0.3,\n    'movement_fraction': 0.5,\n    'adsorption_fraction': 0.08,\n    'interception_alpha': 0.25,\n    'glyphosate_kf_scale': 1.0,\n    'glyphosate_exponent_scale': 1.0,\n    'ampa_kf_scale': 1.0,\n    'ampa_exponent_scale': 1.0\n}\nsensitive_keys = ['k_ref', 'glyphosate_kf_scale', 'glyphosate_exponent_scale', 'ampa_kf_scale', 'ampa_exponent_scale', 'solubility_g_l', 'foliar_half_life_days', 'movement_fraction', 'adsorption_fraction', 'transformation_fraction']\nresponse_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)\nrelative_step = 0.0001\n"
    return [
        {"setup": base + "\nkeys=['k_ref','movement_fraction']; idx=np.arange(9,dtype=int)\n", "call": "pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,1e-4)", "gold_call": "_oracle_pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,1e-4)"},
        {"setup": base + "\nkeys=['k_ref']; idx=np.array([1],dtype=int)\n", "call": "pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,5e-5)", "gold_call": "_oracle_pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,5e-5)"},
        {"setup": base + "\nkeys=['transformation_fraction']; idx=np.array([4,5],dtype=int)\n", "call": "pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,2e-4)", "gold_call": "_oracle_pathway_sensitivity_matrix(forcing,profile,parameters,keys,idx,2e-4)"},
        {"setup": base + "\nkeys=['k_ref']; idx=np.array([1],dtype=int)\ndef run(fn):\n    try: fn(forcing,profile,parameters,keys,idx,0.0); return 0\n    except ValueError: return 1\n", "call": "run(pathway_sensitivity_matrix)", "gold_call": "run(_oracle_pathway_sensitivity_matrix)"},
        {"setup": base + "\nkeys=['k_ref']; idx=np.array([1.0],dtype=float)\ndef run(fn):\n    try: fn(forcing,profile,parameters,keys,idx,1e-4); return 0\n    except ValueError: return 1\n", "call": "run(pathway_sensitivity_matrix)", "gold_call": "run(_oracle_pathway_sensitivity_matrix)"},
    ]
