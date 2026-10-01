"""
Initialize the application and run the daily update across all forcing rows. A correct simulation validates every configuration key and preserves the exact state order without mutating inputs; a failure invalidates the endpoint used by every sensitivity calculation.

The host agroecosystem model advances daily. This reduced simulation retains the paper's new pesticide-fate mechanisms while replacing unavailable full-model hydrology with deterministic forcing arrays.

Returns
-------
np.ndarray with shape (2L + 3,), the final coupled state after all D days
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_glyphosate_ampa_fate(
    forcing: dict,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Run the deterministic coupled fate simulation through all days.
 
    Returns
    -------
    final_state : np.ndarray
        Shape (2L+3,) in the declared state order.
 
    Raises
    ------
    ValueError
        If required keys, shapes, finiteness, cover bounds, application mass,
        or an upstream daily contract is invalid.
    """
    return final_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_simulate_glyphosate_ampa_fate(
    forcing: dict,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Reference multi-day parent--metabolite simulation."""
    required_forcing = ["temperature_c", "moisture", "rainfall", "leaf_area_index", "ground_cover_fraction", "retention_current", "retention_maximum"]
    required_profile = ["layer_thickness", "soil_mass", "field_capacity", "glyphosate_kf", "glyphosate_exponent", "ampa_kf", "ampa_exponent"]
    required_parameters = ["application_mass", "k_ref", "activation_energy", "gas_constant", "reference_temperature_k", "moisture_exponent", "solubility_g_l", "foliar_half_life_days", "transformation_fraction", "movement_fraction", "adsorption_fraction", "interception_alpha", "glyphosate_kf_scale", "glyphosate_exponent_scale", "ampa_kf_scale", "ampa_exponent_scale"]
    if not all(k in forcing for k in required_forcing) or not all(k in profile for k in required_profile) or not all(k in parameters for k in required_parameters):
        raise ValueError("configuration is missing required keys")
    temp = np.asarray(forcing["temperature_c"], dtype=float)
    moisture = np.asarray(forcing["moisture"], dtype=float)
    if temp.ndim != 2 or temp.shape != moisture.shape or temp.shape[0] < 1 or temp.shape[1] < 1:
        raise ValueError("temperature_c and moisture must share a nonempty (D,L) shape")
    days, layers = temp.shape
    for key in required_forcing[2:]:
        arr = np.asarray(forcing[key], dtype=float)
        if arr.shape != (days,) or not np.all(np.isfinite(arr)):
            raise ValueError("every scalar forcing must be finite with shape (D,)")
    for key in required_profile:
        arr = np.asarray(profile[key], dtype=float)
        if arr.shape != (layers,) or not np.all(np.isfinite(arr)):
            raise ValueError("every profile array must be finite with shape (L,)")
    for key in required_parameters:
        if not np.isfinite(parameters[key]):
            raise ValueError("every required parameter must be finite")
    if parameters["application_mass"] < 0.0:
        raise ValueError("application_mass must be nonnegative")
    cover = np.asarray(forcing["ground_cover_fraction"], dtype=float)
    if np.any((cover < 0.0) | (cover > 1.0)):
        raise ValueError("ground_cover_fraction must lie in [0,1]")
 
    state = np.zeros(2 * layers + 3, dtype=float)
    state[0] = cover[0] * parameters["application_mass"]
    state[1] = (1.0 - cover[0]) * parameters["application_mass"]
    for day in range(days):
        state = _oracle_daily_glyphosate_ampa_update(
            state, temp[day], moisture[day],
            float(np.asarray(forcing["rainfall"], dtype=float)[day]),
            float(np.asarray(forcing["leaf_area_index"], dtype=float)[day]),
            float(cover[day]),
            float(np.asarray(forcing["retention_current"], dtype=float)[day]),
            float(np.asarray(forcing["retention_maximum"], dtype=float)[day]),
            profile, parameters,
        )
    return state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    base = "import numpy as np\nforcing = {\n    'temperature_c': np.array([[18.0, 16.5, 15.0], [20.0, 18.2, 16.4], [22.0, 19.5, 17.0], [19.0, 17.5, 16.0], [24.0, 21.0, 18.5], [21.0, 19.0, 17.2], [17.0, 16.0, 15.2]], dtype=np.float64),\n    'moisture': np.array([[0.19, 0.24, 0.29], [0.2, 0.25, 0.3], [0.28, 0.3, 0.32], [0.24, 0.28, 0.31], [0.33, 0.34, 0.36], [0.27, 0.31, 0.34], [0.25, 0.29, 0.33]], dtype=np.float64),\n    'rainfall': np.array([0.0, 6.0, 18.0, 0.0, 32.0, 4.0, 15.0], dtype=np.float64),\n    'leaf_area_index': np.array([2.2, 2.18, 2.15, 2.12, 2.08, 2.04, 2.0], dtype=np.float64),\n    'ground_cover_fraction': np.array([0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66], dtype=np.float64),\n    'retention_current': np.array([35.0, 38.0, 52.0, 44.0, 68.0, 48.0, 57.0], dtype=np.float64),\n    'retention_maximum': np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0], dtype=np.float64)\n}\nprofile = {\n    'layer_thickness': np.array([1.0, 4.0, 10.0], dtype=np.float64),\n    'soil_mass': np.array([13.0, 52.0, 130.0], dtype=np.float64),\n    'field_capacity': np.array([0.31, 0.33, 0.35], dtype=np.float64),\n    'glyphosate_kf': np.array([38.37, 38.37, 38.37], dtype=np.float64),\n    'glyphosate_exponent': np.array([1.28, 1.28, 1.28], dtype=np.float64),\n    'ampa_kf': np.array([22.0, 26.0, 31.0], dtype=np.float64),\n    'ampa_exponent': np.array([1.16, 1.18, 1.2], dtype=np.float64)\n}\nparameters = {\n    'application_mass': 144.0,\n    'k_ref': 0.165,\n    'activation_energy': 54000.0,\n    'gas_constant': 8.314,\n    'reference_temperature_k': 293.15,\n    'moisture_exponent': 0.7,\n    'solubility_g_l': 12.0,\n    'foliar_half_life_days': 10.6,\n    'transformation_fraction': 0.3,\n    'movement_fraction': 0.5,\n    'adsorption_fraction': 0.08,\n    'interception_alpha': 0.25,\n    'glyphosate_kf_scale': 1.0,\n    'glyphosate_exponent_scale': 1.0,\n    'ampa_kf_scale': 1.0,\n    'ampa_exponent_scale': 1.0\n}\nsensitive_keys = ['k_ref', 'glyphosate_kf_scale', 'glyphosate_exponent_scale', 'ampa_kf_scale', 'ampa_exponent_scale', 'solubility_g_l', 'foliar_half_life_days', 'movement_fraction', 'adsorption_fraction', 'transformation_fraction']\nresponse_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)\nrelative_step = 0.0001\n"
    return [
        {"setup": base, "call": "simulate_glyphosate_ampa_fate(forcing,profile,parameters)", "gold_call": "_oracle_simulate_glyphosate_ampa_fate(forcing,profile,parameters)"},
        {"setup": base + "\nfor k in list(forcing): forcing[k]=forcing[k][:1] if np.asarray(forcing[k]).ndim==1 else forcing[k][:1,:]\n", "call": "simulate_glyphosate_ampa_fate(forcing,profile,parameters)", "gold_call": "_oracle_simulate_glyphosate_ampa_fate(forcing,profile,parameters)"},
        {"setup": base + "\nparameters['application_mass']=0.0\n", "call": "simulate_glyphosate_ampa_fate(forcing,profile,parameters)", "gold_call": "_oracle_simulate_glyphosate_ampa_fate(forcing,profile,parameters)"},
        {"setup": base + "\ndel forcing['rainfall']\ndef run(fn):\n    try: fn(forcing,profile,parameters); return 0\n    except ValueError: return 1\n", "call": "run(simulate_glyphosate_ampa_fate)", "gold_call": "run(_oracle_simulate_glyphosate_ampa_fate)"},
        {"setup": base + "\nforcing['moisture']=forcing['moisture'][:,:2]\ndef run(fn):\n    try: fn(forcing,profile,parameters); return 0\n    except ValueError: return 1\n", "call": "run(simulate_glyphosate_ampa_fate)", "gold_call": "run(_oracle_simulate_glyphosate_ampa_fate)"},
    ]
