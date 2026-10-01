"""
Advance all canopy, layered parent/metabolite, and export pools through one declared daily event sequence. Passing behavior reuses the earlier process functions, delays new AMPA until after existing-AMPA dissipation, and preserves nonnegative mass bookkeeping; failure changes the coupled state even when individual formulas are correct.

The paper's added mechanisms interact within a daily time step: rainfall transfers canopy residue, environmental conditions drive degradation, parent loss forms metabolite, nonlinear partition determines pore-water concentration, and preferential flow redistributes or exports dissolved mass.

Returns
-------
np.ndarray with shape (2L + 3,), containing [canopy GLY, L soil GLY, L soil AMPA, cumulative GLY export, cumulative AMPA export]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def daily_glyphosate_ampa_update(
    state: "np.ndarray",
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    retention_current: float,
    retention_maximum: float,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Advance the coupled parent--metabolite state by one day.
 
    State order is [canopy GLY, L soil GLY, L soil AMPA, cumulative GLY
    export, cumulative AMPA export]. Dictionaries use the prompt keys and are
    not mutated.
 
    Returns
    -------
    next_state : np.ndarray
        Nonnegative state with shape (2L+3,) in the same order.
 
    Raises
    ------
    ValueError
        If state/profile/daily shapes, finiteness, water volume, or an upstream
        process contract is invalid.
    """
    return next_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_daily_glyphosate_ampa_update(
    state: "np.ndarray",
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    retention_current: float,
    retention_maximum: float,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    thickness = np.asarray(profile["layer_thickness"], dtype=float)
    layers = thickness.size
    y = np.asarray(state, dtype=float).copy()
    if y.shape != (2 * layers + 3,) or not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("state has an invalid shape or value")
    soil_mass = np.asarray(profile["soil_mass"], dtype=float)
    field_capacity = np.asarray(profile["field_capacity"], dtype=float)
    gly_kf = np.asarray(profile["glyphosate_kf"], dtype=float) * float(parameters["glyphosate_kf_scale"])
    gly_n = np.asarray(profile["glyphosate_exponent"], dtype=float) * float(parameters["glyphosate_exponent_scale"])
    ampa_kf = np.asarray(profile["ampa_kf"], dtype=float) * float(parameters["ampa_kf_scale"])
    ampa_n = np.asarray(profile["ampa_exponent"], dtype=float) * float(parameters["ampa_exponent_scale"])
    for arr in [soil_mass, field_capacity, gly_kf, gly_n, ampa_kf, ampa_n, np.asarray(temperature_c), np.asarray(moisture)]:
        if np.asarray(arr).shape != (layers,):
            raise ValueError("all profile and daily layer arrays must have shape (L,)")
    canopy = y[0]
    gly = y[1 : 1 + layers].copy()
    ampa = y[1 + layers : 1 + 2 * layers].copy()
    export_gly, export_ampa = y[-2:]
 
    canopy_result = _oracle_canopy_washoff(
        canopy,
        rainfall,
        leaf_area_index,
        ground_cover_fraction,
        parameters["solubility_g_l"],
        parameters["foliar_half_life_days"],
        parameters["interception_alpha"],
    )
    canopy = canopy_result[0]
    gly[0] += canopy_result[1]
 
    rate = _oracle_environmental_degradation_rate(
        temperature_c,
        moisture,
        field_capacity,
        parameters["k_ref"],
        parameters["activation_energy"],
        parameters["gas_constant"],
        parameters["reference_temperature_k"],
        parameters["moisture_exponent"],
    )
    gly_after = gly * np.exp(-rate)
    gly_degraded = gly - gly_after
    ampa_after = ampa * np.exp(-rate) + parameters["transformation_fraction"] * gly_degraded
    gly = gly_after
    ampa = ampa_after
 
    water_volume = np.asarray(moisture, dtype=float) * thickness * 10.0
    if np.any(water_volume <= 0.0):
        raise ValueError("daily moisture must imply positive layer water volumes")
    for solute, kf, exponent, export_name in [
        (gly, gly_kf, gly_n, "gly"),
        (ampa, ampa_kf, ampa_n, "ampa"),
    ]:
        aqueous = np.array([
            _oracle_freundlich_aqueous_concentration(
                solute[i], water_volume[i], soil_mass[i], kf[i], exponent[i], 1.0e-13, 300
            )
            for i in range(layers)
        ])
        available = water_volume[0] * aqueous[0]
        routed = _oracle_preferential_bypass(
            aqueous[0],
            available,
            rainfall,
            retention_current,
            retention_maximum,
            parameters["movement_fraction"],
            parameters["adsorption_fraction"],
            thickness,
        )
        bypass = routed[1]
        deposits = routed[2 : 2 + layers]
        exported = routed[-1]
        solute[0] -= bypass
        solute += deposits
        solute[solute < 0.0] = 0.0
        if export_name == "gly":
            export_gly += exported
        else:
            export_ampa += exported
 
    return np.concatenate(([canopy], gly, ampa, [export_gly, export_ampa])).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    base = "import numpy as np\nforcing = {\n    'temperature_c': np.array([[18.0, 16.5, 15.0], [20.0, 18.2, 16.4], [22.0, 19.5, 17.0], [19.0, 17.5, 16.0], [24.0, 21.0, 18.5], [21.0, 19.0, 17.2], [17.0, 16.0, 15.2]], dtype=np.float64),\n    'moisture': np.array([[0.19, 0.24, 0.29], [0.2, 0.25, 0.3], [0.28, 0.3, 0.32], [0.24, 0.28, 0.31], [0.33, 0.34, 0.36], [0.27, 0.31, 0.34], [0.25, 0.29, 0.33]], dtype=np.float64),\n    'rainfall': np.array([0.0, 6.0, 18.0, 0.0, 32.0, 4.0, 15.0], dtype=np.float64),\n    'leaf_area_index': np.array([2.2, 2.18, 2.15, 2.12, 2.08, 2.04, 2.0], dtype=np.float64),\n    'ground_cover_fraction': np.array([0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66], dtype=np.float64),\n    'retention_current': np.array([35.0, 38.0, 52.0, 44.0, 68.0, 48.0, 57.0], dtype=np.float64),\n    'retention_maximum': np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0], dtype=np.float64)\n}\nprofile = {\n    'layer_thickness': np.array([1.0, 4.0, 10.0], dtype=np.float64),\n    'soil_mass': np.array([13.0, 52.0, 130.0], dtype=np.float64),\n    'field_capacity': np.array([0.31, 0.33, 0.35], dtype=np.float64),\n    'glyphosate_kf': np.array([38.37, 38.37, 38.37], dtype=np.float64),\n    'glyphosate_exponent': np.array([1.28, 1.28, 1.28], dtype=np.float64),\n    'ampa_kf': np.array([22.0, 26.0, 31.0], dtype=np.float64),\n    'ampa_exponent': np.array([1.16, 1.18, 1.2], dtype=np.float64)\n}\nparameters = {\n    'application_mass': 144.0,\n    'k_ref': 0.165,\n    'activation_energy': 54000.0,\n    'gas_constant': 8.314,\n    'reference_temperature_k': 293.15,\n    'moisture_exponent': 0.7,\n    'solubility_g_l': 12.0,\n    'foliar_half_life_days': 10.6,\n    'transformation_fraction': 0.3,\n    'movement_fraction': 0.5,\n    'adsorption_fraction': 0.08,\n    'interception_alpha': 0.25,\n    'glyphosate_kf_scale': 1.0,\n    'glyphosate_exponent_scale': 1.0,\n    'ampa_kf_scale': 1.0,\n    'ampa_exponent_scale': 1.0\n}\nsensitive_keys = ['k_ref', 'glyphosate_kf_scale', 'glyphosate_exponent_scale', 'ampa_kf_scale', 'ampa_exponent_scale', 'solubility_g_l', 'foliar_half_life_days', 'movement_fraction', 'adsorption_fraction', 'transformation_fraction']\nresponse_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)\nrelative_step = 0.0001\n"
    return [
        {"setup": base + "\nL=3\nstate=np.zeros(9); state[0]=80.; state[1]=40.\n", "call": "daily_glyphosate_ampa_update(state,forcing['temperature_c'][2],forcing['moisture'][2],18.,2.15,.70,52.,80.,profile,parameters)", "gold_call": "_oracle_daily_glyphosate_ampa_update(state.copy(),forcing['temperature_c'][2].copy(),forcing['moisture'][2].copy(),18.,2.15,.70,52.,80.,profile,parameters)"},
        {"setup": base + "\nstate=np.zeros(9); state[1]=40.\n", "call": "daily_glyphosate_ampa_update(state,forcing['temperature_c'][0],forcing['moisture'][0],0.,0.,0.,0.,80.,profile,parameters)", "gold_call": "_oracle_daily_glyphosate_ampa_update(state.copy(),forcing['temperature_c'][0].copy(),forcing['moisture'][0].copy(),0.,0.,0.,0.,80.,profile,parameters)"},
        {"setup": base + "\nstate=np.zeros(9); state[0]=80.; state[1]=40.\ntheta=np.array([.40,.45,.50])\n", "call": "daily_glyphosate_ampa_update(state,forcing['temperature_c'][4],theta,32.,2.08,.68,68.,80.,profile,parameters)", "gold_call": "_oracle_daily_glyphosate_ampa_update(state.copy(),forcing['temperature_c'][4].copy(),theta.copy(),32.,2.08,.68,68.,80.,profile,parameters)"},
        {"setup": base + "\nstate=np.zeros(8)\ndef run(fn):\n    try: fn(state,forcing['temperature_c'][0],forcing['moisture'][0],1.,1.,.5,20.,80.,profile,parameters); return 0\n    except ValueError: return 1\n", "call": "run(daily_glyphosate_ampa_update)", "gold_call": "run(_oracle_daily_glyphosate_ampa_update)"},
        {"setup": base + "\nstate=np.zeros(9); state[1]=1.\ntheta=np.array([0.,.2,.2])\ndef run(fn):\n    try: fn(state,forcing['temperature_c'][0],theta,1.,1.,.5,20.,80.,profile,parameters); return 0\n    except ValueError: return 1\n", "call": "run(daily_glyphosate_ampa_update)", "gold_call": "run(_oracle_daily_glyphosate_ampa_update)"},
    ]
