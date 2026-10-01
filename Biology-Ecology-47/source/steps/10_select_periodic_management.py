"""
Run the complete periodic-disturbance management pipeline.

The orchestrator calls every earlier public stage in order, unpacks the coupled state, preserves the shared common-effort profiles, and returns the selected plan's regret-controlled decision score as one float

Returns
-------
return one float: the selected plan's regret-controlled decision score.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_periodic_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", durations: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", policies: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float, contraction_gain: float, stability_limit: float, chi_radius: float, regret_weight: float, regret_limit: float, tie_tolerance: float) -> float:
    """Run the full ecology pipeline and return the selected regret-controlled decision score.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_select_periodic_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", durations: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", policies: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float, contraction_gain: float, stability_limit: float, chi_radius: float, regret_weight: float, regret_limit: float, tie_tolerance: float) -> float:
    niche = _oracle_noncircular_niche_profile(resource_matrix, k)
    calibration = _oracle_petra_calibration(reference, calibration_targets, candidates, max_steps)
    candidate = candidates[int(calibration[0])]
    signal = _oracle_petra_residual_signal(reference, disturbed, candidate, max_steps, indices, guild_map)
    packed = _oracle_assemble_periodic_system(base_competition, base_designs, base_loadings, base_reserves, pulse_decay_base, climate_covariance_base, idiosyncratic_noise_base, niche, signal, coefficients)
    B0 = np.asarray(base_competition, dtype=np.float64)
    D0 = np.asarray(base_designs, dtype=np.float64)
    U0 = np.asarray(base_loadings, dtype=np.float64)
    reserve0 = np.asarray(base_reserves, dtype=np.float64)
    pulse0 = np.asarray(pulse_decay_base, dtype=np.float64)
    climate0 = np.asarray(climate_covariance_base, dtype=np.float64)
    idio0 = np.asarray(idiosyncratic_noise_base, dtype=np.float64)
    sizes = [B0.size, D0.size, U0.size, reserve0.size, pulse0.size, climate0.size, idio0.size]
    cuts = np.cumsum([0] + sizes)
    B = packed[cuts[0]:cuts[1]].reshape(B0.shape)
    D = packed[cuts[1]:cuts[2]].reshape(D0.shape)
    U = packed[cuts[2]:cuts[3]].reshape(U0.shape)
    reserve = packed[cuts[3]:cuts[4]].reshape(reserve0.shape)
    pulse = packed[cuts[4]:cuts[5]].reshape(pulse0.shape)
    climate = packed[cuts[5]:cuts[6]].reshape(climate0.shape)
    idio = packed[cuts[6]:cuts[7]].reshape(idio0.shape)
    certificates = _oracle_plan_certificates(B, D, buffer, minimum_effort)
    profiles = _oracle_management_profile_table(B, D, U, reserve, pulse, climate, idio, durations, signal[4:], certificates, upper_efforts, policies, costs, contraction_gain, stability_limit, chi_radius)
    selected = _oracle_select_regret_controlled_plan(profiles, regret_weight, regret_limit, tie_tolerance)
    return float(selected[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = "import numpy as np\nresource_matrix = [[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0], [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]]\nk = 10000.0\n\nreference_trajectories = [[[62, 26, 8, 3, 1], [56, 28, 10, 4, 2], [48, 31, 12, 6, 3], [38, 32, 17, 8, 5], [28, 30, 22, 12, 8], [21, 25, 25, 19, 10], [17, 20, 26, 23, 14], [13, 18, 23, 26, 20]], [[65, 24, 7, 3, 1], [57, 28, 9, 4, 2], [49, 31, 12, 5, 3], [39, 31, 17, 8, 5], [29, 30, 21, 12, 8], [21, 26, 25, 17, 11], [16, 21, 26, 22, 15], [12, 18, 24, 26, 20]], [[60, 28, 8, 3, 1], [53, 30, 11, 4, 2], [46, 32, 13, 6, 3], [37, 31, 18, 9, 5], [29, 28, 22, 13, 8], [22, 24, 24, 19, 11], [18, 20, 25, 23, 15], [14, 18, 23, 25, 20]], [[66, 23, 7, 3, 1], [58, 27, 9, 4, 2], [49, 30, 12, 6, 3], [39, 30, 17, 9, 5], [29, 29, 20, 14, 8], [21, 25, 24, 19, 11], [16, 21, 25, 22, 16], [11, 19, 24, 25, 21]], [[59, 27, 10, 3, 1], [53, 29, 12, 4, 2], [45, 32, 14, 6, 3], [37, 32, 17, 9, 5], [29, 29, 22, 12, 8], [23, 24, 24, 18, 11], [18, 20, 26, 21, 15], [15, 16, 24, 25, 20]], [[63, 27, 6, 3, 1], [56, 29, 9, 4, 2], [47, 32, 12, 6, 3], [38, 32, 17, 8, 5], [28, 30, 21, 13, 8], [21, 25, 25, 18, 11], [17, 20, 26, 22, 15], [13, 17, 24, 26, 20]]]\ncalibration_targets = [[53, 30, 10, 5, 2], [36, 31, 18, 10, 5], [20, 24, 25, 19, 12]]\ndisturbed_trajectory = [[56, 28, 10, 4, 2], [46, 31, 14, 6, 3], [18, 17, 18, 25, 22], [20, 20, 20, 23, 17], [22, 22, 21, 20, 15], [20, 23, 23, 19, 15]]\ncandidates = [(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps = 4\nstate_indices = [1, 2, 4]  # pre-disturbance, disturbance, assessment\nguild_to_species = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]]\n\nbase_competition = [[[0.0, 0.515, 0.858, 1.099, 0.88, 0.244], [0.218, 0.0, 0.792, 0.794, 0.891, 0.134], [1.02, 0.121, 0.0, 0.869, 0.803, 0.199], [0.396, 0.345, 0.452, 0.0, 0.859, 0.543], [0.876, 0.526, 0.397, 0.614, 0.0, 0.387], [0.398, 0.691, 0.511, 0.179, 0.331, 0.0]], [[0.0, 0.685, 1.068, 1.013, 0.914, 0.3], [0.205, 0.0, 0.669, 0.87, 0.757, 0.15], [1.302, 0.093, 0.0, 0.783, 0.697, 0.25], [0.292, 0.416, 0.471, 0.0, 0.952, 0.72], [0.828, 0.54, 0.443, 0.606, 0.0, 0.406], [0.372, 0.636, 0.487, 0.133, 0.421, 0.0]], [[0.0, 0.628, 1.112, 1.418, 1.106, 0.223], [0.226, 0.0, 1.133, 0.992, 1.014, 0.177], [1.352, 0.158, 0.0, 1.025, 0.93, 0.232], [0.337, 0.409, 0.347, 0.0, 1.136, 0.366], [1.115, 0.593, 0.438, 0.666, 0.0, 0.469], [0.465, 0.774, 0.403, 0.219, 0.304, 0.0]]]\nbase_designs = [[0.614, 1.117, 0.817, 1.164, 1.254, 0.973], [1.459, 0.848, 1.019, 0.854, 1.278, 0.643], [0.595, 1.436, 0.855, 0.827, 0.9, 0.781], [0.922, 1.044, 0.85, 1.174, 1.472, 0.922], [1.222, 1.284, 0.928, 0.63, 1.255, 0.553], [1.044, 1.404, 0.966, 0.579, 0.665, 0.716]]\nbase_loadings = [[[0.306, 0.292], [-0.313, 0.23], [-0.395, 0.32], [0.382, 0.177], [0.361, 0.427], [-0.416, -0.493]], [[-0.198, 0.303], [-0.129, 0.568], [-0.205, -0.096], [0.252, 0.326], [-0.297, 0.106], [-0.056, -0.095]], [[-0.008, 0.031], [-0.332, 0.017], [0.395, 0.785], [0.069, 0.69], [0.083, -0.445], [-0.045, -0.223]]]\nbase_reserves = [0.012, 0.015, 0.013, 0.010, 0.014, 0.012]\nupper_efforts = [8.0, 7.6, 8.4, 7.9, 8.2, 7.7]\nplan_costs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\nplan_policy = [[1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06]]  # gain, effort penalty, covariance penalty, Floquet penalty, curvature\nbuffer = 1.2\nminimum_effort = 0.2\npulse_decay_base = [[0.36, 0.41, 0.34, 0.44, 0.39, 0.33], [0.43, 0.31, 0.40, 0.35, 0.46, 0.29], [0.32, 0.47, 0.37, 0.30, 0.41, 0.38], [0.39, 0.35, 0.29, 0.46, 0.33, 0.42], [0.45, 0.38, 0.43, 0.28, 0.36, 0.31], [0.34, 0.44, 0.32, 0.40, 0.30, 0.47]]\nclimate_covariance_base = [[[1.0, 0.72], [0.72, 1.25]], [[0.85, -0.63], [-0.63, 1.35]], [[1.30, 0.81], [0.81, 0.95]]]\nidiosyncratic_noise_base = [[0.0018, 0.0022, 0.0016, 0.0025, 0.0019, 0.0021], [0.0021, 0.0017, 0.0024, 0.0018, 0.0023, 0.0016], [0.0016, 0.0025, 0.0020, 0.0022, 0.0017, 0.0024]]\nseason_durations = [[0.43, 0.57], [0.51, 0.49], [0.61, 0.39]]\ncontraction_gain = 0.045\nstability_limit = 0.91\nchi_radius = 2.447746830680816\nregret_weight = 0.06\nregret_limit = 0.75\ntie_tolerance = 1.0e-9\ncoefficients = [1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5, 0.55, 0.9, 0.45]\n"
    return [
        {"setup": common, "call": 'select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "gold_call": '_oracle_select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "tol": 2e-8},
        {"setup": common + "\nregret_weight=0.0", "call": 'select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "gold_call": '_oracle_select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "tol": 2e-8},
        {"setup": common + "\nidiosyncratic_noise_base=np.asarray(idiosyncratic_noise_base)*1.15", "call": 'select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "gold_call": '_oracle_select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)', "tol": 2e-8},
        {"setup": common + "\ncoefficients=np.asarray(coefficients)[:-1]\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: " + 'select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)' + ")", "gold_call": "caught(lambda: " + '_oracle_select_periodic_management(resource_matrix,k,reference_trajectories,calibration_targets,disturbed_trajectory,candidates,max_steps,state_indices,guild_to_species,base_competition,base_designs,base_loadings,base_reserves,pulse_decay_base,climate_covariance_base,idiosyncratic_noise_base,season_durations,upper_efforts,plan_costs,plan_policy,coefficients,buffer,minimum_effort,contraction_gain,stability_limit,chi_radius,regret_weight,regret_limit,tie_tolerance)' + ")", "tol": 0.0},
    ]
