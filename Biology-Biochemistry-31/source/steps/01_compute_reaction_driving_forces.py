"""
Compute the forward driving force for every aligned metabolite-state and reaction pair.

concentration_states_mM contains one row per metabolite state and one column per internal metabolite. All entries are concentrations in mM.

stoichiometric_columns contains one row per metabolite and one column per reaction. standard_gibbs_kj is aligned with the reaction columns and contains transformed standard Gibbs free energies in kJ mol^-1.

Convert concentrations from mM to M before taking natural logarithms. For state s and reaction r, compute

\[
\Delta_rG_{s,r}=\Delta_rG^{\circ\prime}_r+RT\sum_m N_{m,r}\ln(x_{s,m}),
\qquad f_{s,r}=-\Delta_rG_{s,r}.
\]

Preserve state and reaction order. Return the complete two-dimensional forward-driving-force array in kJ mol^-1.

concentration_states_mM must be finite, strictly positive, non-empty, and two-dimensional. stoichiometric_columns and standard_gibbs_kj must be finite and dimensionally aligned. temperature and gas_constant must be finite strictly positive real scalars. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array of shape (n_states, n_reactions) containing forward driving forces in kJ mol^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_reaction_driving_forces(concentration_states_mM: np.ndarray, stoichiometric_columns: np.ndarray, standard_gibbs_kj: np.ndarray, temperature: float, gas_constant: float) -> np.ndarray:
    """Compute aligned forward reaction driving forces."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_reaction_driving_forces(concentration_states_mM, stoichiometric_columns, standard_gibbs_kj, temperature, gas_constant):
    import numpy as np
    concentrations = np.asarray(concentration_states_mM, dtype=float)
    stoichiometry = np.asarray(stoichiometric_columns, dtype=float)
    standard = np.asarray(standard_gibbs_kj, dtype=float)
    if concentrations.ndim != 2 or min(concentrations.shape) < 1:
        raise ValueError('concentration_states_mM must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError('concentration_states_mM must contain finite strictly positive values')
    if stoichiometry.ndim != 2 or stoichiometry.shape[0] != concentrations.shape[1] or stoichiometry.shape[1] < 1 or (not np.all(np.isfinite(stoichiometry))):
        raise ValueError('stoichiometric_columns must be finite and aligned with metabolite columns')
    if standard.ndim != 1 or standard.size != stoichiometry.shape[1] or (not np.all(np.isfinite(standard))):
        raise ValueError('standard_gibbs_kj must be finite and aligned with reaction columns')
    try:
        t = float(temperature)
        r = float(gas_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError('temperature and gas_constant must be real scalars') from exc
    if not np.isfinite(t) or t <= 0.0 or (not np.isfinite(r)) or (r <= 0.0):
        raise ValueError('temperature and gas_constant must be finite and strictly positive')
    log_molar = np.log(concentrations * 0.001)
    delta_g = standard[None, :] + r * t * (log_molar @ stoichiometry)
    return -delta_g

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nconcentration_states_mM = np.array([[2.0, 5.0, 1.5], [4.0, 1.0, 3.0], [0.8, 2.2, 6.0]], dtype=float)\nstoichiometric_columns = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]], dtype=float)\nstandard_gibbs_kj = np.array([-3.2, -5.1], dtype=float)\ntemperature = 298.15\ngas_constant = 8.314462618e-3\n', 'call': 'compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[2.0, 3.0], [4.0, 1.0], [0.5, 8.0]], dtype=float)\nstoichiometric_columns = np.array([[-2.0, -1.0, 1.0], [1.0, 1.0, -2.0]], dtype=float)\nstandard_gibbs_kj = np.array([-10.0, -3.0, 2.5], dtype=float)\ntemperature = 310.0\ngas_constant = 8.314462618e-3\n', 'call': 'compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[1.2, 4.5, 2.1], [5.8, 0.9, 3.7]], dtype=float)\nstoichiometric_columns = np.array([[-1.0, 0.0, -2.0], [1.0, -1.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)\nstandard_gibbs_kj = np.array([-4.2, -6.7, -1.5], dtype=float)\nmp = np.array([2, 0, 1], dtype=int)\nrp = np.array([1, 2, 0], dtype=int)\nconcentration_states_mM = concentration_states_mM[:, mp]\nstoichiometric_columns = stoichiometric_columns[mp][:, rp]\nstandard_gibbs_kj = standard_gibbs_kj[rp]\ntemperature = 305.0\ngas_constant = 8.314462618e-3\n', 'call': 'compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_reaction_driving_forces(concentration_states_mM.copy(), stoichiometric_columns.copy(), standard_gibbs_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[2.0, 0.0]], dtype=float)\nstoichiometric_columns = np.array([[-1.0], [1.0]], dtype=float)\nstandard_gibbs_kj = np.array([-3.0], dtype=float)\ntemperature = 298.15\ngas_constant = 8.314462618e-3\ndef candidate_wrapper():\n    try:\n        compute_reaction_driving_forces(concentration_states_mM, stoichiometric_columns, standard_gibbs_kj, temperature, gas_constant)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef gold_wrapper():\n    try:\n        _oracle_compute_reaction_driving_forces(concentration_states_mM, stoichiometric_columns, standard_gibbs_kj, temperature, gas_constant)\n    except ValueError:\n        return 1.0\n    return 0.0\n', 'call': 'candidate_wrapper()', 'gold_call': 'gold_wrapper()'}]
