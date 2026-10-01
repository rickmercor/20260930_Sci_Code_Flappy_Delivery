"""
Compute thermodynamic efficiency factors from forward driving forces.

driving_forces_kj contains one row per state and one column per reaction, in kJ mol^-1.

For every entry compute

\[
\gamma=1-\exp\left(-\frac{f}{RT}\right).
\]

Do not clip, floor, or otherwise alter the mathematical result. Negative driving force may therefore produce a negative factor; downstream eligibility determines whether that state/reaction combination is usable.

Preserve input shape and order. driving_forces_kj must be a finite, non-empty, two-dimensional array. temperature and gas_constant must be finite strictly positive real scalars. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array with the same shape as driving_forces_kj.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_thermodynamic_efficiency_factors(driving_forces_kj: np.ndarray, temperature: float, gas_constant: float) -> np.ndarray:
    """Convert forward driving forces into thermodynamic efficiency factors."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_thermodynamic_efficiency_factors(driving_forces_kj, temperature, gas_constant):
    import numpy as np
    forces = np.asarray(driving_forces_kj, dtype=float)
    if forces.ndim != 2 or min(forces.shape) < 1 or (not np.all(np.isfinite(forces))):
        raise ValueError('driving_forces_kj must be a finite non-empty two-dimensional array')
    try:
        t = float(temperature)
        r = float(gas_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError('temperature and gas_constant must be real scalars') from exc
    if not np.isfinite(t) or t <= 0.0 or (not np.isfinite(r)) or (r <= 0.0):
        raise ValueError('temperature and gas_constant must be finite and strictly positive')
    return -np.expm1(-forces / (r * t))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ndriving_forces_kj = np.array([[4.9, 2.1, 6.0, 0.5], [1.2, -0.4, 3.3, 8.0], [0.0, 5.5, -1.1, 2.4]], dtype=float)\ntemperature = 298.15\ngas_constant = 8.314462618e-3\n', 'call': 'compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\ndriving_forces_kj = np.array([[1.0e-12, 0.0, -1.0e-12], [0.25, -0.5, 3.0]], dtype=float)\ntemperature = 310.0\ngas_constant = 8.314462618e-3\n', 'call': 'compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\ndriving_forces_kj = np.array([[2.4, 5.8, 1.1, -0.3], [7.2, 0.6, 4.4, 3.1]], dtype=float)\nrp = np.array([3, 0, 2, 1], dtype=int)\ndriving_forces_kj = driving_forces_kj[:, rp]\ntemperature = 303.0\ngas_constant = 8.314462618e-3\n', 'call': 'compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)', 'gold_call': '_oracle_compute_thermodynamic_efficiency_factors(driving_forces_kj.copy(), temperature, gas_constant)'}, {'setup': 'import numpy as np\ndriving_forces_kj = np.array([[2.0, 3.0]], dtype=float)\ntemperature = 0.0\ngas_constant = 8.314462618e-3\ndef candidate_wrapper():\n    try:\n        compute_thermodynamic_efficiency_factors(driving_forces_kj, temperature, gas_constant)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef gold_wrapper():\n    try:\n        _oracle_compute_thermodynamic_efficiency_factors(driving_forces_kj, temperature, gas_constant)\n    except ValueError:\n        return 1.0\n    return 0.0\n', 'call': 'candidate_wrapper()', 'gold_call': 'gold_wrapper()'}]
