"""
Compute the source's temperature- and moisture-adjusted first-order degradation coefficient for every soil layer. A correct result converts positive Celsius temperatures to Kelvin only inside the Arrhenius term, caps moisture at field capacity, and returns zero at nonpositive Celsius temperature; a failure changes dissipation and every downstream parent/metabolite mass.

The paper replaces a fixed pesticide half-life with an environmental rate that combines Arrhenius temperature dependence and the Walker soil-moisture correction. This makes daily dissipation responsive to pedoclimatic conditions.

Returns
-------
np.ndarray with shape (L,), the daily first-order degradation coefficients in layer order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def environmental_degradation_rate(
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    field_capacity: "np.ndarray",
    k_ref: float,
    activation_energy: float,
    gas_constant: float,
    reference_temperature_k: float,
    moisture_exponent: float,
) -> "np.ndarray":
    """Return layerwise environment-adjusted daily degradation rates.
 
    Parameters
    ----------
    temperature_c, moisture, field_capacity : np.ndarray
        Finite one-dimensional arrays with the same nonempty shape. Moisture
        is nonnegative and field capacity is strictly positive.
    k_ref : float
        Finite nonnegative reference rate in d^-1.
    activation_energy : float
        Finite nonnegative activation energy in J mol^-1.
    gas_constant : float
        Finite positive gas constant in J mol^-1 K^-1.
    reference_temperature_k : float
        Finite positive reference temperature in kelvin.
    moisture_exponent : float
        Finite nonnegative Walker exponent.
 
    Returns
    -------
    rates : np.ndarray
        Daily rate coefficients with shape (L,).
 
    Raises
    ------
    ValueError
        If shapes, finiteness, or stated value ranges are violated.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_environmental_degradation_rate(
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    field_capacity: "np.ndarray",
    k_ref: float,
    activation_energy: float,
    gas_constant: float,
    reference_temperature_k: float,
    moisture_exponent: float,
) -> "np.ndarray":
    t = np.asarray(temperature_c, dtype=float)
    theta = np.asarray(moisture, dtype=float)
    fc = np.asarray(field_capacity, dtype=float)
    if t.shape != theta.shape or t.shape != fc.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("temperature, moisture, and field_capacity must share a nonempty 1D shape")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(fc)):
        raise ValueError("environmental arrays must be finite")
    if np.any(theta < 0.0) or np.any(fc <= 0.0):
        raise ValueError("moisture must be nonnegative and field_capacity positive")
    scalars = [k_ref, activation_energy, gas_constant, reference_temperature_k, moisture_exponent]
    if not all(np.isfinite(x) for x in scalars):
        raise ValueError("kinetic parameters must be finite")
    if k_ref < 0.0 or activation_energy < 0.0 or gas_constant <= 0.0 or reference_temperature_k <= 0.0 or moisture_exponent < 0.0:
        raise ValueError("kinetic parameters are outside their allowed ranges")
    out = np.zeros_like(t, dtype=float)
    mask = t > 0.0
    if np.any(mask):
        tk = t[mask] + 273.15
        walker = np.minimum(theta[mask] / fc[mask], 1.0) ** moisture_exponent
        arrhenius = np.exp(
            activation_energy * (tk - reference_temperature_k)
            / (gas_constant * tk * reference_temperature_k)
        )
        out[mask] = k_ref * arrhenius * walker
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {"setup": """import numpy as np
t=np.array([18.,24.,10.]); th=np.array([.19,.36,.10]); fc=np.array([.31,.35,.30])
""", "call": "environmental_degradation_rate(t,th,fc,.165,54000.,8.314,293.15,.7)", "gold_call": "_oracle_environmental_degradation_rate(t.copy(),th.copy(),fc.copy(),.165,54000.,8.314,293.15,.7)"},
        {"setup": """import numpy as np
t=np.array([0.,-2.]); th=np.array([.2,.2]); fc=np.array([.3,.3])
""", "call": "environmental_degradation_rate(t,th,fc,.165,54000.,8.314,293.15,.7)", "gold_call": "_oracle_environmental_degradation_rate(t.copy(),th.copy(),fc.copy(),.165,54000.,8.314,293.15,.7)"},
        {"setup": """import numpy as np
t=np.array([20.]); th=np.array([.6]); fc=np.array([.3])
""", "call": "environmental_degradation_rate(t,th,fc,.165,54000.,8.314,293.15,.7)", "gold_call": "_oracle_environmental_degradation_rate(t.copy(),th.copy(),fc.copy(),.165,54000.,8.314,293.15,.7)"},
        {"setup": """import numpy as np
t=np.array([20.,21.]); th=np.array([.2]); fc=np.array([.3,.3])
def run(fn):
    try: fn(t,th,fc,.165,54000.,8.314,293.15,.7); return 0
    except ValueError: return 1
""", "call": "run(environmental_degradation_rate)", "gold_call": "run(_oracle_environmental_degradation_rate)"},
        {"setup": """import numpy as np
t=np.array([20.]); th=np.array([.2]); fc=np.array([0.])
def run(fn):
    try: fn(t,th,fc,.165,54000.,8.314,293.15,.7); return 0
    except ValueError: return 1
""", "call": "run(environmental_degradation_rate)", "gold_call": "run(_oracle_environmental_degradation_rate)"},
    ]
