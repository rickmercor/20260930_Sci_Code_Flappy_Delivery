"""
Evaluate the recrossing-corrected normal-crossing transmission probability.

Apply the source's normal-crossing Holstein treatment to every supplied
velocity.  Preserve its limiting conventions: finite coupling at zero
velocity gives unit transmission, while zero coupling gives zero.  Velocities
must be finite and nonnegative, coupling finite and nonnegative, and the
gradient gap strictly positive.  Invalid inputs raise ``ValueError``.

Returns
-------
np.ndarray of dimensionless Holstein-corrected probabilities, with the same shape as velocity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_holstein_transmission(
    velocity: 'np.ndarray',
    coupling: float,
    gradient_gap: float,
) -> 'np.ndarray':
    """Return the Holstein-corrected transmission at each velocity.

    Parameters
    ----------
    velocity
        Nonempty finite real vector of nonnegative crossing velocities in
        bohr per atomic unit of time.
    coupling
        Finite real nonnegative diabatic coupling in ``E_h``.
    gradient_gap
        Finite real strictly positive gradient-difference norm in
        ``E_h bohr^-1``.

    Returns
    -------
    np.ndarray
        Dimensionless Holstein-corrected probabilities with the same shape as
        ``velocity``.

    Raises
    ------
    ValueError
        If the vector or either scalar lies outside its stated domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _step4_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _oracle_compute_holstein_transmission(
    velocity: 'np.ndarray',
    coupling: float,
    gradient_gap: float,
) -> 'np.ndarray':
    raw_velocity = np.asarray(velocity)
    if (
        raw_velocity.ndim != 1
        or raw_velocity.size == 0
        or not np.issubdtype(raw_velocity.dtype, np.number)
        or not np.isrealobj(raw_velocity)
        or np.any(~np.isfinite(raw_velocity))
    ):
        raise ValueError("velocity must be a finite real vector")
    velocities = np.asarray(raw_velocity, dtype=float)
    if np.any(velocities < 0.0):
        raise ValueError("velocity must be nonnegative")
    coupling_value = _step4_real_scalar(coupling, "coupling")
    gradient_gap_value = _step4_real_scalar(gradient_gap, "gradient_gap")
    if coupling_value < 0.0 or gradient_gap_value <= 0.0:
        raise ValueError("coupling must be nonnegative and gradient_gap positive")
    if coupling_value == 0.0:
        return np.zeros_like(velocities)
    single_crossing = np.ones_like(velocities)
    positive = velocities > 0.0
    with np.errstate(over="ignore"):
        exponent = (
            -2.0
            * np.pi
            * np.square(np.float64(coupling_value))
            / (gradient_gap_value * velocities[positive])
        )
    single_crossing[positive] = -np.expm1(exponent)
    return 2.0 * single_crossing / (1.0 + single_crossing)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nv=np.array([0.0,1e-4,5e-4,2e-3])",
            "call": "compute_holstein_transmission(v.copy(),3.5e-4,0.08)",
            "gold_call": "_oracle_compute_holstein_transmission(v.copy(),3.5e-4,0.08)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nv=np.array([0.0,0.2,1.0])",
            "call": "compute_holstein_transmission(v.copy(),0.0,0.3)",
            "gold_call": "_oracle_compute_holstein_transmission(v.copy(),0.0,0.3)",
            "tol": 0.0,
        },
        {
            "setup": "import numpy as np\nv=np.array([1e-15,1e-10,1e-6])",
            "call": "compute_holstein_transmission(v.copy(),1e-8,2.5)",
            "gold_call": "_oracle_compute_holstein_transmission(v.copy(),1e-8,2.5)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nv=np.array([-0.1,0.2])\ndef check(fn):\n try: fn(v.copy(),1e-4,0.1)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(compute_holstein_transmission)",
            "gold_call": "check(_oracle_compute_holstein_transmission)",
            "tol": 0.0,
        },
    ]
