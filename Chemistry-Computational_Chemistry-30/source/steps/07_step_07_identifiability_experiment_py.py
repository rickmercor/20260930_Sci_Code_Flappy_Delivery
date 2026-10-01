"""
Explain why one-temperature trajectories cannot separate the energy gap and common multiplier, then implement thermal_identification for the proposed additional measurement: recover the gap and multiplier from two calibrated kinetic factors at distinct temperatures, assuming both parameters remain constant. Reject zero factors, which carry no finite identifiable gap under this protocol.

The generator depends on Δ and α only through z=α exp(−Δ/θ), where θ=k_BT. Measurements at two distinct temperatures can separate Δ and α if both remain temperature-independent and the other kinetic rates are independently determined at each temperature.

Returns
-------
np.ndarray, shape (2,), float64: [gap_eV, dimensionless_alpha] from two temperatures
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def thermal_identification(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Separate gap and prefactor using two calibrated kinetic factors.

    theta and z are real arrays of shape (2,). theta contains distinct
    positive kBT values in eV; z contains strictly positive measured
    factors k_rISC1/k_rISC2 (=k_rIC/k_IC under this model).
    Assume the same nonnegative gap and positive alpha at both
    temperatures, with other rates independently measured. Infer them
    from log(z)=log(alpha)-gap/theta. Either temperature order is valid.
    A single temperature, or alpha=0 at both temperatures, cannot
    identify the gap; zero factors are therefore outside this contract.

    Returns
    -------
    np.ndarray
        Shape (2,), [gap_eV,alpha], gap>=0 and alpha>0.

    Raises
    ------
    ValueError
        If inputs are not finite real arrays of shape (2,), a theta or
        z value is nonpositive, the two theta values are equal, the
        inferred gap is negative, or results are nonfinite or alpha
        is zero after floating-point underflow.
    """
    return np.empty(2, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_thermal_identification(theta,z):
    import numpy as np
    try:
        if np.iscomplexobj(theta) or np.iscomplexobj(z):
            raise ValueError("real inputs required")
        t = np.asarray(theta,dtype=float)
        f = np.asarray(z,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real arrays required") from exc
    if t.shape != (2,) or f.shape != (2,):
        raise ValueError("two temperatures and factors required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)) or np.any(t<=0) or np.any(f<=0) or t[0]==t[1]:
        raise ValueError("finite positive factors and distinct temperatures required")
    with np.errstate(over="ignore",divide="ignore",under="ignore",invalid="ignore"):
        logs = np.log(f)
        gap = (logs[1]-logs[0])/(1.0/t[0]-1.0/t[1])
        alpha = np.exp(logs[0]+gap/t[0])
    if not np.isfinite(gap) or not np.isfinite(alpha) or gap<0 or alpha<=0:
        raise ValueError("nonnegative finite gap and positive finite alpha required")
    return np.array([gap,alpha],dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "import numpy as np\nt = np.array([.0256,.0300])\nz = 4e-5*np.exp(-.041/t)",
            "call": "thermal_identification(t,z)",
            "gold_call": "_oracle_thermal_identification(t,z)"
        },
        {
            "setup": "import numpy as np\nt = np.array([.02,.04])\nz = np.array([.2,.2])",
            "call": "thermal_identification(t,z)",
            "gold_call": "_oracle_thermal_identification(t,z)"
        },
        {
            "setup": "import numpy as np\nt = np.array([.04,.02])\nz = .5*np.exp(-.2/t)",
            "call": "thermal_identification(t,z)",
            "gold_call": "_oracle_thermal_identification(t,z)"
        },
        {
            "setup": "import numpy as np\nt = np.array([.0256,.0256])\nz = np.array([.001,.001])\ndef _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(thermal_identification, t,z)",
            "gold_call": "_expect_value_error(_oracle_thermal_identification, t,z)"
        },
        {
            "setup": "import numpy as np\nt = np.array([.02,.04])\nz = np.array([0.,0.])\ndef _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(thermal_identification, t,z)",
            "gold_call": "_expect_value_error(_oracle_thermal_identification, t,z)"
        }
    ]
