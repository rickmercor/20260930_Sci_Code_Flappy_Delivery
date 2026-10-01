"""
Evaluate the directional energy quantities.

Directional reaction states carry the energy estimate used in the source thermodynamic comparison. Energies retain the orientation of the stoichiometric columns.

Returns
-------
A finite ndarray with shape directional.shape[:-1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_panel(
    directional: "np.ndarray", gas_constant: float, temperature: float
) -> "np.ndarray":
    """
    directional is a positive finite nonempty array whose last axis is [forward,
    reverse]. gas_constant is positive and finite in kJ mol^-1 K^-1; temperature is
    positive and finite in K. Return the source directional energy statistic, in
    kJ mol^-1, with the last pair axis removed. The pipeline compares this statistic
    only at active net reactions. Its value at a blocked net reaction does not
    determine that reaction's physiological Gibbs energy. Invalid inputs raise
    ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_energy_panel(
    directional: "np.ndarray", gas_constant: float, temperature: float
) -> "np.ndarray":
    import numpy as np

    p = np.asarray(directional, dtype=float)
    R = float(gas_constant)
    T = float(temperature)
    if (
        p.ndim < 1
        or p.shape[-1] != 2
        or p.size == 0
        or (not np.isfinite(p).all())
        or np.any(p <= 0)
        or (not np.isfinite(R))
        or (not np.isfinite(T))
        or (R <= 0)
        or (T <= 0)
    ):
        raise ValueError("Positive paired rates, R and T required")
    return R * T * (np.log(p[..., 1]) - np.log(p[..., 0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\np = np.array([[3.0, 1.0], [2.0, 5.0]])",
            "call": "energy_panel(p,.00831446261815324,298.15)",
            "gold_call": "_oracle_energy_panel(p,.00831446261815324,298.15)",
        },
        {
            "setup": "import numpy as np\np = np.array([[2.0, 2.0]])",
            "call": "energy_panel(p,1.,1.)",
            "gold_call": "_oracle_energy_panel(p,1.,1.)",
        },
        {
            "setup": "import numpy as np\np = np.array([[1e-200, 1e+200]])",
            "call": "energy_panel(p,1.,1.)",
            "gold_call": "_oracle_energy_panel(p,1.,1.)",
        },
        {
            "setup": "import numpy as np\n"
            "p = np.array([[0.0, 1.0]])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(energy_panel, (p,1.0,1.0,))",
            "gold_call": "_error_check(_oracle_energy_panel, (p,1.0,1.0,))",
            "tol": 0.0,
        },
    ]
