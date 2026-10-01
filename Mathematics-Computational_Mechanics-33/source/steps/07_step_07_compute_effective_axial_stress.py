"""
Effective stress extraction from the converged enriched displacement field.




Each cached element stress matrix maps its standard and scaled enriched nodal displacements to the integral of the local Mandel stress. Adding those fluctuation contributions to the macroscopic-strain stress integral gives the cell average because the benchmark cell has unit volume. Under unit axial Mandel strain, the first component is the effective axial stiffness C_eff,1111.




Inputs

------

element_stress : np.ndarray of shape (e, 6, 24)

element_dofs : np.ndarray of shape (e, 24)

stress_macro : np.ndarray of shape (6,)

displacement : np.ndarray of shape (d,)




Returns

-------

effective : dict

    Keys effective_stress and effective_axial_stiffness, with shapes and dtypes specified below.

Returns
-------
dict with float64 array effective_stress (6,) and native float effective_axial_stiffness.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_effective_axial_stress(
    element_stress: np.ndarray,
    element_dofs: np.ndarray,
    stress_macro: np.ndarray,
    displacement: np.ndarray,
) -> dict:
    """Accumulate the effective Mandel stress and axial stiffness.

    Parameters
    ----------
    element_stress : np.ndarray
        Cached element stress maps with shape (e, 6, 24).
    element_dofs : np.ndarray
        Global degree-of-freedom maps, with -1 for inactive entries.
    stress_macro : np.ndarray
        Integrated stress due to the prescribed macroscopic strain.
    displacement : np.ndarray
        Converged standard and scaled enriched displacement vector.

    Returns
    -------
    effective : dict
        Keys effective_stress and effective_axial_stiffness, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If element_stress is not shape (e, 6, 24), if element_dofs is not
        shape (e, 24), if stress_macro is not shape (6,) or displacement is
        not one-dimensional, or if any non-negative dof index is out of range
        for displacement.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_effective_axial_stress(
    element_stress: np.ndarray,
    element_dofs: np.ndarray,
    stress_macro: np.ndarray,
    displacement: np.ndarray,
) -> dict:
    """Reference implementation."""
    element_stress = np.asarray(element_stress, dtype=float)
    element_dofs = np.asarray(element_dofs, dtype=int)
    stress_macro = np.asarray(stress_macro, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if element_stress.ndim != 3 or element_stress.shape[1:] != (6, 24):
        raise ValueError("element_stress must have shape (e, 6, 24)")
    if element_dofs.shape != (element_stress.shape[0], 24):
        raise ValueError("element_dofs must have shape (e, 24)")
    if stress_macro.shape != (6,) or displacement.ndim != 1:
        raise ValueError("stress_macro and displacement have invalid shapes")
    if np.any(element_dofs >= displacement.size):
        raise ValueError("element_dofs contains an out-of-range index")
    effective_stress = stress_macro.copy()
    for matrix, dofs in zip(element_stress, element_dofs):
        active = dofs >= 0
        local = np.zeros(24, dtype=float)
        local[active] = displacement[dofs[active]]
        effective_stress += matrix @ local
    return {
        "effective_stress": effective_stress,
        "effective_axial_stiffness": float(effective_stress[0]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
element_stress = np.zeros((2, 6, 24))
element_stress[0, 0, 0] = 2.0
element_stress[1, 0, 0] = -0.5
element_stress[1, 3, 1] = 1.25
element_dofs = np.tile(np.arange(24, dtype=int), (2, 1))
stress_macro = np.array([10., 4., 4., 0., 0., 0.])
displacement = np.linspace(-0.2, 0.3, 24)
def summarize(result):
    return (
        np.round(result["effective_stress"], 12).tolist(),
        round(result["effective_axial_stiffness"], 12),
    )
""",
            "call": "summarize(compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement))",
            "gold_call": "summarize(_oracle_compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement))",
        },
        {
            "setup": """import numpy as np
element_stress = np.zeros((1, 6, 24))
element_dofs = -np.ones((1, 24), dtype=int)
stress_macro = np.arange(6, dtype=float)
displacement = np.zeros(1)
def summarize(result):
    return (
        result["effective_stress"].tolist(),
        result["effective_axial_stiffness"],
    )
""",
            "call": "summarize(compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement))",
            "gold_call": "summarize(_oracle_compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement))",
        },
        {
            "setup": """import numpy as np
element_stress = np.zeros((1, 6, 24))
element_dofs = np.zeros((2, 24), dtype=int)
stress_macro = np.zeros(6)
displacement = np.zeros(1)
def run_model():
    try:
        compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_effective_axial_stress(element_stress, element_dofs, stress_macro, displacement)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
