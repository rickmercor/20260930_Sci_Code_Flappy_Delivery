"""
Express the local mode ratio of the interface as the shear share of the interfacial energy along a proportional separation path.

The energy stored per unit area splits into a normal part $k_n \delta_n^2/2$ and a shear part $k_s \delta_s^2/2$, so with $\delta_s = r\delta_n$ the shear share is $B = k_s r^2/(k_n + k_s r^2)$. Because the ratio $r$ is constant along a proportional path, $B$ is independent of how far the interface has been pulled and can be evaluated from the stiffnesses alone.

Returns
-------
float: the local mode ratio $B$, the shear fraction of the interfacial energy, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_mode_energy_ratio(k_n: float, k_s: float, disp_ratio: float) -> float:
    """Compute the shear share of the interfacial energy.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).

    Returns
    -------
    b_ratio : float
        Local mode ratio, the shear fraction of the stored interfacial
        energy, in [0, 1) as a native Python float.

    Raises ValueError if ``k_n`` or ``k_s`` is not a finite real number greater
    than zero, or if ``disp_ratio`` is not a finite real number greater than or
    equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_mixed_mode_energy_ratio(k_n: float, k_s: float, disp_ratio: float) -> float:

    for name, val in (("k_n", k_n), ("k_s", k_s)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(disp_ratio, bool) or not isinstance(disp_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("disp_ratio must be a real number")
    disp_ratio = float(disp_ratio)
    if not np.isfinite(disp_ratio) or disp_ratio < 0.0:
        raise ValueError("disp_ratio must be a finite number >= 0")

    k_n = float(k_n)
    k_s = float(k_s)

    # Both energy contributions carry the same delta_n**2, which cancels.
    shear_energy = k_s * disp_ratio ** 2
    return float(shear_energy / (k_n + shear_energy))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: pure opening gives B = 0 exactly, and equal
        # stiffnesses with equal jumps split the energy evenly, B = 1/2. With
        # k_s*r**2 = 3*k_n the shear share is 3/4. All three from the closed
        # form, independently of the oracle.
        {
            "setup": """import numpy as np
EXPECTED = float(0.0 + 10.0 * 0.5 + 100.0 * 0.75)
""",
            "call": ("float(mixed_mode_energy_ratio(10000.0, 25000.0, 0.0)"
                     " + 10.0 * mixed_mode_energy_ratio(8000.0, 8000.0, 1.0)"
                     " + 100.0 * mixed_mode_energy_ratio(8000.0, 24000.0, 1.0))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a mildly shear-biased interface (normal scenario) ---
        {
            "setup": """import numpy as np
k_n = 12406.5
k_s = 17233.2
""",
            "call": "mixed_mode_energy_ratio(k_n, k_s, 0.4)",
            "gold_call": "_oracle_mixed_mode_energy_ratio(k_n, k_s, 0.4)",
        },
        # --- Valid: an interface far stiffer in shear than in opening ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_energy_ratio(15201.4, 61042.8, 0.6)",
            "gold_call": "_oracle_mixed_mode_energy_ratio(15201.4, 61042.8, 0.6)",
        },
        # --- Boundary: a vanishing tangential jump ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_energy_ratio(11000.0, 19000.0, 0.0)",
            "gold_call": "_oracle_mixed_mode_energy_ratio(11000.0, 19000.0, 0.0)",
        },
        # --- Edge: an almost pure shear path ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_energy_ratio(11000.0, 19000.0, 400.0)",
            "gold_call": "_oracle_mixed_mode_energy_ratio(11000.0, 19000.0, 400.0)",
        },
        # --- Invalid: negative separation ratio ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixed_mode_energy_ratio(11000.0, 19000.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixed_mode_energy_ratio(11000.0, 19000.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive normal stiffness ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixed_mode_energy_ratio(0.0, 19000.0, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixed_mode_energy_ratio(0.0, 19000.0, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
