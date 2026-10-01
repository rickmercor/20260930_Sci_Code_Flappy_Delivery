"""
Close the bilinear traction-separation law by locating the effective separation at which the cohesive element loses all load-carrying capacity.

The bilinear law is written in the effective separation and its elastic branch must store the same energy as the two-component response, which along a proportional path with $r = \delta_s/\delta_n$ gives the effective stiffness $k_{\mathrm{eff}} = (k_n + k_s r^2)/(1 + r^2)$. The triangle under that law has area equal to the mixed-mode toughness, so $k_{\mathrm{eff}}\,\delta_{\mathrm{onset}}\,\delta_{\mathrm{failure}}/2 = g_c$ fixes the failure separation.

Returns
-------
float: the effective separation at complete decohesion, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bilinear_failure_separation(k_n: float, k_s: float, disp_ratio: float,
                                delta_onset: float, g_c: float) -> float:
    """Compute the effective separation at complete decohesion.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).
    delta_onset : float
        Effective separation at damage onset (delta_onset > 0).
    g_c : float
        Mixed-mode interlaminar fracture toughness (g_c > 0).

    Returns
    -------
    delta_failure : float
        Effective separation at which the traction returns to zero, as a
        native Python float.

    Raises ValueError if ``k_n``, ``k_s``, ``delta_onset`` or ``g_c`` is not a
    finite real number greater than zero, if ``disp_ratio`` is not a finite real
    number greater than or equal to zero, or if the failure separation would
    fall below ``delta_onset``.

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
def _oracle_bilinear_failure_separation(k_n: float, k_s: float, disp_ratio: float,
                                        delta_onset: float, g_c: float) -> float:
    positives = {"k_n": k_n, "k_s": k_s, "delta_onset": delta_onset, "g_c": g_c}
    for name, val in positives.items():
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
    delta_onset = float(delta_onset)
    g_c = float(g_c)

    # Energy-equivalent secant stiffness of the elastic branch.
    k_eff = (k_n + k_s * disp_ratio ** 2) / (1.0 + disp_ratio ** 2)

    # Area of the bilinear triangle equals the mixed-mode toughness.
    delta_failure = 2.0 * g_c / (k_eff * delta_onset)
    if delta_failure < delta_onset:
        raise ValueError("the failure separation must not fall below the onset separation")
    return float(delta_failure)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned value: a pure-opening path collapses the law to the
        # familiar triangle of height tau_ic, so with k_n = 12000, onset
        # 32.6/12000 and g_c = 0.305 the failure separation must be
        # 2*0.305/32.6 independent of the stiffness. Verified here at two very
        # different normal stiffnesses.
        {
            "setup": """import numpy as np
EXPECTED = float(2.0 * 0.305 / 32.6 * 1001.0)
""",
            "call": ("float(bilinear_failure_separation(12000.0, 20000.0, 0.0, 32.6 / 12000.0, 0.305)"
                     " + 1000.0 * bilinear_failure_separation(60000.0, 20000.0, 0.0, 32.6 / 60000.0, 0.305))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a moderately mixed path on a stiff interface ---
        {
            "setup": """import numpy as np
k_n = 12406.5
k_s = 17233.2
delta_onset = 0.0024150
g_c = 0.4471200
""",
            "call": "bilinear_failure_separation(k_n, k_s, 0.4, delta_onset, g_c)",
            "gold_call": "_oracle_bilinear_failure_separation(k_n, k_s, 0.4, delta_onset, g_c)",
        },
        # --- Valid: a shear-dominated path with a tougher interface ---
        {
            "setup": """import numpy as np
""",
            "call": "bilinear_failure_separation(15201.4, 61042.8, 3.0, 0.00086, 0.7)",
            "gold_call": "_oracle_bilinear_failure_separation(15201.4, 61042.8, 3.0, 0.00086, 0.7)",
        },
        # --- Boundary: equal stiffnesses, where the effective stiffness is
        # independent of the separation ratio ---
        {
            "setup": """import numpy as np
""",
            "call": "bilinear_failure_separation(25000.0, 25000.0, 1.0, 0.0017, 0.4)",
            "gold_call": "_oracle_bilinear_failure_separation(25000.0, 25000.0, 1.0, 0.0017, 0.4)",
        },
        # --- Edge: onset and failure separation almost coincide ---
        {
            "setup": """import numpy as np
""",
            "call": "bilinear_failure_separation(20000.0, 20000.0, 0.5, 0.01, 1.0000005)",
            "gold_call": "_oracle_bilinear_failure_separation(20000.0, 20000.0, 0.5, 0.01, 1.0000005)",
        },
        # --- Invalid: a toughness too small to sustain the elastic branch ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bilinear_failure_separation(20000.0, 20000.0, 0.5, 0.01, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bilinear_failure_separation(20000.0, 20000.0, 0.5, 0.01, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive onset separation ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bilinear_failure_separation(12000.0, 20000.0, 0.75, 0.0, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bilinear_failure_separation(12000.0, 20000.0, 0.75, 0.0, 0.3)
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
