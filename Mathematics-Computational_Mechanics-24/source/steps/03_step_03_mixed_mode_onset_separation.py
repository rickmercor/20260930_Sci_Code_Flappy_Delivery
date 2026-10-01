"""
Locate damage onset along a proportional mixed-mode separation path through the quadratic interaction of the two interfacial strengths.

On a proportional path the normal and tangential jumps keep a fixed ratio $r = \delta_s/\delta_n$, so with the effective separation $\delta = \sqrt{\delta_n^2 + \delta_s^2}$ they are $\delta_n = \delta/\sqrt{1 + r^2}$ and $\delta_s = r\delta/\sqrt{1 + r^2}$. Damage starts where the quadratic interaction $(k_n \delta_n/\tau_{Ic})^2 + (k_s \delta_s/\tau_{IIc})^2 = 1$ is first reached.

Returns
-------
float: the effective separation $\sqrt{\delta_n^2 + \delta_s^2}$ at mixed-mode damage onset, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_mode_onset_separation(k_n: float, k_s: float, tau_ic: float,
                                tau_iic: float, disp_ratio: float) -> float:
    """Compute the effective separation at mixed-mode damage onset.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    tau_ic : float
        Interlaminar strength in opening (tau_ic > 0).
    tau_iic : float
        Interlaminar strength in shearing (tau_iic > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).

    Returns
    -------
    delta_onset : float
        Effective separation at damage onset, as a native Python float.

    Raises ValueError if ``k_n``, ``k_s``, ``tau_ic`` or ``tau_iic`` is not a
    finite real number greater than zero, or if ``disp_ratio`` is not a finite
    real number greater than or equal to zero.

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
def _oracle_mixed_mode_onset_separation(k_n: float, k_s: float, tau_ic: float,
                                        tau_iic: float, disp_ratio: float) -> float:

    positives = {"k_n": k_n, "k_s": k_s, "tau_ic": tau_ic, "tau_iic": tau_iic}
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
    tau_ic = float(tau_ic)
    tau_iic = float(tau_iic)

    # Along the proportional path both jumps are fixed fractions of the
    # effective separation, so the quadratic criterion becomes linear in
    # delta**2 and can be inverted directly.
    norm = np.sqrt(1.0 + disp_ratio ** 2)
    normal_term = k_n / (tau_ic * norm)
    shear_term = k_s * disp_ratio / (tau_iic * norm)
    return float(1.0 / np.sqrt(normal_term ** 2 + shear_term ** 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the two pure-mode limits are exact. At
        # disp_ratio = 0 the onset separation is tau_ic/k_n = 30/12000 =
        # 0.0025, and as the path becomes pure shear the limit is
        # tau_iic/k_s = 60/20000 = 0.003. Probed here at disp_ratio = 0 and at
        # 1e8, where the shear limit is reached to better than 1e-12 relative.
        {
            "setup": """import numpy as np
EXPECTED = float(30.0 / 12000.0 + 1000.0 * 60.0 / 20000.0)
""",
            "call": ("float(mixed_mode_onset_separation(12000.0, 20000.0, 30.0, 60.0, 0.0)"
                     " + 1000.0 * mixed_mode_onset_separation(12000.0, 20000.0, 30.0, 60.0, 1.0e8))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a stiff interface on a moderately mixed path ---
        {
            "setup": """import numpy as np
k_n = 12406.5
k_s = 17233.2
""",
            "call": "mixed_mode_onset_separation(k_n, k_s, 32.6, 98.0, 0.4)",
            "gold_call": "_oracle_mixed_mode_onset_separation(k_n, k_s, 32.6, 98.0, 0.4)",
        },
        # --- Valid: the same strengths with a far more compliant interface ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_onset_separation(1200.0, 900.0, 30.0, 60.0, 2.5)",
            "gold_call": "_oracle_mixed_mode_onset_separation(1200.0, 900.0, 30.0, 60.0, 2.5)",
        },
        # --- Boundary: equal stiffnesses and equal jumps ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_onset_separation(50000.0, 50000.0, 30.0, 60.0, 1.0)",
            "gold_call": "_oracle_mixed_mode_onset_separation(50000.0, 50000.0, 30.0, 60.0, 1.0)",
        },
        # --- Edge: a strongly shear-dominated path ---
        {
            "setup": """import numpy as np
""",
            "call": "mixed_mode_onset_separation(15201.4, 61042.8, 30.0, 60.0, 12.0)",
            "gold_call": "_oracle_mixed_mode_onset_separation(15201.4, 61042.8, 30.0, 60.0, 12.0)",
        },
        # --- Invalid: negative separation ratio ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixed_mode_onset_separation(12000.0, 20000.0, 30.0, 60.0, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixed_mode_onset_separation(12000.0, 20000.0, 30.0, 60.0, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero interlaminar strength ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixed_mode_onset_separation(12000.0, 20000.0, 0.0, 60.0, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixed_mode_onset_separation(12000.0, 20000.0, 0.0, 60.0, 0.75)
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
