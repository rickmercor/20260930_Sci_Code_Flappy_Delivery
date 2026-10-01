"""
Reduce one arm of the joint to the single dimensionless number it contributes to the interfacial compliance of the mid-plane cohesive element for a given fracture mode.

The source paper defines this reduction through its mode-dependent reconstruction of the arm's interlaminar field. Apply that construction to the supplied arm geometry and preserve its normalization and sampling conventions exactly.

Returns
-------
float: the summed dimensionless interface stress ratios of the arm's represented resin-rich layers, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def arm_ratio_sum(mode: str, n_cohesive: int, n_total: int, t_ply: float) -> float:
    r"""Evaluate the source-defined dimensionless arm contribution.

    Parameters
    ----------
    mode : str
        Reconstruction mode, either ``"opening"`` or ``"shear"``.
    n_cohesive : int
        Number of plies in the arm block represented at the mid-plane
        interface (n_cohesive >= 1).
    n_total : int
        Total number of plies in the complete arm (n_total >= n_cohesive).
    t_ply : float
        Thickness of a single ply (t_ply > 0).

    Returns
    -------
    ratio_sum : float
        Dimensionless arm contribution prescribed by the source construction,
        as a native Python float.

    Raises ValueError if ``mode`` is not ``"opening"`` or ``"shear"``, if
    ``n_cohesive`` or ``n_total`` is not an integer, if ``n_cohesive`` < 1, if
    ``n_total`` < ``n_cohesive``, or if ``t_ply`` is not a finite real number
    greater than zero.

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
def _oracle_arm_ratio_sum(mode: str, n_cohesive: int, n_total: int, t_ply: float) -> float:
    if not isinstance(mode, str):
        raise ValueError("mode must be a string")
    if mode not in ("opening", "shear"):
        raise ValueError("mode must be either 'opening' or 'shear'")
    for name, val in (("n_cohesive", n_cohesive), ("n_total", n_total)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    n_cohesive = int(n_cohesive)
    n_total = int(n_total)
    if n_cohesive < 1:
        raise ValueError("n_cohesive must be >= 1")
    if n_total < n_cohesive:
        raise ValueError("n_total must be >= n_cohesive")
    if isinstance(t_ply, bool) or not isinstance(t_ply, (int, float, np.integer, np.floating)):
        raise ValueError("t_ply must be a real number")
    t_ply = float(t_ply)
    if not np.isfinite(t_ply) or t_ply <= 0.0:
        raise ValueError("t_ply must be a finite number > 0")

    # One resin-rich layer per represented ply, the first one lying on the
    # cohesive interface; the polynomials are anchored on the traction-free
    # surface of the whole half-stack.
    z_layers = np.arange(n_cohesive, dtype=float) * t_ply
    h = n_total * t_ply

    # Rows of the linear system act on c = [a1, a2, a3, a4] through
    #   tau(z)   = a1*z**2   + a2*z      + a3
    #   sigma(z) = a1*z**3/3 + a2*z**2/2 + a3*z + a4
    tau_0 = np.array([0.0, 0.0, 1.0, 0.0])
    tau_h = np.array([h ** 2, h, 1.0, 0.0])
    sig_0 = np.array([0.0, 0.0, 0.0, 1.0])
    sig_h = np.array([h ** 3 / 3.0, h ** 2 / 2.0, h, 1.0])

    # Both stresses are traction free on the outer surface. At the cohesive
    # interface the mode decides which stress carries the normalised peak and
    # which one vanishes.
    if mode == "opening":
        a_mat = np.vstack([sig_0, tau_0, tau_h, sig_h])
    else:
        a_mat = np.vstack([tau_0, sig_0, tau_h, sig_h])
    rhs = np.array([1.0, 0.0, 0.0, 0.0])
    a1, a2, a3, a4 = np.linalg.solve(a_mat, rhs)

    if mode == "opening":
        row = a1 * z_layers ** 3 / 3.0 + a2 * z_layers ** 2 / 2.0 + a3 * z_layers + a4
    else:
        row = a1 * z_layers ** 2 + a2 * z_layers + a3

    peak = float(row[0])
    if peak == 0.0:
        raise ValueError("the driving stress on the cohesive interface must be non-zero")
    return float(np.sum(row / peak))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: an unreinforced arm of n plies has normal ratios
        # 2*u**3 - 3*u**2 + 1 and shear ratios 3*u**2 - 4*u + 1 at u = i/n,
        # i = 0..n-1, whose closed-form sums are (n + 1)/2 and (n + 1)/(2*n).
        # For n = 12 that is 6.5 and 13/24, computed here independently of the
        # oracle. Shifting the first layer off the interface, anchoring on the
        # wrong surface, or using a positive shear profile all fail.
        {
            "setup": """import numpy as np
n = 12
EXPECTED = float((n + 1) / 2.0 + 100.0 * (n + 1) / (2.0 * n))
""",
            "call": ("float(arm_ratio_sum('opening', n, n, 0.1875)"
                     " + 100.0 * arm_ratio_sum('shear', n, n, 0.1875))"),
            "gold_call": "EXPECTED",
        },
        # --- Pinned value: a reinforced arm, 4 represented plies inside a 7-ply
        # half-stack, evaluated from the closed-form profiles at u = i/7.
        {
            "setup": """import numpy as np
u = np.arange(4, dtype=float) / 7.0
EXPECTED = float(np.sum(2.0 * u ** 3 - 3.0 * u ** 2 + 1.0)
                 + 100.0 * np.sum(3.0 * u ** 2 - 4.0 * u + 1.0))
""",
            "call": ("float(arm_ratio_sum('opening', 4, 7, 0.25)"
                     " + 100.0 * arm_ratio_sum('shear', 4, 7, 0.25))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: reinforced arm in opening, outer surface beyond the last layer ---
        {
            "setup": """import numpy as np
""",
            "call": "arm_ratio_sum('opening', 13, 19, 0.15)",
            "gold_call": "_oracle_arm_ratio_sum('opening', 13, 19, 0.15)",
        },
        # --- Valid: reinforced arm in shear, whose ratios change sign past a third of the arm ---
        {
            "setup": """import numpy as np
""",
            "call": "arm_ratio_sum('shear', 9, 15, 0.15)",
            "gold_call": "_oracle_arm_ratio_sum('shear', 9, 15, 0.15)",
        },
        # --- Boundary: a single represented layer gives exactly one in either mode ---
        {
            "setup": """import numpy as np
""",
            "call": "float(arm_ratio_sum('opening', 1, 12, 0.125) + 10.0 * arm_ratio_sum('shear', 1, 1, 0.125))",
            "gold_call": "float(_oracle_arm_ratio_sum('opening', 1, 12, 0.125) + 10.0 * _oracle_arm_ratio_sum('shear', 1, 1, 0.125))",
        },
        # --- Edge: the result is independent of the ply thickness ---
        {
            "setup": """import numpy as np
""",
            "call": "float(arm_ratio_sum('shear', 7, 23, 0.125) + 10.0 * arm_ratio_sum('shear', 7, 23, 1.9))",
            "gold_call": "float(_oracle_arm_ratio_sum('shear', 7, 23, 0.125) + 10.0 * _oracle_arm_ratio_sum('shear', 7, 23, 1.9))",
        },
        # --- Invalid: unknown mode string ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        arm_ratio_sum('mixed', 9, 15, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_arm_ratio_sum('mixed', 9, 15, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer total plies than represented resin-rich layers ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        arm_ratio_sum('opening', 9, 4, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_arm_ratio_sum('opening', 9, 4, 0.2)
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
