"""
Interpolate the two pure-mode interlaminar toughnesses to the mixed-mode toughness at the local mode ratio of the interface.

Delamination in unidirectional carbon/epoxy follows the Benzeggagh-Kenane interpolation $G_c = G_{Ic} + (G_{IIc} - G_{Ic})\,B^{\eta}$, in which $B$ is the shear fraction of the interfacial energy and $\eta$ is a material exponent fitted to mixed-mode bending tests. The interpolation returns the pure-mode values at $B = 0$ and $B = 1$ and is monotone between them.

Returns
-------
float: the mixed-mode interlaminar fracture toughness, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def benzeggagh_kenane_toughness(g_ic: float, g_iic: float, eta: float,
                                b_ratio: float) -> float:
    """Interpolate the mixed-mode interlaminar fracture toughness.

    Parameters
    ----------
    g_ic : float
        Interlaminar fracture toughness in opening (g_ic > 0).
    g_iic : float
        Interlaminar fracture toughness in shearing (g_iic > 0).
    eta : float
        Benzeggagh-Kenane material exponent (eta > 0).
    b_ratio : float
        Local mode ratio, the shear fraction of the interfacial energy, in
        [0, 1].

    Returns
    -------
    g_c : float
        Mixed-mode interlaminar fracture toughness, as a native Python float.

    Raises ValueError if ``g_ic``, ``g_iic`` or ``eta`` is not a finite real
    number greater than zero, or if ``b_ratio`` is not a finite real number in
    [0, 1].

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
def _oracle_benzeggagh_kenane_toughness(g_ic: float, g_iic: float, eta: float,
                                        b_ratio: float) -> float:
    
    for name, val in (("g_ic", g_ic), ("g_iic", g_iic), ("eta", eta)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(b_ratio, bool) or not isinstance(b_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("b_ratio must be a real number")
    b_ratio = float(b_ratio)
    if not np.isfinite(b_ratio) or b_ratio < 0.0 or b_ratio > 1.0:
        raise ValueError("b_ratio must be a finite number in [0, 1]")

    g_ic = float(g_ic)
    g_iic = float(g_iic)
    eta = float(eta)

    return float(g_ic + (g_iic - g_ic) * b_ratio ** eta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the interpolation must return the pure-mode
        # toughnesses at its ends, 0.212 at B = 0 and 0.774 at B = 1, and at
        # B = 1/2 with eta = 1 it must return the arithmetic mean 0.493. The
        # three probes together catch a swapped pair or a misplaced exponent.
        {
            "setup": """import numpy as np
EXPECTED = float(0.212 + 10.0 * 0.774 + 100.0 * 0.493)
""",
            "call": ("float(benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 0.0)"
                     " + 10.0 * benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 1.0)"
                     " + 100.0 * benzeggagh_kenane_toughness(0.212, 0.774, 1.0, 0.5))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a near-balanced local mode ratio (normal scenario) ---
        {
            "setup": """import numpy as np
b_ratio = 0.5617
""",
            "call": "benzeggagh_kenane_toughness(0.212, 0.774, 2.1, b_ratio)",
            "gold_call": "_oracle_benzeggagh_kenane_toughness(0.212, 0.774, 2.1, b_ratio)",
        },
        # --- Valid: a tougher system with a lower exponent ---
        {
            "setup": """import numpy as np
""",
            "call": "benzeggagh_kenane_toughness(0.305, 2.77, 1.62, 0.31)",
            "gold_call": "_oracle_benzeggagh_kenane_toughness(0.305, 2.77, 1.62, 0.31)",
        },
        # --- Boundary: pure opening, where the exponent plays no role ---
        {
            "setup": """import numpy as np
""",
            "call": "benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 0.0)",
            "gold_call": "_oracle_benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 0.0)",
        },
        # --- Edge: a mode ratio very close to pure shear ---
        {
            "setup": """import numpy as np
""",
            "call": "benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 0.999999)",
            "gold_call": "_oracle_benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 0.999999)",
        },
        # --- Invalid: mode ratio outside the unit interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 1.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_benzeggagh_kenane_toughness(0.212, 0.774, 2.1, 1.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive exponent ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        benzeggagh_kenane_toughness(0.212, 0.774, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_benzeggagh_kenane_toughness(0.212, 0.774, 0.0, 0.5)
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
