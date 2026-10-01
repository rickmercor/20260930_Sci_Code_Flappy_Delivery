"""
Convert the intrinsic drift-to-diffusion ratio of a branch into the boundary-sensitive figure of merit that its terminals actually experience.

The intrinsic parameter extracted from the equivalence describes the medium, but a device is judged by what its ports see, and that depends on how it is driven. The natural figure of merit for a two-terminal diffusive element under fixed terminal potentials compares the two flux densities obtained by exchanging those potentials. For a reciprocal element the two are equal and opposite, so their sum vanishes; any nonzero sum is a direct measure of the failure of reciprocity. Normalising that sum by the larger of the two magnitudes rather than by their difference or their mean keeps the measure bounded: normalising by the difference would diverge exactly where the element is most strongly rectifying, whereas this choice saturates.




Both exchanged fluxes are available in closed form from the equivalent medium, since its steady-state flux is a fixed linear combination of the two terminal potentials with coefficients built from the exponential of the drift-to-diffusion ratio across the branch. Their sum turns out to involve the terminal potentials only through their sum, and their difference only through their difference, so the figure of merit reduces to two ingredients: the sign of the product of the potential sum with the departure of that exponential from unity, and the magnitude of the ratio between the difference-driven and sum-driven parts. The effective conductivity and the effective advective coefficient cancel out entirely, which is what makes the quantity computable from the intrinsic ratio and the terminal values alone. Three limits expose an incorrect construction. When the two potentials sum to zero the measure vanishes identically even though the medium still carries an effective drift, a statement about the drive rather than the material. When the two potentials are equal it saturates at twice the sign factor, the extreme case in which one direction passes flux and the other does not. And when the medium is reciprocal, so that the exponential factor equals one, it vanishes again; a construction that mishandles this limit typically produces a division by zero rather than the zero that belongs there.

Returns
-------
float: the rectification ratio of the branch under the given terminal # potentials, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_rectification_ratio(advection_ratio: float,
                                length: float,
                                phi_high: float,
                                phi_low: float) -> float:
    """Evaluate the rectification ratio of a branch under fixed potentials.

    Parameters
    ----------
    advection_ratio : float
        Ratio of the effective advective coefficient to the effective
        conductivity, in inverse metres.
    length : float
        Separation of the two terminals in metres, strictly positive.
    phi_high : float
        Potential imposed at the first terminal.
    phi_low : float
        Potential imposed at the second terminal.

    Returns
    -------
    rectification : float
        Sum of the two time-averaged flux densities obtained by exchanging the
        terminal potentials, divided by the larger of their magnitudes, as a
        native Python float.

    Raises
    ------
    ValueError
        If advection_ratio, phi_high or phi_low is not a finite number, or if
        length is not a finite number > 0.
    """
    return rectification  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_rectification_ratio(advection_ratio: float,
                                        length: float,
                                        phi_high: float,
                                        phi_low: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("advection_ratio", advection_ratio), ("phi_high", phi_high),
                        ("phi_low", phi_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if not (isinstance(length, (int, float, np.floating, np.integer))
            and np.isfinite(length) and float(length) > 0.0):
        raise ValueError("length must be a finite number > 0")

    exponent = float(advection_ratio) * float(length)
    terminal_scale = max(abs(float(phi_high)), abs(float(phi_low)))
    if terminal_scale == 0.0 or exponent == 0.0:
        return 0.0
    # Normalising the terminal potentials and using tanh(x/2) evaluates the
    # algebraically equivalent (exp(x)-1)/(exp(x)+1) without overflow.
    high = float(phi_high) / terminal_scale
    low = float(phi_low) / terminal_scale
    total = high + low
    if total == 0.0:
        return 0.0
    difference = high - low
    scaled_balance = abs(total) * abs(float(np.tanh(0.5 * exponent)))
    if scaled_balance == 0.0:
        return 0.0
    direction = float(np.sign(total) * np.sign(exponent))
    return float(2.0 * direction * scaled_balance
                 / (scaled_balance + abs(difference)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a branch whose effective drift opposes the imposed gradient ---
        {
            "setup": """import numpy as np
advection_ratio = -0.33729580147263514
length = 1.0
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Valid: the same terminals seen by a medium whose drift runs with the gradient ---
        {
            "setup": """import numpy as np
advection_ratio = 0.21357902488611340
length = 1.0
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Boundary: terminal potentials summing to zero ---
        {
            "setup": """import numpy as np
advection_ratio = -0.33729580147263514
length = 1.0
phi_high = 20.0
phi_low = -20.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Boundary: equal terminal potentials, where the measure saturates ---
        {
            "setup": """import numpy as np
advection_ratio = 0.9
length = 2.0
phi_high = 15.0
phi_low = 15.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Edge: a reciprocal medium, where the measure vanishes ---
        {
            "setup": """import numpy as np
advection_ratio = 0.0
length = 1.0
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Boundary: the smallest subnormal drift underflows to reciprocity ---
        {
            "setup": """import numpy as np
advection_ratio = np.nextafter(0.0, 1.0)
length = 1.0
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Edge: strongly biased medium with negative terminal potentials ---
        {
            "setup": """import numpy as np
advection_ratio = -3.5
length = 0.8
phi_high = -2.0
phi_low = -18.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Edge: an exponent far beyond the direct exponential range ---
        {
            "setup": """import numpy as np
advection_ratio = 1000.0
length = 1.0
phi_high = 30.0
phi_low = 10.0
""",
            "call": "float(1.0 + 1.0e6 * compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low))",
        },
        # --- Invalid: non-positive branch length ---
        {
            "setup": """import numpy as np
advection_ratio = -0.2
length = -1.0
phi_high = 30.0
phi_low = 10.0
def run_model():
    try:
        compute_rectification_ratio(advection_ratio, length, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite advection ratio ---
        {
            "setup": """import numpy as np
advection_ratio = float("nan")
length = 1.0
phi_high = 30.0
phi_low = 10.0
def run_model():
    try:
        compute_rectification_ratio(advection_ratio, length, phi_high, phi_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_rectification_ratio(advection_ratio, length, phi_high, phi_low)
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
