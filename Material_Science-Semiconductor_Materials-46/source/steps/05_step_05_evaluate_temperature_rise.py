"""
Sum the double eigenmode series to return the transient temperature of the via above ambient at one position and one instant.

With the axial eigenvalues, the layered radial eigenfunctions, the decay rates and the expansion amplitudes all in hand, the temperature difference field is the double sum of amplitude times radial eigenfunction times axial eigenfunction times a decaying exponential. Two properties of that sum matter in practice. The first is that the exponential is common to all layers by construction, so evaluating the field in the copper and in the silicon uses the same time factor and differs only through the layer's radial amplitudes; a field assembled with layer-specific exponents would violate the interface conditions at every instant except the initial one.




The second is that convergence is extremely uneven in the two indices and improves rapidly with time. The radial spectrum of a via whose cross-section is a few tens of micrometres across is very sparse at the bottom and then jumps: for the benchmark geometry the slowest radial mode decays in a fraction of a millisecond while the third already decays in nanoseconds, so at the times at which thermal stress is evaluated only the first one or two radial modes survive and the remainder are numerically zero. The axial series behaves like an ordinary cosine series of a uniform initial state and converges slowly at the initial instant, where the pinned face forces a discontinuity, but the exponential damping removes that difficulty within a few microseconds.




Because a truncated series still satisfies the governing equation, the insulated faces, the pinned face and every interface condition exactly, all of the truncation error appears as an error in the initial condition alone. That gives a cheap and stringent verification: summing the series at a time close to zero must return the uniform initial value everywhere except in a thin layer near the pinned face, and the radial factor alone must reconstruct unity across all layers, which tests the layered eigenfunctions and the heat-capacity-weighted norm together.

Returns
-------
float: the temperature of the via above the ambient temperature in K at the requested point and instant, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_temperature_rise(axial_modes: np.ndarray, decay_rates: np.ndarray,
                              eigenfunctions: np.ndarray,
                              coefficients: np.ndarray, layer: int,
                              radius: float, axial_position: float,
                              time: float) -> float:
    """Return the temperature above ambient at one point and one instant.

    Parameters
    ----------
    axial_modes : np.ndarray
        Array of shape (n_axial, 3) holding the axial eigenvalue, square norm
        and eigenfunction integral of every axial mode.
    decay_rates : np.ndarray
        Array of shape (n_axial, n_radial) holding the decay rate in 1/s of
        every mode.
    eigenfunctions : np.ndarray
        Array of shape (n_axial, n_radial, n_layers, 3) holding the squared
        radial parameter and the two radial amplitudes of every layer for
        every mode.
    coefficients : np.ndarray
        Array of shape (n_axial, n_radial) holding the expansion amplitude in
        K of every mode.
    layer : int
        Index of the layer containing the evaluation point, 0-based.
    radius : float
        Radial coordinate of the evaluation point in m (radius >= 0).
    axial_position : float
        Axial coordinate of the evaluation point in m (axial_position >= 0).
    time : float
        Time since the start of the transient in s (time >= 0).

    Returns
    -------
    temperature_rise : float
        Temperature above ambient in K, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return temperature_rise  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_temperature_rise(axial_modes: np.ndarray,
                                      decay_rates: np.ndarray,
                                      eigenfunctions: np.ndarray,
                                      coefficients: np.ndarray, layer: int,
                                      radius: float, axial_position: float,
                                      time: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, j0, k0, y0

    axial_modes = np.asarray(axial_modes, dtype=float)
    decay_rates = np.asarray(decay_rates, dtype=float)
    eigenfunctions = np.asarray(eigenfunctions, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)

    if axial_modes.ndim != 2 or axial_modes.shape[1] != 3:
        raise ValueError("axial_modes must have shape (n_axial, 3)")
    n_axial = axial_modes.shape[0]
    if decay_rates.ndim != 2 or decay_rates.shape[0] != n_axial:
        raise ValueError("decay_rates must have shape (n_axial, n_radial)")
    n_radial = decay_rates.shape[1]
    if coefficients.shape != (n_axial, n_radial):
        raise ValueError("coefficients must have shape (n_axial, n_radial)")
    if (eigenfunctions.ndim != 4 or eigenfunctions.shape[:2] != (n_axial, n_radial)
            or eigenfunctions.shape[3] != 3):
        raise ValueError("eigenfunctions must have shape (n_axial, n_radial, n_layers, 3)")
    n_layers = eigenfunctions.shape[2]
    if not (isinstance(layer, (int, np.integer)) and not isinstance(layer, bool)
            and 0 <= int(layer) < n_layers):
        raise ValueError("layer must be an integer index of an existing layer")
    for name, value in (("radius", radius), ("axial_position", axial_position),
                        ("time", time)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")

    layer = int(layer)
    radius = float(radius)
    axial_position = float(axial_position)
    time = float(time)

    def _radial_value(mu, amp_a, amp_b, r):
        """Radial eigenfunction at a radius, regular on the axis."""
        if r == 0.0:
            return amp_a
        if mu > 0.0:
            x = np.sqrt(mu) * r
            return amp_a * j0(x) + amp_b * y0(x)
        if mu < 0.0:
            x = np.sqrt(-mu) * r
            return amp_a * i0(x) + amp_b * k0(x)
        return amp_a + amp_b * np.log(r)

    total = 0.0
    for m in range(n_axial):
        eta = float(axial_modes[m, 0])
        radial_sum = 0.0
        for n in range(n_radial):
            mu, amp_a, amp_b = (float(eigenfunctions[m, n, layer, 0]),
                                float(eigenfunctions[m, n, layer, 1]),
                                float(eigenfunctions[m, n, layer, 2]))
            shape = _radial_value(mu, amp_a, amp_b, radius)
            radial_sum += (float(coefficients[m, n]) * shape
                           * np.exp(-float(decay_rates[m, n]) * time))
        total += radial_sum * np.cos(eta * axial_position)

    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: single mode evaluated inside the metal core (normal scenario) ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.981633974483, 1.0e-4, 1.2732395447351628e-04]])
decay_rates = np.array([[5953.409648825929]])
eigenfunctions = np.array([[[[-1.0342822695e+07, 1.0000000000e+00, 0.0000000000e+00],
                             [ 6.7677263268e+09, 1.0260381408e+00, 1.3245291089e+00],
                             [ 1.2975309035e+07, 1.0318116676e+00, 6.4857891790e-03]]]])
coefficients = np.array([[126.22851522849304]])
layer = 0
radius = 5.0e-6
axial_position = 200.0e-6 / 3.0
time = 5.0e-5
""",
            "call": "evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
            "gold_call": "_oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
        },
        # --- Valid: same modal data evaluated in the outer silicon layer ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.981633974483, 1.0e-4, 1.2732395447351628e-04],
                        [23561.944901923447, 1.0e-4, -4.2441318157838763e-05]])
decay_rates = np.array([[5953.409648825929, 126964.8635947617],
                        [53527.0, 6.0e5]])
eigenfunctions = np.tile(np.array([[-1.0342822695e+07, 1.0000000000e+00, 0.0000000000e+00],
                                   [ 6.7677263268e+09, 1.0260381408e+00, 1.3245291089e+00],
                                   [ 1.2975309035e+07, 1.0318116676e+00, 6.4857891790e-03]]),
                         (2, 2, 1, 1))
coefficients = np.array([[126.22851522849304, 1.0875424964178686],
                         [-42.0, -0.35]])
layer = 2
radius = 20.0e-6
axial_position = 50.0e-6
time = 1.0e-5
""",
            "call": "evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
            "gold_call": "_oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
        },
        # --- Boundary: on the axis, where the singular radial solution is absent ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.981633974483, 1.0e-4, 1.2732395447351628e-04]])
decay_rates = np.array([[5953.409648825929]])
eigenfunctions = np.array([[[[-1.0342822695e+07, 1.0, 0.0]]]])
coefficients = np.array([[126.22851522849304]])
layer = 0
radius = 0.0
axial_position = 0.0
time = 0.0
""",
            "call": "evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
            "gold_call": "_oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
        },
        # --- Edge: a long time, at which every mode has decayed away ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.981633974483, 1.0e-4, 1.2732395447351628e-04]])
decay_rates = np.array([[5953.409648825929]])
eigenfunctions = np.array([[[[-1.0342822695e+07, 1.0, 0.0]]]])
coefficients = np.array([[126.22851522849304]])
layer = 0
radius = 0.0
axial_position = 66.66666666666667e-6
time = 1.0
""",
            "call": "evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
            "gold_call": "_oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, layer, radius, axial_position, time)",
        },
        # --- Invalid: layer index outside the stack ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.98, 1.0e-4, 1.27e-4]])
decay_rates = np.array([[5953.4]])
eigenfunctions = np.array([[[[-1.0e7, 1.0, 0.0]]]])
coefficients = np.array([[126.0]])
def run_model():
    try:
        evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, 3, 0.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, 3, 0.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative time ---
        {
            "setup": """import numpy as np
axial_modes = np.array([[7853.98, 1.0e-4, 1.27e-4]])
decay_rates = np.array([[5953.4]])
eigenfunctions = np.array([[[[-1.0e7, 1.0, 0.0]]]])
coefficients = np.array([[126.0]])
def run_model():
    try:
        evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, 0, 0.0, 0.0, -1.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_temperature_rise(axial_modes, decay_rates, eigenfunctions, coefficients, 0, 0.0, 0.0, -1.0e-6)
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
