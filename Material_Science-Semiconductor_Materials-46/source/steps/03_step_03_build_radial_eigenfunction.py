"""
Return, for one decay rate of one axial mode, each layer's radial parameter and the two amplitudes that define its radial eigenfunction, normalised so that the amplitude of the regular solution on the axis is unity.

An eigenvalue of the layered radial problem is only half of what the temperature expansion needs; the other half is the eigenfunction itself, which is piecewise defined and has two amplitudes per layer. The layer radial parameters follow from the shared decay rate: each equals the decay rate divided by that layer's thermal diffusivity, less the square of the axial eigenvalue, so a layer whose diffusivity is small carries a large radial parameter and oscillates rapidly across its thickness, while a layer whose diffusivity is large may carry a negative parameter and vary exponentially instead.




The amplitudes are fixed layer by layer rather than all at once. On the axis the second solution of the Bessel equation is singular and must be discarded, which leaves a single amplitude that can be set to unity because an eigenfunction is defined only up to scale; every downstream quantity that uses the eigenfunction is homogeneous of degree zero in that scale, since the expansion coefficient divides by the same norm it multiplies. Each interface then supplies two conditions, continuity of temperature and continuity of the radial heat flux, and those two conditions determine the two amplitudes of the next layer from the value and the conductivity-weighted slope carried out of the previous one.




The two-by-two system solved at each interface is never singular: its determinant is the layer conductivity times the Wronskian of the pair of Bessel solutions, which is a fixed non-zero multiple of the reciprocal of the radius for both the oscillatory and the exponential branch. That is why building the eigenfunction by outward propagation is numerically preferable to assembling and inverting one global matrix over all layers, whose conditioning degrades quickly when the layer parameters differ by an order of magnitude, as they do whenever a thin, low-diffusivity liner separates two good conductors.

Returns
-------
np.ndarray of shape (n_layers, 3), float: squared radial parameter (1/m^2) and the two radial amplitudes (dimensionless) of every layer, in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_radial_eigenfunction(radii: np.ndarray, conductivities: np.ndarray,
                               diffusivities: np.ndarray, eta: float,
                               decay_rate: float) -> np.ndarray:
    """Return the radial parameter and the two amplitudes of every layer.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive.
    conductivities : np.ndarray
        Thermal conductivity of each layer in W/(m K), shape (n_layers,),
        all positive.
    diffusivities : np.ndarray
        Thermal diffusivity of each layer in m^2/s, shape (n_layers,),
        all positive.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    decay_rate : float
        Decay rate of the mode in 1/s (decay_rate > 0).

    Returns
    -------
    eigenfunction : np.ndarray
        Array of shape (n_layers, 3) whose columns are the squared radial
        parameter of the layer in 1/m^2, the amplitude of the regular radial
        solution and the amplitude of the singular radial solution.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return eigenfunction  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_radial_eigenfunction(radii: np.ndarray,
                                       conductivities: np.ndarray,
                                       diffusivities: np.ndarray, eta: float,
                                       decay_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    for name, value in (("eta", eta), ("decay_rate", decay_rate)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    radii = np.asarray(radii, dtype=float).ravel()
    conductivities = np.asarray(conductivities, dtype=float).ravel()
    diffusivities = np.asarray(diffusivities, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    if conductivities.size != n_layers or diffusivities.size != n_layers:
        raise ValueError("radii, conductivities and diffusivities must agree in length")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    for name, array in (("conductivities", conductivities),
                        ("diffusivities", diffusivities)):
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError(f"{name} must be finite and positive")

    eta = float(eta)
    decay_rate = float(decay_rate)

    def _basis(mu, radius):
        """Pair of independent radial solutions and their derivatives."""
        if mu > 0.0:
            b = np.sqrt(mu)
            x = b * radius
            return j0(x), -b * j1(x), y0(x), -b * y1(x)
        if mu < 0.0:
            b = np.sqrt(-mu)
            x = b * radius
            return i0(x), b * i1(x), k0(x), -b * k1(x)
        return 1.0, 0.0, np.log(radius), 1.0 / radius

    # A common temporal exponent ties the layer parameters to the decay rate.
    mu = decay_rate / diffusivities - eta ** 2

    amplitudes = np.zeros((n_layers, 2))
    amplitudes[0, 0] = 1.0
    for i in range(n_layers - 1):
        radius = radii[i]
        ca, cap, da, dap = _basis(mu[i], radius)
        value = amplitudes[i, 0] * ca + amplitudes[i, 1] * da
        slope = amplitudes[i, 0] * cap + amplitudes[i, 1] * dap
        cb, cbp, db, dbp = _basis(mu[i + 1], radius)
        matrix = np.array([[cb, db],
                           [conductivities[i + 1] * cbp,
                            conductivities[i + 1] * dbp]])
        amplitudes[i + 1] = np.linalg.solve(
            matrix, np.array([value, conductivities[i] * slope]))

    return np.column_stack([mu, amplitudes[:, 0], amplitudes[:, 1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: slowest mode of the benchmark via (normal scenario) ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = np.pi / (2.0 * 200.0e-6)
decay_rate = 5953.409648825929
""",
            "call": "build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
            "gold_call": "_oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
        },
        # --- Valid: a fast, fully oscillatory mode of the same via ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = np.pi / (2.0 * 200.0e-6)
decay_rate = 6764124.44072
""",
            "call": "build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
            "gold_call": "_oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
        },
        # --- Boundary: a homogeneous cylinder needs no interface transfer ---
        {
            "setup": """import numpy as np
radii = np.array([25.0e-6])
conductivities = np.array([130.0])
diffusivities = np.array([130.0 / (2329.0 * 700.0)])
eta = np.pi / (2.0 * 200.0e-6)
decay_rate = 1.0e6
""",
            "call": "build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
            "gold_call": "_oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
        },
        # --- Edge: decay rate below every threshold, so no layer oscillates ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = np.pi / (2.0 * 200.0e-6)
decay_rate = 10.0
""",
            "call": "build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
            "gold_call": "_oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, eta, decay_rate)",
        },
        # --- Invalid: mismatched property array lengths ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4])
diffusivities = np.array([1.0e-4, 1.0e-6, 8.0e-5])
def run_model():
    try:
        build_radial_eigenfunction(radii, conductivities, diffusivities, 7853.98, 5953.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, 7853.98, 5953.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive decay rate ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 25.0e-6])
conductivities = np.array([400.0, 130.0])
diffusivities = np.array([1.0e-4, 8.0e-5])
def run_model():
    try:
        build_radial_eigenfunction(radii, conductivities, diffusivities, 7853.98, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_radial_eigenfunction(radii, conductivities, diffusivities, 7853.98, -1.0)
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
