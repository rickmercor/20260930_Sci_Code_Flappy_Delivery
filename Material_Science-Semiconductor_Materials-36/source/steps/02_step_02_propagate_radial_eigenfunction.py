"""
Propagate one radial eigenfunction of the layered via outward from the axis, returning the two Bessel amplitudes and the separation constant of every material layer for a trial thermal decay rate.

Inside layer i the radial factor of one temperature mode satisfies R_i'' + R_i'/r + beta_i^2 R_i = 0, where beta_i^2 is the radial separation constant of that layer for the trial decay rate mu of the whole stack and the axial eigenvalue eta. The sign of beta_i^2 is not assumed: it can differ from layer to layer for the same mu, and it decides which pair of Bessel functions describes the layer.



Return convention. In a layer with beta_i^2 > 0 the radial factor is R_i(r) = A_i J_0(sqrt(beta_i^2) r) + B_i Y_0(sqrt(beta_i^2) r), and in a layer with beta_i^2 < 0 it is R_i(r) = A_i I_0(sqrt(-beta_i^2) r) + B_i K_0(sqrt(-beta_i^2) r). The innermost layer contains the axis and is normalised to A_1 = 1, B_1 = 0. The amplitudes of every other layer are those for which the temperature and the radial heat flux are continuous across each interface r = r_i, which is an ideal thermal contact. The outer boundary condition at r = r_l is not imposed here, so a solution is returned for any trial decay rate.

Returns
-------
np.ndarray of shape (l, 3), float: per layer the two radial amplitudes and the separation constant beta_i^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def propagate_radial_eigenfunction(radii: np.ndarray, conductivity: np.ndarray,
                                   diffusivity: np.ndarray, axial_eigenvalue: float,
                                   decay_rate: float) -> np.ndarray:
    """Amplitudes and separation constants of one radial eigenfunction.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing
        and strictly positive, with l >= 1.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta in inverse metres (eta > 0).
    decay_rate : float
        Trial thermal decay rate mu in inverse seconds (mu > 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (l, 3). Row i holds the first amplitude A_i, the second
        amplitude B_i and the separation constant beta_i^2 of layer i. The
        innermost layer is normalised to A_1 = 1, B_1 = 0.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; or if axial_eigenvalue or decay_rate
        is not a finite number greater than 0.
    """
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_radial_eigenfunction(radii: np.ndarray,
                                           conductivity: np.ndarray,
                                           diffusivity: np.ndarray,
                                           axial_eigenvalue: float,
                                           decay_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    radii = np.asarray(radii, dtype=float).ravel()
    conductivity = np.asarray(conductivity, dtype=float).ravel()
    diffusivity = np.asarray(diffusivity, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1 or conductivity.size != n_layers or diffusivity.size != n_layers:
        raise ValueError("radii, conductivity and diffusivity must share one length l >= 1")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not (np.all(np.isfinite(conductivity)) and np.all(conductivity > 0.0)):
        raise ValueError("conductivity entries must be finite and > 0")
    if not (np.all(np.isfinite(diffusivity)) and np.all(diffusivity > 0.0)):
        raise ValueError("diffusivity entries must be finite and > 0")
    for name, value in (("axial_eigenvalue", axial_eigenvalue),
                        ("decay_rate", decay_rate)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    eta = float(axial_eigenvalue)
    mu = float(decay_rate)

    def _basis(q, r):
        """Value and radial derivative of the two solutions of the radial ODE."""
        r = np.asarray(r, dtype=float)
        if q > 0.0:
            c = np.sqrt(q)
            return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
        if q < 0.0:
            c = np.sqrt(-q)
            return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
        return np.ones_like(r), np.log(r), np.zeros_like(r), 1.0 / r

    # The common decay rate fixes every layer separation constant at once.
    separation = mu / diffusivity - eta ** 2

    coefficients = np.zeros((n_layers, 3), dtype=float)
    coefficients[0] = (1.0, 0.0, separation[0])
    for i in range(n_layers - 1):
        radius = radii[i]
        u, v, du, dv = _basis(separation[i], radius)
        value = coefficients[i, 0] * u + coefficients[i, 1] * v
        flux = conductivity[i] * (coefficients[i, 0] * du + coefficients[i, 1] * dv)
        u_out, v_out, du_out, dv_out = _basis(separation[i + 1], radius)
        matrix = np.array([[u_out, v_out],
                           [conductivity[i + 1] * du_out, conductivity[i + 1] * dv_out]])
        pair = np.linalg.solve(matrix, np.array([value, flux]))
        coefficients[i + 1] = (pair[0], pair[1], separation[i + 1])

    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark copper / oxide / silicon via, slowest mode ---
        {
            "setup": """import numpy as np
def digest(*parts):
    # The harness compares plain numbers, so each array this step returns is
    # reduced to two descriptors of order one: a normalized weighted sum that
    # pins down its shape, and the log of its weighted magnitude that pins down
    # its scale. Both stay of order one whatever the component is worth, so an
    # error in a small component cannot hide behind a large one.
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = np.pi / (2.0 * 200.0e-6)
mu = 5701.567849
""",
            "call": "digest(*np.asarray(propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
            "gold_call": "digest(*np.asarray(_oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
        },
        # --- Valid: high decay rate, every layer oscillatory ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = 3.0 * np.pi / (2.0 * 200.0e-6)
mu = 5.0e7
""",
            "call": "digest(*np.asarray(propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
            "gold_call": "digest(*np.asarray(_oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
        },
        # --- Boundary: single homogeneous layer, nothing to propagate ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([30.0e-6])
conductivity = np.array([130.0])
diffusivity = np.array([130.0 / (2329.0 * 700.0)])
eta = np.pi / (2.0 * 200.0e-6)
mu = 1.0e5
""",
            "call": "digest(*np.asarray(propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
            "gold_call": "digest(*np.asarray(_oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
        },
        # --- Edge: five layers including a tantalum barrier and a nitride liner ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([2.5e-6, 2.6e-6, 2.75e-6, 2.9e-6, 5.0e-6])
conductivity = np.array([400.0, 57.5, 20.0, 1.4, 130.0])
density = np.array([8960.0, 16650.0, 3100.0, 2200.0, 2329.0])
capacity = np.array([385.0, 140.0, 700.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = np.pi / (2.0 * 50.0e-6)
mu = 2.0e5
""",
            "call": "digest(*np.asarray(propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
            "gold_call": "digest(*np.asarray(_oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, eta, mu)).T)",
        },
        # --- Consistency: temperature and flux must be continuous at both interfaces ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = np.pi / (2.0 * 200.0e-6)
mu = 5701.567849
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def check(fn):
    co = np.asarray(fn(radii, conductivity, diffusivity, eta, mu), dtype=float)
    worst_value = 0.0
    worst_flux = 0.0
    for i in range(2):
        r = radii[i]
        u, v, du, dv = basis(co[i, 2], r)
        u2, v2, du2, dv2 = basis(co[i + 1, 2], r)
        left = co[i, 0] * u + co[i, 1] * v
        right = co[i + 1, 0] * u2 + co[i + 1, 1] * v2
        fl = conductivity[i] * (co[i, 0] * du + co[i, 1] * dv)
        fr = conductivity[i + 1] * (co[i + 1, 0] * du2 + co[i + 1, 1] * dv2)
        worst_value = max(worst_value, abs(left - right) / max(abs(left), 1e-30))
        worst_flux = max(worst_flux, abs(fl - fr) / max(abs(fl), 1e-30))
    return int(worst_value < 1e-10), int(worst_flux < 1e-10)
""",
            "call": "check(propagate_radial_eigenfunction)",
            "gold_call": "check(_oracle_propagate_radial_eigenfunction)",
        },
        # --- Consistency: the copper core separation constant is negative for the slow mode ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = np.pi / (2.0 * 200.0e-6)
def check(fn):
    co = np.asarray(fn(radii, conductivity, diffusivity, eta, 5701.567849), dtype=float)
    return int(co[0, 2] < 0.0), int(co[1, 2] > 0.0), int(co[2, 2] > 0.0)
""",
            "call": "check(propagate_radial_eigenfunction)",
            "gold_call": "check(_oracle_propagate_radial_eigenfunction)",
        },
        # --- Invalid: radii not strictly increasing ---
        {
            "setup": """import numpy as np
radii = np.array([16.0e-6, 15.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
def run_model():
    try:
        propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 5701.6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 5701.6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mismatched property lengths ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
def run_model():
    try:
        propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 5701.6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 5701.6)
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
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
def run_model():
    try:
        propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_propagate_radial_eigenfunction(radii, conductivity, diffusivity, 7854.0, 0.0)
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
