"""
Project a uniform initial temperature rise onto the layered eigenfunctions of one axial mode and return the expansion coefficient of every radial mode.

The excess temperature is expanded as theta(r, z, t) = sum over m and n of c_mn R_mn(r) cos(eta_m z) exp(-mu_mn t), with the radial eigenfunctions R_mn in the convention of sub-problem 02 (A_1 = 1, B_1 = 0, evaluated layer by layer) and cos(eta_m z) the axial eigenfunction of squared norm N_z. For one axial mode m this function returns the coefficients c_mn, in kelvin, for which the series reproduces a uniform initial rise theta_0 at t = 0; each coefficient therefore contains both the axial and the radial projection of theta_0. The eigen_coefficients argument holds, for each radial mode, the amplitudes and the separation constant of every layer. The projection integrals have no closed form once the layers carry different separation constants, so they may be evaluated numerically, but to a relative accuracy of about 1e-10 or better.

Returns
-------
np.ndarray of shape (n_radial,), float: the expansion coefficient of every radial mode of this axial mode, in kelvin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_modal_coefficients(radii: np.ndarray, conductivity: np.ndarray,
                               diffusivity: np.ndarray, height: float,
                               axial_eigenvalue: float, axial_norm: float,
                               eigen_coefficients: np.ndarray,
                               initial_rise: float) -> np.ndarray:
    """Expansion coefficients of a uniform initial temperature rise.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    height : float
        Total via height h in metres (h > 0).
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (eta > 0).
    axial_norm : float
        Squared norm N_z of the axial eigenfunction in metres (> 0).
    eigen_coefficients : np.ndarray
        Shape (n_radial, l, 3). For each radial mode, the two amplitudes and
        the separation constant of every layer.
    initial_rise : float
        Uniform initial temperature rise above ambient in kelvin.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (n_radial,) holding the expansion coefficients in
        kelvin.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; if eigen_coefficients is not finite
        or does not have shape (n_radial, l, 3) with n_radial >= 1; if
        height, axial_eigenvalue or axial_norm is not a finite number
        greater than 0; or if initial_rise is not a finite number.
    """
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_modal_coefficients(radii: np.ndarray, conductivity: np.ndarray,
                                       diffusivity: np.ndarray, height: float,
                                       axial_eigenvalue: float, axial_norm: float,
                                       eigen_coefficients: np.ndarray,
                                       initial_rise: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, j0, k0, y0

    radii = np.asarray(radii, dtype=float).ravel()
    conductivity = np.asarray(conductivity, dtype=float).ravel()
    diffusivity = np.asarray(diffusivity, dtype=float).ravel()
    eigen_coefficients = np.asarray(eigen_coefficients, dtype=float)
    n_layers = radii.size
    if n_layers < 1 or conductivity.size != n_layers or diffusivity.size != n_layers:
        raise ValueError("radii, conductivity and diffusivity must share one length l >= 1")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not (np.all(np.isfinite(conductivity)) and np.all(conductivity > 0.0)):
        raise ValueError("conductivity entries must be finite and > 0")
    if not (np.all(np.isfinite(diffusivity)) and np.all(diffusivity > 0.0)):
        raise ValueError("diffusivity entries must be finite and > 0")
    if eigen_coefficients.ndim != 3 or eigen_coefficients.shape[1] != n_layers \
            or eigen_coefficients.shape[2] != 3 or eigen_coefficients.shape[0] < 1:
        raise ValueError("eigen_coefficients must have shape (n_radial, l, 3) with n_radial >= 1")
    if not np.all(np.isfinite(eigen_coefficients)):
        raise ValueError("eigen_coefficients must be finite")
    for name, value in (("height", height), ("axial_eigenvalue", axial_eigenvalue),
                        ("axial_norm", axial_norm)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(initial_rise, (int, float, np.floating, np.integer))
            and np.isfinite(initial_rise)):
        raise ValueError("initial_rise must be a finite number")

    height = float(height)
    eta = float(axial_eigenvalue)
    axial_norm = float(axial_norm)
    initial_rise = float(initial_rise)

    def _values(q, r):
        """The two solutions of the radial ODE, on the branch fixed by sign(q)."""
        if q > 0.0:
            c = np.sqrt(q)
            return j0(c * r), y0(c * r)
        if q < 0.0:
            c = np.sqrt(-q)
            return i0(c * r), k0(c * r)
        return np.ones_like(r), np.log(r)

    # The radial operator is self adjoint under the volumetric heat capacity,
    # which is the conductivity divided by the diffusivity.
    weight = conductivity / diffusivity
    edges = np.concatenate(([0.0], radii))
    axial_factor = (np.sin(eta * height) / eta) / axial_norm

    n_radial = eigen_coefficients.shape[0]
    coefficients = np.zeros(n_radial, dtype=float)
    for n in range(n_radial):
        squared_norm = 0.0
        projection = 0.0
        for i in range(n_layers):
            lo, hi = edges[i], edges[i + 1]
            separation = eigen_coefficients[n, i, 2]
            wavenumber = np.sqrt(abs(separation))
            n_nodes = int(min(4096, max(96, np.ceil(24.0 * wavenumber * (hi - lo) / np.pi))))
            nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
            r = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
            quad = 0.5 * (hi - lo) * weights
            first, second = _values(separation, r)
            shape = eigen_coefficients[n, i, 0] * first + eigen_coefficients[n, i, 1] * second
            squared_norm += weight[i] * float(np.sum(quad * r * shape * shape))
            projection += weight[i] * float(np.sum(quad * r * shape))
        coefficients[n] = initial_rise * projection * axial_factor / squared_norm

    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark via, lowest axial mode (normal scenario) ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
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
height = 200.0e-6
eta = np.pi / (2.0 * height)
axial_norm = 0.5 * height
mus = np.array([5.701567849e3, 9.697475846e4, 3.966379306e6, 7.131998992e6])
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def propagate(mu):
    sep = mu / diffusivity - eta ** 2
    out = np.zeros((3, 3))
    out[0] = (1.0, 0.0, sep[0])
    for i in range(2):
        r = radii[i]
        u, v, du, dv = basis(sep[i], r)
        val = out[i, 0] * u + out[i, 1] * v
        flx = conductivity[i] * (out[i, 0] * du + out[i, 1] * dv)
        u2, v2, du2, dv2 = basis(sep[i + 1], r)
        m = np.array([[u2, v2], [conductivity[i + 1] * du2, conductivity[i + 1] * dv2]])
        a, b = np.linalg.solve(m, np.array([val, flx]))
        out[i + 1] = (a, b, sep[i + 1])
    return out
eig = np.array([propagate(m) for m in mus])
""",
            "call": "digest(compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0))",
            "gold_call": "digest(_oracle_compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0))",
        },
        # --- Valid: third axial mode, sign of the axial projection flips ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
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
height = 200.0e-6
eta = 5.0 * np.pi / (2.0 * height)
axial_norm = 0.5 * height
mus = np.array([1.4254e5, 3.3216e5, 4.1108e6])
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def propagate(mu):
    sep = mu / diffusivity - eta ** 2
    out = np.zeros((3, 3))
    out[0] = (1.0, 0.0, sep[0])
    for i in range(2):
        r = radii[i]
        u, v, du, dv = basis(sep[i], r)
        val = out[i, 0] * u + out[i, 1] * v
        flx = conductivity[i] * (out[i, 0] * du + out[i, 1] * dv)
        u2, v2, du2, dv2 = basis(sep[i + 1], r)
        m = np.array([[u2, v2], [conductivity[i + 1] * du2, conductivity[i + 1] * dv2]])
        a, b = np.linalg.solve(m, np.array([val, flx]))
        out[i + 1] = (a, b, sep[i + 1])
    return out
eig = np.array([propagate(m) for m in mus])
""",
            "call": "digest(compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0))",
            "gold_call": "digest(_oracle_compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0))",
        },
        # --- Boundary: one layer, one mode, ordinary Bessel branch ---
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
height = 200.0e-6
eta = np.pi / (2.0 * height)
axial_norm = 0.5 * height
eig = np.array([[[1.0, 0.0, (3.8317059702075125 / 30.0e-6) ** 2]]])
""",
            # The exact coefficient of this J1-zero mode is 0, so the raw value is compared
            # with an absolute tolerance instead of through the log-magnitude digest.
            "call": "compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0)",
            "gold_call": "_oracle_compute_modal_coefficients(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0)",
            "tol": 1e-8,
        },
        # --- Consistency: a homogeneous cylinder rejects every non constant radial mode ---
        {
            "setup": """import numpy as np
from scipy.special import jn_zeros
radii = np.array([30.0e-6])
conductivity = np.array([130.0])
diffusivity = np.array([130.0 / (2329.0 * 700.0)])
height = 200.0e-6
eta = np.pi / (2.0 * height)
axial_norm = 0.5 * height
beta = jn_zeros(1, 3) / radii[-1]
eig = np.array([[[1.0, 0.0, b ** 2]] for b in beta])
flat = np.array([[[1.0, 0.0, 0.0]]])
def check(fn):
    higher = np.asarray(fn(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0), dtype=float)
    constant = np.asarray(fn(radii, conductivity, diffusivity, height, eta, axial_norm, flat, 100.0), dtype=float)
    exact = 100.0 * (np.sin(eta * height) / eta) / axial_norm
    return (int(np.max(np.abs(higher)) < 1e-8),
            int(abs(constant[0] - exact) / abs(exact) < 1e-12))
""",
            "call": "check(compute_modal_coefficients)",
            "gold_call": "check(_oracle_compute_modal_coefficients)",
        },
        # --- Consistency: coefficients scale linearly with the initial rise ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
conductivity = np.array([130.0])
diffusivity = np.array([130.0 / (2329.0 * 700.0)])
height = 200.0e-6
eta = np.pi / (2.0 * height)
axial_norm = 0.5 * height
eig = np.array([[[1.0, 0.0, -eta ** 2]], [[1.0, 0.0, 1.0e10]]])
def check(fn):
    a = np.asarray(fn(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 100.0), dtype=float)
    b = np.asarray(fn(radii, conductivity, diffusivity, height, eta, axial_norm, eig, 250.0), dtype=float)
    return int(np.max(np.abs(2.5 * a - b)) < 1e-10 * max(np.max(np.abs(b)), 1.0))
""",
            "call": "check(compute_modal_coefficients)",
            "gold_call": "check(_oracle_compute_modal_coefficients)",
        },
        # --- Invalid: eigen_coefficients with the wrong trailing dimension ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
eig = np.zeros((2, 3, 2))
def run_model():
    try:
        compute_modal_coefficients(radii, conductivity, diffusivity, 200.0e-6, 7854.0, 1.0e-4, eig, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_modal_coefficients(radii, conductivity, diffusivity, 200.0e-6, 7854.0, 1.0e-4, eig, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive axial norm ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
conductivity = np.array([130.0])
diffusivity = np.array([7.97e-5])
eig = np.array([[[1.0, 0.0, 1.0e10]]])
def run_model():
    try:
        compute_modal_coefficients(radii, conductivity, diffusivity, 200.0e-6, 7854.0, 0.0, eig, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_modal_coefficients(radii, conductivity, diffusivity, 200.0e-6, 7854.0, 0.0, eig, 100.0)
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
