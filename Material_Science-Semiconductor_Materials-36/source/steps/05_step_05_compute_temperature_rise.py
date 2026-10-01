"""
Evaluate the transient excess temperature of the layered via at one point and one instant by summing the assembled double eigenfunction series.

With the axial eigenvalues, the decay rates, the layer amplitudes and the expansion coefficients all in hand, the excess temperature above ambient is the double series theta(r, z, t) = sum over m and n of c_mn R_imn(r) cos(eta_m z) exp(-mu_mn t), where i is the index of the layer that contains r. Adding the ambient temperature back recovers the absolute temperature. Two features of this series deserve attention.

First, the temporal factor carries no layer index. That is the whole point of coupling the layers through a single decay rate: a layer-dependent exponent would satisfy the interface conditions at one instant only, and the interface temperature and flux would drift apart as soon as the solution was marched forward. The radial shape R_imn does depend on the layer, so the profile has a slope discontinuity at every interface, set by the conductivity ratio, while the value itself stays continuous.

Second, the series is strongly graded in time. For the leading axial mode of a via of these proportions the first radial overtone decays about seventeen times faster than the fundamental and carries only a small coefficient, and the higher overtones decay two to three orders of magnitude faster, so within a few tens of microseconds the radial content collapses onto the single slowest mode and the profile becomes almost flat in r, with only a small step across the low conductivity liner surviving. The axial content decays as the square of the odd integers, so a handful of axial modes suffices except in the first instants, when the truncated series shows the usual overshoot of a constant expanded on a basis pinned to zero at the sink. Any evaluation at a finite time after the quench is therefore insensitive to the truncation, which is what makes a modal answer reproducible across implementations that keep different numbers of modes.

Returns
-------
float: the excess temperature above ambient at the requested point and instant, in kelvin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_temperature_rise(radii: np.ndarray, axial_eigenvalues: np.ndarray,
                             decay_rates: np.ndarray,
                             modal_coefficients: np.ndarray,
                             eigen_coefficients: np.ndarray, radius: float,
                             depth: float, time: float) -> float:
    """Excess temperature of the layered via at one point and one instant.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing.
    axial_eigenvalues : np.ndarray
        Shape (n_axial,). Axial eigenvalues in inverse metres.
    decay_rates : np.ndarray
        Shape (n_axial, n_radial). Thermal decay rates in inverse seconds.
    modal_coefficients : np.ndarray
        Shape (n_axial, n_radial). Expansion coefficients in kelvin.
    eigen_coefficients : np.ndarray
        Shape (n_axial, n_radial, l, 3). Per mode and layer, the two radial
        amplitudes and the separation constant.
    radius : float
        Radial coordinate in metres, 0 < radius <= radii[-1].
    depth : float
        Axial coordinate in metres, measured from the insulated face (>= 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).

    Returns
    -------
    rise : float
        Excess temperature above ambient in kelvin, as a native Python float.

    Raises
    ------
    ValueError
        If radii or axial_eigenvalues is empty; if radii is not finite,
        positive and strictly increasing; if decay_rates does not have
        shape (n_axial, n_radial); if modal_coefficients does not have
        the same shape as decay_rates; if eigen_coefficients does not
        have shape (n_axial, n_radial, l, 3); if radius, depth or time
        is not a finite number; if radius lies outside the half-open
        interval (0, radii[-1]]; or if depth or time is negative.
    """
    return rise  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_temperature_rise(radii: np.ndarray,
                                     axial_eigenvalues: np.ndarray,
                                     decay_rates: np.ndarray,
                                     modal_coefficients: np.ndarray,
                                     eigen_coefficients: np.ndarray, radius: float,
                                     depth: float, time: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, j0, k0, y0

    radii = np.asarray(radii, dtype=float).ravel()
    axial_eigenvalues = np.asarray(axial_eigenvalues, dtype=float).ravel()
    decay_rates = np.asarray(decay_rates, dtype=float)
    modal_coefficients = np.asarray(modal_coefficients, dtype=float)
    eigen_coefficients = np.asarray(eigen_coefficients, dtype=float)
    n_layers = radii.size
    n_axial = axial_eigenvalues.size
    if n_layers < 1 or n_axial < 1:
        raise ValueError("radii and axial_eigenvalues must both be non-empty")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if decay_rates.ndim != 2 or decay_rates.shape[0] != n_axial:
        raise ValueError("decay_rates must have shape (n_axial, n_radial)")
    if modal_coefficients.shape != decay_rates.shape:
        raise ValueError("modal_coefficients must have the same shape as decay_rates")
    if eigen_coefficients.shape != decay_rates.shape + (n_layers, 3):
        raise ValueError("eigen_coefficients must have shape (n_axial, n_radial, l, 3)")
    for name, value in (("radius", radius), ("depth", depth), ("time", time)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(radius) <= 0.0 or float(radius) > radii[-1]:
        raise ValueError("radius must lie in the half-open interval (0, radii[-1]]")
    if float(depth) < 0.0 or float(time) < 0.0:
        raise ValueError("depth and time must be >= 0")

    radius = float(radius)
    depth = float(depth)
    time = float(time)

    def _values(q, r):
        """The two solutions of the radial ODE, on the branch fixed by sign(q)."""
        if q > 0.0:
            c = np.sqrt(q)
            return j0(c * r), y0(c * r)
        if q < 0.0:
            c = np.sqrt(-q)
            return i0(c * r), k0(c * r)
        return 1.0, np.log(r)

    layer = int(min(np.searchsorted(radii, radius, side="left"), n_layers - 1))

    rise = 0.0
    n_radial = decay_rates.shape[1]
    for m in range(n_axial):
        axial = np.cos(axial_eigenvalues[m] * depth)
        for n in range(n_radial):
            amp_a, amp_b, separation = eigen_coefficients[m, n, layer]
            first, second = _values(separation, radius)
            shape = amp_a * first + amp_b * second
            rise += (modal_coefficients[m, n] * shape * axial
                     * np.exp(-decay_rates[m, n] * time))

    return float(rise)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark via at the mid plane depth, 0.05 ms after the quench ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
height = 200.0e-6
etas = np.array([(2.0 * m - 1.0) * np.pi / (2.0 * height) for m in (1, 2, 3)])
mus = np.array([[5.701567849e3, 9.697475846e4],
                [4.910052825e4, 1.552574424e5],
                [1.425416862e5, 3.321641351e5]])
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def propagate(eta, mu):
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
eig = np.array([[propagate(etas[m], mus[m, n]) for n in range(2)] for m in range(3)])
coef = np.array([[125.5, 1.78], [-37.0, 0.6], [17.0, 0.2]])
""",
            "call": "compute_temperature_rise(radii, etas, mus, coef, eig, 20.0e-6, height / 3.0, 5.0e-5)",
            "gold_call": "_oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 20.0e-6, height / 3.0, 5.0e-5)",
        },
        # --- Valid: same expansion evaluated inside the copper core at t = 0 ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
height = 200.0e-6
etas = np.array([(2.0 * m - 1.0) * np.pi / (2.0 * height) for m in (1, 2)])
mus = np.array([[5.701567849e3, 9.697475846e4],
                [4.910052825e4, 1.552574424e5]])
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def propagate(eta, mu):
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
eig = np.array([[propagate(etas[m], mus[m, n]) for n in range(2)] for m in range(2)])
coef = np.array([[125.5, 1.78], [-37.0, 0.6]])
""",
            "call": "compute_temperature_rise(radii, etas, mus, coef, eig, 5.0e-6, 0.0, 0.0)",
            "gold_call": "_oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 5.0e-6, 0.0, 0.0)",
        },
        # --- Boundary: exactly on the heat sink, where every axial mode vanishes ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
height = 200.0e-6
etas = np.array([(2.0 * m - 1.0) * np.pi / (2.0 * height) for m in (1, 2, 3, 4)])
mus = np.full((4, 1), 1.0e4)
coef = np.array([[100.0], [-33.0], [20.0], [-14.0]])
eig = np.ones((4, 1, 1, 3))
eig[..., 0] = 1.0
eig[..., 1] = 0.0
eig[..., 2] = -1.0e6
""",
            "call": "compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, height, 1.0e-5)",
            "gold_call": "_oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, height, 1.0e-5)",
        },
        # --- Edge: very long time, every mode exponentially dead ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
height = 200.0e-6
etas = np.array([np.pi / (2.0 * height)])
mus = np.array([[5.7e3]])
coef = np.array([[125.5]])
eig = np.array([[[[1.0, 0.0, -1.0e6]]]])
""",
            "call": "compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, 1.0)",
            "gold_call": "_oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, 1.0)",
        },
        # --- Consistency: continuity of temperature across the oxide interfaces ---
        {
            "setup": """import numpy as np
from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
height = 200.0e-6
etas = np.array([np.pi / (2.0 * height)])
mus = np.array([[5.701567849e3]])
def basis(q, r):
    if q > 0.0:
        c = np.sqrt(q)
        return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
    c = np.sqrt(-q)
    return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
def propagate(eta, mu):
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
eig = np.array([[propagate(etas[0], mus[0, 0])]])
coef = np.array([[125.5]])
def check(fn):
    worst = 0.0
    for r in radii[:2]:
        inner = fn(radii, etas, mus, coef, eig, r * (1.0 - 1.0e-9), height / 3.0, 5.0e-5)
        outer = fn(radii, etas, mus, coef, eig, r * (1.0 + 1.0e-9), height / 3.0, 5.0e-5)
        worst = max(worst, abs(inner - outer) / abs(inner))
    return int(worst < 1e-7)
""",
            "call": "check(compute_temperature_rise)",
            "gold_call": "check(_oracle_compute_temperature_rise)",
        },
        # --- Invalid: radius outside the via ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
etas = np.array([7854.0])
mus = np.array([[5.7e3]])
coef = np.array([[125.5]])
eig = np.array([[[[1.0, 0.0, -1.0e6]]]])
def run_model():
    try:
        compute_temperature_rise(radii, etas, mus, coef, eig, 31.0e-6, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 31.0e-6, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: coefficient table inconsistent with the decay-rate table ---
        {
            "setup": """import numpy as np
radii = np.array([30.0e-6])
etas = np.array([7854.0])
mus = np.array([[5.7e3, 9.7e4]])
coef = np.array([[125.5]])
eig = np.zeros((1, 2, 1, 3))
def run_model():
    try:
        compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, 0.0)
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
radii = np.array([30.0e-6])
etas = np.array([7854.0])
mus = np.array([[5.7e3]])
coef = np.array([[125.5]])
eig = np.array([[[[1.0, 0.0, -1.0e6]]]])
def run_model():
    try:
        compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, -1.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_temperature_rise(radii, etas, mus, coef, eig, 10.0e-6, 0.0, -1.0e-6)
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
