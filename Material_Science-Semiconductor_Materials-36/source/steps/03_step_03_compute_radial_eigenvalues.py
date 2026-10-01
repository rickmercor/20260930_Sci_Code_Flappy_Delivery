"""
Locate the leading thermal decay rates of the layered via for one axial mode by finding the zeros of the outward radial slope at the adiabatic outer surface.

The admissible thermal decay rates of one axial mode are the trial rates mu for which the radial eigenfunction propagated outward from the axis, as in sub-problem 02, also satisfies the adiabatic condition dR_l/dr = 0 at the outer surface r = r_l. The function returns the n_modes smallest admissible rates in ascending order, counting every admissible rate, including one whose eigenfunction is flat in r. The search for them is sampled with samples_per_half_wave points per expected spacing between consecutive rates; the returned rates must not depend on that density once every rate has been bracketed, and a ValueError is raised if fewer than n_modes rates are bracketed.

Returns
-------
np.ndarray of shape (n_modes,), float: the ascending thermal decay rates mu in inverse seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_radial_eigenvalues(radii: np.ndarray, conductivity: np.ndarray,
                               diffusivity: np.ndarray, axial_eigenvalue: float,
                               n_modes: int,
                               samples_per_half_wave: int = 24) -> np.ndarray:
    """Leading thermal decay rates of the layered via for one axial mode.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing
        and strictly positive.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta in inverse metres (eta > 0).
    n_modes : int
        Number of radial modes to return (n_modes >= 1).
    samples_per_half_wave : int
        Number of scan samples per expected root spacing (>= 4).

    Returns
    -------
    decay_rates : np.ndarray
        Array of shape (n_modes,) holding the decay rates mu in inverse
        seconds, ascending.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; if axial_eigenvalue is not a finite
        number greater than 0; if n_modes is not an integer greater than
        or equal to 1; if samples_per_half_wave is not an integer
        greater than or equal to 4; or if the scan fails to bracket the
        requested number of modes.
    """
    return decay_rates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_radial_eigenvalues(radii: np.ndarray, conductivity: np.ndarray,
                                       diffusivity: np.ndarray,
                                       axial_eigenvalue: float, n_modes: int,
                                       samples_per_half_wave: int = 24) -> np.ndarray:
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
    if not (isinstance(axial_eigenvalue, (int, float, np.floating, np.integer))
            and np.isfinite(axial_eigenvalue) and float(axial_eigenvalue) > 0.0):
        raise ValueError("axial_eigenvalue must be a finite number > 0")
    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    if not (isinstance(samples_per_half_wave, (int, np.integer))
            and not isinstance(samples_per_half_wave, bool)
            and int(samples_per_half_wave) >= 4):
        raise ValueError("samples_per_half_wave must be an integer >= 4")

    eta = float(axial_eigenvalue)
    n_modes = int(n_modes)
    samples = int(samples_per_half_wave)

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

    def _outer_slope(mu):
        """Radial slope at the adiabatic outer surface for a trial decay rate."""
        separation = mu / diffusivity - eta ** 2
        amp_a, amp_b = 1.0, 0.0
        for i in range(n_layers - 1):
            radius = radii[i]
            u, v, du, dv = _basis(separation[i], radius)
            value = amp_a * u + amp_b * v
            flux = conductivity[i] * (amp_a * du + amp_b * dv)
            u_out, v_out, du_out, dv_out = _basis(separation[i + 1], radius)
            matrix = np.array([[u_out, v_out],
                               [conductivity[i + 1] * du_out,
                                conductivity[i + 1] * dv_out]])
            amp_a, amp_b = np.linalg.solve(matrix, np.array([value, flux]))
        _, _, du, dv = _basis(separation[-1], radii[-1])
        return amp_a * du + amp_b * dv

    # Root spacing follows the optical thickness of the stack in the variable
    # s = sqrt(mu / kappa_1), in which the roots are nearly equispaced.
    thickness = np.diff(np.concatenate(([0.0], radii)))
    path = float(np.sum(thickness * np.sqrt(diffusivity[0] / diffusivity)))
    step = np.pi / (samples * path)

    roots = []
    s_prev = 1.0e-9
    f_prev = _outer_slope(diffusivity[0] * s_prev ** 2)
    guard = 0
    while len(roots) < n_modes:
        guard += 1
        if guard > 4000000:
            raise ValueError("scan failed to bracket the requested number of modes")
        s_next = s_prev + step
        mu_next = diffusivity[0] * s_next ** 2
        f_next = _outer_slope(mu_next)
        if np.isfinite(f_prev) and np.isfinite(f_next) and f_prev * f_next < 0.0:
            lo = diffusivity[0] * s_prev ** 2
            hi = mu_next
            f_lo = f_prev
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                f_mid = _outer_slope(mid)
                if f_mid == 0.0:
                    lo = hi = mid
                    break
                if f_lo * f_mid < 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            roots.append(0.5 * (lo + hi))
        s_prev, f_prev = s_next, f_next

    return np.array(roots, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark via, lowest axial mode (normal scenario) ---
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
""",
            "call": "digest(compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 6))",
            "gold_call": "digest(_oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 6))",
        },
        # --- Valid: fifth axial mode of the same via ---
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
eta = 9.0 * np.pi / (2.0 * 200.0e-6)
""",
            "call": "digest(compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 4))",
            "gold_call": "digest(_oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 4))",
        },
        # --- Boundary: one mode only, from a single homogeneous cylinder ---
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
""",
            "call": "digest(compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 1))",
            "gold_call": "digest(_oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, eta, 1))",
        },
        # --- Consistency: a homogeneous cylinder has mu = kappa (beta^2 + eta^2) with J1(beta r) = 0, flat mode beta = 0 first ---
        {
            "setup": """import numpy as np
from scipy.special import jn_zeros
radii = np.array([30.0e-6])
conductivity = np.array([130.0])
kappa = 130.0 / (2329.0 * 700.0)
diffusivity = np.array([kappa])
eta = np.pi / (2.0 * 200.0e-6)
def check(fn):
    mus = np.asarray(fn(radii, conductivity, diffusivity, eta, 4), dtype=float)
    beta = np.concatenate(([0.0], jn_zeros(1, 3))) / radii[-1]
    exact = kappa * (beta ** 2 + eta ** 2)
    return int(np.max(np.abs(mus - exact) / exact) < 1e-10)
""",
            "call": "check(compute_radial_eigenvalues)",
            "gold_call": "check(_oracle_compute_radial_eigenvalues)",
        },
        # --- Consistency: the roots are ascending and insensitive to the scan density ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
density = np.array([8960.0, 2200.0, 2329.0])
capacity = np.array([385.0, 730.0, 700.0])
diffusivity = conductivity / (density * capacity)
eta = np.pi / (2.0 * 200.0e-6)
def check(fn):
    coarse = np.asarray(fn(radii, conductivity, diffusivity, eta, 5, 12), dtype=float)
    fine = np.asarray(fn(radii, conductivity, diffusivity, eta, 5, 64), dtype=float)
    ascending = int(np.all(np.diff(coarse) > 0.0))
    stable = int(np.max(np.abs(coarse - fine) / fine) < 1e-11)
    return ascending, stable
""",
            "call": "check(compute_radial_eigenvalues)",
            "gold_call": "check(_oracle_compute_radial_eigenvalues)",
        },
        # --- Invalid: fewer than one mode requested ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
def run_model():
    try:
        compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative diffusivity ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, -8.72e-7, 7.97e-5])
def run_model():
    try:
        compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: scan density below the admissible floor ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
conductivity = np.array([400.0, 1.4, 130.0])
diffusivity = np.array([1.16e-4, 8.72e-7, 7.97e-5])
def run_model():
    try:
        compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 3, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_radial_eigenvalues(radii, conductivity, diffusivity, 7854.0, 3, 1)
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
