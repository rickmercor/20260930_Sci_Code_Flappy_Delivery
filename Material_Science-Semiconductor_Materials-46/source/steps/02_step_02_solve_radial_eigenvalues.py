"""
Return the slowest decay rates of the layered via for one axial mode, obtained as the eigenvalues of the radial problem that couples all material layers through the interface conditions.

Once the axial dependence has been separated, what remains in each layer is a zero-order Bessel equation whose parameter is the layer's own separation constant. Those constants cannot be chosen independently: a single product solution must decay in time at one rate throughout the whole via, otherwise the interface conditions, which have to hold at every instant, would be satisfied only at isolated times. Requiring a common temporal exponent ties the layer constants to one another through the layer thermal diffusivities and turns what looks like a family of independent radial problems into a single eigenvalue problem for that shared decay rate. This is the step that makes an exact layered solution possible, and it is also the step that makes the layer parameters unequal in magnitude, since a slow layer needs a much larger radial parameter than a fast one to decay at the same rate.




The interface conditions are continuity of temperature and continuity of the radial heat flux, so a solution can be propagated outwards: start from the regular solution on the axis, where the second Bessel function of each layer is inadmissible, carry the temperature and the flux across each interface, and re-express them in the next layer's basis. The remaining condition, an insulated outer cylindrical surface, is then a scalar function of the trial decay rate whose zeros are the eigenvalues, and propagating rather than assembling a global determinant keeps the search well conditioned because every interface transfer is a two-by-two solve whose determinant is fixed by the Bessel Wronskian and can never vanish.




A subtlety governs the whole search. A layer's radial parameter is positive only when the trial decay rate exceeds that layer's thermal diffusivity times the square of the axial eigenvalue; below that value the parameter is negative and the layer's radial behaviour is exponential rather than oscillatory, described by modified Bessel functions. Each layer therefore contributes its own threshold, the scalar residual is smooth only between consecutive thresholds, and the search must be conducted interval by interval. Ignoring this and looking only for oscillatory solutions loses the slowest-decaying eigenvalue of the via whenever the material on the axis is the one with the highest diffusivity, which is exactly the mode that dominates the transient.

Returns
-------
np.ndarray of shape (n_modes,), float: the n_modes smallest decay rates in 1/s for the given axial mode, sorted ascending, in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_radial_eigenvalues(radii: np.ndarray, conductivities: np.ndarray,
                             diffusivities: np.ndarray, eta: float,
                             n_modes: int) -> np.ndarray:
    """Return the slowest decay rates of the layered via for one axial mode.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive; the innermost layer starts on the axis.
    conductivities : np.ndarray
        Thermal conductivity of each layer in W/(m K), shape (n_layers,),
        all positive.
    diffusivities : np.ndarray
        Thermal diffusivity of each layer in m^2/s, shape (n_layers,),
        all positive.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    n_modes : int
        Number of decay rates to return, ordered by increasing value
        (n_modes >= 1).

    Returns
    -------
    decay_rates : np.ndarray
        Array of shape (n_modes,) holding the decay rates in 1/s, ascending.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the
        search cannot locate n_modes decay rates; invalid input must raise
        ValueError rather than return a sentinel value.
    """
    return decay_rates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_radial_eigenvalues(radii: np.ndarray,
                                     conductivities: np.ndarray,
                                     diffusivities: np.ndarray, eta: float,
                                     n_modes: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    if not (isinstance(eta, (int, float, np.floating, np.integer))
            and not isinstance(eta, bool)
            and np.isfinite(eta) and float(eta) > 0.0):
        raise ValueError("eta must be a finite number > 0")

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
    n_modes = int(n_modes)
    inner = np.concatenate([[0.0], radii[:-1]])
    layer_thresholds = diffusivities * eta ** 2

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

    def _residual(decay):
        """Normalised radial derivative at the insulated outer surface."""
        # Form the difference before dividing. At decay = D_i * eta^2 this
        # produces an exact zero for the flat radial branch instead of losing
        # it to cancellation in decay / D_i - eta^2.
        mu = (decay - layer_thresholds) / diffusivities
        amp_a, amp_b = 1.0, 0.0
        for i in range(n_layers - 1):
            radius = radii[i]
            ca, cap, da, dap = _basis(mu[i], radius)
            value = amp_a * ca + amp_b * da
            slope = amp_a * cap + amp_b * dap
            cb, cbp, db, dbp = _basis(mu[i + 1], radius)
            matrix = np.array([[cb, db],
                               [conductivities[i + 1] * cbp,
                                conductivities[i + 1] * dbp]])
            amp_a, amp_b = np.linalg.solve(
                matrix, np.array([value, conductivities[i] * slope]))
        ca, cap, da, dap = _basis(mu[-1], radii[-1])
        value = amp_a * ca + amp_b * da
        slope = amp_a * cap + amp_b * dap
        scale = abs(slope) + abs(value) * np.sqrt(abs(mu[-1]))
        if scale == 0.0:
            return 0.0
        return slope / scale

    def _bisect(lo, hi):
        """Refine a bracketed sign change of the residual."""
        f_lo = _residual(lo)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = _residual(mid)
            if f_mid == 0.0 or (hi - lo) <= 1.0e-15 * abs(mid):
                return mid
            if (f_lo > 0.0) != (f_mid > 0.0):
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    thresholds = np.sort(layer_thresholds)
    roots = []

    # Below the highest threshold at least one layer is non-oscillatory, and
    # the residual is smooth only between consecutive thresholds.
    edges = np.concatenate([[0.0], thresholds])
    for lo_edge, hi_edge in zip(edges[:-1], edges[1:]):
        if not hi_edge > lo_edge:
            continue
        grid = np.linspace(lo_edge + 1.0e-9 * (hi_edge - lo_edge),
                           hi_edge - 1.0e-9 * (hi_edge - lo_edge), 512)
        values = np.array([_residual(g) for g in grid])
        for j in range(grid.size - 1):
            if (values[j] > 0.0) != (values[j + 1] > 0.0):
                roots.append(_bisect(grid[j], grid[j + 1]))

    # A threshold itself is an eigenvalue when the outer layer is flat there,
    # which is what happens for a homogeneous cylinder.
    for level in np.unique(thresholds):
        if abs(_residual(level)) < 1.0e-12:
            roots.append(float(level))

    # Above every threshold each layer oscillates, and the natural scan
    # variable is the radial parameter of the innermost layer.
    reference = diffusivities[0]
    path = float(np.sum(np.sqrt(reference / diffusivities) * (radii - inner)))
    step = np.pi / path / 64.0
    start = float(thresholds[-1]) * (1.0 + 1.0e-12)
    previous = _residual(start)
    previous_decay = start
    scan = np.sqrt(start / reference)
    # Cost note: the step is one 64th of the shortest oscillation in the stack,
    # so consecutive eigenvalues are about 64 steps apart and the loop exits
    # after roughly 64 * (n_modes + 2) iterations -- a few hundred for any
    # request this task makes. The 400000 bound is only a guard against a
    # pathological property set that never brackets a sign change; reaching it
    # costs about a second and then raises below rather than looping forever.
    for _ in range(400000):
        if len(roots) >= n_modes + 2:
            break
        scan += step
        decay = reference * scan ** 2
        current = _residual(decay)
        if (previous > 0.0) != (current > 0.0):
            roots.append(_bisect(previous_decay, decay))
        previous, previous_decay = current, decay

    roots = np.sort(np.array(roots, dtype=float))
    if roots.size < n_modes:
        raise ValueError("the eigenvalue search did not locate enough decay rates")
    return roots[:n_modes]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark copper / oxide / silicon via (normal scenario) ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = np.pi / (2.0 * 200.0e-6)
n_modes = 6
""",
            "call": "solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
            "gold_call": "_oracle_solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
        },
        # --- Valid: higher axial mode of the same via ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = 9.0 * np.pi / (2.0 * 200.0e-6)
n_modes = 4
""",
            "call": "solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
            "gold_call": "_oracle_solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
        },
        # --- Boundary: exact regression for the flat homogeneous mode ---
        {
            "setup": """import numpy as np
radii = np.array([1.0e-5])
conductivities = np.array([10.0])
diffusivities = np.array([1.0e-5])
eta = 1000.0
n_modes = 4
""",
            "call": "solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)[0]",
            "gold_call": "10.0",
        },
        # --- Edge: four layers including a very thin, very slow barrier ---
        {
            "setup": """import numpy as np
radii = np.array([2.5e-6, 2.6e-6, 2.9e-6, 10.0e-6])
conductivities = np.array([400.0, 57.5, 1.4, 130.0])
capacities = np.array([8960.0 * 385.0, 16650.0 * 140.0, 2200.0 * 730.0, 2329.0 * 700.0])
diffusivities = conductivities / capacities
eta = np.pi / (2.0 * 50.0e-6)
n_modes = 3
""",
            "call": "solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
            "gold_call": "_oracle_solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, n_modes)",
        },
        # --- Invalid: radii not strictly increasing ---
        {
            "setup": """import numpy as np
radii = np.array([16.0e-6, 15.0e-6, 25.0e-6])
conductivities = np.array([400.0, 1.4, 130.0])
diffusivities = np.array([1.0e-4, 1.0e-6, 8.0e-5])
eta = 7853.981633974483
def run_model():
    try:
        solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive thermal diffusivity ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 25.0e-6])
conductivities = np.array([400.0, 130.0])
diffusivities = np.array([1.0e-4, 0.0])
eta = 7853.981633974483
def run_model():
    try:
        solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_radial_eigenvalues(radii, conductivities, diffusivities, eta, 3)
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
