#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_axial_modes(height: float, n_modes: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    if not (isinstance(height, (int, float, np.floating, np.integer))
            and not isinstance(height, bool)
            and np.isfinite(height) and float(height) > 0.0):
        raise ValueError("height must be a finite number > 0")

    height = float(height)
    n_modes = int(n_modes)

    # Insulated at z = 0 and pinned at z = h selects the cosines whose
    # quarter wavelengths fit an odd number of times into the height.
    index = np.arange(1, n_modes + 1, dtype=float)
    eta = (2.0 * index - 1.0) * np.pi / (2.0 * height)

    # The cosines are orthogonal on the height with a norm of half the height,
    # and their plain integral alternates in sign with the mode index.
    norm = np.full(n_modes, 0.5 * height)
    moment = np.sin(eta * height) / eta

    return np.column_stack([eta, norm, moment])

def solve_radial_eigenvalues(radii: np.ndarray,
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

def build_radial_eigenfunction(radii: np.ndarray,
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

def compute_expansion_coefficient(radii: np.ndarray,
                                          capacities: np.ndarray,
                                          eigenfunction: np.ndarray,
                                          axial_mode: np.ndarray,
                                          initial_rise: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    if not (isinstance(initial_rise, (int, float, np.floating, np.integer))
            and not isinstance(initial_rise, bool) and np.isfinite(initial_rise)):
        raise ValueError("initial_rise must be a finite number")

    radii = np.asarray(radii, dtype=float).ravel()
    capacities = np.asarray(capacities, dtype=float).ravel()
    eigenfunction = np.asarray(eigenfunction, dtype=float)
    axial_mode = np.asarray(axial_mode, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    if capacities.size != n_layers:
        raise ValueError("radii and capacities must agree in length")
    if eigenfunction.shape != (n_layers, 3):
        raise ValueError("eigenfunction must have shape (n_layers, 3)")
    if axial_mode.size != 3:
        raise ValueError("axial_mode must have three entries")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(capacities)) or np.any(capacities <= 0.0):
        raise ValueError("capacities must be finite and positive")

    eta, axial_norm, axial_moment = (float(axial_mode[0]), float(axial_mode[1]),
                                     float(axial_mode[2]))
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError("the axial eigenvalue must be finite and > 0")
    if not (np.isfinite(axial_norm) and axial_norm > 0.0):
        raise ValueError("the axial square norm must be finite and > 0")

    inner = np.concatenate([[0.0], radii[:-1]])

    def _value_and_slope(mu, amp_a, amp_b, radius):
        """Radial eigenfunction and its derivative at a radius."""
        if radius == 0.0:
            return amp_a, 0.0
        if mu > 0.0:
            b = np.sqrt(mu)
            x = b * radius
            return (amp_a * j0(x) + amp_b * y0(x),
                    -b * (amp_a * j1(x) + amp_b * y1(x)))
        if mu < 0.0:
            b = np.sqrt(-mu)
            x = b * radius
            return (amp_a * i0(x) + amp_b * k0(x),
                    b * (amp_a * i1(x) - amp_b * k1(x)))
        return amp_a + amp_b * np.log(radius), amp_b / radius

    def _moments(mu, amp_a, amp_b, radius):
        """Antiderivatives of r*R and r*R^2 evaluated at a radius."""
        if radius == 0.0:
            return 0.0, 0.0
        if mu != 0.0:
            value, slope = _value_and_slope(mu, amp_a, amp_b, radius)
            return (-radius * slope / mu,
                    0.5 * radius ** 2 * (slope ** 2 / mu + value ** 2))
        log_r = np.log(radius)
        first = (0.5 * amp_a * radius ** 2
                 + amp_b * (0.5 * radius ** 2 * log_r - 0.25 * radius ** 2))
        second = (0.5 * amp_a ** 2 * radius ** 2
                  + amp_a * amp_b * (radius ** 2 * log_r - 0.5 * radius ** 2)
                  + amp_b ** 2 * (0.5 * radius ** 2 * log_r ** 2
                                  - 0.5 * radius ** 2 * log_r + 0.25 * radius ** 2))
        return first, second

    # Both radial integrals are weighted by the volumetric heat capacity,
    # which is the weight that makes the layered radial operator self-adjoint.
    radial_moment = 0.0
    radial_norm = 0.0
    for i in range(n_layers):
        mu, amp_a, amp_b = (float(eigenfunction[i, 0]), float(eigenfunction[i, 1]),
                            float(eigenfunction[i, 2]))
        outer_first, outer_second = _moments(mu, amp_a, amp_b, radii[i])
        inner_first, inner_second = _moments(mu, amp_a, amp_b, inner[i])
        radial_moment += capacities[i] * (outer_first - inner_first)
        radial_norm += capacities[i] * (outer_second - inner_second)

    if radial_norm == 0.0:
        raise ValueError("the radial square norm of the eigenfunction vanishes")

    return float(initial_rise) * axial_moment * radial_moment / (axial_norm * radial_norm)

def evaluate_temperature_rise(axial_modes: np.ndarray,
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

def compute_thermal_source_terms(decay_rates: np.ndarray,
                                         eigenfunctions: np.ndarray,
                                         coefficients: np.ndarray, eta: float,
                                         temperature_rise: float,
                                         thermal_expansion: float,
                                         poisson: float, shear_modulus: float,
                                         layer: int, radius: float,
                                         time: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    decay_rates = np.asarray(decay_rates, dtype=float).ravel()
    eigenfunctions = np.asarray(eigenfunctions, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).ravel()

    n_radial = decay_rates.size
    if n_radial < 1:
        raise ValueError("decay_rates must hold at least one radial mode")
    if coefficients.size != n_radial:
        raise ValueError("decay_rates and coefficients must agree in length")
    if (eigenfunctions.ndim != 3 or eigenfunctions.shape[0] != n_radial
            or eigenfunctions.shape[2] != 3):
        raise ValueError("eigenfunctions must have shape (n_radial, n_layers, 3)")
    n_layers = eigenfunctions.shape[1]
    if not (isinstance(layer, (int, np.integer)) and not isinstance(layer, bool)
            and 0 <= int(layer) < n_layers):
        raise ValueError("layer must be an integer index of an existing layer")
    if not (isinstance(eta, (int, float, np.floating, np.integer))
            and not isinstance(eta, bool) and np.isfinite(eta) and float(eta) > 0.0):
        raise ValueError("eta must be a finite number > 0")
    if not (isinstance(shear_modulus, (int, float, np.floating, np.integer))
            and not isinstance(shear_modulus, bool) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    if not (isinstance(poisson, (int, float, np.floating, np.integer))
            and not isinstance(poisson, bool) and np.isfinite(poisson)
            and -1.0 < float(poisson) < 0.5):
        raise ValueError("poisson must be a finite number in (-1, 0.5)")
    if not (isinstance(thermal_expansion, (int, float, np.floating, np.integer))
            and not isinstance(thermal_expansion, bool)
            and np.isfinite(thermal_expansion)):
        raise ValueError("thermal_expansion must be a finite number")
    if not (isinstance(temperature_rise, (int, float, np.floating, np.integer))
            and not isinstance(temperature_rise, bool)
            and np.isfinite(temperature_rise)):
        raise ValueError("temperature_rise must be a finite number")
    for name, value in (("radius", radius), ("time", time)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")

    layer = int(layer)
    eta = float(eta)
    radius = float(radius)
    time = float(time)
    shear_modulus = float(shear_modulus)
    poisson = float(poisson)
    temperature_rise = float(temperature_rise)

    def _radial_derivatives(mu, amp_a, amp_b, r):
        """Radial eigenfunction, slope and curvature at a radius."""
        if r == 0.0:
            return amp_a, 0.0, -0.5 * mu * amp_a
        if mu > 0.0:
            b = np.sqrt(mu)
            x = b * r
            value = amp_a * j0(x) + amp_b * y0(x)
            slope = -b * (amp_a * j1(x) + amp_b * y1(x))
        elif mu < 0.0:
            b = np.sqrt(-mu)
            x = b * r
            value = amp_a * i0(x) + amp_b * k0(x)
            slope = b * (amp_a * i1(x) - amp_b * k1(x))
        else:
            value = amp_a + amp_b * np.log(r)
            slope = amp_b / r
        return value, slope, -slope / r - mu * value

    # Free thermal dilatation of a constrained isotropic solid.
    modulus = thermal_expansion * (1.0 + poisson) / (1.0 - poisson)

    sum_value = sum_slope = sum_curvature = sum_slope_over_r = 0.0
    for n in range(n_radial):
        mu, amp_a, amp_b = (float(eigenfunctions[n, layer, 0]),
                            float(eigenfunctions[n, layer, 1]),
                            float(eigenfunctions[n, layer, 2]))
        value, slope, curvature = _radial_derivatives(mu, amp_a, amp_b, radius)
        weight = float(coefficients[n]) * np.exp(-float(decay_rates[n]) * time)
        # Dividing by the Laplacian eigenvalue inverts the potential equation.
        potential = weight / (mu + eta ** 2)
        sum_value += potential * value
        sum_slope += potential * slope
        sum_curvature += potential * curvature
        sum_slope_over_r += potential * (curvature if radius == 0.0 else slope / radius)

    # The undifferentiated thermal dilatation, supplied by the temperature step.
    dilatation = modulus * temperature_rise

    two_g = 2.0 * shear_modulus
    return np.array([
        -modulus * sum_slope,
        modulus * eta * sum_value,
        two_g * (-modulus * sum_curvature - dilatation),
        two_g * (-modulus * sum_slope_over_r - dilatation),
        two_g * (modulus * eta ** 2 * sum_value - dilatation),
        two_g * modulus * eta * sum_slope,
    ])

def assemble_love_system(radii: np.ndarray, poisson: np.ndarray,
                                 shear_moduli: np.ndarray, eta: float,
                                 thermal_terms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    radii = np.asarray(radii, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    shear_moduli = np.asarray(shear_moduli, dtype=float).ravel()
    thermal_terms = np.asarray(thermal_terms, dtype=float)

    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    if poisson.size != n_layers or shear_moduli.size != n_layers:
        raise ValueError("radii, poisson and shear_moduli must agree in length")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(poisson)) or np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("every Poisson's ratio must be finite and lie in (-1, 0.5)")
    if not np.all(np.isfinite(shear_moduli)) or np.any(shear_moduli <= 0.0):
        raise ValueError("shear_moduli must be finite and positive")
    if thermal_terms.shape != (n_layers, 2, 6):
        raise ValueError("thermal_terms must have shape (n_layers, 2, 6)")
    if not (isinstance(eta, (int, float, np.floating, np.integer))
            and not isinstance(eta, bool) and np.isfinite(eta) and float(eta) > 0.0):
        raise ValueError("eta must be a finite number > 0")

    eta = float(eta)

    def _shapes(radius, on_axis):
        """Rows of the biharmonic shapes and their derivatives at a radius."""
        x = eta * radius
        bess_i0, bess_i1, bess_k0, bess_k1 = i0(x), i1(x), k0(x), k1(x)
        shape = np.array([bess_i0, bess_k0, x * bess_i1, x * bess_k1])
        slope = eta * np.array([bess_i1, -bess_k1, x * bess_i0, -x * bess_k0])
        curvature = eta ** 2 * np.array([bess_i0 - bess_i1 / x,
                                         bess_k0 + bess_k1 / x,
                                         bess_i0 + x * bess_i1,
                                         -bess_k0 + x * bess_k1])
        laplacian = 2.0 * eta ** 2 * np.array([0.0, 0.0, bess_i0, -bess_k0])
        laplacian_slope = 2.0 * eta ** 3 * np.array([0.0, 0.0, bess_i1, bess_k1])
        rows = (shape, slope, curvature, laplacian, laplacian_slope)
        if on_axis:
            return tuple(row[[0, 2]] for row in rows)
        return rows

    widths = [2] + [4] * (n_layers - 1)
    offsets = np.cumsum([0] + widths)
    ndof = int(offsets[-1])
    system = np.zeros((ndof, ndof + 1))

    row = 0
    for i in range(n_layers - 1):
        radius = radii[i]
        f_a, fp_a, fpp_a, g_a, gp_a = _shapes(radius, i == 0)
        f_b, fp_b, fpp_b, g_b, gp_b = _shapes(radius, False)
        slice_a = slice(offsets[i], offsets[i + 1])
        slice_b = slice(offsets[i + 1], offsets[i + 2])
        term_a = thermal_terms[i, 1]
        term_b = thermal_terms[i + 1, 0]
        two_g_a = 2.0 * shear_moduli[i]
        two_g_b = 2.0 * shear_moduli[i + 1]

        # Continuity of radial displacement.
        system[row, slice_a] = -eta * fp_a
        system[row, slice_b] = eta * fp_b
        system[row, -1] = term_b[0] - term_a[0]
        row += 1
        # Continuity of axial displacement.
        system[row, slice_a] = 2.0 * (1.0 - poisson[i]) * g_a + eta ** 2 * f_a
        system[row, slice_b] = -(2.0 * (1.0 - poisson[i + 1]) * g_b + eta ** 2 * f_b)
        system[row, -1] = term_b[1] - term_a[1]
        row += 1
        # Continuity of radial normal stress.
        system[row, slice_a] = two_g_a * eta * (poisson[i] * g_a - fpp_a)
        system[row, slice_b] = -two_g_b * eta * (poisson[i + 1] * g_b - fpp_b)
        system[row, -1] = term_b[2] - term_a[2]
        row += 1
        # Continuity of shear stress.
        system[row, slice_a] = two_g_a * ((1.0 - poisson[i]) * gp_a + eta ** 2 * fp_a)
        system[row, slice_b] = -two_g_b * ((1.0 - poisson[i + 1]) * gp_b
                                           + eta ** 2 * fp_b)
        system[row, -1] = term_b[5] - term_a[5]
        row += 1

    last = n_layers - 1
    radius = radii[last]
    _f_o, fp_o, fpp_o, g_o, gp_o = _shapes(radius, last == 0)
    slice_o = slice(offsets[last], offsets[last + 1])
    two_g_o = 2.0 * shear_moduli[last]
    term_o = thermal_terms[last, 1]

    # Traction-free outer cylindrical surface.
    system[row, slice_o] = two_g_o * eta * (poisson[last] * g_o - fpp_o)
    system[row, -1] = -term_o[2]
    row += 1
    system[row, slice_o] = two_g_o * ((1.0 - poisson[last]) * gp_o + eta ** 2 * fp_o)
    system[row, -1] = -term_o[5]

    return system

def solve_love_coefficients(system: np.ndarray,
                                    n_layers: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    if not (isinstance(n_layers, (int, np.integer)) and not isinstance(n_layers, bool)
            and int(n_layers) >= 1):
        raise ValueError("n_layers must be an integer >= 1")

    n_layers = int(n_layers)
    system = np.asarray(system, dtype=float)
    ndof = 4 * n_layers - 2
    if system.shape != (ndof, ndof + 1):
        raise ValueError("system must have shape (4*n_layers-2, 4*n_layers-1)")
    if not np.all(np.isfinite(system)):
        raise ValueError("system must be finite")

    matrix = system[:, :ndof]
    rhs = system[:, ndof]
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("the interface system is singular") from error

    # The innermost layer keeps only the two regular shapes.
    widths = [2] + [4] * (n_layers - 1)
    offsets = np.cumsum([0] + widths)
    amplitudes = np.zeros((n_layers, 4))
    for i in range(n_layers):
        block = solution[offsets[i]:offsets[i + 1]]
        if i == 0:
            amplitudes[i, 0] = block[0]
            amplitudes[i, 2] = block[1]
        else:
            amplitudes[i] = block

    return amplitudes

def evaluate_stress_components(love_coefficients: np.ndarray,
                                       thermal_terms: np.ndarray, eta: float,
                                       poisson: float, shear_modulus: float,
                                       radius: float,
                                       axial_position: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    love_coefficients = np.asarray(love_coefficients, dtype=float).ravel()
    thermal_terms = np.asarray(thermal_terms, dtype=float).ravel()
    if love_coefficients.size != 4:
        raise ValueError("love_coefficients must hold four amplitudes")
    if thermal_terms.size != 6:
        raise ValueError("thermal_terms must hold six entries")
    if not np.all(np.isfinite(love_coefficients)) or not np.all(np.isfinite(thermal_terms)):
        raise ValueError("love_coefficients and thermal_terms must be finite")
    if not (isinstance(eta, (int, float, np.floating, np.integer))
            and not isinstance(eta, bool) and np.isfinite(eta) and float(eta) > 0.0):
        raise ValueError("eta must be a finite number > 0")
    if not (isinstance(poisson, (int, float, np.floating, np.integer))
            and not isinstance(poisson, bool) and np.isfinite(poisson)
            and -1.0 < float(poisson) < 0.5):
        raise ValueError("poisson must be a finite number in (-1, 0.5)")
    if not (isinstance(shear_modulus, (int, float, np.floating, np.integer))
            and not isinstance(shear_modulus, bool) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    for name, value in (("radius", radius), ("axial_position", axial_position)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")

    eta = float(eta)
    poisson = float(poisson)
    shear_modulus = float(shear_modulus)
    radius = float(radius)
    axial_position = float(axial_position)

    if radius == 0.0:
        if love_coefficients[1] != 0.0 or love_coefficients[3] != 0.0:
            raise ValueError("the singular amplitudes must vanish on the axis")
        shape = np.array([1.0, 0.0, 0.0, 0.0])
        slope = np.zeros(4)
        curvature = eta ** 2 * np.array([0.5, 0.0, 1.0, 0.0])
        slope_over_r = curvature.copy()
        laplacian = 2.0 * eta ** 2 * np.array([0.0, 0.0, 1.0, 0.0])
        laplacian_slope = np.zeros(4)
    else:
        x = eta * radius
        bess_i0, bess_i1, bess_k0, bess_k1 = i0(x), i1(x), k0(x), k1(x)
        shape = np.array([bess_i0, bess_k0, x * bess_i1, x * bess_k1])
        slope = eta * np.array([bess_i1, -bess_k1, x * bess_i0, -x * bess_k0])
        curvature = eta ** 2 * np.array([bess_i0 - bess_i1 / x,
                                         bess_k0 + bess_k1 / x,
                                         bess_i0 + x * bess_i1,
                                         -bess_k0 + x * bess_k1])
        slope_over_r = slope / radius
        laplacian = 2.0 * eta ** 2 * np.array([0.0, 0.0, bess_i0, -bess_k0])
        laplacian_slope = 2.0 * eta ** 3 * np.array([0.0, 0.0, bess_i1, bess_k1])

    value_f = float(shape @ love_coefficients)
    value_fp = float(slope @ love_coefficients)
    value_fpp = float(curvature @ love_coefficients)
    value_fpr = float(slope_over_r @ love_coefficients)
    value_g = float(laplacian @ love_coefficients)
    value_gp = float(laplacian_slope @ love_coefficients)

    two_g = 2.0 * shear_modulus
    cosine = np.cos(eta * axial_position)
    sine = np.sin(eta * axial_position)

    radial = (two_g * eta * (poisson * value_g - value_fpp) + thermal_terms[2]) * cosine
    hoop = (two_g * eta * (poisson * value_g - value_fpr) + thermal_terms[3]) * cosine
    axial = (two_g * eta * ((2.0 - poisson) * value_g + eta ** 2 * value_f)
             + thermal_terms[4]) * cosine
    shear = (two_g * ((1.0 - poisson) * value_gp + eta ** 2 * value_fp)
             + thermal_terms[5]) * sine

    return np.array([radial, hoop, axial, shear])

def run_tsv_thermal_stress_pipeline(
        radii: tuple = (15.0e-6, 16.0e-6, 25.0e-6),
        conductivities: tuple = (400.0, 1.4, 130.0),
        densities: tuple = (8960.0, 2200.0, 2329.0),
        specific_heats: tuple = (385.0, 730.0, 700.0),
        expansions: tuple = (17.0e-6, 0.5e-6, 2.6e-6),
        youngs_moduli: tuple = (110.0e9, 70.0e9, 170.0e9),
        poisson: tuple = (0.35, 0.17, 0.28),
        height: float = 200.0e-6, initial_rise: float = 100.0,
        plane_fraction: float = 0.9, time: float = 5.0e-5,
        n_axial: int = 12, n_radial: int = 6, n_samples: int = 401) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-09. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the
    #    public function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    axial_modes_of = _resolve_step(
        "compute_axial_modes", "*compute_axial_modes*.py")
    eigenvalues_of = _resolve_step(
        "solve_radial_eigenvalues", "*solve_radial_eigenvalues*.py")
    eigenfunction_of = _resolve_step(
        "build_radial_eigenfunction", "*build_radial_eigenfunction*.py")
    amplitude_of = _resolve_step(
        "compute_expansion_coefficient", "*compute_expansion_coefficient*.py")
    temperature_of = _resolve_step(
        "evaluate_temperature_rise", "*evaluate_temperature_rise*.py")
    thermal_of = _resolve_step(
        "compute_thermal_source_terms", "*compute_thermal_source_terms*.py")
    system_of = _resolve_step(
        "assemble_love_system", "*assemble_love_system*.py")
    solve_of = _resolve_step(
        "solve_love_coefficients", "*solve_love_coefficients*.py")
    stress_of = _resolve_step(
        "evaluate_stress_components", "*evaluate_stress_components*.py")

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_axial", n_axial, 1), ("n_radial", n_radial, 1),
                               ("n_samples", n_samples, 2)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("height", height),):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(time, (int, float, np.floating, np.integer))
            and not isinstance(time, bool) and np.isfinite(time) and float(time) >= 0.0):
        raise ValueError("time must be a finite number >= 0")
    if not (isinstance(initial_rise, (int, float, np.floating, np.integer))
            and not isinstance(initial_rise, bool) and np.isfinite(initial_rise)):
        raise ValueError("initial_rise must be a finite number")
    if not (isinstance(plane_fraction, (int, float, np.floating, np.integer))
            and not isinstance(plane_fraction, bool) and np.isfinite(plane_fraction)
            and 0.0 <= float(plane_fraction) <= 1.0):
        raise ValueError("plane_fraction must be a finite number in [0, 1]")

    radii = np.asarray(radii, dtype=float).ravel()
    conductivities = np.asarray(conductivities, dtype=float).ravel()
    densities = np.asarray(densities, dtype=float).ravel()
    specific_heats = np.asarray(specific_heats, dtype=float).ravel()
    expansions = np.asarray(expansions, dtype=float).ravel()
    youngs_moduli = np.asarray(youngs_moduli, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    for name, array in (("conductivities", conductivities), ("densities", densities),
                        ("specific_heats", specific_heats), ("expansions", expansions),
                        ("youngs_moduli", youngs_moduli), ("poisson", poisson)):
        if array.size != n_layers:
            raise ValueError(f"{name} must have one entry per layer")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be positive and strictly increasing")
    for name, array in (("conductivities", conductivities), ("densities", densities),
                        ("specific_heats", specific_heats),
                        ("youngs_moduli", youngs_moduli)):
        if np.any(array <= 0.0):
            raise ValueError(f"{name} must be positive")
    if np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("every Poisson's ratio must lie in (-1, 0.5)")

    height = float(height)
    time = float(time)
    n_axial = int(n_axial)
    n_radial = int(n_radial)
    n_samples = int(n_samples)
    inner = np.concatenate([[0.0], radii[:-1]])
    plane = float(plane_fraction) * height

    # -- Derived layer properties.
    capacities = densities * specific_heats
    diffusivities = conductivities / capacities
    shear_moduli = youngs_moduli / (2.0 * (1.0 + poisson))

    # -- Sub-problems 01-04: the modal description of the temperature field.
    axial = np.asarray(axial_modes_of(height, n_axial), dtype=float)
    decays = np.zeros((n_axial, n_radial))
    eigen = np.zeros((n_axial, n_radial, n_layers, 3))
    amplitudes = np.zeros((n_axial, n_radial))
    for m in range(n_axial):
        eta = float(axial[m, 0])
        decays[m] = np.asarray(eigenvalues_of(radii, conductivities, diffusivities,
                                              eta, n_radial), dtype=float)
        for n in range(n_radial):
            eigen[m, n] = np.asarray(
                eigenfunction_of(radii, conductivities, diffusivities, eta,
                                 float(decays[m, n])), dtype=float)
            amplitudes[m, n] = float(
                amplitude_of(radii, capacities, eigen[m, n], axial[m], initial_rise))

    # -- Sub-problem 05: the modal temperature at a radius, stripped of its
    #    axial factor, which is what an axial position of zero returns.
    def _temperature(m, layer, radius):
        return float(temperature_of(axial[m:m + 1], decays[m:m + 1],
                                    eigen[m:m + 1], amplitudes[m:m + 1],
                                    layer, float(radius), 0.0, time))

    # -- Sub-problems 05-08: the complementary amplitudes of every axial mode.
    love = np.zeros((n_axial, n_layers, 4))
    for m in range(n_axial):
        eta = float(axial[m, 0])
        terms = np.zeros((n_layers, 2, 6))
        for i in range(n_layers):
            for j, radius in enumerate((inner[i], radii[i])):
                terms[i, j] = np.asarray(
                    thermal_of(decays[m], eigen[m], amplitudes[m], eta,
                               _temperature(m, i, radius),
                               float(expansions[i]), float(poisson[i]),
                               float(shear_moduli[i]), i, float(radius), time),
                    dtype=float)
        system = system_of(radii, poisson, shear_moduli, eta, terms)
        love[m] = np.asarray(solve_of(system, n_layers), dtype=float)

    # -- Sub-problem 09: sweep the cross-section layer by layer.
    peak = 0.0
    for i in range(n_layers):
        for radius in np.linspace(inner[i], radii[i], n_samples):
            total = np.zeros(4)
            for m in range(n_axial):
                eta = float(axial[m, 0])
                terms = np.asarray(
                    thermal_of(decays[m], eigen[m], amplitudes[m], eta,
                               _temperature(m, i, radius),
                               float(expansions[i]), float(poisson[i]),
                               float(shear_moduli[i]), i, float(radius), time),
                    dtype=float)
                total = total + np.asarray(
                    stress_of(love[m, i], terms, eta, float(poisson[i]),
                              float(shear_moduli[i]), float(radius), plane),
                    dtype=float)
            equivalent = np.sqrt(
                0.5 * ((total[0] - total[1]) ** 2 + (total[1] - total[2]) ** 2
                       + (total[2] - total[0]) ** 2) + 3.0 * total[3] ** 2)
            peak = max(peak, float(equivalent))

    return peak / 1.0e6
SCICODE_GOLD_EOF
