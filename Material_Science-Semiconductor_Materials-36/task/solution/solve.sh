#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_axial_eigenmodes(height: float, n_modes: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(height, (int, float)) and np.isfinite(height)
            and float(height) > 0.0):
        raise ValueError("height must be a finite number > 0")
    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")

    height = float(height)
    n_modes = int(n_modes)

    order = np.arange(1, n_modes + 1, dtype=float)
    # Insulated at z = 0 and held at ambient at z = h gives odd quarter waves.
    eigenvalues = (2.0 * order - 1.0) * np.pi / (2.0 * height)
    # The oscillatory part of the norm vanishes because eta_m h is an odd
    # multiple of pi / 2, so every squared norm is exactly h / 2.
    norms = np.full(n_modes, 0.5 * height, dtype=float)

    return np.column_stack([eigenvalues, norms])

def propagate_radial_eigenfunction(radii: np.ndarray,
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

import numpy as np

def compute_radial_eigenvalues(radii: np.ndarray, conductivity: np.ndarray,
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

import numpy as np

def compute_modal_coefficients(radii: np.ndarray, conductivity: np.ndarray,
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

import numpy as np

def compute_temperature_rise(radii: np.ndarray,
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

import numpy as np

def compute_potential_stress_terms(radius: float, axial_eigenvalue: float,
                                           decay_rates: np.ndarray,
                                           modal_coefficients: np.ndarray,
                                           layer_eigen_coefficients: np.ndarray,
                                           expansion: float, poisson: float,
                                           shear: float,
                                           time: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    decay_rates = np.asarray(decay_rates, dtype=float).ravel()
    modal_coefficients = np.asarray(modal_coefficients, dtype=float).ravel()
    layer_eigen_coefficients = np.asarray(layer_eigen_coefficients, dtype=float)
    n_radial = decay_rates.size
    if n_radial < 1 or modal_coefficients.size != n_radial:
        raise ValueError("decay_rates and modal_coefficients must share one length >= 1")
    if layer_eigen_coefficients.shape != (n_radial, 3):
        raise ValueError("layer_eigen_coefficients must have shape (n_radial, 3)")
    if not (np.all(np.isfinite(decay_rates)) and np.all(decay_rates > 0.0)):
        raise ValueError("decay_rates entries must be finite and > 0")
    if not np.all(np.isfinite(modal_coefficients)):
        raise ValueError("modal_coefficients must be finite")
    if not np.all(np.isfinite(layer_eigen_coefficients)):
        raise ValueError("layer_eigen_coefficients must be finite")
    for name, value in (("radius", radius), ("axial_eigenvalue", axial_eigenvalue),
                        ("expansion", expansion), ("shear", shear)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(poisson, (int, float, np.floating, np.integer))
            and np.isfinite(poisson) and 0.0 < float(poisson) < 0.5):
        raise ValueError("poisson must be a finite number in the open interval (0, 0.5)")
    if not (isinstance(time, (int, float, np.floating, np.integer))
            and np.isfinite(time) and float(time) >= 0.0):
        raise ValueError("time must be a finite number >= 0")

    radius = float(radius)
    eta = float(axial_eigenvalue)
    time = float(time)
    shear = float(shear)

    def _basis(q, r):
        """Value and radial derivative of the two solutions of the radial ODE."""
        if q > 0.0:
            c = np.sqrt(q)
            return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
        if q < 0.0:
            c = np.sqrt(-q)
            return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
        return 1.0, np.log(r), 0.0, 1.0 / r

    thermal_factor = float(expansion) * (1.0 + float(poisson)) / (1.0 - float(poisson))

    terms = np.zeros(6, dtype=float)
    for n in range(n_radial):
        amp_a, amp_b, separation = layer_eigen_coefficients[n]
        first, second, d_first, d_second = _basis(separation, radius)
        shape = amp_a * first + amp_b * second
        slope = amp_a * d_first + amp_b * d_second
        # The radial factor solves R'' + R'/r + beta^2 R = 0, so the curvature
        # is available in closed form and needs no numerical differentiation.
        curvature = -slope / radius - separation * shape
        laplace_eigenvalue = separation + eta ** 2
        weight = modal_coefficients[n] * np.exp(-decay_rates[n] * time)
        potential = -thermal_factor * weight / laplace_eigenvalue
        terms[0] += potential * slope
        terms[1] += -potential * eta * shape
        terms[2] += 2.0 * shear * (potential * curvature - thermal_factor * weight * shape)
        terms[3] += 2.0 * shear * (potential * slope / radius
                                   - thermal_factor * weight * shape)
        terms[4] += 2.0 * shear * (-eta ** 2 * potential * shape
                                   - thermal_factor * weight * shape)
        terms[5] += 2.0 * shear * (-eta * potential * slope)

    return terms

import numpy as np

def compute_love_mode_coefficients(radii: np.ndarray, poisson: np.ndarray,
                                           shear: np.ndarray,
                                           axial_eigenvalue: float,
                                           potential_terms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    radii = np.asarray(radii, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    shear = np.asarray(shear, dtype=float).ravel()
    potential_terms = np.asarray(potential_terms, dtype=float)
    n_layers = radii.size
    if n_layers < 2 or poisson.size != n_layers or shear.size != n_layers:
        raise ValueError("radii, poisson and shear must share one length l >= 2")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not (np.all(np.isfinite(poisson)) and np.all(poisson > 0.0) and np.all(poisson < 0.5)):
        raise ValueError("poisson entries must be finite and in the open interval (0, 0.5)")
    if not (np.all(np.isfinite(shear)) and np.all(shear > 0.0)):
        raise ValueError("shear entries must be finite and > 0")
    if potential_terms.shape != (n_layers, 2, 6) or not np.all(np.isfinite(potential_terms)):
        raise ValueError("potential_terms must be a finite array of shape (l, 2, 6)")
    if not (isinstance(axial_eigenvalue, (int, float, np.floating, np.integer))
            and np.isfinite(axial_eigenvalue) and float(axial_eigenvalue) > 0.0):
        raise ValueError("axial_eigenvalue must be a finite number > 0")

    eta = float(axial_eigenvalue)

    def _love_rows(radius, nu, modulus):
        """Rows of amplitudes per unit E, F, P and Q at one radius of one layer."""
        x = eta * radius
        bessel_i0, bessel_i1 = i0(x), i1(x)
        bessel_k0, bessel_k1 = k0(x), k1(x)
        eta2, eta3 = eta ** 2, eta ** 3
        radial = np.array([-eta2 * bessel_i1, eta2 * bessel_k1,
                           -eta2 * x * bessel_i0, eta2 * x * bessel_k0])
        axial = np.array([eta2 * bessel_i0, eta2 * bessel_k0,
                          4.0 * (1.0 - nu) * eta2 * bessel_i0 + eta2 * x * bessel_i1,
                          -4.0 * (1.0 - nu) * eta2 * bessel_k0 + eta2 * x * bessel_k1])
        normal = 2.0 * modulus * np.array([
            -eta3 * (bessel_i0 - bessel_i1 / x),
            -eta3 * (bessel_k0 + bessel_k1 / x),
            2.0 * nu * eta3 * bessel_i0 - eta3 * (bessel_i0 + x * bessel_i1),
            -2.0 * nu * eta3 * bessel_k0 - eta3 * (-bessel_k0 + x * bessel_k1)])
        hoop = 2.0 * modulus * np.array([
            -eta2 / radius * bessel_i1,
            eta2 / radius * bessel_k1,
            2.0 * nu * eta3 * bessel_i0 - eta2 / radius * x * bessel_i0,
            -2.0 * nu * eta3 * bessel_k0 + eta2 / radius * x * bessel_k0])
        axial_normal = 2.0 * modulus * np.array([
            eta3 * bessel_i0, eta3 * bessel_k0,
            2.0 * (2.0 - nu) * eta3 * bessel_i0 + eta3 * x * bessel_i1,
            -2.0 * (2.0 - nu) * eta3 * bessel_k0 + eta3 * x * bessel_k1])
        shear_row = 2.0 * modulus * np.array([
            eta3 * bessel_i1, -eta3 * bessel_k1,
            2.0 * (1.0 - nu) * eta3 * bessel_i1 + eta3 * x * bessel_i0,
            2.0 * (1.0 - nu) * eta3 * bessel_k1 - eta3 * x * bessel_k0])
        rows = np.array([radial, axial, normal, hoop, axial_normal, shear_row])
        return rows

    def _columns(i):
        return (0, 2) if i == 0 else (2 + 4 * (i - 1), 2 + 4 * i)

    def _block(i, radius):
        rows = _love_rows(radius, poisson[i], shear[i])
        return rows[:, [0, 2]] if i == 0 else rows

    size = 4 * n_layers - 2
    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    row = 0
    # Radial displacement, axial displacement, radial stress and shear stress
    # are continuous across every bonded interface. Hoop and axial stress jump.
    for i in range(n_layers - 1):
        radius = radii[i]
        inner_block = _block(i, radius)
        outer_block = _block(i + 1, radius)
        inner_start, inner_stop = _columns(i)
        outer_start, outer_stop = _columns(i + 1)
        for component in (0, 1, 2, 5):
            matrix[row, inner_start:inner_stop] = inner_block[component]
            matrix[row, outer_start:outer_stop] = -outer_block[component]
            rhs[row] = potential_terms[i + 1, 1, component] - potential_terms[i, 0, component]
            row += 1
    # The outer cylindrical surface is traction free.
    radius = radii[-1]
    outer_block = _block(n_layers - 1, radius)
    start, stop = _columns(n_layers - 1)
    for component in (2, 5):
        matrix[row, start:stop] = outer_block[component]
        rhs[row] = -potential_terms[n_layers - 1, 0, component]
        row += 1

    # Displacement rows and traction rows differ by many orders of magnitude,
    # so the system is equilibrated by rows and then by columns before solving.
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(row_scale <= 0.0):
        raise ValueError("interface system has an identically zero row")
    matrix = matrix / row_scale[:, None]
    rhs = rhs / row_scale
    column_scale = np.max(np.abs(matrix), axis=0)
    if np.any(column_scale <= 0.0):
        raise ValueError("interface system has an identically zero column")

    return np.linalg.solve(matrix / column_scale, rhs) / column_scale

import numpy as np 

def assemble_mode_stress(radius: float, radii: np.ndarray,
                                 poisson: np.ndarray, shear: np.ndarray,
                                 axial_eigenvalue: float,
                                 love_coefficients: np.ndarray,
                                 potential_terms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    radii = np.asarray(radii, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    shear = np.asarray(shear, dtype=float).ravel()
    love_coefficients = np.asarray(love_coefficients, dtype=float).ravel()
    potential_terms = np.asarray(potential_terms, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 2 or poisson.size != n_layers or shear.size != n_layers:
        raise ValueError("radii, poisson and shear must share one length l >= 2")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not (np.all(np.isfinite(poisson)) and np.all(poisson > 0.0) and np.all(poisson < 0.5)):
        raise ValueError("poisson entries must be finite and in the open interval (0, 0.5)")
    if not (np.all(np.isfinite(shear)) and np.all(shear > 0.0)):
        raise ValueError("shear entries must be finite and > 0")
    if love_coefficients.size != 4 * n_layers - 2 or not np.all(np.isfinite(love_coefficients)):
        raise ValueError("love_coefficients must be a finite array of length 4 * l - 2")
    if potential_terms.size != 6 or not np.all(np.isfinite(potential_terms)):
        raise ValueError("potential_terms must be a finite array of length 6")
    if not (isinstance(axial_eigenvalue, (int, float, np.floating, np.integer))
            and np.isfinite(axial_eigenvalue) and float(axial_eigenvalue) > 0.0):
        raise ValueError("axial_eigenvalue must be a finite number > 0")
    if not (isinstance(radius, (int, float, np.floating, np.integer))
            and np.isfinite(radius)):
        raise ValueError("radius must be a finite number")
    if float(radius) <= 0.0 or float(radius) > radii[-1]:
        raise ValueError("radius must lie in the half-open interval (0, radii[-1]]")

    radius = float(radius)
    eta = float(axial_eigenvalue)

    layer = int(min(np.searchsorted(radii, radius, side="left"), n_layers - 1))
    nu = poisson[layer]
    modulus = shear[layer]

    x = eta * radius
    bessel_i0, bessel_i1 = i0(x), i1(x)
    bessel_k0, bessel_k1 = k0(x), k1(x)
    eta2, eta3 = eta ** 2, eta ** 3

    normal = 2.0 * modulus * np.array([
        -eta3 * (bessel_i0 - bessel_i1 / x),
        -eta3 * (bessel_k0 + bessel_k1 / x),
        2.0 * nu * eta3 * bessel_i0 - eta3 * (bessel_i0 + x * bessel_i1),
        -2.0 * nu * eta3 * bessel_k0 - eta3 * (-bessel_k0 + x * bessel_k1)])
    hoop = 2.0 * modulus * np.array([
        -eta2 / radius * bessel_i1,
        eta2 / radius * bessel_k1,
        2.0 * nu * eta3 * bessel_i0 - eta2 / radius * x * bessel_i0,
        -2.0 * nu * eta3 * bessel_k0 + eta2 / radius * x * bessel_k0])
    axial = 2.0 * modulus * np.array([
        eta3 * bessel_i0, eta3 * bessel_k0,
        2.0 * (2.0 - nu) * eta3 * bessel_i0 + eta3 * x * bessel_i1,
        -2.0 * (2.0 - nu) * eta3 * bessel_k0 + eta3 * x * bessel_k1])
    shear_row = 2.0 * modulus * np.array([
        eta3 * bessel_i1, -eta3 * bessel_k1,
        2.0 * (1.0 - nu) * eta3 * bessel_i1 + eta3 * x * bessel_i0,
        2.0 * (1.0 - nu) * eta3 * bessel_k1 - eta3 * x * bessel_k0])

    if layer == 0:
        # The innermost layer carries only the coefficients bounded on the axis.
        active = np.array([love_coefficients[0], 0.0, love_coefficients[1], 0.0])
    else:
        start = 2 + 4 * (layer - 1)
        active = love_coefficients[start:start + 4]

    return np.array([
        float(normal @ active + potential_terms[2]),
        float(hoop @ active + potential_terms[3]),
        float(axial @ active + potential_terms[4]),
        float(shear_row @ active + potential_terms[5])])

def run_mobility_shift_pipeline(radius: float = 20.0e-6,
                                        depth: float = 200.0e-6 / 3.0,
                                        time: float = 5.0e-5,
                                        n_axial: int = 8, n_radial: int = 8,
                                        radii=(15.0e-6, 16.0e-6, 30.0e-6),
                                        height: float = 200.0e-6,
                                        conductivity=(400.0, 1.4, 130.0),
                                        density=(8960.0, 2200.0, 2329.0),
                                        heat_capacity=(385.0, 730.0, 700.0),
                                        expansion=(17.0e-6, 0.5e-6, 2.6e-6),
                                        young=(110.0e9, 70.0e9, 170.0e9),
                                        poisson=(0.35, 0.17, 0.28),
                                        initial_rise: float = 100.0,
                                        piezo_coefficient: float = 71.8e-11,
                                        orientation_factor: float = 1.0) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- The oracles of sub-problems 01-08 share this namespace.
    axial_modes = compute_axial_eigenmodes
    propagate = propagate_radial_eigenfunction
    radial_modes = compute_radial_eigenvalues
    modal_coefficients = compute_modal_coefficients
    temperature_rise = compute_temperature_rise
    potential_terms = compute_potential_stress_terms
    love_coefficients = compute_love_mode_coefficients
    mode_stress = assemble_mode_stress

    # -- Validate the orchestrator inputs.
    radii = np.asarray(radii, dtype=float).ravel()
    conductivity = np.asarray(conductivity, dtype=float).ravel()
    density = np.asarray(density, dtype=float).ravel()
    heat_capacity = np.asarray(heat_capacity, dtype=float).ravel()
    expansion = np.asarray(expansion, dtype=float).ravel()
    young = np.asarray(young, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 2:
        raise ValueError("the via must have at least two layers")
    for name, table in (("conductivity", conductivity), ("density", density),
                        ("heat_capacity", heat_capacity), ("expansion", expansion),
                        ("young", young), ("poisson", poisson)):
        if table.size != n_layers:
            raise ValueError(f"{name} must have one entry per layer")
        if not np.all(np.isfinite(table)) or np.any(table <= 0.0):
            raise ValueError(f"{name} entries must be finite and > 0")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if np.any(poisson >= 0.5):
        raise ValueError("poisson entries must be below 0.5")
    for name, value, floor in (("n_axial", n_axial, 1), ("n_radial", n_radial, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("height", height), ("initial_rise", initial_rise),
                        ("piezo_coefficient", piezo_coefficient)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("radius", radius), ("depth", depth), ("time", time),
                        ("orientation_factor", orientation_factor)):
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
    n_axial = int(n_axial)
    n_radial = int(n_radial)

    diffusivity = conductivity / (density * heat_capacity)
    shear = young / (2.0 * (1.0 + poisson))
    inner_radii = np.concatenate(([0.0], radii[:-1]))
    layer_of_query = int(min(np.searchsorted(radii, radius, side="left"), n_layers - 1))

    # -- Sub-problem 01: axial eigenvalues and their squared norms.
    axial = np.asarray(axial_modes(float(height), n_axial), dtype=float)

    # The temperature evaluation of sub-problem 05 takes the whole basis at
    # once, so the per-mode tables are collected while the modes are built.
    decay_table = np.zeros((n_axial, n_radial), dtype=float)
    coefficient_table = np.zeros((n_axial, n_radial), dtype=float)
    eigen_table = np.zeros((n_axial, n_radial, n_layers, 3), dtype=float)
    potential_trace = 0.0

    stress = np.zeros(4, dtype=float)
    for m in range(n_axial):
        eta = float(axial[m, 0])
        axial_norm = float(axial[m, 1])

        # -- Sub-problem 03: thermal decay rates of this axial mode.
        rates = np.asarray(radial_modes(radii, conductivity, diffusivity, eta,
                                        n_radial), dtype=float)
        # -- Sub-problem 02: layer amplitudes of each radial eigenfunction.
        eigen = np.array([np.asarray(propagate(radii, conductivity, diffusivity,
                                               eta, float(rate)), dtype=float)
                          for rate in rates])
        # -- Sub-problem 04: projection of the uniform initial rise.
        coefficients = np.asarray(modal_coefficients(radii, conductivity, diffusivity,
                                                     float(height), eta, axial_norm,
                                                     eigen, float(initial_rise)),
                                  dtype=float)
        decay_table[m] = rates
        coefficient_table[m] = coefficients
        eigen_table[m] = eigen

        # -- Sub-problem 06: potential amplitudes at both faces of every layer.
        boundary_terms = np.zeros((n_layers, 2, 6), dtype=float)
        for i in range(n_layers):
            boundary_terms[i, 0] = potential_terms(
                float(radii[i]), eta, rates, coefficients, eigen[:, i, :],
                float(expansion[i]), float(poisson[i]), float(shear[i]), time)
            if i > 0:
                boundary_terms[i, 1] = potential_terms(
                    float(inner_radii[i]), eta, rates, coefficients, eigen[:, i, :],
                    float(expansion[i]), float(poisson[i]), float(shear[i]), time)

        # -- Sub-problem 07: Love coefficients enforcing the interfaces and surface.
        love = np.asarray(love_coefficients(radii, poisson, shear, eta,
                                            boundary_terms), dtype=float)

        # -- Sub-problem 06 again, now at the query radius, then sub-problem 08.
        query_terms = potential_terms(
            radius, eta, rates, coefficients, eigen[:, layer_of_query, :],
            float(expansion[layer_of_query]), float(poisson[layer_of_query]),
            float(shear[layer_of_query]), time)
        amplitudes = np.asarray(mode_stress(radius, radii, poisson, shear, eta,
                                            love, query_terms), dtype=float)
        potential_trace += float(query_terms[2] + query_terms[3]
                                 + query_terms[4]) * np.cos(eta * depth)

        # The three normal components ride on cos(eta z), the shear on sin(eta z).
        stress += amplitudes * np.array([np.cos(eta * depth), np.cos(eta * depth),
                                         np.cos(eta * depth), np.sin(eta * depth)])

    # -- Sub-problem 05: the temperature the stress state is read against. The
    #    potential is built so that its Laplacian returns the temperature, so the
    #    trace of the potential part of the stress at the query point is exactly
    #    -4 G alpha (1 + nu) / (1 - nu) theta there. Checking the two against one
    #    another ties the assembled mechanics to the temperature series.
    rise = float(temperature_rise(radii, axial[:, 0], decay_table,
                                  coefficient_table, eigen_table, radius, depth,
                                  time))
    expected_trace = (-4.0 * float(shear[layer_of_query])
                      * float(expansion[layer_of_query])
                      * (1.0 + float(poisson[layer_of_query]))
                      / (1.0 - float(poisson[layer_of_query])) * rise)
    scale = max(abs(potential_trace), abs(expected_trace))
    if abs(potential_trace - expected_trace) > 1.0e-6 * scale:
        raise ValueError("the temperature series and the thermoelastic potential "
                         "disagree at the query point")

    # -- Piezoresistive conversion: dmu/mu = -drho/rho, drho/rho = Pi * beta * sigma_rr.
    return float(-100.0 * float(piezo_coefficient) * float(orientation_factor) * stress[0])
SCICODE_GOLD_EOF
