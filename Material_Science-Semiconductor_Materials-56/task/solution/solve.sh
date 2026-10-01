#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_spectral_model(n_shells: int, omega_max: float,
                                 number_density: float,
                                 temperature: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    hbar = 1.054571817e-34
    k_boltzmann = 1.380649e-23

    if not (isinstance(n_shells, (int, np.integer)) and not isinstance(n_shells, bool)
            and int(n_shells) >= 1):
        raise ValueError("n_shells must be an integer >= 1")
    for name, value in (("omega_max", omega_max),
                        ("number_density", number_density),
                        ("temperature", temperature)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    n_shells = int(n_shells)
    omega_max = float(omega_max)
    number_density = float(number_density)
    temperature = float(temperature)

    # Debye sphere holding exactly three modes per atom.
    k_debye = (6.0 * np.pi ** 2 * number_density) ** (1.0 / 3.0)
    dk = k_debye / n_shells
    wavevector = (np.arange(n_shells, dtype=float) + 0.5) * dk

    # Born-von Karman dispersion and its analytic derivative.
    phase = 0.5 * np.pi * wavevector / k_debye
    omega = omega_max * np.sin(phase)
    velocity = omega_max * (0.5 * np.pi / k_debye) * np.cos(phase)

    # Mode density of the isotropic sphere times the Einstein heat capacity.
    x = hbar * omega / (k_boltzmann * temperature)
    ex = np.exp(x)
    mode_density = 3.0 * wavevector ** 2 * dk / (2.0 * np.pi ** 2)
    capacity = mode_density * k_boltzmann * x ** 2 * ex / (ex - 1.0) ** 2

    return np.column_stack([omega, velocity, capacity])

def calibrate_umklapp_coefficient(spectral: np.ndarray, impurity: float,
                                          kappa_bulk: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    table = np.asarray(spectral, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("spectral must be a 2D array of shape (n_shells, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("spectral must be finite")
    if np.any(table[:, 0] <= 0.0) or np.any(table[:, 2] < 0.0):
        raise ValueError("spectral frequencies must be > 0 and heat capacities >= 0")
    for name, value, floor in (("impurity", impurity, 0.0),
                               ("kappa_bulk", kappa_bulk, None)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(impurity) < 0.0:
        raise ValueError("impurity must be a finite number >= 0")
    if float(kappa_bulk) <= 0.0:
        raise ValueError("kappa_bulk must be a finite number > 0")

    omega, velocity, capacity = table[:, 0], table[:, 1], table[:, 2]
    impurity = float(impurity)
    kappa_bulk = float(kappa_bulk)

    def _residual(exponent):
        # Kinetic-theory conductivity of the sampled spectrum, minus the target.
        lifetime = 1.0 / (impurity * omega ** 4 + (10.0 ** exponent) * omega ** 2)
        return float(np.sum(capacity * velocity ** 2 * lifetime) / 3.0) - kappa_bulk

    low, high = -40.0, 20.0
    if _residual(low) < 0.0 or _residual(high) > 0.0:
        raise ValueError("no admissible umklapp coefficient in the search bracket")

    # The conductivity decreases monotonically with the coefficient, so plain
    # bisection on the exponent is unconditionally convergent.
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _residual(mid) > 0.0:
            low = mid
        else:
            high = mid

    return float(10.0 ** (0.5 * (low + high)))

def discretize_phonon_bands(spectral: np.ndarray, impurity: float,
                                    umklapp: float, n_bands: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    table = np.asarray(spectral, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("spectral must be a 2D array of shape (n_shells, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("spectral must be finite")
    if np.any(table[:, 0] <= 0.0):
        raise ValueError("spectral frequencies must be > 0")
    if not (isinstance(impurity, (int, float, np.floating, np.integer))
            and not isinstance(impurity, bool)
            and np.isfinite(impurity) and float(impurity) >= 0.0):
        raise ValueError("impurity must be a finite number >= 0")
    if not (isinstance(umklapp, (int, float, np.floating, np.integer))
            and not isinstance(umklapp, bool)
            and np.isfinite(umklapp) and float(umklapp) > 0.0):
        raise ValueError("umklapp must be a finite number > 0")
    if not (isinstance(n_bands, (int, np.integer)) and not isinstance(n_bands, bool)
            and 1 <= int(n_bands) <= table.shape[0]):
        raise ValueError("n_bands must be an integer in [1, n_shells]")

    omega, velocity, capacity = table[:, 0], table[:, 1], table[:, 2]
    n_bands = int(n_bands)
    lifetime = 1.0 / (float(impurity) * omega ** 4 + float(umklapp) * omega ** 2)
    mean_free_path = velocity * lifetime
    conductivity = capacity * velocity ** 2 * lifetime / 3.0

    if not np.all(conductivity > 0.0):
        raise ValueError("every shell must carry a strictly positive conductivity")

    # Cut the mean-free-path-ordered spectrum at equally spaced levels of the
    # accumulated conductivity; a shell belongs to the band that contains the
    # accumulated fraction reached once that shell is included.
    order = np.argsort(mean_free_path, kind="stable")
    accumulated = np.cumsum(conductivity[order])
    label = np.minimum((n_bands * accumulated / accumulated[-1]).astype(int),
                       n_bands - 1)

    bands = np.zeros((n_bands, 3), dtype=float)
    for index in range(n_bands):
        members = order[label == index]
        if members.size == 0:
            raise ValueError("band discretisation produced an empty band")
        band_capacity = float(capacity[members].sum())
        band_conductivity = float(conductivity[members].sum())
        if band_capacity <= 0.0:
            raise ValueError("band discretisation produced a band of zero heat capacity")
        band_velocity = float(np.sum(capacity[members] * velocity[members])
                              / band_capacity)
        band_lifetime = 3.0 * band_conductivity / (band_capacity * band_velocity ** 2)
        bands[index] = (band_capacity, band_velocity, band_lifetime)

    return bands

def build_angular_quadrature(n_dirs: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    if not (isinstance(n_dirs, (int, np.integer)) and not isinstance(n_dirs, bool)
            and int(n_dirs) >= 2 and int(n_dirs) % 2 == 0):
        raise ValueError("n_dirs must be an even integer >= 2")

    cosines, weights = np.polynomial.legendre.leggauss(int(n_dirs))

    # The azimuthal integral is exact and contributes 2*pi, so the weights
    # sum to the 4*pi steradians of the full sphere.
    return np.column_stack([cosines, 2.0 * np.pi * weights])

def assemble_transport_operator(bands: np.ndarray,
                                        quadrature: np.ndarray,
                                        n_cells: int,
                                        cell_size: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))):
        raise ValueError("bands and quadrature must be finite")
    if np.any(band_table[:, 1] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band velocities and relaxation times must be > 0")
    if np.any(np.abs(quad[:, 0]) > 1.0):
        raise ValueError("direction cosines must lie in [-1, 1]")
    if not (isinstance(n_cells, (int, np.integer)) and not isinstance(n_cells, bool)
            and int(n_cells) >= 1):
        raise ValueError("n_cells must be an integer >= 1")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

    n_cells = int(n_cells)
    operators = np.zeros(
        (band_table.shape[0], quad.shape[0], n_cells, n_cells), dtype=float)
    index = np.arange(n_cells)
    for b in range(band_table.shape[0]):
        velocity = band_table[b, 1]
        relaxation_time = band_table[b, 2]
        for i in range(quad.shape[0]):
            advection = velocity * quad[i, 0] / float(cell_size)
            operator = operators[b, i]
            operator[index, index] = 1.0 / relaxation_time + abs(advection)

            # Upwinded face flux: the inflow neighbour is on the left for
            # positive velocity components and on the right otherwise. The
            # incoming deviational energy at a thermalising wall is zero.
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection

    return operators

def distribute_mode_source(bands: np.ndarray, heat_source: np.ndarray,
                                   solid_angle: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    source = np.asarray(heat_source, dtype=float)
    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if source.ndim != 1 or source.size < 1:
        raise ValueError("heat_source must be a 1D array of length >= 1")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(source))):
        raise ValueError("bands and heat_source must be finite")
    if np.any(band_table[:, 0] <= 0.0):
        raise ValueError("band heat capacities must be > 0")
    if not (isinstance(solid_angle, (int, float, np.floating, np.integer))
            and not isinstance(solid_angle, bool)
            and np.isfinite(solid_angle) and float(solid_angle) > 0.0):
        raise ValueError("solid_angle must be a finite number > 0")

    capacity = band_table[:, 0]

    # Equilibrium source: shared among bands by heat capacity, spread
    # isotropically over the ordinates by the total angular weight.
    share = capacity / capacity.sum() / float(solid_angle)

    return np.outer(share, source)

def solve_transport_sweep(bands: np.ndarray, quadrature: np.ndarray,
                                  transport_operators: np.ndarray,
                                  mode_source: np.ndarray,
                                  lattice_temperature: np.ndarray) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    operators = np.asarray(transport_operators, dtype=float)
    source = np.asarray(mode_source, dtype=float)
    temperature = np.asarray(lattice_temperature, dtype=float)

    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if temperature.ndim != 1 or temperature.size < 1:
        raise ValueError("lattice_temperature must be a 1D array of length >= 1")
    if source.shape != (band_table.shape[0], temperature.size):
        raise ValueError("mode_source must have shape (n_bands, n_cells)")
    expected_operator_shape = (band_table.shape[0], quad.shape[0],
                               temperature.size, temperature.size)
    if operators.shape != expected_operator_shape:
        raise ValueError(
            "transport_operators must have shape "
            "(n_bands, n_dirs, n_cells, n_cells)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))
            and np.all(np.isfinite(operators)) and np.all(np.isfinite(source))
            and np.all(np.isfinite(temperature))):
        raise ValueError("all inputs must be finite")
    if np.any(band_table[:, 1] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band velocities and relaxation times must be > 0")
    if np.any(np.abs(quad[:, 0]) > 1.0):
        raise ValueError("direction cosines must lie in [-1, 1]")
    n_bands = band_table.shape[0]
    n_dirs = quad.shape[0]
    n_cells = temperature.size
    solid_angle = float(quad[:, 1].sum())
    if not solid_angle > 0.0:
        raise ValueError("the quadrature weights must sum to a positive solid angle")

    diagonal = np.diagonal(operators, axis1=2, axis2=3)
    if np.any(diagonal <= 0.0):
        raise ValueError("transport operator diagonals must be > 0")

    solution = np.empty((n_bands, n_dirs, n_cells), dtype=float)
    for b in range(n_bands):
        capacity, velocity, relaxation_time = band_table[b]
        # Relaxation towards the local equilibrium plus the isotropic source.
        rhs = capacity * temperature / solid_angle / relaxation_time + source[b]
        for i in range(n_dirs):
            matrix = operators[b, i]
            values = np.empty(n_cells, dtype=float)
            if quad[i, 0] > 0.0:
                values[0] = rhs[0] / matrix[0, 0]
                for cell in range(1, n_cells):
                    values[cell] = (rhs[cell]
                                    - matrix[cell, cell - 1] * values[cell - 1]) \
                                   / matrix[cell, cell]
            elif quad[i, 0] < 0.0:
                values[-1] = rhs[-1] / matrix[-1, -1]
                for cell in range(n_cells - 2, -1, -1):
                    values[cell] = (rhs[cell]
                                    - matrix[cell, cell + 1] * values[cell + 1]) \
                                   / matrix[cell, cell]
            else:
                values[:] = rhs / diagonal[b, i]
            solution[b, i] = values

    return solution

def compute_macroscopic_fields(bands: np.ndarray, quadrature: np.ndarray,
                                       energy: np.ndarray,
                                       cell_size: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    field = np.asarray(energy, dtype=float)

    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if field.ndim != 3 or field.shape[:2] != (band_table.shape[0], quad.shape[0]) \
            or field.shape[2] < 1:
        raise ValueError("energy must have shape (n_bands, n_dirs, n_cells)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))
            and np.all(np.isfinite(field))):
        raise ValueError("all inputs must be finite")
    if np.any(band_table[:, 0] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band heat capacities and relaxation times must be > 0")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

    capacity, velocity, relaxation = band_table[:, 0], band_table[:, 1], band_table[:, 2]
    cosine, weight = quad[:, 0], quad[:, 1]
    n_cells = field.shape[2]

    # Collision-moment closure for the lattice temperature.
    angular = np.einsum("i,bic->bc", weight, field)
    denominator = float(np.sum(capacity / relaxation))
    if denominator <= 0.0:
        raise ValueError("the collision closure has a non-positive denominator")
    lattice = np.einsum("b,bc->c", 1.0 / relaxation, angular) / denominator

    # First angular moment: the heat flux along the transport coordinate.
    flux = np.einsum("i,b,bic->c", weight * cosine, velocity, field)

    # Divergence from the same upwinded face fluxes the transport uses, with
    # zero deviational energy entering through a thermalising wall.
    coefficient = np.outer(velocity, cosine) * weight[None, :]
    face = np.zeros(n_cells + 1, dtype=float)
    face[1:] += np.einsum("bi,bic->c", np.where(coefficient > 0.0, coefficient, 0.0),
                          field)
    face[:-1] += np.einsum("bi,bic->c", np.where(coefficient < 0.0, coefficient, 0.0),
                           field)
    divergence = (face[1:] - face[:-1]) / float(cell_size)

    return np.vstack([lattice, flux, divergence])

def update_macroscopic_temperature(kappa_bulk: float,
                                           heat_source: np.ndarray,
                                           flux_divergence: np.ndarray,
                                           previous_temperature: np.ndarray,
                                           cell_size: float) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _laplacian(n_cells, dx):
        """Cell-centred Laplacian with the walls held at zero rise."""
        matrix = np.zeros((n_cells, n_cells), dtype=float)
        inverse = 1.0 / dx ** 2
        for cell in range(n_cells):
            # Left face: a wall face has a half-cell gradient arm.
            matrix[cell, cell] -= 2.0 * inverse if cell == 0 else inverse
            if cell > 0:
                matrix[cell, cell - 1] += inverse
            # Right face.
            matrix[cell, cell] -= 2.0 * inverse if cell == n_cells - 1 else inverse
            if cell < n_cells - 1:
                matrix[cell, cell + 1] += inverse
        return matrix

    source = np.asarray(heat_source, dtype=float)
    divergence = np.asarray(flux_divergence, dtype=float)
    previous = np.asarray(previous_temperature, dtype=float)

    if source.ndim != 1 or source.size < 1:
        raise ValueError("heat_source must be a 1D array of length >= 1")
    if divergence.shape != source.shape:
        raise ValueError("flux_divergence must have the same shape as heat_source")
    if previous.shape != source.shape:
        raise ValueError("previous_temperature must have the same shape as heat_source")
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(divergence))
            and np.all(np.isfinite(previous))):
        raise ValueError("all field inputs must be finite")
    if not (isinstance(kappa_bulk, (int, float, np.floating, np.integer))
            and not isinstance(kappa_bulk, bool)
            and np.isfinite(kappa_bulk) and float(kappa_bulk) > 0.0):
        raise ValueError("kappa_bulk must be a finite number > 0")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

    kappa_bulk = float(kappa_bulk)
    laplacian = _laplacian(source.size, float(cell_size))

    # Diffusion driven by the dissipation minus the divergence of the
    # non-Fourier flux, the latter evaluated at the previous iterate.
    right_hand = source - divergence - kappa_bulk * (laplacian @ previous)

    return np.linalg.solve(-kappa_bulk * laplacian, right_hand)

def run_self_heating_pipeline(length: float = 2.0e-8, n_cells: int = 200,
                                      n_bands: int = 12, n_dirs: int = 16,
                                      n_shells: int = 400,
                                      omega_max: float = 56548667764616.27,
                                      number_density: float = 5.0e28,
                                      temperature: float = 300.0,
                                      impurity: float = 1.32e-45,
                                      kappa_bulk: float = 148.0,
                                      power_density: float = 1.5e19,
                                      hotspot_fraction: float = 0.2,
                                      tol: float = 1.0e-12,
                                      max_iter: int = 5000,
                                      scheme: str = "sequential") -> float:
    import numpy as np

    # The grading harness concatenates the sub-problems into one namespace.
    # Bind the preceding oracles directly so the final reference pipeline
    # cannot fall back to candidate code or depend on the filesystem.
    build_spectrum = build_spectral_model
    calibrate = calibrate_umklapp_coefficient
    discretize = discretize_phonon_bands
    build_quadrature = build_angular_quadrature
    assemble = assemble_transport_operator
    split_source = distribute_mode_source
    sweep = solve_transport_sweep
    moments = compute_macroscopic_fields
    macroscopic = update_macroscopic_temperature

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_cells", n_cells, 1), ("n_bands", n_bands, 1),
                               ("n_dirs", n_dirs, 2), ("n_shells", n_shells, 1),
                               ("max_iter", max_iter, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    if int(n_dirs) % 2 != 0:
        raise ValueError("n_dirs must be an even integer >= 2")
    if int(n_bands) > int(n_shells):
        raise ValueError("n_bands must not exceed n_shells")
    for name, value in (("length", length), ("omega_max", omega_max),
                        ("number_density", number_density),
                        ("temperature", temperature), ("kappa_bulk", kappa_bulk),
                        ("power_density", power_density), ("tol", tol)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(impurity, (int, float, np.floating, np.integer))
            and not isinstance(impurity, bool)
            and np.isfinite(impurity) and float(impurity) >= 0.0):
        raise ValueError("impurity must be a finite number >= 0")
    if not (isinstance(hotspot_fraction, (int, float, np.floating, np.integer))
            and not isinstance(hotspot_fraction, bool)
            and np.isfinite(hotspot_fraction)
            and 0.0 < float(hotspot_fraction) <= 1.0):
        raise ValueError("hotspot_fraction must be a finite number in (0, 1]")
    if scheme not in ("sequential", "synthetic"):
        raise ValueError("scheme must be either 'sequential' or 'synthetic'")

    n_cells = int(n_cells)
    length = float(length)

    # -- Sub-problems 01-04: material model, band reduction and ordinate set.
    spectral = build_spectrum(int(n_shells), float(omega_max),
                              float(number_density), float(temperature))
    umklapp = calibrate(spectral, float(impurity), float(kappa_bulk))
    bands = discretize(spectral, float(impurity), umklapp, int(n_bands))
    quadrature = build_quadrature(int(n_dirs))
    solid_angle = float(np.asarray(quadrature, dtype=float)[:, 1].sum())

    # -- Geometry: a centred hot spot on a uniform mesh.
    cell_size = length / n_cells
    centres = (np.arange(n_cells) + 0.5) * cell_size
    inside = (np.abs(centres - 0.5 * length)
              <= 0.5 * float(hotspot_fraction) * length + 1.0e-12 * length)
    heat_source = np.where(inside, float(power_density), 0.0)

    # -- Sub-problem 05: assemble each temperature-independent matrix once.
    #    Step 07 reuses this bank for every right-hand-side substitution.
    operators = assemble(bands, quadrature, n_cells, cell_size)

    # -- Sub-problem 06: share the dissipation among bands and directions.
    mode_source = split_source(bands, heat_source, solid_angle)

    # -- Sub-problem 09 with no kinetic input: the Fourier reference.
    zeros = np.zeros(n_cells)
    fourier = macroscopic(float(kappa_bulk), heat_source, zeros, zeros, cell_size)
    peak_fourier = float(np.max(fourier))
    if not peak_fourier > 0.0:
        raise ValueError("the Fourier reference has no positive temperature rise")

    # -- Sub-problems 05, 07, 08: outer iteration on the kinetic solution.
    field = fourier.copy()
    for _ in range(int(max_iter)):
        energy = sweep(bands, quadrature, operators, mode_source, field)
        fields = moments(bands, quadrature, energy, cell_size)
        if scheme == "sequential":
            updated = fields[0]
        else:
            updated = macroscopic(float(kappa_bulk), heat_source, fields[2],
                                  field, cell_size)
        scale = float(np.max(np.abs(updated)))
        residual = float(np.max(np.abs(updated - field))) / max(scale, 1.0e-300)
        field = updated
        if residual <= float(tol):
            break

    return float(np.max(field) / peak_fourier)
SCICODE_GOLD_EOF
