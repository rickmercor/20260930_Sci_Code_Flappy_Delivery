#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_emission_rate(temperature: np.ndarray, activation_energy: float,
                                  sigma_inf: float, mass_ratio: float = 0.063) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    for name, value in (("activation_energy", activation_energy),
                        ("sigma_inf", sigma_inf), ("mass_ratio", mass_ratio)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value) or float(value) <= 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    temperature = np.asarray(temperature, dtype=float)
    if temperature.size == 0 or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be non-empty and finite")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature must be strictly positive")

    mass = float(mass_ratio) * m_e

    # Root-mean-square thermal velocity of the free carrier, m/s to cm/s.
    v_thermal = np.sqrt(3.0 * kb_j * temperature / mass) * 100.0

    # Effective density of states of the receiving band, converted to 1/cm^3.
    n_states = 2.0 * (2.0 * np.pi * mass * kb_j * temperature
                      / h_planck ** 2) ** 1.5 * 1e-6

    boltzmann = np.exp(-float(activation_energy) / (kb_ev * temperature))
    return float(sigma_inf) * v_thermal * n_states * boltzmann

def generate_capacitance_transients(temperature: np.ndarray, activation_energy: np.ndarray,
                                            sigma_inf: np.ndarray, deflection: np.ndarray,
                                            base_capacitance: float, sampling_rate: float,
                                            n_samples: int, noise_ref: float,
                                            temperature_ref: float, seed: int,
                                            mass_ratio: float = 0.063) -> np.ndarray:
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    temperature = np.asarray(temperature, dtype=float)
    activation_energy = np.atleast_1d(np.asarray(activation_energy, dtype=float))
    sigma_inf = np.atleast_1d(np.asarray(sigma_inf, dtype=float))
    deflection = np.atleast_1d(np.asarray(deflection, dtype=float))

    if not (isinstance(n_samples, (int, np.integer)) and not isinstance(n_samples, bool)
            and int(n_samples) >= 1):
        raise ValueError("n_samples must be an integer >= 1")
    if not np.isfinite(sampling_rate) or float(sampling_rate) <= 0.0:
        raise ValueError("sampling_rate must be a finite number > 0")
    if float(noise_ref) < 0.0:
        raise ValueError("noise_ref must be non-negative")
    if activation_energy.size != sigma_inf.size or activation_energy.size != deflection.size:
        raise ValueError("level parameter arrays must have matching lengths")
    if temperature.size == 0 or np.any(temperature <= 0.0):
        raise ValueError("temperature must be non-empty and strictly positive")

    mass = float(mass_ratio) * m_e
    n_samples = int(n_samples)
    times = np.arange(1, n_samples + 1) / float(sampling_rate)
    rng = np.random.default_rng(seed)

    transients = np.empty((temperature.size, n_samples), dtype=float)
    for i, temp in enumerate(temperature):
        v_thermal = np.sqrt(3.0 * kb_j * temp / mass) * 100.0
        n_states = 2.0 * (2.0 * np.pi * mass * kb_j * temp / h_planck ** 2) ** 1.5 * 1e-6
        signal = np.full(n_samples, float(base_capacitance))
        for k in range(activation_energy.size):
            rate = (sigma_inf[k] * v_thermal * n_states
                    * np.exp(-activation_energy[k] / (kb_ev * temp)))
            signal = signal + deflection[k] * np.exp(-rate * times)
        scale = float(noise_ref) * np.sqrt(temp / float(temperature_ref))
        transients[i] = signal + rng.standard_normal(n_samples) * scale

    return transients

def bin_transient_logarithmic(times: np.ndarray, capacitance: np.ndarray,
                                      n_bins: int) -> np.ndarray:
    import numpy as np

    times = np.asarray(times, dtype=float)
    capacitance = np.asarray(capacitance, dtype=float)

    if not (isinstance(n_bins, (int, np.integer)) and not isinstance(n_bins, bool)
            and int(n_bins) >= 1):
        raise ValueError("n_bins must be an integer >= 1")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array of at least 2 points")
    if times.shape != capacitance.shape:
        raise ValueError("times and capacitance must have the same shape")
    if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly positive and strictly increasing")

    n_bins = int(n_bins)

    # Interval edges are geometric between the first and the last sample, so
    # every decade of the record receives the same number of intervals.
    edges = np.geomspace(times[0], times[-1], n_bins + 1)

    # digitize against the interior edges assigns the first and last samples to
    # the first and last interval without any out-of-range index.
    index = np.clip(np.digitize(times, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for j in range(n_bins):
        mask = index == j
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((float(times[mask].mean()),
                     float(capacitance[mask].mean()),
                     float(count)))

    return np.array(rows, dtype=float)

def build_inversion_system(binned: np.ndarray, emission_grid: np.ndarray) -> np.ndarray:
    import numpy as np

    binned = np.asarray(binned, dtype=float)
    emission_grid = np.asarray(emission_grid, dtype=float)

    if (binned.ndim not in (2, 3) or binned.shape[-1] != 3
            or binned.shape[-2] < 1 or (binned.ndim == 3 and binned.shape[0] < 1)):
        raise ValueError(
            "binned must have shape (n_kept, 3) or (n_records, n_kept, 3)")
    if emission_grid.ndim != 1 or emission_grid.size < 1:
        raise ValueError("emission_grid must be a non-empty one-dimensional array")
    if np.any(emission_grid <= 0.0) or not np.all(np.isfinite(emission_grid)):
        raise ValueError("emission_grid entries must be finite and > 0")
    if np.any(binned[..., 2] <= 0.0):
        raise ValueError("bin sample counts must be strictly positive")

    times = binned[..., 0]
    values = binned[..., 1]
    counts = binned[..., 2]

    # The last binned value is the only baseline estimate that does not
    # presuppose the decomposition; negating makes the residual positive for
    # majority carrier capture.
    residual = -(values - values[..., -1, None])

    # Exact benchmark row multiplier.  This retains the relative precision of
    # the bin means but deliberately excludes the common per-transient noise
    # scale, whose inclusion would rescale the residual against the fixed
    # Tikhonov penalty used by the next step.
    weights = np.sqrt(counts)

    kernel = np.exp(-times[..., None] * emission_grid)

    system = np.empty(times.shape + (emission_grid.size + 1,), dtype=float)
    system[..., :-1] = kernel * weights[..., None]
    system[..., -1] = residual * weights

    return system

def solve_regularized_spectrum(system: np.ndarray, regularization: float) -> np.ndarray:
    import numpy as np
    from scipy.optimize import nnls

    system = np.asarray(system, dtype=float)

    if (system.ndim not in (2, 3) or system.shape[-1] < 2
            or system.shape[-2] < 1 or (system.ndim == 3 and system.shape[0] < 1)):
        raise ValueError(
            "system must have shape (n_kept, n_rates + 1) or "
            "(n_records, n_kept, n_rates + 1)")
    if not np.all(np.isfinite(system)):
        raise ValueError("system must be finite")
    if (isinstance(regularization, bool)
            or not isinstance(regularization, (int, float, np.floating, np.integer))
            or not np.isfinite(regularization) or float(regularization) < 0.0):
        raise ValueError("regularization must be a finite number >= 0")

    def solve_one(one_system):
        kernel = one_system[:, :-1]
        rhs = one_system[:, -1]
        n_rates = kernel.shape[1]

        # The penalised problem is an ordinary least-squares problem on the
        # kernel stacked on lambda times the identity, with zeros appended to b.
        stacked = np.vstack([kernel, float(regularization) * np.eye(n_rates)])
        padded = np.concatenate([rhs, np.zeros(n_rates)])

        # For lambda > 0 the ridge term, not non-negativity, makes the
        # quadratic objective strictly convex.  At zero, uniqueness is not
        # guaranteed for a rank-deficient kernel.
        density, _ = nnls(stacked, padded)
        return density

    if system.ndim == 2:
        return solve_one(system)
    return np.stack([solve_one(one_system) for one_system in system])

def extract_spectral_peaks(density: np.ndarray, emission_grid: np.ndarray,
                                   n_peaks: int) -> np.ndarray:
    import numpy as np

    density = np.asarray(density, dtype=float)
    emission_grid = np.asarray(emission_grid, dtype=float)

    if not (isinstance(n_peaks, (int, np.integer)) and not isinstance(n_peaks, bool)
            and int(n_peaks) >= 1):
        raise ValueError("n_peaks must be an integer >= 1")
    if density.ndim != 1 or density.shape != emission_grid.shape:
        raise ValueError("density and emission_grid must be one-dimensional and equal length")
    if density.size < 3:
        raise ValueError("density must hold at least 3 grid points")
    if np.any(emission_grid <= 0.0) or np.any(np.diff(emission_grid) <= 0.0):
        raise ValueError("emission_grid must be strictly positive and strictly increasing")
    if np.any(density < 0.0):
        raise ValueError("density must be non-negative")

    n_grid = density.size
    log_rate = np.log(emission_grid)

    # Interior local maxima, kept by descending height so that a weak feature
    # beside a strong one is still selected in the right order.
    maxima = [j for j in range(1, n_grid - 1)
              if density[j] > density[j - 1] and density[j] >= density[j + 1]
              and density[j] > 0.0]
    maxima.sort(key=lambda j: -density[j])
    maxima = sorted(maxima[:int(n_peaks)])

    if not maxima:
        return np.zeros((0, 2), dtype=float)

    # Treat bounds as half-open slice starts.  A separating minimum is the
    # first point of the cluster on its right, so no point is double counted.
    bounds = [0]
    for left, right in zip(maxima[:-1], maxima[1:]):
        bounds.append(left + int(np.argmin(density[left:right + 1])))
    bounds.append(n_grid)

    rows = []
    for m in range(len(maxima)):
        low, high = bounds[m], bounds[m + 1]
        segment = density[low:high]
        weight = segment.sum()
        rate = float(np.exp((segment * log_rate[low:high]).sum() / weight))
        rows.append((rate, float(weight)))

    return np.array(rows, dtype=float)

def fit_arrhenius_parameters(temperature: np.ndarray, emission_rate: np.ndarray,
                                     mass_ratio: float = 0.063) -> np.ndarray:
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    temperature = np.asarray(temperature, dtype=float)
    emission_rate = np.asarray(emission_rate, dtype=float)

    if (isinstance(mass_ratio, bool)
            or not isinstance(mass_ratio, (int, float, np.floating, np.integer))
            or not np.isfinite(mass_ratio) or float(mass_ratio) <= 0.0):
        raise ValueError("mass_ratio must be a finite number > 0")
    if temperature.ndim != 1 or temperature.shape != emission_rate.shape:
        raise ValueError("temperature and emission_rate must be one-dimensional and equal length")
    if temperature.size < 2 or np.unique(temperature).size < 2:
        raise ValueError("at least two distinct temperatures are required")
    if np.any(temperature <= 0.0) or np.any(emission_rate <= 0.0):
        raise ValueError("temperature and emission_rate must be strictly positive")

    mass = float(mass_ratio) * m_e

    # Temperature-independent constant left in the prefactor once the rate has
    # been divided by the square of the temperature.
    prefactor = (np.sqrt(3.0 * kb_j / mass) * 100.0
                 * 2.0 * (2.0 * np.pi * mass * kb_j / h_planck ** 2) ** 1.5 * 1e-6)

    abscissa = 1.0 / (kb_ev * temperature)
    ordinate = np.log(emission_rate / temperature ** 2)

    slope, intercept = np.polyfit(abscissa, ordinate, 1)

    depth = -float(slope)
    cross_section = float(np.exp(intercept) / prefactor)

    return np.array([depth, cross_section], dtype=float)

def fit_constrained_amplitudes(binned: np.ndarray, rates: np.ndarray) -> np.ndarray:
    import numpy as np

    binned = np.asarray(binned, dtype=float)
    rates = np.asarray(rates, dtype=float)

    single = binned.ndim == 2 and rates.ndim == 1
    batch = binned.ndim == 3 and rates.ndim == 2
    if not single and not batch:
        raise ValueError(
            "use binned/rates shapes (n_kept, 3)/(n_rates,) or "
            "(n_records, n_kept, 3)/(n_records, n_rates)")
    if binned.shape[-1] != 3 or binned.shape[-2] < 1:
        raise ValueError("the final binned axis must have length 3")
    if rates.shape[-1] < 1:
        raise ValueError("rates must contain at least one rate per record")
    if batch and (binned.shape[0] < 1 or rates.shape[0] != binned.shape[0]):
        raise ValueError("binned and rates must have the same non-zero batch size")
    if np.any(rates <= 0.0) or not np.all(np.isfinite(rates)):
        raise ValueError("rates must be finite and strictly positive")
    n_rates = rates.shape[-1]
    if binned.shape[-2] < n_rates + 1:
        raise ValueError("at least as many binned points as fitted coefficients are required")
    if np.any(binned[..., 2] <= 0.0):
        raise ValueError("bin sample counts must be strictly positive")

    def fit_one(one_binned, one_rates):
        times = one_binned[:, 0]
        values = one_binned[:, 1]
        weights = np.sqrt(one_binned[:, 2])

        # One exponential column per level plus a constant column, so the
        # quiescent capacitance is fitted rather than read off the last sample.
        columns = [np.exp(-rate * times) for rate in one_rates]
        columns.append(np.ones_like(times))
        design = np.column_stack(columns)
        solution, *_ = np.linalg.lstsq(
            design * weights[:, None], values * weights, rcond=None)

        components = np.empty(one_rates.size + 1, dtype=float)
        components[:-1] = np.abs(solution[:-1])
        components[-1] = solution[-1]
        return components

    if single:
        return fit_one(binned, rates)
    return np.stack([fit_one(binned[i], rates[i])
                     for i in range(binned.shape[0])])

def compute_defect_concentration(deflection: np.ndarray, doping_density: float,
                                         base_capacitance: float,
                                         correction: float = 1.0) -> np.ndarray:
    import numpy as np

    for name, value in (("doping_density", doping_density),
                        ("base_capacitance", base_capacitance),
                        ("correction", correction)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value) or float(value) <= 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    deflection = np.atleast_1d(np.asarray(deflection, dtype=float))
    if deflection.size == 0 or not np.all(np.isfinite(deflection)):
        raise ValueError("deflection must be non-empty and finite")

    # Expanding the junction capacitance to first order in the emitted charge
    # density gives a factor of two from the square-root dependence of the
    # capacitance on the charge the space charge region contains.
    ratio = np.abs(deflection) / float(base_capacitance)

    return 2.0 * float(doping_density) * ratio * float(correction)

def average_active_window(temperature: np.ndarray, deflection: np.ndarray,
                                  window_low: float, window_high: float) -> np.ndarray:
    import numpy as np

    temperature = np.asarray(temperature, dtype=float)
    deflection = np.asarray(deflection, dtype=float)

    if deflection.ndim != 2:
        raise ValueError("deflection must have shape (n_temperatures, n_levels)")
    if temperature.ndim != 1 or temperature.size != deflection.shape[0]:
        raise ValueError("temperature length must match the first axis of deflection")
    for name, value in (("window_low", window_low), ("window_high", window_high)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(window_high) < float(window_low):
        raise ValueError("window_high must not be below window_low")

    inside = (temperature >= float(window_low)) & (temperature <= float(window_high))
    if not np.any(inside):
        raise ValueError("the active window contains no measured temperature")

    selected = deflection[inside]

    mean = selected.mean(axis=0)
    # Population standard deviation, so a window holding one temperature gives
    # a scatter of exactly zero rather than an undefined value.
    scatter = selected.std(axis=0)

    summary = np.empty((deflection.shape[1], 2), dtype=float)
    summary[:, 0] = mean
    # Guard the division so a level recovered as identically zero reports an
    # infinite relative scatter rather than raising a divide warning.
    nonzero = mean != 0.0
    summary[:, 1] = np.inf
    summary[nonzero, 1] = scatter[nonzero] / np.abs(mean[nonzero])

    return summary

def run_dlts_concentration_pipeline(temperature: np.ndarray = None,
                                            activation_energy: tuple = (0.711, 0.658),
                                            sigma_inf: tuple = (1.8e-15, 9.1e-15),
                                            deflection: tuple = (-0.75, -0.05),
                                            base_capacitance: float = 204.5,
                                            doping_density: float = 2.0e16,
                                            sampling_rate: float = 1.0e5,
                                            n_samples: int = 100000,
                                            noise_ref: float = 0.0032,
                                            temperature_ref: float = 350.0,
                                            seed: int = 20260722,
                                            n_bins: int = 100,
                                            rate_min: float = 0.1,
                                            rate_max: float = 1.0e5,
                                            n_rates: int = 150,
                                            regularization: float = 0.01,
                                            window_low: float = 370.0,
                                            window_high: float = 450.0,
                                            mass_ratio: float = 0.063) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-10. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary). Only the
    #     implementations are ever accepted: falling back to a public
    #    name would let this gold solution run on the candidate's code.
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
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    rate_of = _resolve_step(
        "compute_emission_rate", "*compute_emission_rate*.py")
    transients_of = _resolve_step(
        "generate_capacitance_transients", "*generate_capacitance_transients*.py")
    bin_of = _resolve_step(
        "bin_transient_logarithmic", "*bin_transient_logarithmic*.py")
    system_of = _resolve_step(
        "build_inversion_system", "*build_inversion_system*.py")
    density_of = _resolve_step(
        "solve_regularized_spectrum", "*solve_regularized_spectrum*.py")
    features_of = _resolve_step(
        "extract_spectral_peaks", "*extract_spectral_peaks*.py")
    arrhenius_of = _resolve_step(
        "fit_arrhenius_parameters", "*fit_arrhenius_parameters*.py")
    amplitudes_of = _resolve_step(
        "fit_constrained_amplitudes", "*fit_constrained_amplitudes*.py")
    concentration_of = _resolve_step(
        "compute_defect_concentration", "*compute_defect_concentration*.py")
    window_of = _resolve_step(
        "average_active_window", "*average_active_window*.py")

    # -- Validate the orchestrator inputs.
    if temperature is None:
        temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
    temperature = np.asarray(temperature, dtype=float)
    activation_energy = np.atleast_1d(np.asarray(activation_energy, dtype=float))
    sigma_inf = np.atleast_1d(np.asarray(sigma_inf, dtype=float))
    deflection = np.atleast_1d(np.asarray(deflection, dtype=float))
    n_levels = activation_energy.size
    if n_levels < 2:
        raise ValueError("at least two levels are required to form a ratio")
    if sigma_inf.size != n_levels or deflection.size != n_levels:
        raise ValueError("level parameter arrays must have matching lengths")

    grid = np.geomspace(float(rate_min), float(rate_max), int(n_rates))
    times = np.arange(1, int(n_samples) + 1) / float(sampling_rate)

    inside = np.nonzero((temperature >= float(window_low))
                        & (temperature <= float(window_high)))[0]
    if inside.size < 2:
        raise ValueError("the active window must contain at least two temperatures")

    # -- Sub-problems 02-03: synthesise every transient and bin it.
    records = transients_of(temperature, activation_energy, sigma_inf, deflection,
                            base_capacitance, sampling_rate, n_samples, noise_ref,
                            temperature_ref, seed, mass_ratio)
    binned = [np.asarray(bin_of(times, records[i], n_bins), dtype=float)
              for i in range(temperature.size)]

    # -- Sub-problems 04-06: build and solve all active inverse problems as a
    #    batch, then locate the features of each independent density.
    active_binned = np.stack([binned[i] for i in inside])
    systems = np.asarray(system_of(active_binned, grid), dtype=float)
    densities = np.asarray(density_of(systems, regularization), dtype=float)
    rates = np.empty((inside.size, n_levels))
    for row in range(inside.size):
        found = np.asarray(features_of(densities[row], grid, n_levels), dtype=float)
        if found.shape[0] < n_levels:
            raise ValueError("a level was not resolved inside the active window")
        rates[row] = found[:, 0]

    # -- Sub-problem 07: regress the located rates to depths and cross sections.
    parameters = np.array([np.asarray(arrhenius_of(temperature[inside], rates[:, m],
                                                   mass_ratio), dtype=float)
                           for m in range(n_levels)])

    # -- Sub-problems 01 and 08: one fixed-rate row and one independent
    #    amplitude/baseline fit per active temperature.
    fixed = np.column_stack([
        np.asarray(rate_of(temperature[inside], parameters[m, 0],
                           parameters[m, 1], mass_ratio), dtype=float)
        for m in range(n_levels)
    ])
    fitted = np.asarray(amplitudes_of(active_binned, fixed), dtype=float)[:, :-1]

    # -- Sub-problems 09-10: average over the window and form the ratio.
    summary = np.asarray(window_of(temperature[inside], fitted,
                                   window_low, window_high), dtype=float)
    concentration = np.asarray(concentration_of(summary[:, 0], doping_density,
                                                base_capacitance, 1.0), dtype=float)

    return float(concentration.max() / concentration.min())
SCICODE_GOLD_EOF
