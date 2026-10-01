"""
Chain the sub-problem functions 01-10 end to end on a synthetic multi-level capacitance data set and return the ratio of the largest recovered apparent trap concentration to the smallest, with every partial-probing correction set to one.

The full procedure chains the preceding steps in a fixed order. Every measured temperature contributes one generated and binned transient. The active-window records are stacked in acquisition order and passed through the batched interfaces of the inversion-system builder and non-negative spectral solver; each batch row remains an independent inverse problem on the common emission-rate grid. The strongest features of each resulting density give that temperature's emission rates. Those rates are regressed to a trap depth and a capture cross section for each level. Only then are the amplitudes determined, by recomputing a row of fixed rates for every active temperature and passing those rates with the same batch of binned transients to the independent fixed-rate fits. Final averaging also uses only the active window.




The order matters, and so does the refusal to shortcut it. Reading the amplitudes off the spectral density is the obvious shortcut, and it fails in exactly the regime of interest, because the penalty that stabilises the inversion also reshapes a weak feature standing beside a strong one. Reusing the per-temperature rates instead of the regressed ones is the other shortcut, and it injects the scatter of the inversion directly into the amplitudes rather than letting a regression over the whole window average it away. What survives both refusals is a rate estimate smoothed across temperature and an amplitude estimate obtained in the domain where the concentration is defined.




The remaining amplitude error couples statistical rate estimation to fixed-rate sensitivity. The weaker level's rates are the least well determined by the noisy regularised inverse, so its regressed depth carries the largest residual error; constraining its amplitude to that imperfect emission rate then forces a compensating shift onto the amplitude. The fixed-rate fit therefore propagates the weak rate error rather than creating a bias independent of noise. Its apparent concentration is recovered less accurately than the strong level's, while a noise-free control retains only the much smaller effects of discretisation and residual rate mismatch. The physical conversion contains a level-dependent partial-probing correction; because the junction inputs required to evaluate it are absent, this benchmark sets the correction of every level to one and does not claim that the corrected factors cancel.

Returns
-------
float: dimensionless ratio of the largest to the smallest recovered apparent trap concentration, with every partial-probing correction set to one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

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
    """Return the largest-to-smallest recovered apparent concentration ratio.

    Parameters
    ----------
    temperature : np.ndarray
        Measured temperatures in K, in acquisition order. None selects the
        benchmark grid, 300 to 480 K inclusive in steps of 3 K.
    activation_energy : tuple
        Trap depth of each level in eV used to synthesise the data. Must
        contain at least two values.
    sigma_inf : tuple
        Infinite-temperature capture cross section of each level in cm^2 used
        to synthesise the data. Must have the same length as activation_energy.
    deflection : tuple
        Capacitance deflection of each level in pF used to synthesise the data.
        Must have the same length as activation_energy.
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF.
    doping_density : float
        Shallow doping density of the probed layer in 1/cm^3.
    sampling_rate : float
        Sampling rate of the acquisition in Hz.
    n_samples : int
        Number of samples per transient.
    noise_ref : float
        Noise standard deviation in pF at the reference temperature.
    temperature_ref : float
        Reference temperature in K for the noise scaling.
    seed : int
        Seed of the NumPy default random generator.
    n_bins : int
        Number of logarithmically spaced time intervals used to bin each
        transient.
    rate_min, rate_max : float
        Bounds in 1/s of the geometric emission-rate grid.
    n_rates : int
        Number of points on the emission-rate grid.
    regularization : float
        Tikhonov regularization parameter of the inversion.
    window_low, window_high : float
        Bounds in K of the active temperature window.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass.

    Returns
    -------
    ratio : float
        Largest recovered apparent trap concentration divided by the smallest
        across all supplied levels, with every partial-probing correction set
        to one. For the two-level benchmark this is the stronger-to-weaker
        ratio requested in the problem statement.

    Raises
    ------
    ValueError
        If fewer than two levels are supplied, the level-parameter arrays have
        different lengths, or any other argument is outside the stated domain.
    """
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_dlts_concentration_pipeline(temperature: np.ndarray = None,
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
    #    _oracle_ implementations are ever accepted: falling back to a public
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
        "_oracle_compute_emission_rate", "*compute_emission_rate*.py")
    transients_of = _resolve_step(
        "_oracle_generate_capacitance_transients", "*generate_capacitance_transients*.py")
    bin_of = _resolve_step(
        "_oracle_bin_transient_logarithmic", "*bin_transient_logarithmic*.py")
    system_of = _resolve_step(
        "_oracle_build_inversion_system", "*build_inversion_system*.py")
    density_of = _resolve_step(
        "_oracle_solve_regularized_spectrum", "*solve_regularized_spectrum*.py")
    features_of = _resolve_step(
        "_oracle_extract_spectral_peaks", "*extract_spectral_peaks*.py")
    arrhenius_of = _resolve_step(
        "_oracle_fit_arrhenius_parameters", "*fit_arrhenius_parameters*.py")
    amplitudes_of = _resolve_step(
        "_oracle_fit_constrained_amplitudes", "*fit_constrained_amplitudes*.py")
    concentration_of = _resolve_step(
        "_oracle_compute_defect_concentration", "*compute_defect_concentration*.py")
    window_of = _resolve_step(
        "_oracle_average_active_window", "*average_active_window*.py")

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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: reduced acquisition, exercises every stage end to end ---
        {
            "setup": """import numpy as np
temperature = np.arange(366.0, 456.0 + 1e-9, 6.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
""",
            "call": "run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
            "gold_call": "_oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
        },
        # --- Valid: noise-free data isolates residual discretisation and rate mismatch ---
        {
            "setup": """import numpy as np
temperature = np.arange(372.0, 450.0 + 1e-9, 6.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
""",
            "call": "run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70, noise_ref=0.0)",
            "gold_call": "_oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70, noise_ref=0.0)",
        },
        # --- Valid: a milder amplitude imbalance is recovered more accurately ---
        {
            "setup": """import numpy as np
temperature = np.arange(366.0, 456.0 + 1e-9, 6.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.25])
""",
            "call": "run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
            "gold_call": "_oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
        },
        # --- Edge: a stronger penalty broadens the features and shifts the ratio ---
        {
            "setup": """import numpy as np
temperature = np.arange(366.0, 456.0 + 1e-9, 6.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
""",
            "call": "run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70, regularization=0.05)",
            "gold_call": "_oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70, regularization=0.05)",
        },
        # --- Valid: three resolved levels exercise general level-axis composition ---
        {
            "setup": """import numpy as np
temperature = np.arange(366.0, 456.0 + 1e-9, 6.0)
ea = np.array([0.74, 0.68, 0.61])
sg = np.array([2.0e-15, 4.0e-15, 8.0e-15])
dc = np.array([-0.25, -0.45, -0.12])
""",
            "call": "run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
            "gold_call": "_oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=20000, n_bins=70)",
        },
        # --- Invalid: a single level cannot form a concentration ratio ---
        {
            "setup": """import numpy as np
temperature = np.arange(372.0, 450.0 + 1e-9, 6.0)
def run_model():
    try:
        run_dlts_concentration_pipeline(temperature, np.array([0.711]), np.array([1.8e-15]), np.array([-0.75]), n_samples=5000, n_bins=50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_dlts_concentration_pipeline(temperature, np.array([0.711]), np.array([1.8e-15]), np.array([-0.75]), n_samples=5000, n_bins=50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an active window holding fewer than two temperatures ---
        {
            "setup": """import numpy as np
temperature = np.arange(372.0, 450.0 + 1e-9, 6.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
def run_model():
    try:
        run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=5000, n_bins=50, window_low=400.0, window_high=401.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_dlts_concentration_pipeline(temperature, ea, sg, dc, n_samples=5000, n_bins=50, window_low=400.0, window_high=401.0)
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
