"""
Chain sub-problems 01-10 end to end under a total sampling-cost budget, then construct the largest affordable equal-bin control and return its RMSD minus the adaptive RMSD.

The protocol is a closed loop. It starts from a handful of windows placed inside the two stable states, because those are the only regions a plain unbiased simulation can locate in advance and because anchoring both ends fixes the free-energy difference the quadrature is trying to estimate. At each iteration it asks the current surrogate three questions before spending any simulation time: how uncertain the free-energy integral still is, where a new window would reduce that uncertainty most once the preference for low-free-energy regions has been folded in, and whether the reduction on offer is still worth a simulation. Only then is a window run and the surrogate updated.




Three guards close the loop, each scale-free. The first two are checked before a candidate is chosen: one compares the posterior standard deviation of the integral with the magnitude of the current estimate, the other compares the surviving posterior variance with the prior variance of the integral, so a run that has already removed almost all of its initial uncertainty stops. The third is checked after the winner is known and compares the variance reduction that winner buys against the largest reduction any single observation could ever buy. A fourth guard removes candidates that cannot be paid for from the remaining total-cost budget; when none remain, the run is cost-limited. The initial windows count against that same budget, whereas the source experiment's production-query count remains a hard cap.




The third guard needs a scale against which a variance reduction can be called small, and that scale is the largest reduction any one observation could ever buy on this interval before any window has been run. Under a shared prospective noise level it is attained at the midpoint of the range, the point least predictable from the boundaries; when prospective precision varies across the candidate menu, the best such prior reduction over the menu takes its place.




Two invariants are asserted rather than assumed. The domain is divided by a contiguous interval partition (four equal intervals by default), and the full-range prior and posterior integral variance are recovered by summing every entry of the corresponding interval covariance matrices; summing only their diagonals would incorrectly treat neighbouring free-energy increments as independent. The local posterior means must likewise sum to the full-range estimate, regardless of how many intervals the partition contains. Existing windows may have different observation variances, while every newly acquired window uses the declared prospective variance; after acquisition that prospective variance becomes part of the growing observation-noise vector passed to all later GP operations. The doubly integrated covariance cannot exceed the width of the range times the largest kernel mean on it, which ties the two embeddings together. And the posterior mean must reproduce each measured mean force within the declared interpolation tolerance before it is integrated.




Accuracy is finally quantified as the root-mean-square deviation, over the nodes of the reconstruction grid, between a reconstructed profile and the reference profile obtained by evaluating the model surface itself on that grid, each measured from its own minimum. The control uses the largest number of equal-width-bin midpoints whose total cost fits the same budget. Returning control RMSD minus adaptive RMSD makes a positive result mean that the cost-aware adaptive design was more accurate in this one deterministic benchmark.

Returns
-------
float: uniform-control RMSD minus adaptive RMSD, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_bayesian_umbrella_quadrature(initial_centers: np.ndarray = None, n_queries: int = 15,
                                     kappa: float = 1.0, surface: tuple = None,
                                     domain: tuple = None, n_grid: int = 100,
                                     weight: float = 0.45, variance: float = 0.25,
                                     lengthscale: float = 20.0,
                                     noise: np.ndarray = 1.0e-3,
                                     rel_tol: float = 0.02, var_floor: float = 0.01,
                                     gain_tol: float = 0.02,
                                     interp_tol: float = 0.05,
                                     candidate_noise: np.ndarray = None,
                                     interval_edges: np.ndarray = None,
                                     total_cost_budget: float = 17.0) -> float:
    """Run cost-aware adaptive and matched uniform umbrella protocols.

    Parameters
    ----------
    initial_centers : np.ndarray, optional
        Array of shape (m,) holding the collective-variable values of the
        windows the run is initialised with. Defaults to the four
        initialisation windows of the benchmark.
    n_queries : int
        Maximum number of windows acquired after initialisation
        (n_queries >= 0).
    kappa : float
        Harmonic force constant used in every restrained window (kappa > 0).
    surface : tuple, optional
        The three coefficients (gamma, dmu, offset) of the model free energy.
        Defaults to the benchmark surface.
    domain : tuple, optional
        The pair (lower, upper) bounding the collective variable. Defaults to
        the benchmark range.
    n_grid : int
        Number of uniformly spaced nodes used both as the candidate menu and
        as the reconstruction grid (n_grid >= 2).
    weight : float
        Weight of the exploitation term, in the closed interval [0, 1].
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance shared by the initial
        windows or an array of shape (m,) giving one variance per initial
        window.
    rel_tol : float
        Relative convergence tolerance on the posterior standard deviation of
        the integral (rel_tol >= 0).
    var_floor : float
        Stop once the posterior variance of the integral falls to this
        fraction of its prior variance (var_floor >= 0).
    gain_tol : float
        Stop once the winning candidate's variance reduction falls to this
        fraction of the largest reduction one observation could buy
        (gain_tol >= 0).
    interp_tol : float
        Largest admissible mismatch between the posterior mean and a measured
        mean force at the same window (interp_tol > 0).
    candidate_noise : float or np.ndarray, optional
        Non-negative observation-noise variance assigned to every newly
        acquired window, either as a shared scalar or an array of shape
        (n_grid,) aligned with the complete candidate menu. The value at the
        selected node is appended to the observation-noise history. If
        omitted, the scalar value of noise is reused. It must be supplied
        explicitly when noise is window-specific. For the equal-bin control,
        a vector schedule is linearly interpolated from the candidate grid.
    interval_edges : np.ndarray, optional
        Strictly increasing one-dimensional partition edges beginning at the
        domain lower bound and ending at its upper bound. At least two edges
        are required. Defaults to the five edges of four equal intervals.
    total_cost_budget : float
        Strictly positive budget shared by the adaptive and uniform designs.
        Every adaptive initial and production window counts against it. Window
        cost on a domain of width L and midpoint m is
        1 + 0.6 exp(-((s-m)/(5L/32))**2)
        + 0.35 sin(4 pi (s-lower)/L)**2, which reduces to the prompt's stated
        schedule on the default domain.

    Raises
    ------
    ValueError
        If the initial design is empty or outside the domain, n_queries is not
        a non-negative integer, surface or domain has the wrong length, a
        stopping tolerance is non-finite or negative, interp_tol is non-finite
        or non-positive, total_cost_budget is invalid or cannot fund the
        initial adaptive design or one control window, the
        domain is invalid, observation or candidate noise is invalid,
        interval_edges is not a contiguous partition of the domain, the
        kernel embeddings are inconsistent, the GP fails its interpolation
        check, an interval mean/covariance or another batched delegated result
        has the wrong shape, or a delegated step rejects another invalid
        benchmark parameter.

    Returns
    -------
    rmsd_contrast : float
        Uniform-control RMSD minus adaptive RMSD over the grid, as a native
        Python float. Positive values favour the adaptive design.
    """
    return rmsd_contrast  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_bayesian_umbrella_quadrature(initial_centers: np.ndarray = None, n_queries: int = 15,
                                             kappa: float = 1.0, surface: tuple = None,
                                             domain: tuple = None, n_grid: int = 100,
                                             weight: float = 0.45, variance: float = 0.25,
                                             lengthscale: float = 20.0,
                                             noise: np.ndarray = 1.0e-3,
                                             rel_tol: float = 0.02, var_floor: float = 0.01,
                                             gain_tol: float = 0.02,
                                             interp_tol: float = 0.05,
                                             candidate_noise: np.ndarray = None,
                                             interval_edges: np.ndarray = None,
                                             total_cost_budget: float = 17.0) -> float:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Sub-problems 01-10 are reached by their oracle names in the
    # concatenated namespace the harness builds; the public names are never
    # used, because in a shared namespace those belong to the candidate and the
    # gold side must not execute candidate code.
    import numpy as np

    umbrella_mean_force = _oracle_compute_umbrella_mean_force
    kernel_embedding = _oracle_compute_kernel_embedding
    initial_integral_variance = _oracle_compute_initial_integral_variance
    posterior_mean_gradient = _oracle_compute_posterior_mean_gradient
    integral_posterior_mean = _oracle_compute_integral_posterior_mean
    integral_posterior_variance = _oracle_compute_integral_posterior_variance
    ivr_acquisition = _oracle_compute_ivr_acquisition
    free_energy_profile_value = _oracle_compute_free_energy_profile_value
    combined_acquisition = _oracle_compute_combined_acquisition
    next_umbrella_center = _oracle_select_next_umbrella_center

    # -- Benchmark configuration of the task.
    if initial_centers is None:
        initial_centers = np.array([1.6, 3.0, 284.16, 285.0])
    if surface is None:
        surface = (2.5, 0.4, 8.0)
    if domain is None:
        domain = (0.0, 288.0)

    try:
        initial_centers = np.asarray(initial_centers, dtype=float).ravel()
    except (TypeError, ValueError):
        raise ValueError("initial_centers must contain real numbers")
    if initial_centers.size < 1:
        raise ValueError("at least one initialisation window is required")
    if not np.all(np.isfinite(initial_centers)):
        raise ValueError("initial_centers must contain finite real numbers")
    if isinstance(n_queries, bool) or not isinstance(n_queries, (int, np.integer)) or int(n_queries) < 0:
        raise ValueError("n_queries must be a non-negative integer")
    if (isinstance(n_grid, bool) or
            not isinstance(n_grid, (int, np.integer)) or int(n_grid) < 2):
        raise ValueError("n_grid must be an integer of at least 2")
    try:
        surface = tuple(surface)
    except (TypeError, ValueError):
        raise ValueError("surface must hold the three coefficients (gamma, dmu, offset)")
    try:
        domain = tuple(domain)
    except (TypeError, ValueError):
        raise ValueError("domain must hold the pair (lower, upper)")
    if len(surface) != 3:
        raise ValueError("surface must hold the three coefficients (gamma, dmu, offset)")
    if len(domain) != 2:
        raise ValueError("domain must hold the pair (lower, upper)")
    scalar_types = (int, float, np.integer, np.floating)
    for name, value in (("rel_tol", rel_tol), ("var_floor", var_floor),
                        ("gain_tol", gain_tol), ("interp_tol", interp_tol),
                        ("total_cost_budget", total_cost_budget)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, scalar_types) or
                not np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite real scalar")
    for name, value in (("rel_tol", rel_tol), ("var_floor", var_floor), ("gain_tol", gain_tol)):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if interp_tol <= 0.0:
        raise ValueError("interp_tol must be > 0")
    if total_cost_budget <= 0.0:
        raise ValueError("total_cost_budget must be > 0")

    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, scalar_types)
           for v in surface + domain):
        raise ValueError("surface and domain must contain real scalars")
    gamma, dmu, offset = (float(v) for v in surface)
    lower, upper = (float(v) for v in domain)
    if not np.all(np.isfinite((gamma, dmu, offset, lower, upper))):
        raise ValueError("surface and domain must contain finite real scalars")
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if np.any(initial_centers < lower) or np.any(initial_centers > upper):
        raise ValueError("every initialisation window must lie inside the range")

    try:
        initial_noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    noise_was_scalar = initial_noise_array.ndim == 0
    if noise_was_scalar:
        initial_noise_vector = np.full(initial_centers.size, float(initial_noise_array))
    elif initial_noise_array.shape == initial_centers.shape:
        initial_noise_vector = initial_noise_array.astype(float, copy=False)
    else:
        raise ValueError("noise must be scalar or match initial_centers")
    if (not np.all(np.isfinite(initial_noise_vector)) or
            np.any(initial_noise_vector < 0.0)):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with window-specific noise")
        candidate_noise_array = np.asarray(float(initial_noise_array))
    else:
        try:
            raw_candidate_noise = np.asarray(candidate_noise)
            if np.issubdtype(raw_candidate_noise.dtype, np.bool_):
                raise ValueError("candidate_noise must contain real numbers")
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    candidate_noise_was_scalar = candidate_noise_array.ndim == 0
    if candidate_noise_array.ndim == 0:
        candidate_noise_grid = np.full(int(n_grid), float(candidate_noise_array))
    elif candidate_noise_array.shape == (int(n_grid),):
        candidate_noise_grid = candidate_noise_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_noise must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_noise_grid)) or
            np.any(candidate_noise_grid < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    if interval_edges is None:
        partition_edges = np.linspace(lower, upper, 5)
    else:
        try:
            partition_edges = np.asarray(interval_edges, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("interval_edges must contain real numbers")
        if partition_edges.ndim != 1 or partition_edges.size < 2:
            raise ValueError("interval_edges must be one-dimensional with at least two entries")
        if not np.all(np.isfinite(partition_edges)):
            raise ValueError("interval_edges must contain only finite entries")
        if np.any(np.diff(partition_edges) <= 0.0):
            raise ValueError("interval_edges must be strictly increasing")
        endpoint_tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
        if (not np.isclose(partition_edges[0], lower, rtol=0.0,
                           atol=endpoint_tolerance) or
                not np.isclose(partition_edges[-1], upper, rtol=0.0,
                               atol=endpoint_tolerance)):
            raise ValueError("interval_edges must span the complete domain")
        partition_edges = partition_edges.astype(float, copy=True)
        partition_edges[0] = lower
        partition_edges[-1] = upper

    grid = np.linspace(lower, upper, int(n_grid))
    width = upper - lower

    def _sampling_cost(location):
        """Dimensionless location-dependent effort used by both designs."""
        location = np.asarray(location, dtype=float)
        midpoint = 0.5 * (lower + upper)
        return (1.0
                + 0.6 * np.exp(-((location - midpoint) / (5.0 * width / 32.0)) ** 2)
                + 0.35 * np.sin(4.0 * np.pi * (location - lower) / width) ** 2)

    candidate_cost_grid = np.asarray(_sampling_cost(grid), dtype=float)
    if (candidate_cost_grid.shape != grid.shape
            or not np.all(np.isfinite(candidate_cost_grid))
            or np.any(candidate_cost_grid <= 0.0)):
        raise ValueError("the candidate cost schedule is invalid")
    integration_intervals = np.column_stack(
        (partition_edges[:-1], partition_edges[1:]))
    interval_count = integration_intervals.shape[0]

    # -- Sub-problems 02-03: the two kernel embeddings, and the scale-free
    #    reference they define for the acquisition. The doubly integrated
    #    covariance cannot exceed the range width times the largest kernel
    #    mean on the range; a violation means the two are inconsistent.
    prior_covariance = np.asarray(initial_integral_variance(
        lower, upper, variance, lengthscale, integration_intervals), dtype=float)
    if (prior_covariance.shape != (interval_count, interval_count)
            or not np.all(np.isfinite(prior_covariance))):
        raise ValueError("the prior interval covariance has the wrong shape")
    prior_variance = float(prior_covariance.sum())
    peak_embedding = float(kernel_embedding(0.5 * (lower + upper), lower, upper,
                                            variance, lengthscale))
    if prior_variance > (upper - lower) * peak_embedding * (1.0 + 1.0e-9):
        raise ValueError("the two kernel embeddings are mutually inconsistent")
    if candidate_noise_was_scalar:
        largest_gain = peak_embedding ** 2 / (
            float(variance) + float(candidate_noise_array))
    else:
        grid_embedding = np.asarray(kernel_embedding(
            grid, lower, upper, variance, lengthscale), dtype=float)
        if (grid_embedding.shape != grid.shape or
                not np.all(np.isfinite(grid_embedding))):
            raise ValueError("the batched kernel embedding has the wrong shape")
        largest_gain = float(np.max(
            grid_embedding ** 2 / (float(variance) + candidate_noise_grid)))

    # -- Sub-problem 01: evaluate the initial design through its batched
    #    interface; later acquisitions use the same function on a scalar.
    centers = [float(c) for c in initial_centers]
    initial_forces = np.asarray(
        umbrella_mean_force(np.asarray(centers), kappa, gamma, dmu, offset),
        dtype=float)
    if initial_forces.shape != (len(centers),):
        raise ValueError("the batched mean-force result has the wrong shape")
    forces = [float(value) for value in initial_forces]
    noise_values = [float(value) for value in initial_noise_vector]
    spent_cost = float(np.sum(_sampling_cost(np.asarray(centers, dtype=float))))
    if spent_cost > float(total_cost_budget) + 1.0e-12:
        raise ValueError("total_cost_budget cannot fund the initial adaptive design")

    for _ in range(int(n_queries)):
        # -- Sub-problems 05-06: the two convergence diagnostics.
        observation_noise = np.asarray(noise_values, dtype=float)
        interval_means = np.asarray(integral_posterior_mean(
            np.array(centers), np.array(forces), lower, upper,
            variance, lengthscale, observation_noise,
            integration_intervals), dtype=float)
        interval_covariance = np.asarray(integral_posterior_variance(
            np.array(centers), lower, upper, variance, lengthscale,
            observation_noise, integration_intervals), dtype=float)
        if (interval_means.shape != (interval_count,)
                or interval_covariance.shape != (interval_count, interval_count)
                or not np.all(np.isfinite(interval_means))
                or not np.all(np.isfinite(interval_covariance))):
            raise ValueError("the posterior interval diagnostics have the wrong shape")
        estimate = float(interval_means.sum())
        spread = float(interval_covariance.sum())
        if estimate != 0.0 and np.sqrt(max(spread, 0.0)) / abs(estimate) <= float(rel_tol):
            break
        if spread <= float(var_floor) * prior_variance:
            break

        remaining_cost = float(total_cost_budget) - spent_cost
        sampled = np.isclose(grid[:, None], np.asarray(centers)[None, :],
                             rtol=0.0, atol=1.0e-9).any(axis=1)
        affordable = candidate_cost_grid <= remaining_cost + 1.0e-12
        if not np.any((~sampled) & affordable):
            break

        # -- Sub-problem 10: the winning candidate.
        nxt = float(next_umbrella_center(np.array(centers), np.array(forces), int(n_grid),
                                         weight, lower, upper, variance,
                                         lengthscale, observation_noise,
                                         candidate_noise_grid,
                                         candidate_cost_grid, remaining_cost))
        node = int(np.argmin(np.abs(grid - nxt)))
        selected_noise = float(candidate_noise_grid[node])

        # -- Sub-problem 09: its combined score, which a degenerate menu makes
        #    non-finite, and sub-problem 07: the raw variance reduction behind
        #    that score, compared against the best any one observation can buy.
        if not np.isfinite(combined_acquisition(np.array(centers), np.array(forces), node,
                                                int(n_grid), weight, lower, upper,
                                                variance, lengthscale,
                                                observation_noise,
                                                candidate_noise_grid,
                                                candidate_cost_grid)):
            break
        gain = float(ivr_acquisition(np.array(centers), nxt, lower, upper,
                                     variance, lengthscale, observation_noise,
                                     selected_noise))
        if gain <= float(gain_tol) * largest_gain:
            break

        # -- Sub-problem 04: the surrogate must interpolate its own
        #    observations to within the white-noise level before it is trusted
        #    to predict anywhere else.
        for index in (0, len(centers) - 1):
            predicted = float(posterior_mean_gradient(np.array(centers), np.array(forces),
                                                      centers[index], variance,
                                                      lengthscale,
                                                      observation_noise))
            if abs(predicted - forces[index]) > float(interp_tol):
                raise ValueError("the surrogate does not interpolate its own observations")

        centers.append(nxt)
        forces.append(float(umbrella_mean_force(nxt, kappa, gamma, dmu, offset)))
        noise_values.append(selected_noise)
        spent_cost += float(candidate_cost_grid[node])

    # -- Sub-problem 08: the reconstructed profile on the grid.
    reconstructed = np.asarray(
        free_energy_profile_value(np.array(centers), np.array(forces),
                                  np.arange(int(n_grid), dtype=int),
                                  int(n_grid), lower, upper, variance,
                                  lengthscale, np.asarray(noise_values,
                                                         dtype=float)),
        dtype=float)
    if reconstructed.shape != grid.shape:
        raise ValueError("the batched reconstructed profile has the wrong shape")

    reference = gamma * (grid + offset) ** (2.0 / 3.0) - dmu * grid
    reference = reference - reference.min()

    adaptive_rmsd = float(np.sqrt(np.mean((reconstructed - reference) ** 2)))

    # Cost-matched control: choose the largest equal-bin midpoint design whose
    # complete cost fits the same total budget. Searching up to n_grid keeps
    # the control finite and deterministic for any valid benchmark grid.
    uniform_centers = None
    for count in range(1, int(n_grid) + 1):
        trial = lower + (np.arange(count, dtype=float) + 0.5) * width / count
        if float(np.sum(_sampling_cost(trial))) <= float(total_cost_budget) + 1.0e-12:
            uniform_centers = trial
    if uniform_centers is None:
        raise ValueError("total_cost_budget cannot fund one uniform-control window")
    uniform_forces = np.asarray(
        umbrella_mean_force(uniform_centers, kappa, gamma, dmu, offset), dtype=float)
    if uniform_forces.shape != uniform_centers.shape:
        raise ValueError("the uniform-control mean-force result has the wrong shape")
    if candidate_noise_was_scalar:
        uniform_noise = np.full(uniform_centers.size, float(candidate_noise_array))
    else:
        uniform_noise = np.interp(uniform_centers, grid, candidate_noise_grid)
    uniform_reconstructed = np.asarray(
        free_energy_profile_value(uniform_centers, uniform_forces,
                                  np.arange(int(n_grid), dtype=int), int(n_grid),
                                  lower, upper, variance, lengthscale,
                                  uniform_noise), dtype=float)
    if uniform_reconstructed.shape != grid.shape:
        raise ValueError("the uniform-control profile has the wrong shape")
    uniform_rmsd = float(np.sqrt(np.mean((uniform_reconstructed - reference) ** 2)))

    return float(uniform_rmsd - adaptive_rmsd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the benchmark run of the task (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature()",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature()",
        },
        # --- Integration: a shorter budget, stopped well before convergence ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature(None, 5)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(None, 5)",
        },
        # --- Integration (boundary): no acquisitions at all, only the initialisation ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature(None, 0)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(None, 0)",
        },
        # --- Integration (boundary): a loose relative tolerance stops the loop early ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 5.0)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 5.0)",
        },
        # --- Integration (boundary): a high variance floor stops the loop early ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.3)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.3)",
        },
        # --- Integration (boundary): a high gain tolerance stops the loop early ---
        {
            "setup": """import numpy as np
""",
            "call": "run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.01, 0.5)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.01, 0.5)",
        },
        # --- Integration (edge): a different surface, domain and coarse candidate menu ---
        {
            "setup": """import numpy as np
initial_centers = np.array([2.0, 5.0, 95.0, 98.0])
surface = (4.0, 0.55, 3.0)
domain = (0.0, 100.0)
""",
            "call": "run_bayesian_umbrella_quadrature(initial_centers, 10, 2.5, surface, domain, 41, 0.2, 0.5, 12.0, 1.0e-4)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(initial_centers, 10, 2.5, surface, domain, 41, 0.2, 0.5, 12.0, 1.0e-4)",
        },
        # --- Integration (edge): heterogeneous initial noise and a nonuniform partition ---
        {
            "setup": """import numpy as np
initial_centers = np.array([2.0, 5.0, 95.0, 98.0])
surface = (4.0, 0.55, 3.0)
domain = (0.0, 100.0)
noise = np.array([2.0e-5, 8.0e-4, 3.0e-4, 1.0e-5])
candidate_noise = np.geomspace(2.0e-5, 2.0e-3, 41)
interval_edges = np.array([0.0, 9.0, 31.0, 63.0, 84.0, 100.0])
""",
            "call": "run_bayesian_umbrella_quadrature(initial_centers=initial_centers, n_queries=9, kappa=2.5, surface=surface, domain=domain, n_grid=41, weight=0.2, variance=0.5, lengthscale=12.0, noise=noise, candidate_noise=candidate_noise, interval_edges=interval_edges)",
            "gold_call": "_oracle_run_bayesian_umbrella_quadrature(initial_centers=initial_centers, n_queries=9, kappa=2.5, surface=surface, domain=domain, n_grid=41, weight=0.2, variance=0.5, lengthscale=12.0, noise=noise, candidate_noise=candidate_noise, interval_edges=interval_edges)",
        },
        # --- Invalid: a negative query budget ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bayesian_umbrella_quadrature(None, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(None, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: window-specific initial noise needs a prospective variance ---
        {
            "setup": """import numpy as np
noise = np.array([1.0e-5, 2.0e-3, 4.0e-4, 8.0e-5])
def run_model():
    try:
        run_bayesian_umbrella_quadrature(noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: interval edges do not span the declared domain ---
        {
            "setup": """import numpy as np
interval_edges = np.array([5.0, 80.0, 160.0, 288.0])
def run_model():
    try:
        run_bayesian_umbrella_quadrature(interval_edges=interval_edges)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(interval_edges=interval_edges)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an initialisation window outside the collective-variable range ---
        {
            "setup": """import numpy as np
initial_centers = np.array([1.6, 3.0, 284.16, 400.0])
def run_model():
    try:
        run_bayesian_umbrella_quadrature(initial_centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(initial_centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a scalar cannot supply the three surface coefficients ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bayesian_umbrella_quadrature(surface=2.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(surface=2.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite stopping tolerance is not meaningful ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bayesian_umbrella_quadrature(rel_tol=np.nan)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(rel_tol=np.nan)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive interpolation tolerance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.01, 0.02, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bayesian_umbrella_quadrature(None, 15, 1.0, None, None, 100, 0.45, 0.25, 20.0, 1.0e-3, 0.02, 0.01, 0.02, 0.0)
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
