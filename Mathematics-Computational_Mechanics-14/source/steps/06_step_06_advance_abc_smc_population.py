"""
The update sorts particles by discrepancy, retains the best fraction, resamples the remainder, constructs the scale-aware Gaussian random-walk covariance, estimates a trial acceptance rate, and applies the adaptive number of ABC-MCMC moves.

Advance one adaptive ABC-SMC population for the hierarchical affinity model.

Returns
-------
np.ndarray, four diagnostics followed by updated particles and discrepancies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_abc_smc_population(
    theta: np.ndarray,
    rho: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    n_grid: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Advance one adaptive ABC-SMC population and pack the numeric state.

    Parameters
    ----------
    theta : np.ndarray
        Current particle matrix with shape (dimension, n_particles).
    rho : np.ndarray
        Current discrepancy for each particle.
    observed : np.ndarray
        Observed fluorescence array with shape (time, replicate, cell).
    times : np.ndarray
        Observation times in seconds.
    n_cells : int
        Simulated cells per replicate and time.
    n_prep : int
        Pilot cells per fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    cell_calibration : np.ndarray
        Empirical cell-autofluorescence sample.
    particle_calibration : np.ndarray
        Empirical single-particle fluorescence sample.
    n_grid : int
        Number of empirical-CDF grid points.
    r_trial : int
        Trial ABC-MCMC moves per resampled particle.
    c_smc : float
        Target probability bound for a particle never moving.
    split_fraction : float
        Fraction of particles resampled and moved.
    max_moves : int
        Maximum total ABC-MCMC moves per resampled particle.
    rng : np.random.Generator
        Shared PCG64 stream for resampling and move-stage randomness.

    Notes
    -----
    Resample the rejected particle indices first. Process those particles in
    ascending position order, drawing each Gaussian proposal as $\theta+Lz$ with
    the lower Cholesky factor of the divisor-$n-1$ scaled covariance before its support
    check and drawing a Metropolis uniform only for a finite-prior proposal.
    Simulate the proposed data only after that Metropolis screen passes.

    Returns
    -------
    result : np.ndarray
        Packed vector containing four update diagnostics, the updated particle
        matrix, and the updated discrepancies. The first four entries are
        $(\epsilon,\widehat p_{\mathrm{trial}},R,\widehat p_{\mathrm{extra}})$.

    Raises
    ------
    ValueError
        If theta and rho have inconsistent shapes, there are fewer than two
        particles, split_fraction is outside (0, 1) or empties a subset,
        r_trial is below 1, c_smc is outside (0, 1), or max_moves is below
        r_trial.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _arithmetic_to_log_parameters(mean, sd):
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)):
        raise ValueError("mean and sd must be finite")
    if np.any(mean <= 0.0) or np.any(sd < 0.0):
        raise ValueError("mean must be positive and sd must be non-negative")
    log_sd = np.sqrt(np.log1p((sd / mean) ** 2))
    log_mean = np.log(mean) - 0.5 * log_sd**2
    return log_mean, log_sd


def _parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k):
    mu_r = np.asarray(mu_r, dtype=float)
    mu_k = np.asarray(mu_k, dtype=float)
    if mu_r.ndim != 1 or mu_k.ndim != 1 or mu_r.size == 0 or mu_r.shape != mu_k.shape:
        raise ValueError("mu_r and mu_k must be non-empty one-dimensional arrays of equal length")
    mr, sr = _arithmetic_to_log_parameters(mu_r, float(sigma_r))
    mk, sk = _arithmetic_to_log_parameters(mu_k, float(sigma_k))
    return np.column_stack((mr, sr, mk, sk)).astype(float)


def _solve_particle_association(time, association_rate, carrying_capacity):
    t, r, k = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(association_rate, dtype=float),
        np.asarray(carrying_capacity, dtype=float),
    )
    if np.any(k <= 0.0):
        raise ValueError("carrying capacity must be positive")
    volume = 1.01e-6
    u0 = 9.95e7
    coverage = 1.0
    surface_area = 5.31e-5
    vu0 = volume * u0
    out = np.empty_like(t, dtype=float)
    singular = np.isclose(k, vu0, rtol=1.0e-12, atol=1.0e-14)
    regular = ~singular
    if np.any(regular):
        kr = k[regular]
        rr = r[regular]
        tr = t[regular]
        expo = np.exp(
            -rr * coverage * surface_area * (vu0 - kr) * tr / (kr * volume)
        )
        out[regular] = vu0 * (1.0 - (vu0 - kr) / (vu0 - kr * expo))
    if np.any(singular):
        ks = k[singular]
        rs = r[singular]
        ts = t[singular]
        out[singular] = (
            rs * coverage * surface_area * u0**2 * volume * ts
            / (ks + rs * coverage * surface_area * u0 * ts)
        )
    return out


def _compute_shared_environment(log_params, times, n_prep, fine_step, rng):
    log_params = np.asarray(log_params, dtype=float)
    times = np.asarray(times, dtype=float)
    if log_params.shape != (4,):
        raise ValueError("log_params must have shape (4,)")
    if times.ndim != 1 or times.size == 0 or np.any(times < 0.0):
        raise ValueError("times must be a non-empty non-negative one-dimensional array")
    if int(n_prep) < 1 or float(fine_step) <= 0.0:
        raise ValueError("n_prep and fine_step must be positive")
    mr, sr, mk, sk = log_params
    if sr < 0.0 or sk < 0.0:
        raise ValueError("log-normal scales must be non-negative")

    end = float(np.max(times))
    count = int(round(end / float(fine_step)))
    fine = float(fine_step) * np.arange(count + 1, dtype=float)
    if not np.all(np.isclose(times[:, None], fine[None, :], rtol=0.0, atol=1.0e-9).any(axis=1)):
        raise ValueError("all observation times must lie on the fine grid")

    volume = 1.01e-6
    u0 = 9.95e7
    ubar = np.empty(fine.size, dtype=float)
    for i, t in enumerate(fine):
        r = rng.lognormal(mean=mr, sigma=sr, size=int(n_prep))
        k = rng.lognormal(mean=mk, sigma=sk, size=int(n_prep))
        p = _solve_particle_association(t, r, k)
        ubar[i] = np.mean(u0 - p / volume)

    integral = np.zeros_like(ubar)
    if fine.size > 1:
        increments = 0.5 * (ubar[1:] + ubar[:-1]) * np.diff(fine)
        integral[1:] = np.cumsum(increments)
    indices = [int(np.flatnonzero(np.isclose(fine, t, rtol=0.0, atol=1.0e-9))[0]) for t in times]
    return integral[np.asarray(indices, dtype=int)]


def _sample_particle_fluorescence_vectorised(particle_count, particle_calibration, rng):
    p = np.maximum(0.0, np.asarray(particle_count, dtype=float).ravel())
    calibration = np.asarray(particle_calibration, dtype=float).ravel()
    if calibration.size == 0:
        raise ValueError("particle_calibration must be non-empty")
    n_full = np.floor(p).astype(np.int64)
    frac = p - n_full
    has_fraction = frac > 0.0
    n_draw = n_full + has_fraction.astype(np.int64)
    total_draws = int(np.sum(n_draw))
    if total_draws == 0:
        return np.zeros_like(p)
    draws = calibration[rng.integers(0, calibration.size, size=total_draws)]
    offsets = np.empty(p.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_draw, out=offsets[1:])
    cumulative = np.empty(total_draws + 1, dtype=float)
    cumulative[0] = 0.0
    np.cumsum(draws, out=cumulative[1:])
    segment_sum = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    last = np.zeros_like(p)
    nonempty = n_draw > 0
    last[nonempty] = draws[offsets[1:][nonempty] - 1]
    correction = np.zeros_like(p)
    correction[has_fraction] = (1.0 - frac[has_fraction]) * last[has_fraction]
    return segment_sum - correction


def _simulate_hierarchical_fluorescence(
    log_params,
    times,
    n_cells,
    n_prep,
    fine_step,
    cell_calibration,
    particle_calibration,
    rng,
):
    log_params = np.asarray(log_params, dtype=float)
    times = np.asarray(times, dtype=float)
    cell_calibration = np.asarray(cell_calibration, dtype=float).ravel()
    particle_calibration = np.asarray(particle_calibration, dtype=float).ravel()
    if log_params.ndim != 2 or log_params.shape[1] != 4 or log_params.shape[0] < 1:
        raise ValueError("log_params must have shape (M, 4)")
    if int(n_cells) < 1:
        raise ValueError("n_cells must be positive")
    if cell_calibration.size == 0 or particle_calibration.size == 0:
        raise ValueError("calibration arrays must be non-empty")

    m = log_params.shape[0]
    coverage = 1.0
    surface_area = 5.31e-5
    result = np.empty((times.size, m, int(n_cells)), dtype=float)
    for j in range(m):
        integral = _compute_shared_environment(log_params[j], times, n_prep, fine_step, rng)
        mr, sr, mk, sk = log_params[j]
        for it in range(times.size):
            r = rng.lognormal(mean=mr, sigma=sr, size=int(n_cells))
            k = rng.lognormal(mean=mk, sigma=sk, size=int(n_cells))
            dcell = cell_calibration[rng.integers(0, cell_calibration.size, size=int(n_cells))]
            p = k * (1.0 - np.exp(-coverage * surface_area * r * integral[it] / k))
            result[it, j] = dcell + _sample_particle_fluorescence_vectorised(
                p, particle_calibration, rng
            )
    return result


def _make_cdf_grid(x, n_grid):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("each observed snapshot must contain at least two finite values")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    xp = np.linspace(float(np.min(x)), float(np.max(x)) + 1.0e-3, int(n_grid))
    p = np.searchsorted(np.sort(x), xp, side="left") / float(x.size)
    return xp, p


def _prepare_discrepancy(observed, n_grid):
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 3:
        raise ValueError("observed must have shape (time, replicate, cell)")
    nt, m = observed.shape[:2]
    prepared = []
    for i in range(nt):
        row = []
        for j in range(m):
            row.append(_make_cdf_grid(observed[i, j], n_grid))
        prepared.append(row)
    return prepared


def _anderson_darling_distance(cdf, y):
    xp, p = cdf
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 1 or np.any(~np.isfinite(y)):
        raise ValueError("simulated snapshot must contain finite values")
    f = np.interp(np.sort(y), xp, p, left=p[0], right=p[-1])
    f = np.clip(f, 1.0e-9, 1.0 - 1.0e-9)
    n = y.size
    i = np.arange(1, n + 1, dtype=float)
    value = -n - np.sum(((2.0 * i - 1.0) / n) * (np.log(f) + np.log(1.0 - f[::-1])))
    return float(np.sqrt(max(value, 0.0)))


def _compute_prepared_discrepancy(simulated, prepared):
    simulated = np.asarray(simulated, dtype=float)
    if simulated.ndim != 3:
        raise ValueError("simulated must have shape (time, replicate, cell)")
    if simulated.shape[0] != len(prepared) or simulated.shape[1] != len(prepared[0]):
        raise ValueError("observed and simulated time-replicate dimensions must match")
    total = 0.0
    for i in range(simulated.shape[0]):
        for j in range(simulated.shape[1]):
            total += _anderson_darling_distance(prepared[i][j], simulated[i, j])
    return float(total)


def _safe_uniform(rng, low, high, size=None):
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError("invalid uniform probability interval")
    if high == low:
        if size is None:
            return float(low)
        return np.full(size, low, dtype=float)
    lo = np.nextafter(float(low), float(high))
    hi = np.nextafter(float(high), float(low))
    return rng.uniform(lo, hi, size=size)


def _sample_positive_normal(mean, sd, size, rng):
    mean = float(mean)
    sd = float(sd)
    size = int(size)
    if sd < 0.0 or size < 1:
        raise ValueError("sd must be non-negative and size must be positive")
    if sd == 0.0:
        if mean < 0.0:
            raise ValueError("degenerate positive Normal requires non-negative mean")
        return np.full(size, mean, dtype=float)
    lower_cdf = float(ndtr((0.0 - mean) / sd))
    u = _safe_uniform(rng, lower_cdf, 1.0, size=size)
    return mean + sd * ndtri(u)


def _sample_half_cauchy(scale, rng):
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("half-Cauchy scale must be positive")
    return float(scale * abs(rng.standard_cauchy()))


def _sample_hierarchical_prior(n_replicates, rng):
    m = int(n_replicates)
    if m < 1:
        raise ValueError("n_replicates must be positive")
    sigma_r = rng.uniform(0.0, 2.0e-6)
    sigma_k = rng.uniform(0.0, 5.0)
    mean_r = rng.uniform(0.0, 1.0e-6)
    mean_k = rng.uniform(0.0, 50.0)
    sd_r = _sample_half_cauchy(3.0e-7, rng)
    sd_k = _sample_half_cauchy(5.0, rng)
    theta = np.empty(2 * m + 6, dtype=float)
    theta[:m] = _sample_positive_normal(mean_r, sd_r, m, rng)
    theta[m : 2 * m] = _sample_positive_normal(mean_k, sd_k, m, rng)
    theta[2 * m :] = [sigma_r, sigma_k, mean_r, mean_k, sd_r, sd_k]
    return theta


def _uniform_logpdf(x, lo, hi):
    if not (lo <= x <= hi):
        return -np.inf
    return -math.log(hi - lo)


def _half_cauchy_logpdf(x, scale):
    if x < 0.0 or scale <= 0.0:
        return -np.inf
    return math.log(2.0) - math.log(math.pi * scale) - math.log1p((x / scale) ** 2)


def _normal_logpdf_sum(values, mean, sd):
    values = np.asarray(values, dtype=float)
    if sd < 0.0:
        return -np.inf
    if sd == 0.0:
        return 0.0 if np.allclose(values, mean, rtol=0.0, atol=0.0) else -np.inf
    z = (values - mean) / sd
    return float(-0.5 * np.sum(z**2) - values.size * math.log(sd * math.sqrt(2.0 * math.pi)))


def _log_hierarchical_prior(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 8 or (theta.size - 6) % 2 != 0:
        return -np.inf
    m = (theta.size - 6) // 2
    mu_r = theta[:m]
    mu_k = theta[m : 2 * m]
    sigma_r, sigma_k, mean_r, mean_k, sd_r, sd_k = theta[2 * m :]
    if np.any(~np.isfinite(theta)) or np.any(mu_r < 0.0) or np.any(mu_k < 0.0):
        return -np.inf
    lp = 0.0
    lp += _uniform_logpdf(float(sigma_r), 0.0, 2.0e-6)
    lp += _uniform_logpdf(float(sigma_k), 0.0, 5.0)
    lp += _uniform_logpdf(float(mean_r), 0.0, 1.0e-6)
    lp += _uniform_logpdf(float(mean_k), 0.0, 50.0)
    lp += _half_cauchy_logpdf(float(sd_r), 3.0e-7)
    lp += _half_cauchy_logpdf(float(sd_k), 5.0)
    if not np.isfinite(lp):
        return -np.inf
    lp += _normal_logpdf_sum(mu_r, float(mean_r), float(sd_r))
    lp += _normal_logpdf_sum(mu_k, float(mean_k), float(sd_k))
    return float(lp)


def _regularise_covariance(covariance, relative_jitter=1.0e-12):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    jitter = float(relative_jitter) * diagonal
    jitter = np.where(diagonal > 0.0, jitter, np.finfo(float).tiny)
    return covariance + np.diag(jitter)


def _scaled_cholesky(covariance, relative_jitter=1.0e-12, max_attempts=8):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    sd = np.sqrt(diagonal)
    positive = sd > 0.0
    safe_sd = np.where(positive, sd, 1.0)
    correlation = covariance / np.outer(safe_sd, safe_sd)
    correlation[~positive, :] = 0.0
    correlation[:, ~positive] = 0.0
    np.fill_diagonal(correlation, 1.0)
    correlation = 0.5 * (correlation + correlation.T)
    eye = np.eye(correlation.shape[0])
    for attempt in range(int(max_attempts)):
        jitter = float(relative_jitter) * (10.0**attempt)
        try:
            factor_corr = np.linalg.cholesky(correlation + jitter * eye)
            physical_sd = np.where(positive, sd, math.sqrt(np.finfo(float).tiny))
            return physical_sd[:, None] * factor_corr
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("proposal covariance is not factorisable")


def _proposal_factor(particles):
    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 2 or particles.shape[0] < 1 or particles.shape[1] < 2:
        raise ValueError("particles must have shape (dimension, n_particles) with n_particles >= 2")
    d = particles.shape[0]
    covariance = (2.38**2 / d) * np.cov(particles, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance = _regularise_covariance(covariance, 1.0e-12)
    return _scaled_cholesky(covariance)


def _simulate_from_theta(theta, times, n_cells, n_prep, fine_step, cell_calibration, particle_calibration, rng):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or (theta.size - 6) % 2 != 0:
        raise ValueError("theta must have dimension 2*M+6")
    m = (theta.size - 6) // 2
    mu_r = theta[:m]
    mu_k = theta[m : 2 * m]
    sigma_r, sigma_k = theta[2 * m], theta[2 * m + 1]
    log_params = _parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)
    return _simulate_hierarchical_fluorescence(
        log_params,
        times,
        n_cells,
        n_prep,
        fine_step,
        cell_calibration,
        particle_calibration,
        rng,
    )


def _move_particle(
    state,
    rho_value,
    factor,
    epsilon,
    n_moves,
    prepared,
    times,
    n_cells,
    n_prep,
    fine_step,
    cell_calibration,
    particle_calibration,
    rng,
):
    state = np.asarray(state, dtype=float).copy()
    rho_value = float(rho_value)
    log_prior_state = _log_hierarchical_prior(state)
    accepted = 0
    if not np.isfinite(log_prior_state):
        return state, rho_value, accepted

    for _ in range(int(n_moves)):
        proposal = state + factor @ rng.standard_normal(state.size)
        log_prior_prop = _log_hierarchical_prior(proposal)
        if not np.isfinite(log_prior_prop):
            continue
        log_alpha = log_prior_prop - log_prior_state
        if not np.isfinite(log_alpha) or math.log(rng.uniform()) > min(0.0, log_alpha):
            continue
        simulated = _simulate_from_theta(
            proposal,
            times,
            n_cells,
            n_prep,
            fine_step,
            cell_calibration,
            particle_calibration,
            rng,
        )
        rho_prop = _compute_prepared_discrepancy(simulated, prepared)
        if rho_prop <= epsilon:
            state = proposal
            rho_value = rho_prop
            log_prior_state = log_prior_prop
            accepted += 1
    return state, rho_value, accepted


def _advance_abc_smc_population_core(
    theta,
    rho,
    observed,
    times,
    n_cells,
    n_prep,
    fine_step,
    cell_calibration,
    particle_calibration,
    n_grid,
    r_trial,
    c_smc,
    split_fraction,
    max_moves,
    rng,
):
    theta = np.asarray(theta, dtype=float).copy()
    rho = np.asarray(rho, dtype=float).copy()
    observed = np.asarray(observed, dtype=float)
    if theta.ndim != 2 or rho.shape != (theta.shape[1],):
        raise ValueError("theta and rho shapes are inconsistent")
    if theta.shape[1] < 2 or not 0.0 < float(split_fraction) < 1.0:
        raise ValueError("particle count and split_fraction are invalid")
    if int(r_trial) < 1 or not 0.0 < float(c_smc) < 1.0:
        raise ValueError("r_trial and c_smc are invalid")
    if int(max_moves) < int(r_trial):
        raise ValueError("max_moves must be at least r_trial")

    n_particles = theta.shape[1]
    n_resampled = int(math.floor(float(split_fraction) * n_particles))
    n_retained = n_particles - n_resampled
    if n_resampled < 1 or n_retained < 1:
        raise ValueError("split_fraction produces an empty subset")

    order = np.argsort(rho)
    theta = theta[:, order]
    rho = rho[order]
    epsilon = float(rho[n_retained - 1])

    source = rng.integers(0, n_retained, size=n_resampled)
    theta[:, n_retained:] = theta[:, source]
    rho[n_retained:] = rho[source]

    factor = _proposal_factor(theta)
    prepared = _prepare_discrepancy(observed, n_grid)

    accepted_trial = 0
    for j in range(n_retained, n_particles):
        state, rho_value, accepted = _move_particle(
            theta[:, j],
            rho[j],
            factor,
            epsilon,
            r_trial,
            prepared,
            times,
            n_cells,
            n_prep,
            fine_step,
            cell_calibration,
            particle_calibration,
            rng,
        )
        theta[:, j] = state
        rho[j] = rho_value
        accepted_trial += accepted

    acceptance_trial = accepted_trial / float(int(r_trial) * n_resampled)
    if acceptance_trial <= 0.0 or acceptance_trial >= 1.0:
        r_total = int(r_trial)
    else:
        r_total = int(math.ceil(math.log(float(c_smc)) / math.log1p(-acceptance_trial)))
        r_total = max(int(r_trial), r_total)
    r_total = min(int(max_moves), r_total)
    r_remaining = r_total - int(r_trial)

    accepted_remaining = 0
    if r_remaining > 0:
        for j in range(n_retained, n_particles):
            state, rho_value, accepted = _move_particle(
                theta[:, j],
                rho[j],
                factor,
                epsilon,
                r_remaining,
                prepared,
                times,
                n_cells,
                n_prep,
                fine_step,
                cell_calibration,
                particle_calibration,
                rng,
            )
            theta[:, j] = state
            rho[j] = rho_value
            accepted_remaining += accepted
        acceptance_final = accepted_remaining / float(r_remaining * n_resampled)
    else:
        acceptance_final = acceptance_trial

    meta = np.array([epsilon, acceptance_trial, float(r_total), acceptance_final], dtype=float)
    return theta, rho, meta


def _make_update_fixture(n_particles, n_cells, seed):
    times = np.array([1800.0, 3600.0], dtype=float)
    m = 2
    true_theta = np.array(
        [8.0e-7, 4.0e-7, 10.0, 18.0, 4.0e-7, 2.0, 4.0e-7, 10.0, 0.0, 0.0],
        dtype=float,
    )
    q = (np.arange(24) + 0.5) / 24.0
    z = ndtri(q)
    cell_calibration = np.exp(5.0 + 0.25 * z)
    particle_calibration = np.exp(2.5 + 0.2 * z)
    observed = _simulate_from_theta(
        true_theta,
        times,
        n_cells,
        5,
        900.0,
        cell_calibration,
        particle_calibration,
        np.random.default_rng(int(seed)),
    )
    prepared = _prepare_discrepancy(observed, 101)
    rng = np.random.default_rng(int(seed) + 1)
    theta = np.empty((2 * m + 6, int(n_particles)), dtype=float)
    rho = np.empty(int(n_particles), dtype=float)
    for i in range(int(n_particles)):
        theta[:, i] = _sample_hierarchical_prior(m, rng)
        simulated = _simulate_from_theta(
            theta[:, i],
            times,
            n_cells,
            5,
            900.0,
            cell_calibration,
            particle_calibration,
            rng,
        )
        rho[i] = _compute_prepared_discrepancy(simulated, prepared)
    return theta, rho, observed, times, cell_calibration, particle_calibration


# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import ndtr, ndtri

def _oracle_advance_abc_smc_population(
    theta: np.ndarray,
    rho: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    n_grid: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.asarray(theta).ndim != 2 or np.asarray(rho).shape != (np.asarray(theta).shape[1],):
        raise ValueError("theta and rho shapes are inconsistent")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    theta_new, rho_new, meta = _advance_abc_smc_population_core(
        theta,
        rho,
        observed,
        times,
        n_cells,
        n_prep,
        fine_step,
        cell_calibration,
        particle_calibration,
        n_grid,
        r_trial,
        c_smc,
        split_fraction,
        max_moves,
        rng,
    )
    return np.concatenate((meta, theta_new.ravel(), rho_new)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic tests for one adaptive resample-move population."""
    return [
        {
            "setup": """theta, rho, observed, times, cell_calibration, particle_calibration = _make_update_fixture(12, 8, 31)
rng_call = np.random.default_rng(91)
rng_gold = np.random.default_rng(91)
""",
            "call": "advance_abc_smc_population(theta, rho, observed, times, 8, 5, 900.0, cell_calibration, particle_calibration, 101, 2, 0.5, 0.5, 6, rng_call)",
            "gold_call": "_oracle_advance_abc_smc_population(theta, rho, observed, times, 8, 5, 900.0, cell_calibration, particle_calibration, 101, 2, 0.5, 0.5, 6, rng_gold)",
        },
        {
            "setup": """theta, rho, observed, times, cell_calibration, particle_calibration = _make_update_fixture(10, 6, 52)
rng_call = np.random.default_rng(123)
rng_gold = np.random.default_rng(123)
""",
            "call": "advance_abc_smc_population(theta, rho, observed, times, 6, 5, 900.0, cell_calibration, particle_calibration, 81, 1, 0.4, 0.4, 4, rng_call)",
            "gold_call": "_oracle_advance_abc_smc_population(theta, rho, observed, times, 6, 5, 900.0, cell_calibration, particle_calibration, 81, 1, 0.4, 0.4, 4, rng_gold)",
        },
        {
            "setup": """theta, rho, observed, times, cell_calibration, particle_calibration = _make_update_fixture(14, 7, 77)
rng_call = np.random.default_rng(404)
rng_gold = np.random.default_rng(404)
""",
            "call": "advance_abc_smc_population(theta, rho, observed, times, 7, 5, 900.0, cell_calibration, particle_calibration, 121, 3, 0.2, 0.5, 9, rng_call)",
            "gold_call": "_oracle_advance_abc_smc_population(theta, rho, observed, times, 7, 5, 900.0, cell_calibration, particle_calibration, 121, 3, 0.2, 0.5, 9, rng_gold)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
theta = np.ones((10, 4))
rho = np.arange(4.0)
observed = np.ones((1, 2, 3))
times = np.array([900.0])
cell_calibration = np.array([100.0])
particle_calibration = np.array([10.0])
rng_call = np.random.default_rng(1)
rng_gold = np.random.default_rng(1)
""",
            "call": "value_error_code(lambda: advance_abc_smc_population(theta, rho, observed, times, 3, 2, 900.0, cell_calibration, particle_calibration, 21, 3, 0.2, 0.5, 2, rng_call))",
            "gold_call": "value_error_code(lambda: _oracle_advance_abc_smc_population(theta, rho, observed, times, 3, 2, 900.0, cell_calibration, particle_calibration, 21, 3, 0.2, 0.5, 2, rng_gold))",
        },
    ]
