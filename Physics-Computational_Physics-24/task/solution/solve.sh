#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def evaluate_multiplicity_moments(multiplicity: np.ndarray,
                                          abundance: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    arrays = [np.asarray(a, dtype=float) for a in (multiplicity, abundance)]
    for name, array in zip(("multiplicity", "abundance"), arrays):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    counts, weights = arrays
    if counts.size != weights.size:
        raise ValueError("multiplicity and abundance must have the same length")
    if np.any(counts < 0.0):
        raise ValueError("multiplicity entries must be non-negative")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("multiplicity entries must be integer valued")
    if np.any(weights < 0.0):
        raise ValueError("abundance entries must be non-negative")
    if abs(float(weights.sum()) - 1.0) > 1e-10:
        raise ValueError("abundance entries must sum to one")

    # Mean multiplicity fixes the deterministic prompt production per fission.
    mean_multiplicity = float(weights @ counts)
    # A fission consumes the incident neutron, so the net gain is nu_p - 1 and
    # its mean square is what the neutron noise amplitude is built from.
    mean_square_net_gain = float(weights @ (counts - 1.0) ** 2)

    return np.array([mean_multiplicity, mean_square_net_gain], dtype=float)

def compute_neutron_event_rates(reactivity: float, delayed_fraction: float,
                                        generation_time: float,
                                        multiplicity_moments: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    reactivity = _finite("reactivity", reactivity)
    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    if generation_time <= 0.0:
        raise ValueError("generation_time must be greater than zero")

    moments = np.asarray(multiplicity_moments, dtype=float)
    if moments.shape != (2,):
        raise ValueError("multiplicity_moments must be an array of shape (2,)")
    if not np.all(np.isfinite(moments)):
        raise ValueError("multiplicity_moments must contain only finite entries")
    mean_multiplicity, mean_square_net_gain = float(moments[0]), float(moments[1])
    if mean_multiplicity <= 0.0:
        raise ValueError("the mean prompt multiplicity must be greater than zero")
    if mean_square_net_gain < 0.0:
        raise ValueError("the mean square net prompt gain must be non-negative")

    # Prompt production per neutron per second is the prompt share of the
    # inverse generation time, so the fission rate follows from dividing it by
    # the mean number of prompt neutrons a fission releases.
    fission_rate = (1.0 - delayed_fraction) / (mean_multiplicity * generation_time)
    # Total removal per neutron per second is fixed by the reactivity; what is
    # not fission is capture and leakage.
    loss_rate = (1.0 - reactivity) / generation_time - fission_rate
    if loss_rate < 0.0:
        raise ValueError("the capture-and-leakage rate must be non-negative")

    # The delayed yield is fixed by requiring the fission rate to reproduce the
    # delayed production of the point-kinetic equations.
    delayed_yield = delayed_fraction / (generation_time * fission_rate)

    # Fission and removal both contribute to the variance of the neutron
    # population; fission does so through the mean square of its net gain.
    noise_coefficient = np.sqrt(mean_square_net_gain * fission_rate + loss_rate)

    return np.array([fission_rate, loss_rate, delayed_yield, noise_coefficient],
                    dtype=float)

def build_kinetics_matrix(reactivity: float, delayed_fraction: float,
                                  group_fractions: np.ndarray,
                                  decay_constants: np.ndarray, generation_time: float,
                                  tau_core: float, tau_excore: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    reactivity = _finite("reactivity", reactivity)
    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    tau_core = _finite("tau_core", tau_core)
    tau_excore = _finite("tau_excore", tau_excore)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    for name, value in (("generation_time", generation_time), ("tau_core", tau_core),
                        ("tau_excore", tau_excore)):
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")

    shares = np.asarray(group_fractions, dtype=float)
    lambdas = np.asarray(decay_constants, dtype=float)
    for name, array in (("group_fractions", shares), ("decay_constants", lambdas)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} entries must be strictly positive")
    if shares.size != lambdas.size:
        raise ValueError("group_fractions and decay_constants must have equal length")
    if abs(float(shares.sum()) - 1.0) > 1e-10:
        raise ValueError("group_fractions entries must sum to one")

    n_groups = shares.size
    size = 1 + 2 * n_groups
    matrix = np.zeros((size, size), dtype=float)
    core = slice(1, 1 + n_groups)
    excore = slice(1 + n_groups, size)

    # Neutron balance: prompt multiplication net of the delayed fraction, fed by
    # the decay of the precursors that are inside the core at that instant.
    matrix[0, 0] = (reactivity - delayed_fraction) / generation_time
    matrix[0, core] = lambdas

    # In-core precursors: born from fission, removed by decay and by being
    # carried out of the core, replenished by the returning ex-core inventory.
    matrix[core, 0] = delayed_fraction * shares / generation_time
    core_diagonal = -(lambdas + 1.0 / tau_core)
    excore_diagonal = -(lambdas + 1.0 / tau_excore)
    for j in range(n_groups):
        matrix[1 + j, 1 + j] = core_diagonal[j]
        matrix[1 + j, 1 + n_groups + j] = 1.0 / tau_excore
        # Ex-core precursors: no fission source out there, and their decay
        # releases a neutron outside the chain, so it only removes inventory.
        matrix[1 + n_groups + j, 1 + n_groups + j] = excore_diagonal[j]
        matrix[1 + n_groups + j, 1 + j] = 1.0 / tau_core

    return matrix

def compute_drift_reactivity_loss(kinetics_matrix: np.ndarray,
                                          delayed_fraction: float,
                                          decay_constants: np.ndarray,
                                          generation_time: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    if generation_time <= 0.0:
        raise ValueError("generation_time must be greater than zero")

    lambdas = np.asarray(decay_constants, dtype=float)
    if lambdas.ndim != 1 or lambdas.size < 1:
        raise ValueError("decay_constants must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("decay_constants must contain only finite entries")
    if np.any(lambdas <= 0.0):
        raise ValueError("decay_constants entries must be strictly positive")

    matrix = np.asarray(kinetics_matrix, dtype=float)
    n_groups = lambdas.size
    size = 1 + 2 * n_groups
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("kinetics_matrix must be square of size 1 + 2 * n_groups")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("kinetics_matrix must contain only finite entries")

    # Hold the neutron population at unity and let the precursor rows relax:
    # the resulting inventory is the stationary precursor distribution per
    # neutron, and it is what the neutron balance has to be closed against.
    precursor_block = matrix[1:, 1:]
    fission_source = matrix[1:, 0]
    if abs(float(np.linalg.det(precursor_block))) <= 0.0:
        raise ValueError("the precursor block of kinetics_matrix is singular")
    try:
        inventory = np.linalg.solve(precursor_block, -fission_source)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the precursor block of kinetics_matrix is singular") from exc
    if not np.all(np.isfinite(inventory)) or np.any(inventory <= 0.0):
        raise ValueError("the per-neutron precursor inventory must be strictly positive")

    # Only the in-core inventory feeds neutrons back into the chain, so the
    # delayed fraction the chain actually sees falls short of the nominal one.
    returned = float(lambdas @ inventory[:n_groups])
    return float(delayed_fraction - generation_time * returned)

def solve_steady_state_populations(kinetics_matrix: np.ndarray,
                                           source_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(source_rate, (int, float, np.floating, np.integer))
            and not isinstance(source_rate, bool) and np.isfinite(source_rate)):
        raise ValueError("source_rate must be a finite number")
    source_rate = float(source_rate)
    if source_rate <= 0.0:
        raise ValueError("source_rate must be greater than zero")

    matrix = np.asarray(kinetics_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kinetics_matrix must be a two-dimensional square array")
    size = matrix.shape[0]
    if size < 3 or size % 2 == 0:
        raise ValueError("kinetics_matrix must have odd size of at least three")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("kinetics_matrix must contain only finite entries")

    # A stationary state exists only if every mode of the drift decays; a
    # non-decaying mode means the configuration is at or above critical and the
    # source-driven population grows without bound.
    spectrum = np.linalg.eigvals(matrix)
    if float(np.max(spectrum.real)) >= 0.0:
        raise ValueError("kinetics_matrix has a non-decaying mode, so no stationary state exists")

    # The external source enters the neutron balance only; the precursor rows
    # are driven purely by fission.
    source = np.zeros(size, dtype=float)
    source[0] = source_rate

    try:
        state = np.linalg.solve(matrix, -source)
    except np.linalg.LinAlgError as exc:
        raise ValueError("kinetics_matrix is singular") from exc

    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise ValueError("the stationary state must be strictly positive in every component")

    return state

def build_jump_diffusion_matrix(state: np.ndarray, event_rates: np.ndarray,
                                        group_fractions: np.ndarray,
                                        decay_constants: np.ndarray,
                                        multiplicity_moments: np.ndarray,
                                        tau_core: float, tau_excore: float,
                                        source_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _positive(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
        return float(value)

    tau_core = _positive("tau_core", tau_core)
    tau_excore = _positive("tau_excore", tau_excore)
    source_rate = _positive("source_rate", source_rate)

    shares = np.asarray(group_fractions, dtype=float)
    lambdas = np.asarray(decay_constants, dtype=float)
    for name, array in (("group_fractions", shares), ("decay_constants", lambdas)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} entries must be strictly positive")
    if shares.size != lambdas.size:
        raise ValueError("group_fractions and decay_constants must have equal length")
    if abs(float(shares.sum()) - 1.0) > 1e-10:
        raise ValueError("group_fractions entries must sum to one")

    rates = np.asarray(event_rates, dtype=float)
    moments = np.asarray(multiplicity_moments, dtype=float)
    if rates.shape != (4,) or not np.all(np.isfinite(rates)):
        raise ValueError("event_rates must be a finite array of shape (4,)")
    if np.any(rates[:3] < 0.0):
        raise ValueError("the first three event_rates entries must be non-negative")
    if moments.shape != (2,) or not np.all(np.isfinite(moments)):
        raise ValueError("multiplicity_moments must be a finite array of shape (2,)")

    populations = np.asarray(state, dtype=float)
    n_groups = shares.size
    size = 1 + 2 * n_groups
    if populations.ndim != 1 or populations.size != size:
        raise ValueError("state must be one-dimensional of length 1 + 2 * n_groups")
    if not np.all(np.isfinite(populations)) or np.any(populations < 0.0):
        raise ValueError("state entries must be finite and non-negative")

    fission_rate, loss_rate, delayed_yield = float(rates[0]), float(rates[1]), float(rates[2])
    mean_multiplicity, mean_square_net_gain = float(moments[0]), float(moments[1])
    neutrons = float(populations[0])
    core = populations[1:1 + n_groups]
    excore = populations[1 + n_groups:]
    core_slice = slice(1, 1 + n_groups)
    excore_slice = slice(1 + n_groups, size)

    matrix = np.zeros((size, size), dtype=float)

    # Capture-and-leakage and the external source each move the neutron count by
    # one and nothing else.
    matrix[0, 0] += neutrons * loss_rate + source_rate

    # A fission moves the neutron count by its net prompt gain and simultaneously
    # creates delayed precursors, so it correlates the neutron row with every
    # precursor row. The per-group delayed yield is Poisson, so its own second
    # moment carries an extra diagonal piece.
    group_yield = delayed_yield * shares
    matrix[0, 0] += neutrons * fission_rate * mean_square_net_gain
    cross = neutrons * fission_rate * (mean_multiplicity - 1.0) * group_yield
    matrix[0, core_slice] += cross
    matrix[core_slice, 0] += cross
    matrix[core_slice, core_slice] += neutrons * fission_rate * (
        np.outer(group_yield, group_yield) + np.diag(group_yield))

    # A decay inside the core destroys a precursor and creates a neutron, so it
    # is anti-correlated between the two; a decay in the loop only destroys.
    core_decay = core * lambdas
    matrix[0, 0] += float(core_decay.sum())
    for j in range(n_groups):
        matrix[0, 1 + j] -= core_decay[j]
        matrix[1 + j, 0] -= core_decay[j]
        matrix[1 + j, 1 + j] += core_decay[j]
        matrix[1 + n_groups + j, 1 + n_groups + j] += excore[j] * lambdas[j]

    # A transfer moves one precursor from one region to the other; both
    # directions carry the same increment up to a sign, so their rates add.
    transfer = core / tau_core + excore / tau_excore
    for j in range(n_groups):
        matrix[1 + j, 1 + j] += transfer[j]
        matrix[1 + n_groups + j, 1 + n_groups + j] += transfer[j]
        matrix[1 + j, 1 + n_groups + j] -= transfer[j]
        matrix[1 + n_groups + j, 1 + j] -= transfer[j]

    return 0.5 * (matrix + matrix.T)

def build_diffusive_noise_matrix(state: np.ndarray,
                                         noise_coefficient: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(noise_coefficient, (int, float, np.floating, np.integer))
            and not isinstance(noise_coefficient, bool)
            and np.isfinite(noise_coefficient)):
        raise ValueError("noise_coefficient must be a finite number")
    noise_coefficient = float(noise_coefficient)
    if noise_coefficient < 0.0:
        raise ValueError("noise_coefficient must be non-negative")

    populations = np.asarray(state, dtype=float)
    if populations.ndim != 1:
        raise ValueError("state must be a one-dimensional array")
    size = populations.size
    if size < 3 or size % 2 == 0:
        raise ValueError("state must have odd length of at least three")
    if not np.all(np.isfinite(populations)):
        raise ValueError("state must contain only finite entries")
    if np.any(populations < 0.0):
        raise ValueError("state entries must be non-negative")

    # The single Brownian term acts on the neutron balance alone, with a
    # variance rate proportional to the neutron population; the precursor
    # equations of this description carry no noise at all, so every other entry
    # of the matrix is identically zero.
    matrix = np.zeros((size, size), dtype=float)
    matrix[0, 0] = noise_coefficient ** 2 * float(populations[0])

    return matrix

def solve_stationary_covariance(drift_matrix: np.ndarray,
                                        diffusion_matrix: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    drift = np.asarray(drift_matrix, dtype=float)
    diffusion = np.asarray(diffusion_matrix, dtype=float)
    for name, array in (("drift_matrix", drift), ("diffusion_matrix", diffusion)):
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional square array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if drift.shape != diffusion.shape:
        raise ValueError("drift_matrix and diffusion_matrix must have the same shape")

    scale = float(np.abs(diffusion).max())
    tolerance = 1e-9 * scale if scale > 0.0 else 1e-12
    if float(np.abs(diffusion - diffusion.T).max()) > tolerance:
        raise ValueError("diffusion_matrix must be symmetric")

    # A diffusion matrix is a covariance rate, so it cannot have a materially
    # negative eigenvalue; without this check an indefinite input is accepted
    # and returns something that is not a covariance at all.
    def _reject_indefinite(matrix, name):
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        largest = float(np.abs(eigenvalues).max())
        floor = -1e-9 * largest if largest > 0.0 else -1e-12
        if float(eigenvalues.min()) < floor:
            raise ValueError(f"{name} must be positive semi-definite")

    _reject_indefinite(diffusion, "diffusion_matrix")

    # A unique stationary covariance exists only if the drift has no
    # non-decaying mode.
    spectrum = np.linalg.eigvals(drift)
    if float(np.max(spectrum.real)) >= 0.0:
        raise ValueError("drift_matrix has a non-decaying mode, so no stationary covariance exists")

    # The stationary balance is the linear matrix equation in which the drift
    # acting on the covariance from the left and from the right offsets the
    # diffusion; vectorising it turns the equation into an ordinary linear
    # system whose operator is the Kronecker sum of the drift with itself.
    size = drift.shape[0]
    identity = np.eye(size)
    operator = np.kron(identity, drift) + np.kron(drift, identity)
    try:
        solution = np.linalg.solve(operator, -np.asarray(diffusion).reshape(-1))
    except np.linalg.LinAlgError as exc:
        raise ValueError("the stationary covariance equation is singular") from exc

    covariance = solution.reshape(size, size)
    # Round-off makes the raw solution very slightly asymmetric; the exact
    # solution is symmetric because the diffusion matrix is.
    covariance = 0.5 * (covariance + covariance.T)
    _reject_indefinite(covariance, "the stationary covariance")

    return covariance

def compute_inventory_dispersion_ratio(reference_covariance: np.ndarray,
                                               comparison_covariance: np.ndarray,
                                               weights: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    reference = np.asarray(reference_covariance, dtype=float)
    comparison = np.asarray(comparison_covariance, dtype=float)
    for name, array in (("reference_covariance", reference),
                        ("comparison_covariance", comparison)):
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional square array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        scale = float(np.abs(array).max())
        tolerance = 1e-9 * scale if scale > 0.0 else 1e-12
        if float(np.abs(array - array.T).max()) > tolerance:
            raise ValueError(f"{name} must be symmetric")
    if reference.shape != comparison.shape:
        raise ValueError("the two covariance matrices must have the same shape")

    coefficients = np.asarray(weights, dtype=float)
    if coefficients.ndim != 1 or coefficients.size != reference.shape[0]:
        raise ValueError("weights must be one-dimensional of the same length as the covariances")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weights must contain only finite entries")
    if float(np.abs(coefficients).max()) == 0.0:
        raise ValueError("weights must have at least one non-zero entry")

    # The aggregate is a linear functional of the state, so its variance is the
    # quadratic form of the covariance with the same coefficients; every
    # cross-covariance between the summed components enters here.
    reference_variance = float(coefficients @ reference @ coefficients)
    comparison_variance = float(coefficients @ comparison @ coefficients)
    if reference_variance < 0.0 or comparison_variance < 0.0:
        raise ValueError("a covariance matrix gave a negative variance for the aggregate")
    if comparison_variance == 0.0:
        raise ValueError("the comparison variance vanishes, so the ratio is undefined")

    return float(np.sqrt(reference_variance / comparison_variance))

def run_precursor_dispersion_pipeline(subcriticality: float = 0.025,
                                              tau_core: float = 7.5,
                                              tau_excore: float = 12.5,
                                              source_rate: float = 8800.0,
                                              generation_time: float = 1.0e-3,
                                              delayed_fraction: float = 0.0065,
                                              region: str = "excore") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    for name, value in (("subcriticality", subcriticality), ("tau_core", tau_core),
                        ("tau_excore", tau_excore), ("source_rate", source_rate),
                        ("generation_time", generation_time)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
    if not (isinstance(delayed_fraction, (int, float, np.floating, np.integer))
            and not isinstance(delayed_fraction, bool) and np.isfinite(delayed_fraction)
            and 0.0 < float(delayed_fraction) < 1.0):
        raise ValueError("delayed_fraction must be a finite number strictly between zero and one")
    if region not in ("excore", "core", "neutron"):
        raise ValueError("region must be one of 'excore', 'core' or 'neutron'")

    subcriticality = float(subcriticality)
    tau_core = float(tau_core)
    tau_excore = float(tau_excore)
    source_rate = float(source_rate)
    generation_time = float(generation_time)
    delayed_fraction = float(delayed_fraction)

    # -- Nuclear data of the testbed.
    group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
    decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
    multiplicity = np.arange(6, dtype=float)
    abundance = np.array([0.027, 0.158, 0.339, 0.305, 0.133, 0.038])
    n_groups = decay_constants.size

    # -- Sub-problem 01: the two moments of the prompt fission multiplicity.
    multiplicity_moments = evaluate_multiplicity_moments(multiplicity, abundance)

    # -- Sub-problems 03 and 04: locate the circulating critical point. The
    #    reference assembly is made at zero reactivity because the drift loss is
    #    a property of the precursor rows alone.
    reference_matrix = build_kinetics_matrix(0.0, delayed_fraction, group_fractions,
                                                     decay_constants, generation_time,
                                                     tau_core, tau_excore)
    reactivity_loss = compute_drift_reactivity_loss(reference_matrix, delayed_fraction,
                                                            decay_constants, generation_time)
    reactivity = float(reactivity_loss) - subcriticality

    # -- Sub-problems 02 and 03: the event rates and the drift at the operating
    #    reactivity.
    event_rates = compute_neutron_event_rates(reactivity, delayed_fraction,
                                                      generation_time, multiplicity_moments)
    kinetics_matrix = build_kinetics_matrix(reactivity, delayed_fraction,
                                                    group_fractions, decay_constants,
                                                    generation_time, tau_core, tau_excore)

    # -- Sub-problem 05: the source-driven stationary state the noise sits on.
    state = solve_steady_state_populations(kinetics_matrix, source_rate)

    # -- Sub-problems 06 and 07: the two competing diffusion matrices.
    jump_diffusion = build_jump_diffusion_matrix(state, event_rates, group_fractions,
                                                         decay_constants, multiplicity_moments,
                                                         tau_core, tau_excore, source_rate)
    diffusive_noise = build_diffusive_noise_matrix(state, float(event_rates[3]))

    # -- Sub-problem 08: the stationary covariance of each description.
    jump_covariance = solve_stationary_covariance(kinetics_matrix, jump_diffusion)
    diffusive_covariance = solve_stationary_covariance(kinetics_matrix, diffusive_noise)

    # -- Sub-problem 09: reduce both covariances to the selected aggregate.
    weights = np.zeros(1 + 2 * n_groups, dtype=float)
    if region == "neutron":
        weights[0] = 1.0
    elif region == "core":
        weights[1:1 + n_groups] = 1.0
    else:
        weights[1 + n_groups:] = 1.0

    return float(compute_inventory_dispersion_ratio(jump_covariance,
                                                            diffusive_covariance, weights))
SCICODE_GOLD_EOF
