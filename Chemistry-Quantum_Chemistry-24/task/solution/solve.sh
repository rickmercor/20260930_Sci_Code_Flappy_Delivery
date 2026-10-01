#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def hubbard_hop_channels(
    occupation: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    import numpy as np

    raw_state = np.asarray(occupation, dtype=float)
    if raw_state.ndim != 1 or len(raw_state) == 0 or len(raw_state) % 2:
        raise ValueError("occupation must contain two equal spin blocks")
    if np.any(~np.isfinite(raw_state)) or np.any(
        (raw_state != 0) & (raw_state != 1)
    ):
        raise ValueError("occupation must be binary")
    state = raw_state.astype(int)
    n_sites = len(state) // 2

    raw_edges = np.asarray(edges, dtype=float)
    if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
        raise ValueError("edges must have shape (n_edges, 2)")
    if np.any(~np.isfinite(raw_edges)) or np.any(
        raw_edges != np.floor(raw_edges)
    ):
        raise ValueError("edge indices must be integers")
    edge_array = raw_edges.astype(int)
    seen = set()
    for i, j in edge_array:
        if i == j or min(i, j) < 0 or max(i, j) >= n_sites:
            raise ValueError("invalid edge")
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key in seen:
            raise ValueError("edges must be unique")
        seen.add(key)

    scalars = np.asarray([hopping, interaction, delta_beta], dtype=float)
    if (
        np.any(~np.isfinite(scalars))
        or hopping < 0
        or interaction < 0
        or delta_beta <= 0
    ):
        raise ValueError("invalid Hubbard parameters")

    moves = []
    coefficients = []
    for i, j in edge_array:
        for spin in range(2):
            for source, destination in ((int(i), int(j)), (int(j), int(i))):
                source_orbital = spin * n_sites + source
                destination_orbital = spin * n_sites + destination
                if (
                    state[source_orbital] == 1
                    and state[destination_orbital] == 0
                ):
                    opposite = 1 - spin
                    exponent = (
                        state[opposite * n_sites + source]
                        - state[opposite * n_sites + destination]
                    ) * interaction * delta_beta / 2.0
                    moves.append((spin, source, destination))
                    coefficients.append(
                        hopping * delta_beta * np.exp(exponent)
                    )

    move_array = np.asarray(moves, dtype=int).reshape(-1, 3)
    coefficient_array = np.asarray(coefficients, dtype=float)
    normalization = float(1.0 + np.sum(coefficient_array))
    return move_array, coefficient_array, normalization

def sample_hubbard_step(
    occupation: np.ndarray,
    moves: np.ndarray,
    coefficients: np.ndarray,
    normalization: float,
    interaction: float,
    delta_beta: float,
    uniform: float,
) -> tuple[np.ndarray, float]:
    import numpy as np

    raw_state = np.asarray(occupation, dtype=float)
    if raw_state.ndim != 1 or len(raw_state) == 0 or len(raw_state) % 2:
        raise ValueError("occupation must contain two equal spin blocks")
    if np.any((raw_state != 0) & (raw_state != 1)):
        raise ValueError("occupation must be binary")
    state = raw_state.astype(int)
    n_sites = len(state) // 2
    move_array = np.asarray(moves, dtype=int)
    weights = np.asarray(coefficients, dtype=float)
    if move_array.ndim != 2 or move_array.shape[1:] != (3,):
        raise ValueError("moves must have shape (n_moves, 3)")
    if (
        weights.shape != (len(move_array),)
        or np.any(weights < 0)
        or np.any(~np.isfinite(weights))
    ):
        raise ValueError("invalid coefficients")
    if not np.isfinite(normalization) or not np.isclose(
        normalization, 1.0 + np.sum(weights), rtol=1e-12, atol=1e-14
    ):
        raise ValueError("normalization must equal 1 + sum(coefficients)")
    if interaction < 0 or delta_beta <= 0 or not 0 <= uniform < 1:
        raise ValueError("invalid scalar input")

    double_occupancy = int(np.dot(state[:n_sites], state[n_sites:]))
    diagonal = float(np.exp(-interaction * double_occupancy * delta_beta))
    scaled_uniform = float(uniform * normalization)
    if scaled_uniform < 1.0:
        return state.copy(), float(normalization * diagonal)

    index = int(
        np.searchsorted(
            1.0 + np.cumsum(weights), scaled_uniform, side="right"
        )
    )
    if index >= len(move_array):
        index = len(move_array) - 1
    spin, source, destination = map(int, move_array[index])
    if (
        spin not in (0, 1)
        or min(source, destination) < 0
        or max(source, destination) >= n_sites
    ):
        raise ValueError("invalid hop")
    source_orbital = spin * n_sites + source
    destination_orbital = spin * n_sites + destination
    if state[source_orbital] != 1 or state[destination_orbital] != 0:
        raise ValueError("hop is not allowed by the occupation")
    low, high = sorted((source_orbital, destination_orbital))
    fermion_sign = (
        -1.0 if int(np.sum(state[low + 1 : high])) % 2 else 1.0
    )
    result = state.copy()
    result[source_orbital] = 0
    result[destination_orbital] = 1
    amplitude = -float(normalization) * fermion_sign * diagonal
    return result, float(amplitude)

def propagate_hubbard_walkers(
    initial_states: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    raw_states = np.asarray(initial_states, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if (
        raw_states.ndim != 2
        or len(raw_states) == 0
        or raw_states.shape[1] % 2
    ):
        raise ValueError("initial_states must be a nonempty two-spin array")
    if np.any((raw_states != 0) & (raw_states != 1)):
        raise ValueError("states must be binary")
    if draws.ndim != 2 or draws.shape[0] != len(raw_states):
        raise ValueError("one uniform row is required per walker")
    if np.any(~np.isfinite(draws)) or np.any((draws < 0) | (draws >= 1)):
        raise ValueError("uniforms must lie in [0,1)")
    if hopping < 0 or interaction < 0 or delta_beta <= 0:
        raise ValueError("invalid Hubbard parameters")
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[1:] != (2,):
        raise ValueError("edges must have shape (n_edges, 2)")

    states = raw_states.astype(int).copy()
    amplitudes = np.ones(len(states), dtype=float)
    channel_cache = {}

    for step_index in range(draws.shape[1]):
        for walker in range(len(states)):
            key = tuple(states[walker])
            if key not in channel_cache:
                channel_cache[key] = hubbard_hop_channels(
                    states[walker],
                    edge_array,
                    hopping,
                    interaction,
                    delta_beta,
                )

            moves, coefficients, normalization = channel_cache[key]
            states[walker], factor = sample_hubbard_step(
                states[walker],
                moves,
                coefficients,
                normalization,
                interaction,
                delta_beta,
                draws[walker, step_index],
            )
            amplitudes[walker] *= factor

    return states, amplitudes

def direct_partition_estimate(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> float:
    import numpy as np

    raw_basis = np.asarray(basis_states, dtype=float)
    raw_final = np.asarray(final_states, dtype=float)
    amplitudes = np.asarray(path_amplitudes, dtype=float)
    if raw_basis.ndim != 2 or len(raw_basis) == 0:
        raise ValueError("basis_states must be a nonempty matrix")
    if np.any((raw_basis != 0) & (raw_basis != 1)):
        raise ValueError("basis states must be binary")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != len(raw_basis):
        raise ValueError("path_amplitudes must have shape (dimension, samples)")
    expected_shape = (
        len(raw_basis),
        amplitudes.shape[1],
        raw_basis.shape[1],
    )
    if raw_final.shape != expected_shape:
        raise ValueError("inconsistent trajectory arrays")
    if amplitudes.shape[1] == 0 or np.any(~np.isfinite(amplitudes)):
        raise ValueError("finite nonempty samples required")
    if np.any((raw_final != 0) & (raw_final != 1)):
        raise ValueError("final states must be binary")
    if len({tuple(row) for row in raw_basis.astype(int)}) != len(raw_basis):
        raise ValueError("basis rows must be unique")
    closed = np.all(raw_final == raw_basis[:, None, :], axis=2)
    return float(
        np.sum(np.mean(np.where(closed, amplitudes, 0.0), axis=1))
    )

def midway_amplitude_samples(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> np.ndarray:
    import numpy as np

    raw_basis = np.asarray(basis_states, dtype=float)
    raw_final = np.asarray(final_states, dtype=float)
    amplitudes = np.asarray(path_amplitudes, dtype=float)
    if raw_basis.ndim != 2 or len(raw_basis) == 0:
        raise ValueError("basis_states must be a nonempty matrix")
    if np.any((raw_basis != 0) & (raw_basis != 1)):
        raise ValueError("basis states must be binary")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != len(raw_basis):
        raise ValueError("path_amplitudes must have shape (dimension, samples)")
    expected = (len(raw_basis), amplitudes.shape[1], raw_basis.shape[1])
    if raw_final.shape != expected:
        raise ValueError("inconsistent final-state array")
    if amplitudes.shape[1] == 0 or np.any(~np.isfinite(amplitudes)):
        raise ValueError("finite nonempty samples required")
    if np.any((raw_final != 0) & (raw_final != 1)):
        raise ValueError("final states must be binary")
    basis = raw_basis.astype(int)
    lookup = {tuple(row): index for index, row in enumerate(basis)}
    if len(lookup) != len(basis):
        raise ValueError("basis rows must be unique")

    n_samples = amplitudes.shape[1]
    matrices = np.zeros(
        (n_samples, len(basis), len(basis)), dtype=float
    )
    for source in range(len(basis)):
        for sample in range(n_samples):
            integer_state = raw_final[source, sample].astype(int)
            target = lookup.get(tuple(integer_state))
            if target is None or np.any(
                raw_final[source, sample] != integer_state
            ):
                raise ValueError("a final state is outside the supplied basis")
            matrices[sample, target, source] = amplitudes[source, sample]
    return matrices

def sigma_mtp_estimates(amplitude_samples: np.ndarray) -> np.ndarray:
    import numpy as np

    samples = np.asarray(amplitude_samples)
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError("samples must have shape (M, dimension, dimension)")
    n_samples = len(samples)
    if n_samples < 2 or n_samples % 2:
        raise ValueError("an even positive sample count is required")
    if np.any(~np.isfinite(samples)):
        raise ValueError("amplitudes must be finite")

    def estimate(block):
        mean_amplitude = np.mean(block, axis=0)
        return float(np.sum(np.abs(mean_amplitude) ** 2))

    midpoint = n_samples // 2
    return np.asarray(
        [
            estimate(samples),
            estimate(samples[:midpoint]),
            estimate(samples[midpoint:]),
        ],
        dtype=float,
    )

def jackknife2_mtp(estimates: np.ndarray) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(estimates, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)):
        raise ValueError("three finite estimates are required")
    full, first_half, second_half = map(float, values)
    half_average = 0.5 * (first_half + second_half)
    leading_bias = half_average - full
    corrected = full - leading_bias
    return float(corrected), float(leading_bias)

def _fixed_spin_basis(n_sites, n_up, n_down):
    import itertools
    import numpy as np

    rows = []
    for up_sites in itertools.combinations(range(n_sites), n_up):
        for down_sites in itertools.combinations(range(n_sites), n_down):
            occupation = np.zeros(2 * n_sites, dtype=int)
            occupation[list(up_sites)] = 1
            occupation[
                n_sites + np.asarray(down_sites, dtype=int)
            ] = 1
            rows.append(occupation)

    return np.asarray(rows, dtype=int)


def _apply_fermion_hop(occupation, move):
    import numpy as np

    state = np.asarray(occupation, dtype=int)
    n_sites = len(state) // 2
    spin, source, destination = map(int, move)

    source_orbital = spin * n_sites + source
    destination_orbital = spin * n_sites + destination
    low, high = sorted((source_orbital, destination_orbital))

    sign = (
        -1.0
        if int(np.sum(state[low + 1 : high])) % 2
        else 1.0
    )

    result = state.copy()
    result[source_orbital] = 0
    result[destination_orbital] = 1
    return result, sign


def _exact_trotter_step(
    basis_states,
    edges,
    hopping,
    interaction,
    delta_beta,
):
    import numpy as np

    basis = np.asarray(basis_states, dtype=int)
    lookup = {
        tuple(row): index
        for index, row in enumerate(basis)
    }
    matrix = np.zeros(
        (len(basis), len(basis)),
        dtype=float,
    )
    n_sites = basis.shape[1] // 2

    for source, state in enumerate(basis):
        moves, coefficients, _ = hubbard_hop_channels(
            state,
            edges,
            hopping,
            interaction,
            delta_beta,
        )
        double_occupancy = int(
            np.dot(
                state[:n_sites],
                state[n_sites:],
            )
        )
        diagonal = float(
            np.exp(
                -interaction
                * double_occupancy
                * delta_beta
            )
        )
        matrix[source, source] += diagonal

        for move, coefficient in zip(moves, coefficients):
            target_state, sign = _apply_fermion_hop(
                state,
                move,
            )
            target = lookup[tuple(target_state)]
            matrix[target, source] += (
                -coefficient
                * sign
                * diagonal
            )

    return matrix


def hubbard_rmse_reduction(
    n_sites: int,
    n_up: int,
    n_down: int,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> float:
    """Reference orchestrator over the seven precursor functions."""
    import numpy as np

    if any(
        int(value) != value
        for value in (n_sites, n_up, n_down)
    ):
        raise ValueError(
            "site and particle counts must be integers"
        )

    n_sites = int(n_sites)
    n_up = int(n_up)
    n_down = int(n_down)

    if n_sites < 2 or not (
        0 <= n_up <= n_sites
        and 0 <= n_down <= n_sites
    ):
        raise ValueError("invalid fixed-spin sector")

    parameters = np.asarray(
        [hopping, interaction, delta_beta],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(parameters))
        or hopping < 0
        or interaction < 0
        or delta_beta <= 0
    ):
        raise ValueError("invalid Hubbard parameters")

    raw_edges = np.asarray(edges, dtype=float)
    if (
        raw_edges.ndim != 2
        or raw_edges.shape[1:] != (2,)
    ):
        raise ValueError(
            "edges must have shape (n_edges, 2)"
        )
    if (
        np.any(~np.isfinite(raw_edges))
        or np.any(raw_edges != np.floor(raw_edges))
    ):
        raise ValueError(
            "edge indices must be integers"
        )

    edge_array = raw_edges.astype(int)
    seen = set()
    for i, j in edge_array:
        key = (
            min(int(i), int(j)),
            max(int(i), int(j)),
        )
        if (
            i == j
            or min(i, j) < 0
            or max(i, j) >= n_sites
            or key in seen
        ):
            raise ValueError(
                "edges must be unique, in range, and non-self"
            )
        seen.add(key)

    basis = _fixed_spin_basis(
        n_sites,
        n_up,
        n_down,
    )
    draws = np.asarray(uniforms, dtype=float)

    if (
        draws.ndim != 4
        or draws.shape[1] != len(basis)
    ):
        raise ValueError(
            "uniforms must have axes "
            "(replicate, basis, sample, step)"
        )

    n_replicates, _, n_samples, n_steps = draws.shape
    if (
        n_replicates < 2
        or n_samples < 2
        or n_samples % 2
        or n_steps < 2
        or n_steps % 2
    ):
        raise ValueError(
            "replicates >= 2 and positive even "
            "sample and step counts required"
        )

    if (
        np.any(~np.isfinite(draws))
        or np.any((draws < 0) | (draws >= 1))
    ):
        raise ValueError(
            "uniforms must lie in [0,1)"
        )

    # Exercise Steps 01 and 02 explicitly and require their
    # one-step result to agree with the composed Step 03 result.
    probe_moves, probe_coefficients, probe_normalization = (
        hubbard_hop_channels(
            basis[0],
            edge_array,
            hopping,
            interaction,
            delta_beta,
        )
    )
    probe_state, probe_amplitude = (
        sample_hubbard_step(
            basis[0],
            probe_moves,
            probe_coefficients,
            probe_normalization,
            interaction,
            delta_beta,
            float(draws[0, 0, 0, 0]),
        )
    )

    chained_state, chained_amplitude = (
        propagate_hubbard_walkers(
            basis[:1],
            edge_array,
            hopping,
            interaction,
            delta_beta,
            draws[0, 0, 0, :1].reshape(1, 1),
        )
    )

    if not (
        np.array_equal(
            probe_state,
            chained_state[0],
        )
        and np.isclose(
            probe_amplitude,
            chained_amplitude[0],
            rtol=1e-13,
            atol=1e-13,
        )
    ):
        raise ValueError(
            "primitive and composed one-step "
            "propagators disagree"
        )

    one_step = _exact_trotter_step(
        basis,
        edge_array,
        hopping,
        interaction,
        delta_beta,
    )
    if not np.allclose(
        one_step,
        one_step.T,
        rtol=1e-12,
        atol=1e-13,
    ):
        raise ValueError(
            "the symmetric Trotter matrix lost Hermiticity"
        )

    reference = float(
        np.trace(
            np.linalg.matrix_power(
                one_step,
                n_steps,
            )
        )
    )

    starts = np.repeat(
        basis[:, None, :],
        n_samples,
        axis=1,
    )
    flat_starts = starts.reshape(
        -1,
        2 * n_sites,
    )

    direct_values = np.empty(
        n_replicates,
        dtype=float,
    )
    jackknife_values = np.empty(
        n_replicates,
        dtype=float,
    )
    half_steps = n_steps // 2

    for replicate in range(n_replicates):
        final_states, path_amplitudes = (
            propagate_hubbard_walkers(
                flat_starts,
                edge_array,
                hopping,
                interaction,
                delta_beta,
                draws[replicate].reshape(
                    -1,
                    n_steps,
                ),
            )
        )

        direct_values[replicate] = (
            direct_partition_estimate(
                basis,
                final_states.reshape(
                    len(basis),
                    n_samples,
                    2 * n_sites,
                ),
                path_amplitudes.reshape(
                    len(basis),
                    n_samples,
                ),
            )
        )

        midway_states, midway_amplitudes = (
            propagate_hubbard_walkers(
                flat_starts,
                edge_array,
                hopping,
                interaction,
                delta_beta,
                draws[
                    replicate,
                    :,
                    :,
                    :half_steps,
                ].reshape(
                    -1,
                    half_steps,
                ),
            )
        )

        amplitude_samples = (
            midway_amplitude_samples(
                basis,
                midway_states.reshape(
                    len(basis),
                    n_samples,
                    2 * n_sites,
                ),
                midway_amplitudes.reshape(
                    len(basis),
                    n_samples,
                ),
            )
        )

        estimates = sigma_mtp_estimates(
            amplitude_samples
        )
        jackknife_values[replicate], _ = (
            jackknife2_mtp(estimates)
        )

    direct_rmse = float(
        np.sqrt(
            np.mean(
                (direct_values - reference) ** 2
            )
        )
    )
    jackknife_rmse = float(
        np.sqrt(
            np.mean(
                (jackknife_values - reference) ** 2
            )
        )
    )

    if direct_rmse == 0:
        raise ValueError(
            "percentage RMSE reduction is undefined"
        )

    return float(
        100.0
        * (direct_rmse - jackknife_rmse)
        / direct_rmse
    )


def _benchmark_fixture():
    import numpy as np

    n_sites = 4
    n_up = 2
    n_down = 1
    edges = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        dtype=int,
    )
    uniforms = np.random.default_rng(
        20260821
    ).random(
        (16, 24, 64, 12)
    )

    return (
        n_sites,
        n_up,
        n_down,
        edges,
        1.0,
        2.0,
        0.05,
        uniforms,
    )


def _small_fixture(
    seed,
    n_sites,
    n_up,
    n_down,
    interaction,
    shape,
):
    import numpy as np

    edges = np.asarray(
        [
            [site, site + 1]
            for site in range(n_sites - 1)
        ],
        dtype=int,
    )

    dimension = len(
        _fixed_spin_basis(
            n_sites,
            n_up,
            n_down,
        )
    )
    n_replicates, n_samples, n_steps = shape

    uniforms = np.random.default_rng(
        seed
    ).random(
        (
            n_replicates,
            dimension,
            n_samples,
            n_steps,
        )
    )

    return (
        n_sites,
        n_up,
        n_down,
        edges,
        1.0,
        interaction,
        0.05,
        uniforms,
    )
SCICODE_GOLD_EOF
