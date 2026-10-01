"""
Compose the seven precursor operations and measure the corrected estimator’s population-RMSE improvement.

The complete calculation builds the fixed-spin basis, evaluates the matching deterministic finite-step reference, generates full and midway stochastic trajectories, applies the direct and nonlinear estimators, and performs the source bias correction. Every precursor public function must participate in the calculation. Return the percentage change in population RMSE produced by the corrected estimator.

Returns
-------
Return one finite floating-point value representing the corrected estimator’s percentage population-RMSE change relative to the direct estimator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


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
    """Compose all seven precursor functions and return the corrected estimator's percentage population-RMSE change."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
        moves, coefficients, _ = _oracle_hubbard_hop_channels(
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


def _oracle_hubbard_rmse_reduction(
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
        _oracle_hubbard_hop_channels(
            basis[0],
            edge_array,
            hopping,
            interaction,
            delta_beta,
        )
    )
    probe_state, probe_amplitude = (
        _oracle_sample_hubbard_step(
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
        _oracle_propagate_hubbard_walkers(
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
            _oracle_propagate_hubbard_walkers(
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
            _oracle_direct_partition_estimate(
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
            _oracle_propagate_hubbard_walkers(
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
            _oracle_midway_amplitude_samples(
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

        estimates = _oracle_sigma_mtp_estimates(
            amplitude_samples
        )
        jackknife_values[replicate], _ = (
            _oracle_jackknife2_mtp(estimates)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nargs=_benchmark_fixture()",
            "call": "hubbard_rmse_reduction(*args)",
            "gold_call": "_oracle_hubbard_rmse_reduction(*args)",
        },
        {
            "setup": "import numpy as np\nargs=list(_small_fixture(17,2,1,1,0.,(6,16,4))); args[3]=args[3][::-1].copy(); args[7]=np.nextafter(args[7],0.0); args=tuple(args)",
            "call": "hubbard_rmse_reduction(*args)",
            "gold_call": "_oracle_hubbard_rmse_reduction(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_small_fixture(91,3,1,1,1.5,(5,12,6))",
            "call": "hubbard_rmse_reduction(*args)",
            "gold_call": "_oracle_hubbard_rmse_reduction(*args)",
        },
    ]
