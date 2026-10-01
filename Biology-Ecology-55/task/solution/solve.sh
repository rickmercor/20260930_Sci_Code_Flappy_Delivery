#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from itertools import combinations

import numpy as np  # noqa: E402, F811


def enumerate_supports(n_species: int) -> np.ndarray:
    """Reference implementation."""
    if isinstance(n_species, (bool, np.bool_)) or not isinstance(
        n_species, (int, np.integer)
    ):
        raise ValueError("n_species must be a positive integer")
    n_species = int(n_species)
    if n_species <= 0:
        raise ValueError("n_species must be a positive integer")

    rows = []
    for richness in range(n_species + 1):
        for support in combinations(range(n_species), richness):
            mask = np.zeros(n_species, dtype=int)
            mask[list(support)] = 1
            rows.append(mask)
    return np.vstack(rows)

import numpy as np  # noqa: E402, F811


def solve_support_equilibria(
    interaction: np.ndarray,
    growth: np.ndarray,
    support_masks: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    support_masks = np.asarray(support_masks)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be a square matrix")
    n_species = interaction.shape[0]
    if n_species == 0 or growth.shape != (n_species,):
        raise ValueError("growth must have shape (N,)")
    if support_masks.ndim != 2 or support_masks.shape[1] != n_species:
        raise ValueError("support_masks must have shape (Q, N)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    equilibria = np.full((support_masks.shape[0], n_species), np.nan, dtype=float)
    for row, mask in enumerate(support_masks.astype(bool)):
        resident = np.flatnonzero(mask)
        if resident.size == 0:
            equilibria[row] = 0.0
            continue
        restricted = interaction[np.ix_(resident, resident)]
        try:
            resident_abundance = np.linalg.solve(restricted, -growth[resident])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(resident_abundance)) and np.all(resident_abundance > tol):
            equilibria[row] = 0.0
            equilibria[row, resident] = resident_abundance
    return equilibria

import numpy as np  # noqa: E402, F811


def compute_invasion_rates(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    equilibria = np.asarray(equilibria, dtype=float)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be square")
    n_species = interaction.shape[0]
    if growth.shape != (n_species,):
        raise ValueError("growth must have one entry per species")
    if equilibria.ndim != 2 or equilibria.shape[1] != n_species:
        raise ValueError("equilibria must have shape (n_supports, n_species)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    finite_rows = np.all(np.isfinite(equilibria), axis=1)
    nan_rows = np.all(np.isnan(equilibria), axis=1)
    if not np.all(finite_rows | nan_rows):
        raise ValueError("each equilibrium row must be finite or all-NaN")

    invasion_rates = np.full_like(equilibria, np.nan, dtype=float)
    invasion_rates[finite_rows] = (
        growth[None, :] + equilibria[finite_rows] @ interaction.T
    )
    return invasion_rates

import numpy as np  # noqa: E402, F811


def compute_spectral_abscissae(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    equilibria = np.asarray(equilibria, dtype=float)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be a square matrix")
    n_species = interaction.shape[0]
    if growth.shape != (n_species,):
        raise ValueError("growth must have shape (N,)")
    if equilibria.ndim != 2 or equilibria.shape[1] != n_species:
        raise ValueError("equilibria must have shape (L, N)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    finite_rows = np.all(np.isfinite(equilibria), axis=1)
    invalid_rows = np.all(np.isnan(equilibria), axis=1)
    if not np.all(finite_rows | invalid_rows):
        raise ValueError("each equilibrium row must be finite or entirely NaN")

    spectral_abscissae = np.full(equilibria.shape[0], np.nan, dtype=float)
    for row in np.flatnonzero(finite_rows):
        equilibrium = equilibria[row]
        per_capita_growth = growth + interaction @ equilibrium
        jacobian = np.diag(per_capita_growth) + equilibrium[:, None] * interaction
        eigenvalues = np.linalg.eigvals(jacobian)
        spectral_abscissae[row] = float(np.max(np.real(eigenvalues)))
    return spectral_abscissae

import numpy as np  # noqa: E402, F811


def classify_stable_states(
    support_masks: np.ndarray,
    equilibria: np.ndarray,
    invasion_rates: np.ndarray,
    spectral_abscissae: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Reference implementation."""
    support_masks = np.asarray(support_masks)
    equilibria = np.asarray(equilibria, dtype=float)
    invasion_rates = np.asarray(invasion_rates, dtype=float)
    spectral_abscissae = np.asarray(spectral_abscissae, dtype=float)
    if support_masks.ndim != 2:
        raise ValueError("support_masks must be two dimensional")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if equilibria.shape != support_masks.shape or invasion_rates.shape != support_masks.shape:
        raise ValueError("equilibria and invasion_rates must match support_masks")
    if spectral_abscissae.shape != (support_masks.shape[0],):
        raise ValueError("spectral_abscissae must have shape (Q,)")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    equilibrium_finite = np.all(np.isfinite(equilibria), axis=1)
    equilibrium_nan = np.all(np.isnan(equilibria), axis=1)
    rate_finite = np.all(np.isfinite(invasion_rates), axis=1)
    rate_nan = np.all(np.isnan(invasion_rates), axis=1)
    if not np.all(equilibrium_finite | equilibrium_nan):
        raise ValueError("each equilibrium row must be finite or all NaN")
    if not np.all(rate_finite | rate_nan):
        raise ValueError("each invasion-rate row must be finite or all NaN")
    if not np.array_equal(equilibrium_finite, rate_finite):
        raise ValueError("finite equilibrium and invasion-rate rows must align")
    if not np.array_equal(equilibrium_finite, np.isfinite(spectral_abscissae)):
        raise ValueError("finite equilibria and spectral abscissae must align")

    stable_flags = np.zeros(support_masks.shape[0], dtype=int)
    for row in np.flatnonzero(equilibrium_finite):
        resident = support_masks[row].astype(bool)
        absent = ~resident
        equilibrium = equilibria[row]
        rates = invasion_rates[row]
        support_consistent = np.all(equilibrium[resident] > tol) and np.all(
            np.abs(equilibrium[absent]) <= tol
        )
        steady = np.all(np.abs(rates[resident]) <= tol)
        saturated = np.all(rates[absent] <= tol)
        stable = spectral_abscissae[row] < -tol
        stable_flags[row] = int(support_consistent and steady and saturated and stable)
    return stable_flags

import numpy as np  # noqa: E402, F811


def summarize_system_richness(
    support_masks: np.ndarray,
    stable_flags: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    support_masks = np.asarray(support_masks)
    stable_flags = np.asarray(stable_flags)
    if support_masks.ndim != 2 or support_masks.shape[0] == 0 or support_masks.shape[1] == 0:
        raise ValueError("support_masks must be a nonempty two-dimensional array")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if stable_flags.shape != (support_masks.shape[0],):
        raise ValueError("stable_flags must have one entry per support")
    if not np.all((stable_flags == 0) | (stable_flags == 1)):
        raise ValueError("stable_flags must be binary")

    n_species = support_masks.shape[1]
    richness = np.sum(support_masks, axis=1, dtype=int)
    selected = stable_flags.astype(bool)
    counts = np.bincount(richness[selected], minlength=n_species + 1)
    multiplicity = int(np.sum(stable_flags))
    return np.concatenate((np.array([multiplicity], dtype=int), counts.astype(int)))

import numpy as np  # noqa: E402, F811


def build_system_probability_vectors(system_summaries: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    system_summaries = np.asarray(system_summaries)
    if system_summaries.ndim != 2 or system_summaries.shape[0] == 0:
        raise ValueError("system_summaries must be a non-empty two-dimensional array")
    if system_summaries.shape[1] < 3:
        raise ValueError("system_summaries must include S_p and richness counts")
    if not np.all(np.isfinite(system_summaries)):
        raise ValueError("system_summaries must be finite")
    if not np.all(system_summaries == np.floor(system_summaries)):
        raise ValueError("system_summaries must contain integers")
    if np.any(system_summaries < 0):
        raise ValueError("system_summaries must be non-negative")
    summaries = system_summaries.astype(int)
    if not np.array_equal(summaries[:, 0], np.sum(summaries[:, 1:], axis=1)):
        raise ValueError("S_p must equal the sum of richness counts")

    probability_vectors = np.zeros(summaries.shape, dtype=float)
    has_states = summaries[:, 0] > 0
    probability_vectors[~has_states, 0] = 1.0
    probability_vectors[has_states, 1:] = (
        summaries[has_states, 1:] / summaries[has_states, 0, None]
    )
    return probability_vectors

import numpy as np  # noqa: E402, F811


def compare_richness_weightings(
    system_summaries: np.ndarray,
    probability_vectors: np.ndarray,
    target_richness: int,
) -> np.ndarray:
    """Reference implementation."""
    system_summaries = np.asarray(system_summaries)
    probability_vectors = np.asarray(probability_vectors, dtype=float)
    if system_summaries.ndim != 2 or system_summaries.shape[0] == 0:
        raise ValueError("system_summaries must be a nonempty matrix")
    if system_summaries.shape[1] < 3:
        raise ValueError("system_summaries must include S_p and richness counts")
    if probability_vectors.shape != system_summaries.shape:
        raise ValueError("probability_vectors must match system_summaries")
    if not np.all(np.isfinite(system_summaries)) or not np.all(
        system_summaries == np.floor(system_summaries)
    ):
        raise ValueError("system_summaries must contain finite integers")
    if np.any(system_summaries < 0):
        raise ValueError("system_summaries must be nonnegative")
    system_summaries = system_summaries.astype(int)
    multiplicities = system_summaries[:, 0]
    counts = system_summaries[:, 1:]
    if not np.array_equal(multiplicities, np.sum(counts, axis=1)):
        raise ValueError("S_p must equal the sum of richness counts")
    n_species = system_summaries.shape[1] - 2
    if isinstance(target_richness, (bool, np.bool_)) or not isinstance(
        target_richness, (int, np.integer)
    ):
        raise ValueError("target_richness must be an integer from zero to N")
    target_richness = int(target_richness)
    if target_richness < 0 or target_richness > n_species:
        raise ValueError("target_richness must be an integer from zero to N")
    if not np.all(np.isfinite(probability_vectors)) or np.any(probability_vectors < 0.0):
        raise ValueError("probability_vectors must be finite and nonnegative")
    if not np.allclose(np.sum(probability_vectors, axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("each probability vector must sum to one")

    expected = np.zeros(probability_vectors.shape, dtype=float)
    zero_state = multiplicities == 0
    expected[zero_state, 0] = 1.0
    positive = ~zero_state
    expected[positive, 1:] = counts[positive] / multiplicities[positive, None]
    if not np.allclose(probability_vectors, expected, atol=1e-12, rtol=0.0):
        raise ValueError("probability_vectors are inconsistent with system_summaries")
    if not np.any(positive):
        raise ValueError("at least one system must contain a stable state")

    richness_column = target_richness + 1
    system_centered = float(np.mean(probability_vectors[positive, richness_column]))
    state_centered = float(
        np.sum(system_summaries[positive, richness_column]) / np.sum(multiplicities[positive])
    )
    return np.array(
        [system_centered, state_centered, state_centered - system_centered],
        dtype=float,
    )

import numpy as np  # noqa: E402, F811


def run_full_pipeline(
    interaction_matrices: np.ndarray,
    growth: np.ndarray,
    target_richness: int = 2,
    tol: float = 1e-9,
) -> float:
    """Reference implementation chaining every earlier step."""
    interaction_matrices = np.asarray(interaction_matrices, dtype=float)
    growth = np.asarray(growth, dtype=float)
    if interaction_matrices.ndim != 3 or interaction_matrices.shape[0] == 0:
        raise ValueError("interaction_matrices must be a nonempty three-dimensional array")
    if interaction_matrices.shape[1] == 0 or interaction_matrices.shape[1] != interaction_matrices.shape[2]:
        raise ValueError("each interaction matrix must be nonempty and square")
    n_species = interaction_matrices.shape[1]
    if growth.shape != (n_species,):
        raise ValueError("growth must have one entry per species")
    if not np.all(np.isfinite(interaction_matrices)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction_matrices and growth must be finite")
    if isinstance(target_richness, (bool, np.bool_)) or not isinstance(
        target_richness, (int, np.integer)
    ):
        raise ValueError("target_richness must be an integer from zero to n_species")
    target_richness = int(target_richness)
    if target_richness < 0 or target_richness > n_species:
        raise ValueError("target_richness must be an integer from zero to n_species")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    support_masks = enumerate_supports(n_species)  # noqa: F821
    summaries = []
    for interaction in interaction_matrices:
        equilibria = solve_support_equilibria(  # noqa: F821
            interaction, growth, support_masks, tol
        )
        invasion_rates = compute_invasion_rates(  # noqa: F821
            interaction, growth, equilibria
        )
        spectral_abscissae = compute_spectral_abscissae(  # noqa: F821
            interaction, growth, equilibria
        )
        stable_flags = classify_stable_states(  # noqa: F821
            support_masks, equilibria, invasion_rates, spectral_abscissae, tol
        )
        summaries.append(
            summarize_system_richness(support_masks, stable_flags)  # noqa: F821
        )
    system_summaries = np.asarray(summaries, dtype=int)
    probability_vectors = build_system_probability_vectors(  # noqa: F821
        system_summaries
    )
    comparison = compare_richness_weightings(  # noqa: F821
        system_summaries, probability_vectors, target_richness
    )
    return float(comparison[2])
SCICODE_GOLD_EOF
