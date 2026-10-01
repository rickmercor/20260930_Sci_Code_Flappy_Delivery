#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def compute_cycle_observables(free_energies, temperature: float = 298.15):
    """Compute full-cycle microscopic rates and steady-state observables."""
    import numpy as np

    boltzmann = 1.380649e-23
    planck = 6.62607015e-34
    gas_constant_kcal = 8.314462618 / 4184.0

    energies = np.asarray(free_energies, dtype=float)
    if energies.ndim < 1 or energies.shape[-1] != 6:
        raise ValueError("free_energies must have a trailing axis of length 6")
    if not np.all(np.isfinite(energies)):
        raise ValueError("free_energies must be finite")
    if not (
        isinstance(temperature, (int, float, np.floating, np.integer))
        and not isinstance(temperature, bool)
        and np.isfinite(temperature)
        and float(temperature) > 0.0
    ):
        raise ValueError("temperature must be a finite number > 0")

    temperature = float(temperature)
    prefactor = boltzmann * temperature / planck
    thermal = gas_constant_kcal * temperature

    g_free, g_bind, g_es, g_chem, g_ep, g_release = (
        energies[..., 0],
        energies[..., 1],
        energies[..., 2],
        energies[..., 3],
        energies[..., 4],
        energies[..., 5],
    )

    barriers = np.stack(
        [
            g_bind - g_free,
            g_bind - g_es,
            g_chem - g_es,
            g_chem - g_ep,
            g_release - g_ep,
        ],
        axis=-1,
    )
    rate_constants = prefactor * np.exp(-barriers / thermal)

    k_on = rate_constants[..., 0]
    k_off = rate_constants[..., 1]
    k_chem = rate_constants[..., 2]
    k_rev = rate_constants[..., 3]
    k_rel = rate_constants[..., 4]

    if np.any(k_on <= 0.0):
        raise ValueError("the association rate constant must be > 0")

    # Saturating substrate: normalized ES/EP steady-state balance.
    bound_balance = np.stack(
        [
            np.stack([-k_chem, k_rev + k_rel], axis=-1),
            np.stack([np.ones_like(k_chem), np.ones_like(k_chem)], axis=-1),
        ],
        axis=-2,
    )
    bound_rhs = np.stack(
        [np.zeros_like(k_chem), np.ones_like(k_chem)],
        axis=-1,
    )[..., np.newaxis]

    try:
        bound_occupancy = np.linalg.solve(bound_balance, bound_rhs)[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("the bound-state balance is singular") from exc

    k_cat = k_rel * bound_occupancy[..., 1]

    # Low-substrate ES/EP balance per unit free-enzyme and substrate
    # concentration. Its product flux is kcat / KM.
    dilute_balance = np.stack(
        [
            np.stack([k_off + k_chem, -k_rev], axis=-1),
            np.stack([-k_chem, k_rev + k_rel], axis=-1),
        ],
        axis=-2,
    )
    dilute_rhs = np.stack(
        [k_on, np.zeros_like(k_on)],
        axis=-1,
    )[..., np.newaxis]

    try:
        dilute_occupancy = np.linalg.solve(dilute_balance, dilute_rhs)[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("the low-substrate balance is singular") from exc

    catalytic_efficiency = k_rel * dilute_occupancy[..., 1]
    if (
        np.any(~np.isfinite(k_cat))
        or np.any(~np.isfinite(catalytic_efficiency))
        or np.any(k_cat <= 0.0)
        or np.any(catalytic_efficiency <= 0.0)
    ):
        raise ValueError("the cycle must carry non-zero steady-state flux")

    k_d = k_off / k_on
    k_m = k_cat / catalytic_efficiency

    parameters = np.stack(
        [k_d, k_cat, k_m, catalytic_efficiency],
        axis=-1,
    )
    return rate_constants, parameters

# ORACLE SOLUTION
import numpy as np

def simulate_mutant_free_energies(wt_free_energies, n_mutants: int = 1000,
                                          amplitude: float = 2.0,
                                          seed: int = 314159) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    energies = np.asarray(wt_free_energies, dtype=float)
    if energies.shape != (6,):
        raise ValueError("wt_free_energies must have shape (6,)")
    if not np.all(np.isfinite(energies)):
        raise ValueError("wt_free_energies must be finite")
    if not (isinstance(n_mutants, (int, np.integer)) and not isinstance(n_mutants, bool)
            and int(n_mutants) >= 1):
        raise ValueError("n_mutants must be an integer >= 1")
    if not (isinstance(amplitude, (int, float, np.floating, np.integer))
            and not isinstance(amplitude, bool)
            and np.isfinite(amplitude) and float(amplitude) > 0.0):
        raise ValueError("amplitude must be a finite number > 0")
    if not (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
            and int(seed) >= 0):
        raise ValueError("seed must be an integer >= 0")

    n_mutants = int(n_mutants)
    amplitude = float(amplitude)

    # One block draw, rows indexed by variant and columns by sub state in
    # reaction coordinate order, so the library is reproducible verbatim.
    generator = np.random.default_rng(int(seed))
    perturbations = generator.uniform(-amplitude, amplitude, size=(n_mutants, 6))

    return energies[None, :] + perturbations

import numpy as np
def compute_variant_fold_changes(wt_free_energies, mutant_free_energies,
                                         temperature: float = 298.15):
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    boltzmann = 1.380649e-23
    planck = 6.62607015e-34
    gas_constant_kcal = 8.314462618 / 4184.0

    def _parameters(rates):
        """Steady-state observables of the three-step cycle."""
        k_on, k_off = rates[..., 0], rates[..., 1]
        k_chem, k_rev, k_rel = rates[..., 2], rates[..., 3], rates[..., 4]
        partition = k_chem + k_rev + k_rel
        capture = k_chem * k_rel + k_off * k_rev + k_off * k_rel
        if np.any(partition <= 0.0) or np.any(capture <= 0.0):
            raise ValueError("the cycle must carry non-zero steady-state flux")
        return np.stack([k_off / k_on,
                         k_chem * k_rel / partition,
                         capture / (k_on * partition),
                         k_on * k_chem * k_rel / capture], axis=-1)

    wt = np.asarray(wt_free_energies, dtype=float)
    mutants = np.asarray(mutant_free_energies, dtype=float)
    if wt.shape != (6,):
        raise ValueError("wt_free_energies must have shape (6,)")
    if mutants.ndim != 2 or mutants.shape[1] != 6 or mutants.shape[0] < 1:
        raise ValueError("mutant_free_energies must have shape (n_mutants, 6), n_mutants >= 1")
    if not (np.all(np.isfinite(wt)) and np.all(np.isfinite(mutants))):
        raise ValueError("free energies must be finite")
    if not (isinstance(temperature, (int, float, np.floating, np.integer))
            and not isinstance(temperature, bool)
            and np.isfinite(temperature) and float(temperature) > 0.0):
        raise ValueError("temperature must be a finite number > 0")

    temperature = float(temperature)
    thermal = gas_constant_kcal * temperature
    shifts = mutants - wt[None, :]

    # Each barrier shift is the transition-state shift minus the shift of the
    # ground state the elementary step departs from. Columns of shifts are
    # ordered free enzyme, binding TS, ES, chemical TS, EP, release TS.
    barrier_shifts = np.stack([shifts[:, 1] - shifts[:, 0],
                               shifts[:, 1] - shifts[:, 2],
                               shifts[:, 3] - shifts[:, 2],
                               shifts[:, 3] - shifts[:, 4],
                               shifts[:, 5] - shifts[:, 4]], axis=1)
    rate_folds = np.exp(-barrier_shifts / thermal)

    # The wild-type rate constants come from its own reaction coordinate under
    # the same transition-state-theory convention.
    prefactor = boltzmann * temperature / planck
    barriers = np.stack([wt[1] - wt[0],
                         wt[1] - wt[2],
                         wt[3] - wt[2],
                         wt[3] - wt[4],
                         wt[5] - wt[4]], axis=-1)
    rate_constants_wt = prefactor * np.exp(-barriers / thermal)
    if not np.all(np.isfinite(rate_constants_wt)) or rate_constants_wt[0] <= 0.0:
        raise ValueError("the wild-type reaction coordinate must give usable rate constants")

    # Rebuild each variant's rate constants, then divide the variant's
    # steady-state observables by those of the wild type.
    parameter_folds = (_parameters(rate_constants_wt[None, :] * rate_folds)
                       / _parameters(rate_constants_wt)[None, :])

    return rate_folds, parameter_folds

def compute_double_mutant_epistasis(
    rate_constants_wt,
    rate_folds,
) -> np.ndarray:
    """Compute all four additive-null epistasis matrices from microscopic folds."""
    import numpy as np

    def _parameters(rates):
        """Return KD, kcat, KM, and kcat/KM on the final axis."""
        k_on = rates[..., 0]
        k_off = rates[..., 1]
        k_chem = rates[..., 2]
        k_rev = rates[..., 3]
        k_rel = rates[..., 4]

        partition = k_chem + k_rev + k_rel
        capture = k_chem * k_rel + k_off * k_rev + k_off * k_rel
        if np.any(partition <= 0.0) or np.any(capture <= 0.0):
            raise ValueError("the cycle must carry non-zero steady-state flux")

        return np.stack(
            [
                k_off / k_on,
                k_chem * k_rel / partition,
                capture / (k_on * partition),
                k_on * k_chem * k_rel / capture,
            ],
            axis=-1,
        )

    wt = np.asarray(rate_constants_wt, dtype=float)
    folds = np.asarray(rate_folds, dtype=float)

    if wt.shape != (5,):
        raise ValueError("rate_constants_wt must have shape (5,)")
    if folds.ndim != 2 or folds.shape[0] < 1 or folds.shape[1] != 5:
        raise ValueError(
            "rate_folds must have shape (n_mutants, 5), n_mutants >= 1"
        )
    if not (np.all(np.isfinite(wt)) and np.all(np.isfinite(folds))):
        raise ValueError("inputs must be finite")
    if np.any(wt <= 0.0) or np.any(folds <= 0.0):
        raise ValueError("rate constants and fold changes must be positive")

    # Reconstruct the observable-level expectation from each single mutant.
    wt_parameters = _parameters(wt)
    single_rates = wt[None, :] * folds
    single_parameters = _parameters(single_rates)
    single_parameter_folds = single_parameters / wt_parameters[None, :]

    # Construct every ordered additive-null double mutant microscopically.
    double_rates = (
        wt[None, None, :]
        * folds[:, None, :]
        * folds[None, :, :]
    )
    double_parameters = _parameters(double_rates)

    expected_parameters = (
        wt_parameters[None, None, :]
        * single_parameter_folds[:, None, :]
        * single_parameter_folds[None, :, :]
    )
    if np.any(expected_parameters <= 0.0):
        raise ValueError("the reconstructed null expectation must be positive")

    epistasis = np.moveaxis(
        double_parameters / expected_parameters,
        -1,
        0,
    )
    return epistasis

# ORACLE SOLUTION


def summarize_epistasis(epistasis, threshold: float = 1.5) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    values = np.asarray(epistasis, dtype=float)
    if values.size < 1:
        raise ValueError("epistasis must contain at least one entry")
    if not np.all(np.isfinite(values)):
        raise ValueError("epistasis must be finite")
    if np.any(values <= 0.0):
        raise ValueError("epistasis ratios must be strictly positive")
    if not (isinstance(threshold, (int, float, np.floating, np.integer))
            and not isinstance(threshold, bool)
            and np.isfinite(threshold) and float(threshold) > 1.0):
        raise ValueError("threshold must be a finite number > 1")

    magnitude = np.abs(np.log10(values))
    significant = magnitude >= np.log10(float(threshold))

    fraction_significant = float(np.mean(significant))
    median_magnitude = float(np.median(magnitude))
    if np.any(significant):
        fraction_enhancing = float(np.mean(values[significant] > 1.0))
    else:
        fraction_enhancing = 0.0
    max_magnitude = float(np.max(magnitude))

    return np.array([fraction_significant, median_magnitude,
                     fraction_enhancing, max_magnitude], dtype=float)

import numpy as np 

def decompose_measured_epistasis(rate_table) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _parameters(rates):
        """Steady-state observables of the three-step cycle."""
        k_on, k_off = rates[..., 0], rates[..., 1]
        k_chem, k_rev, k_rel = rates[..., 2], rates[..., 3], rates[..., 4]
        partition = k_chem + k_rev + k_rel
        capture = k_chem * k_rel + k_off * k_rev + k_off * k_rel
        if np.any(partition <= 0.0) or np.any(capture <= 0.0):
            raise ValueError("the cycle must carry non-zero steady-state flux")
        return np.stack([k_off / k_on,
                         k_chem * k_rel / partition,
                         capture / (k_on * partition),
                         k_on * k_chem * k_rel / capture], axis=-1)

    table = np.asarray(rate_table, dtype=float)
    if table.shape != (4, 5):
        raise ValueError("rate_table must have shape (4, 5)")
    if not np.all(np.isfinite(table)):
        raise ValueError("rate_table must be finite")
    if np.any(table < 0.0):
        raise ValueError("rate constants must be non-negative")
    if np.any(table[:, 0] <= 0.0):
        raise ValueError("every association rate constant must be > 0")
    if np.any(table[0] <= 0.0):
        raise ValueError("every wild-type rate constant must be > 0 to define fold-changes")

    wild, first, second = table[0], table[1], table[2]

    # Expectation built one level below the observables: free energy
    # additivity multiplies the rate-constant fold-changes step by step.
    additive_rates = wild * (first / wild) * (second / wild)

    parameters = _parameters(table)
    wt_parameters = parameters[0]
    conventional = wt_parameters * (parameters[1] / wt_parameters) * (parameters[2] / wt_parameters)
    additive = _parameters(additive_rates)
    observed = parameters[3]

    total = observed / conventional
    manufactured = additive / conventional
    genuine = observed / additive

    return np.stack([total, manufactured, genuine], axis=-1)

import numpy as np 

def compute_cryptic_epistasis_score(manufactured_epistasis: float,
                                            median_reference_magnitude: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(manufactured_epistasis, (int, float, np.floating, np.integer))
            and not isinstance(manufactured_epistasis, bool)
            and np.isfinite(manufactured_epistasis) and float(manufactured_epistasis) > 0.0):
        raise ValueError("manufactured_epistasis must be a finite number > 0")
    if not (isinstance(median_reference_magnitude, (int, float, np.floating, np.integer))
            and not isinstance(median_reference_magnitude, bool)
            and np.isfinite(median_reference_magnitude)
            and float(median_reference_magnitude) > 0.0):
        raise ValueError("median_reference_magnitude must be a finite number > 0")

    magnitude = abs(np.log10(float(manufactured_epistasis)))

    return float(magnitude / float(median_reference_magnitude))

def run_cryptic_epistasis_pipeline(
    wt_free_energies=(0.0, 10.0, -5.0, 11.0, -9.0, 9.0),
    rate_table=None,
    n_mutants: int = 1000,
    amplitude: float = 2.0,
    seed: int = 314159,
    temperature: float = 298.15,
    threshold: float = 1.5,
    parameter_index: int = 3,
) -> float:
    """Run the complete cryptic-epistasis calculation."""
    import numpy as np

    if rate_table is None:
        rate_table = np.array(
            [
                [1.20e6, 850.0, 340.0, 4.50, 95.0],
                [7.80e5, 1190.0, 6.12, 5.40, 5.70],
                [2.82e6, 595.0, 34.0, 3.82, 14.2],
                [2.75e6, 2500.0, 2.45, 5.97, 2.22],
            ],
            dtype=float,
        )

    if not (
        isinstance(parameter_index, (int, np.integer))
        and not isinstance(parameter_index, bool)
        and 1 <= int(parameter_index) <= 3
    ):
        raise ValueError("parameter_index must be an integer in 1..3")

    index = int(parameter_index)

    # Step 01: wild-type microscopic rates and observables.
    rate_constants_wt, wt_parameters = compute_cycle_observables(
        wt_free_energies,
        temperature,
    )
    if not np.all(np.isfinite(wt_parameters)):
        raise ValueError(
            "the reference reaction coordinate must give finite observables"
        )

    # Steps 02–03: construct the reference single-mutant library.
    mutant_free_energies = simulate_mutant_free_energies(
        wt_free_energies,
        n_mutants,
        amplitude,
        seed,
    )
    rate_folds, _ = compute_variant_fold_changes(
        wt_free_energies,
        mutant_free_energies,
        temperature,
    )

    # Steps 04–05: derive observable expectations internally from microscopic
    # folds, then select and summarize the requested epistasis distribution.
    all_epistasis = compute_double_mutant_epistasis(
        rate_constants_wt,
        rate_folds,
    )
    epistasis = all_epistasis[index]
    summary = summarize_epistasis(epistasis, threshold)

    # Step 06: split measured double-mutant epistasis into components.
    decomposition = decompose_measured_epistasis(rate_table)

    # Step 07: normalize the manufactured component by the library median.
    score = compute_cryptic_epistasis_score(
        float(decomposition[index, 1]),
        float(summary[1]),
    )

    return float(score)
SCICODE_GOLD_EOF
