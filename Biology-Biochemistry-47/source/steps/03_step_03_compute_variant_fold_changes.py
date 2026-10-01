"""
Convert each variant's perturbed reaction coordinate into its fold-changes in the five microscopic rate constants and in the four steady-state observables of the cycle.

The quantity that propagates through the rest of the analysis is not a variant's absolute rate constant but its fold-change relative to the wild type, because that is the form in which experimental mutational effects are reported and combined. Both outputs of this step are fold-changes in that sense, the variant's value divided by the wild type's, one set over the five microscopic rate constants and one set over the four steady state observables, and both are taken on the same reaction coordinate and at the same temperature.




The wild type enters twice and in the same way each time. Its rate constants are the ones its own reaction coordinate produces under the conventions of the first step of this task, and its steady state observables are the ones those rate constants produce, so no separate wild type measurement is supplied or needed here. A variant is a full vector of shifts over the reaction coordinate rather than a single scalar, since substitutions are known empirically to perturb several states of the catalytic cycle at once rather than one.

Returns
-------
tuple of two np.ndarray: fold-changes of shape (n_mutants, 5) in the microscopic rate constants and of shape (n_mutants, 4) in the steady-state observables.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_variant_fold_changes(wt_free_energies, mutant_free_energies,
                                 temperature: float = 298.15):
    """Convert perturbed reaction coordinates into fold-changes at both levels.

    Parameters
    ----------
    wt_free_energies : array_like
        Six wild-type sub-state free energies in kcal/mol, in reaction
        coordinate order.
    mutant_free_energies : array_like
        Array of shape (n_mutants, 6) holding the perturbed sub-state free
        energies of each variant in kcal/mol, in the same order.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).

    Returns
    -------
    rate_folds : np.ndarray
        Array of shape (n_mutants, 5) holding each variant's fold-change in
        the association, dissociation, forward chemical, reverse chemical and
        release rate constants, in that order.
    parameter_folds : np.ndarray
        Array of shape (n_mutants, 4) holding each variant's fold-change in
        the dissociation constant, turnover number, Michaelis constant and
        specificity constant, in that order.

    Raises
    ------
    ValueError
        If either free-energy array has the wrong shape or holds a non-finite
        entry, if temperature is not a finite number greater than zero, or if
        the wild-type cycle carries no steady-state flux.
    """
    return rate_folds, parameter_folds  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_variant_fold_changes(wt_free_energies, mutant_free_energies,
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: F401  (keeps this field self-contained)

def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: rate-constant fold-changes of a small library (normal scenario) ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
mutant_free_energies = np.array([[0.3, 10.8, -5.4, 11.2, -8.6, 9.4],
                                 [-0.5, 9.1, -4.2, 12.3, -9.7, 8.2],
                                 [0.9, 11.5, -6.1, 10.4, -8.1, 10.3]])
""",
            "call": "compute_variant_fold_changes(wt_free_energies, mutant_free_energies)[0]",
            "gold_call": "_oracle_compute_variant_fold_changes(wt_free_energies, mutant_free_energies)[0]",
        },
        # --- Valid: the steady-state parameter fold-changes of the same library ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
mutant_free_energies = np.array([[0.3, 10.8, -5.4, 11.2, -8.6, 9.4],
                                 [-0.5, 9.1, -4.2, 12.3, -9.7, 8.2],
                                 [0.9, 11.5, -6.1, 10.4, -8.1, 10.3]])
""",
            "call": "compute_variant_fold_changes(wt_free_energies, mutant_free_energies)[1]",
            "gold_call": "_oracle_compute_variant_fold_changes(wt_free_energies, mutant_free_energies)[1]",
        },
        # --- Valid: a different coordinate and temperature, both outputs digested together ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 9.0, -3.5, 13.0, -6.0, 11.5]
mutant_free_energies = np.array([[1.1, 8.2, -3.9, 13.8, -5.2, 12.4],
                                 [-0.7, 9.9, -2.6, 12.1, -6.8, 10.7]])
def digest(fn):
    rate_folds, parameter_folds = fn(wt_free_energies, mutant_free_energies, 310.15)
    return np.array([float(rate_folds.size), float(parameter_folds.size),
                     float(np.sum(np.log10(rate_folds))),
                     float(np.sum(np.log10(parameter_folds))),
                     float(np.sum(parameter_folds) * 1e3)])
""",
            "call": "digest(compute_variant_fold_changes)",
            "gold_call": "digest(_oracle_compute_variant_fold_changes)",
        },
        # --- Boundary: a variant identical to the wild type, so every fold-change is one ---
        {
            "setup": """import numpy as np
wt_free_energies = np.array([0.0, 10.0, -5.0, 11.0, -9.0, 9.0])
mutant_free_energies = np.array([[0.0, 10.0, -5.0, 11.0, -9.0, 9.0],
                                 [2.0, 12.0, -3.0, 13.0, -7.0, 11.0]])
def digest(fn):
    rate_folds, parameter_folds = fn(wt_free_energies, mutant_free_energies)
    return np.array([float(np.sum(np.abs(np.log10(rate_folds))) * 1e6),
                     float(np.sum(np.abs(np.log10(parameter_folds))) * 1e6),
                     float(np.sum(parameter_folds))])
""",
            "call": "digest(compute_variant_fold_changes)",
            "gold_call": "digest(_oracle_compute_variant_fold_changes)",
        },
        # --- Edge: strongly release-limited wild type, so chemical perturbations barely register ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 6.0, -9.0, 16.0]
mutant_free_energies = np.array([[0.0, 10.0, -5.0, 8.5, -9.0, 16.0],
                                 [0.0, 10.0, -5.0, 6.0, -9.0, 14.5]])
def digest(fn):
    rate_folds, parameter_folds = fn(wt_free_energies, mutant_free_energies)
    return np.array([float(np.sum(np.log10(rate_folds))),
                     float(np.sum(np.log10(parameter_folds))),
                     float(parameter_folds[0, 1] * 1e6), float(parameter_folds[1, 1] * 1e3)])
""",
            "call": "digest(compute_variant_fold_changes)",
            "gold_call": "digest(_oracle_compute_variant_fold_changes)",
        },
        # --- Invalid: variant array with the wrong trailing dimension ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_variant_fold_changes([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], np.zeros((4, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_variant_fold_changes([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], np.zeros((4, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite entry in the variant array ---
        {
            "setup": """import numpy as np
bad = np.zeros((2, 6))
bad[1, 3] = np.nan
def run_model():
    try:
        compute_variant_fold_changes([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_variant_fold_changes([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], bad)
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
