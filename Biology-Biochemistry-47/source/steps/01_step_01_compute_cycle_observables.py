"""
Compute the microscopic rate constants and steady-state observables of a reversible three-step enzyme cycle from its six reaction-coordinate free energies.

The reaction coordinate is ordered as free enzyme plus substrate, binding transition state, enzyme–substrate complex, chemical transition state, enzyme–product complex, and product-release transition state. Convert each barrier into its corresponding microscopic rate constant using transition-state theory at the specified temperature and the common prefactor (k_B T/h).

Do not treat the steady-state observables as independent textbook expressions. Derive them from the steady-state balance of the coupled enzyme–substrate and enzyme–product occupancies for the full reversible cycle
[
\mathrm{E}+\mathrm{S}\rightleftharpoons\mathrm{ES}\rightleftharpoons\mathrm{EP}\rightarrow\mathrm{E}+\mathrm{P}.
]
Use that balance to obtain, in order, the dissociation constant (K_D), turnover number (k_{\mathrm{cat}}), Michaelis constant (K_M), and catalytic efficiency (k_{\mathrm{cat}}/K_M). The result must retain both the reverse chemical step and finite product release; do not substitute a two-step Michaelis–Menten approximation.

The function accepts either one reaction coordinate or a batch with a trailing axis of length six. It returns the five microscopic rate constants in the order association, dissociation, forward chemical, reverse chemical, release, followed by the four steady-state observables in the order (K_D), (k_{\mathrm{cat}}), (K_M), and (k_{\mathrm{cat}}/K_M).

An enzyme catalytic cycle is specified completely by the free energy of every species it visits. The mechanism treated here is the one in which substrate binds reversibly, is chemically transformed reversibly on the enzyme, and is then released irreversibly, so the reaction coordinate carries three ground states, namely free enzyme plus substrate, the enzyme substrate complex and the enzyme product complex, separated by three transition states, namely the binding transition state, the chemical transition state and the product release transition state. The reaction coordinate is listed in that alternating order throughout this task.




Two conventions fix the mapping onto rate constants. Barrier crossings are converted by transition state theory at the stated temperature, and the same universal frequency factor is applied to every elementary step, the association step included. That is a deliberate idealisation, since it allows the second order association rate constant to exceed the diffusion limit by orders of magnitude, but it keeps the mapping a single uniform rule and it does not affect quantities in which the association rate constant cancels. Free energies are in kilocalories per mole, the association rate constant is in units of one over molar second and the other four rate constants in units of one over second.




What an experimentalist measures is not a microscopic rate constant but a steady state observable, and the four reported here are the dissociation constant of the enzyme substrate complex, the turnover number, the Michaelis constant and the specificity constant, each defined for this three step mechanism rather than for the two step Michaelis Menten scheme it reduces to when the reverse chemical step vanishes and release becomes instantaneous. How each of them depends on the underlying rate constants is what governs everything downstream, so the map has to be the steady state solution of this cycle and not a textbook shortcut.

Returns
-------
tuple of two np.ndarray: the five microscopic rate constants and four full-cycle steady-state observables for each supplied reaction coordinate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_cycle_observables(free_energies, temperature: float = 298.15):
    """Map reaction-coordinate free energies to full-cycle observables.

    Convert the three transition-state barriers into the association,
    dissociation, forward-chemical, reverse-chemical, and product-release
    rate constants. Then solve the steady-state balance of the coupled ES and
    EP occupancies for the complete reversible three-step cycle before
    calculating its kinetic observables.

    Parameters
    ----------
    free_energies : array_like
        Array whose trailing axis has length 6, holding the Gibbs free
        energies in kcal/mol of free enzyme plus substrate, binding transition
        state, ES, chemical transition state, EP, and release transition
        state, in that order. Leading axes are broadcast over.
    temperature : float
        Absolute temperature in kelvin; must be finite and greater than zero.

    Returns
    -------
    rate_constants : np.ndarray
        Array with the input leading shape and trailing length 5, ordered as
        association, dissociation, forward chemical, reverse chemical, and
        release rate constants.
    parameters : np.ndarray
        Array with the input leading shape and trailing length 4, ordered as
        \(K_D\), \(k_{\mathrm{cat}}\), \(K_M\), and
        \(k_{\mathrm{cat}}/K_M\), each obtained from the full-cycle
        steady-state balance.

    Raises
    ------
    ValueError
        If free_energies lacks a finite trailing axis of length 6,
        temperature is invalid, or the resulting cycle has undefined
        steady-state observables.
    """
    return rate_constants, parameters

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_cycle_observables(free_energies, temperature: float = 298.15):
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

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
import numpy as np  # noqa: F401  (keeps this field self-contained)

def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: rate constants of the reference reaction coordinate (normal scenario) ---
        {
            "setup": """import numpy as np
free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
""",
            "call": "compute_cycle_observables(free_energies)[0]",
            "gold_call": "_oracle_compute_cycle_observables(free_energies)[0]",
        },
        # --- Valid: steady-state observables of the same coordinate ---
        {
            "setup": """import numpy as np
free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
""",
            "call": "compute_cycle_observables(free_energies)[1]",
            "gold_call": "_oracle_compute_cycle_observables(free_energies)[1]",
        },
        # --- Valid: a stack of coordinates at a lower temperature, exercising the broadcast ---
        {
            "setup": """import numpy as np
free_energies = np.array([[0.5, 12.0, -4.0, 13.5, -7.5, 10.5],
                          [0.0, 9.0, -3.5, 13.0, -6.0, 11.5]])
temperature = 277.15
""",
            "call": "compute_cycle_observables(free_energies, temperature)[1]",
            "gold_call": "_oracle_compute_cycle_observables(free_energies, temperature)[1]",
        },
        # --- Boundary: flat reaction coordinate, every barrier equal to zero ---
        {
            "setup": """import numpy as np
def digest(fn):
    rates, params = fn(np.zeros(6, dtype=float))
    return np.array([float(rates.size), float(params.size),
                     float(np.sum(np.log10(rates))), float(np.sum(np.log10(params)))])
""",
            "call": "digest(compute_cycle_observables)",
            "gold_call": "digest(_oracle_compute_cycle_observables)",
        },
        # --- Edge: very deep product well, so the reverse chemical step is negligible ---
        {
            "setup": """import numpy as np
def digest(fn):
    rates, params = fn([0.0, 8.0, -3.0, 14.0, -22.0, 6.0], 310.15)
    return np.array([float(np.sum(np.log10(rates))), float(np.sum(np.log10(params))),
                     float(params[1] * 1e6), float(params[3] * 1e-3)])
""",
            "call": "digest(compute_cycle_observables)",
            "gold_call": "digest(_oracle_compute_cycle_observables)",
        },
        # --- Invalid: wrong number of sub-states ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cycle_observables([0.0, 10.0, -5.0, 11.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cycle_observables([0.0, 10.0, -5.0, 11.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cycle_observables([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cycle_observables([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], 0.0)
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
