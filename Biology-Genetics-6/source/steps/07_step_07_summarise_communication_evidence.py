"""
Put the associational verdict and the posterior inclusion probability on one common odds scale, report the gap between them in orders of magnitude, and express both causal communication coefficients in standard deviation units of pathway activity.

The two analyses being compared return objects of different kinds. One returns a tail probability of a frequentist test statistic, the other a posterior probability that a latent indicator is on. Neither is convertible into the other, and the usual way to make them comparable is the one this comparison adopts: map each verdict onto the unit interval, treating the complement of the tail probability as the associational communication score, and then compare the two scores as odds. The mapping is monotone, so it changes no ordering, and the logarithm of the odds is the natural scale for the comparison because both scores live at the extremes of the unit interval where a difference expressed in probability units is uninformative.




Reading the gap is the point of the exercise. A positive value in base ten logarithms says that the associational analysis carries that many orders of magnitude more evidence for communication than the causal analysis does, and in a cohort constructed with no communication at all the greater part of the gap is manufactured by the latent donor factor that loads on ligand, receptor and pathway alike. The gap is not a defect of the significance threshold, and it does not shrink as donors accumulate: the displacement of the associational ligand coefficient is a function of the confounder loadings and not of cohort size, so its tail probability only falls further as the cohort grows and the gap widens. That is the diagnosis a co-expression screen cannot make about itself.




The standardised effects serve a different purpose, of making estimated effects readable rather than verdicts comparable. Multiplying the main effect by the standard deviation of ligand expression and dividing by the standard deviation of pathway activity expresses it as the change in pathway activity, in its own standard deviation units, produced by a one standard deviation change in ligand expression. The interaction carries two exposures rather than one, so its standardisation multiplies by both exposure standard deviations before dividing by the outcome standard deviation, and it reads as the change in that per standard deviation ligand slope produced by a one standard deviation change in receptor abundance. Both are the form in which effects can be placed side by side across ligand receptor pathway triplets whose genes are measured on entirely different scales, and the form in which an estimate near zero is legible as near zero rather than merely small in whatever units the expression matrix happened to carry.

Returns
-------
np.ndarray of five floats: the gap in base ten logarithms between the associational and causal communication odds, the associational log odds, the causal log odds, the standardised ligand main effect and the standardised interaction effect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def summarise_communication_evidence(naive_summary: np.ndarray,
                                     posterior_inclusion_probability: float,
                                     posterior_ligand_effect: float,
                                     posterior_interaction_effect: float,
                                     ligand_sd: float, receptor_sd: float,
                                     pathway_sd: float) -> np.ndarray:
    '''Compare the associational and causal verdicts on a common odds scale.

    The associational log odds is taken ready made from the second entry of
    naive_summary and is not recomputed here. The causal communication score
    is posterior_inclusion_probability, and both scores are read as odds and
    reported in base ten logarithms. The gap is the associational communication
    odds divided by the causal communication odds, in those same logarithms.
    Both posterior effects are reported in standard deviation units of pathway
    activity.

    Parameters
    ----------
    naive_summary : np.ndarray
        Array of four floats as returned by the associational scoring step,
        whose second entry is the base ten logarithm of the associational
        communication odds.
    posterior_inclusion_probability : float
        Posterior probability that the communication effect vector belongs to
        the slab, strictly between zero and one.
    posterior_ligand_effect : float
        Posterior mean of the causal ligand main effect.
    posterior_interaction_effect : float
        Posterior mean of the causal ligand by receptor interaction effect.
    ligand_sd : float
        Standard deviation of ligand expression across donors, strictly
        positive.
    receptor_sd : float
        Standard deviation of receptor expression across donors, strictly
        positive.
    pathway_sd : float
        Standard deviation of pathway activity across donors, strictly
        positive.

    Returns
    -------
    evidence : np.ndarray
        Array of five floats holding, in order, the gap in base ten logarithms
        between the associational and causal communication odds, the
        associational log odds, the causal log odds, the standardised ligand
        main effect and the standardised interaction effect.

    Raises
    ------
    ValueError
        If naive_summary is not a finite array of four floats, if
        posterior_inclusion_probability is not strictly inside the unit
        interval, if either posterior effect is not finite, or if any of the
        three standard deviations is not a positive finite number.
    '''
    return evidence  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_summarise_communication_evidence(naive_summary: np.ndarray,
                                             posterior_inclusion_probability: float,
                                             posterior_ligand_effect: float,
                                             posterior_interaction_effect: float,
                                             ligand_sd: float, receptor_sd: float,
                                             pathway_sd: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    def _check_real(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    summary = np.asarray(naive_summary, dtype=float).ravel()
    if summary.size != 4:
        raise ValueError("naive_summary must hold exactly four floats")
    if not np.all(np.isfinite(summary)):
        raise ValueError("naive_summary must be finite")
    if isinstance(posterior_inclusion_probability, bool) or not isinstance(
            posterior_inclusion_probability, (int, float, np.integer, np.floating)):
        raise ValueError("posterior_inclusion_probability must be a real number")
    inclusion = float(posterior_inclusion_probability)
    if not (np.isfinite(inclusion) and 0.0 < inclusion < 1.0):
        raise ValueError("posterior_inclusion_probability must lie strictly inside "
                         "the unit interval")
    ligand_effect = _check_real("posterior_ligand_effect", posterior_ligand_effect)
    interaction_effect = _check_real("posterior_interaction_effect",
                                     posterior_interaction_effect)
    ligand_spread = _check_positive("ligand_sd", ligand_sd)
    receptor_spread = _check_positive("receptor_sd", receptor_sd)
    pathway_spread = _check_positive("pathway_sd", pathway_sd)

    # Both verdicts are read as odds, which keeps the comparison finite where a
    # score on the unit interval would saturate.
    naive_log_odds = float(summary[1])
    causal_log_odds = float(np.log10(inclusion / (1.0 - inclusion)))
    gap = naive_log_odds - causal_log_odds

    # The main effect carries one exposure and the interaction carries two, so
    # they are standardised by a different number of exposure spreads.
    standardised_main = ligand_effect * ligand_spread / pathway_spread
    standardised_interaction = (interaction_effect * ligand_spread * receptor_spread
                                / pathway_spread)
    return np.asarray([gap, naive_log_odds, causal_log_odds, standardised_main,
                       standardised_interaction], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a confounded null verdict, where the associational odds are enormous ---
        {
            "setup": """import numpy as np
naive_summary = np.array([12.015896, 5.115466, 0.152915, 0.001757])
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(summarise_communication_evidence(naive_summary, 0.036389, -0.000612, 0.000744, 1.559759, 1.644893, 1.727393))",
            "gold_call": "digest(_oracle_summarise_communication_evidence(naive_summary, 0.036389, -0.000612, 0.000744, 1.559759, 1.644893, 1.727393))",
        },
        # --- Valid: genuine communication, where both analyses agree and the gap is small ---
        {
            "setup": """import numpy as np
naive_summary = np.array([210.5, 40.31, 0.44, 0.31])
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(summarise_communication_evidence(naive_summary, 0.9995, 0.298, 0.331, 1.6, 1.7, 2.1))",
            "gold_call": "digest(_oracle_summarise_communication_evidence(naive_summary, 0.9995, 0.298, 0.331, 1.6, 1.7, 2.1))",
        },
        # --- Boundary: an inclusion probability of one half, where the causal odds vanish ---
        {
            "setup": """import numpy as np
naive_summary = np.array([3.0, 1.25, 0.05, -0.02])
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(summarise_communication_evidence(naive_summary, 0.5, 0.04, -0.02, 1.0, 1.0, 1.0))",
            "gold_call": "digest(_oracle_summarise_communication_evidence(naive_summary, 0.5, 0.04, -0.02, 1.0, 1.0, 1.0))",
        },
        # --- Edge: an inclusion probability far above the associational verdict, so the gap is negative ---
        {
            "setup": """import numpy as np
naive_summary = np.array([1.1, 0.35, 0.02, 0.01])
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(summarise_communication_evidence(naive_summary, 0.999, 0.51, -0.22, 2.4, 0.9, 1.2))",
            "gold_call": "digest(_oracle_summarise_communication_evidence(naive_summary, 0.999, 0.51, -0.22, 2.4, 0.9, 1.2))",
        },
        # --- Invalid: an inclusion probability at zero ---
        {
            "setup": """import numpy as np
naive_summary = np.array([3.0, 1.25, 0.05, -0.02])
def run_model():
    try:
        summarise_communication_evidence(naive_summary, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarise_communication_evidence(naive_summary, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a pathway standard deviation of zero ---
        {
            "setup": """import numpy as np
naive_summary = np.array([3.0, 1.25, 0.05, -0.02])
def run_model():
    try:
        summarise_communication_evidence(naive_summary, 0.3, 0.1, 0.1, 1.0, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarise_communication_evidence(naive_summary, 0.3, 0.1, 0.1, 1.0, 1.0, 0.0)
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
