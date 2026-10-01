"""
Integrate the complete mechanistic workflow for both catalytic-cycle representations and return the deterministic amplification ratio of significant catalytic-efficiency epistasis. The integration must use the same wild-type energetic reference, mutation perturbations, pair construction, significance threshold, and pair ordering for both mechanisms so that their prevalence comparison isolates the effect of kinetic-cycle complexity.

The paper's computational argument links four levels of description: thermodynamic state energies, microscopic transition rates, emergent kinetic parameters, and apparent mutational epistasis. The final comparison requires carrying the additive microscopic perturbation model through both the simple and extended catalytic cycles before applying the same macroscopic null and significance criterion. This end-to-end construction is important because non-specific epistasis is not introduced explicitly at the energetic level; it emerges from the nonlinear mapping between microscopic rate constants and kinetic observables. The extended mechanism additionally demonstrates why parameters such as kcat can acquire epistasis when they depend on multiple microscopic transitions, whereas KD retains its simpler relationship to binding rates.

Returns
-------
float, the deterministic dimensionless ratio of significant catalytic-efficiency epistasis prevalence for the extended kinetic mechanism relative to the simple kinetic mechanism.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_kinetic_complexity_ratio(mutation_deltas: "np.ndarray") -> float:
    """Run both kinetic mechanisms over all 120 mutation pairs.

    Parameters
    ----------
    mutation_deltas : np.ndarray
        Six-component mutation free-energy perturbation matrix for the
        16 substitutions, with shape (16,6).

    Returns
    -------
    float
        Complete-mechanism significant-pair prevalence divided by the
        four-state significant-pair prevalence.

    Raises
    ------
    ValueError
        If mutation_deltas does not have shape (16,6) or contains
        non-finite values.
    """
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_kinetic_complexity_ratio(mutation_deltas: "np.ndarray") -> float:
    import numpy as np

    d = np.asarray(mutation_deltas, dtype=float)

    if d.shape != (16, 6) or not np.all(np.isfinite(d)):
        raise ValueError("mutation_deltas must have shape (16,6)")

    T = 298.15
    R = 1.98720425864083e-3
    kB = 1.380649e-23
    h = 6.62607015e-34

    wt4 = np.array([0.0, 10.0, -5.0, 11.0])
    wt6 = np.array([0.0, 10.0, -5.0, 11.0, -9.0, 9.0])

    pairs = np.array(
        [(i, j) for i in range(16) for j in range(i + 1, 16)],
        dtype=int
    )

    def _run(wt, complex_model):
        deltas = d if complex_model else d[:, :4]

        single_E = np.vstack(
            (wt, wt[None, :] + deltas)
        )

        single_rates = _oracle_compute_tst_rates(
            single_E,
            T,
            R,
            kB,
            h,
            complex_model
        )

        single_obs = _oracle_compute_kinetic_observables(
            single_rates,
            complex_model
        )

        folds = _oracle_compute_single_fold_changes(
            single_obs,
            0
        )

        double_E = _oracle_build_additive_double_energies(
            wt,
            deltas,
            pairs
        )

        double_rates = _oracle_compute_tst_rates(
            double_E,
            T,
            R,
            kB,
            h,
            complex_model
        )

        double_obs = _oracle_compute_kinetic_observables(
            double_rates,
            complex_model
        )

        interactions = _oracle_compute_pair_interactions(
            folds,
            single_obs[0, 2],
            double_obs,
            pairs + 1,
            0,
            2
        )

        return _oracle_classify_epistasis(
            interactions,
            1.5
        )

    simple = _run(wt4, False)
    complete = _run(wt6, True)

    return _oracle_compute_complexity_ratio(
        simple,
        complete
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nd=np.array([[1.579, -1.372, -1.539, -0.468, -0.142, -0.998], [-0.242, -1.868, 1.189, 1.473, 0.964, -0.256], [1.032, -1.408, -1.411, -1.093, -0.442, -0.024], [-0.274, -1.587, -0.994, 1.752, 0.979, -1.726], [1.318, -0.099, -1.016, 1.38, 0.361, 1.092], [-1.411, -0.899, -1.667, -1.419, 0.925, 0.687], [1.747, -0.192, 1.025, 0.405, -0.963, 1.882], [-1.753, -0.155, -0.072, -0.165, 1.732, -1.112], [-1.03, -0.036, 1.174, -0.695, -0.806, 1.161], [-0.837, 0.312, -0.659, 1.69, -0.604, -0.511], [1.457, 1.262, 0.716, -1.579, 1.185, -0.562], [-1.421, -1.247, 1.23, -1.314, -1.224, 1.638], [1.579, -0.988, 0.33, 1.911, 0.736, -0.879], [0.206, -1.158, -1.947, -1.37, 1.933, 1.505], [1.276, -1.811, -0.147, 0.305, 0.54, 1.998], [-0.607, 0.449, -1.039, 1.6, -1.905, -0.369]],dtype=float)\n", "call": "resolve_kinetic_complexity_ratio(d)", "gold_call": "_oracle_resolve_kinetic_complexity_ratio(d)"},
        {"setup": "import numpy as np\nd=np.array([[1.579, -1.372, -1.539, -0.468, -0.142, -0.998], [-0.242, -1.868, 1.189, 1.473, 0.964, -0.256], [1.032, -1.408, -1.411, -1.093, -0.442, -0.024], [-0.274, -1.587, -0.994, 1.752, 0.979, -1.726], [1.318, -0.099, -1.016, 1.38, 0.361, 1.092], [-1.411, -0.899, -1.667, -1.419, 0.925, 0.687], [1.747, -0.192, 1.025, 0.405, -0.963, 1.882], [-1.753, -0.155, -0.072, -0.165, 1.732, -1.112], [-1.03, -0.036, 1.174, -0.695, -0.806, 1.161], [-0.837, 0.312, -0.659, 1.69, -0.604, -0.511], [1.457, 1.262, 0.716, -1.579, 1.185, -0.562], [-1.421, -1.247, 1.23, -1.314, -1.224, 1.638], [1.579, -0.988, 0.33, 1.911, 0.736, -0.879], [0.206, -1.158, -1.947, -1.37, 1.933, 1.505], [1.276, -1.811, -0.147, 0.305, 0.54, 1.998], [-0.607, 0.449, -1.039, 1.6, -1.905, -0.369]],dtype=float)\nperm=np.array([7, 0, 12, 3, 15, 1, 10, 5, 14, 8, 2, 13, 6, 11, 4, 9],dtype=int)\n", "call": "resolve_kinetic_complexity_ratio(d[perm])", "gold_call": "_oracle_resolve_kinetic_complexity_ratio(d[perm])"},
        {"setup": "import numpy as np\nd=0.5*np.array([[1.579, -1.372, -1.539, -0.468, -0.142, -0.998], [-0.242, -1.868, 1.189, 1.473, 0.964, -0.256], [1.032, -1.408, -1.411, -1.093, -0.442, -0.024], [-0.274, -1.587, -0.994, 1.752, 0.979, -1.726], [1.318, -0.099, -1.016, 1.38, 0.361, 1.092], [-1.411, -0.899, -1.667, -1.419, 0.925, 0.687], [1.747, -0.192, 1.025, 0.405, -0.963, 1.882], [-1.753, -0.155, -0.072, -0.165, 1.732, -1.112], [-1.03, -0.036, 1.174, -0.695, -0.806, 1.161], [-0.837, 0.312, -0.659, 1.69, -0.604, -0.511], [1.457, 1.262, 0.716, -1.579, 1.185, -0.562], [-1.421, -1.247, 1.23, -1.314, -1.224, 1.638], [1.579, -0.988, 0.33, 1.911, 0.736, -0.879], [0.206, -1.158, -1.947, -1.37, 1.933, 1.505], [1.276, -1.811, -0.147, 0.305, 0.54, 1.998], [-0.607, 0.449, -1.039, 1.6, -1.905, -0.369]],dtype=float)\n", "call": "resolve_kinetic_complexity_ratio(d)", "gold_call": "_oracle_resolve_kinetic_complexity_ratio(d)"},
        {"setup": "import numpy as np\nd=np.zeros((15,6),float)\ndef run_model():\n    try: resolve_kinetic_complexity_ratio(d); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_resolve_kinetic_complexity_ratio(d); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "run_model()", "gold_call": "run_gold()"},
    ]
