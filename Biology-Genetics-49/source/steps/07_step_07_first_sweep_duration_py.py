"""
Expected duration of the first selective sweep among several mutant strains, under the size response their establishment counts support.

This step returns the deliverable. Establishment experiments with several mutant

strains of the same ancestral population decide how its census size responds to

the mutant frequency, and that response then sets how likely each strain is to

sweep and how long its sweep takes. If new cells of the strains arise at equal

rates, rarely enough that two strains never segregate together, the strain that

fixes first is drawn in proportion to the fixation probabilities, and the expected

duration of that first sweep combines both quantities of every strain under the

decided response.

Returns
-------
float, the expected duration of the first sweep in generations, as a native Python float, accurate to a relative error below 1e-8
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_sweep_duration(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                         selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> float:
    '''Expected duration of the first selective sweep among several mutant strains, under the size response their establishment counts support.

    A haploid ancestral population of census size ancestral_size reproduces
    as a Wright-Fisher population with offspring-number variance
    offspring_variance, in the diffusion approximation of the earlier steps.
    Strain i has the selective advantage selections[i] and the census size
    mutant_sizes[i] when fixed; while a strain segregates, the census size
    follows one of the five responses of the first step, the same for every
    strain. Each strain was introduced `introductions` times, each time as a
    single cell at frequency 1 / ancestral_size, and fixed fixations[i] times.
    Decide the response by the largest joint binomial log-likelihood of the
    counts, the earliest in the order "arithmetic", "geometric", "harmonic",
    "root_mean_square", "s_shaped" on a tie. Under the decided response, with
    u_i the fixation probability of a single cell of strain i and t_i its
    conditional mean time to fixation, return sum_i u_i t_i / sum_i u_i, the
    expected duration of the first sweep when cells of the strains arise at
    equal rates and two strains never segregate together.

    Parameters
    ----------
    fixations : np.ndarray
        Shape (m,), m >= 1; integer counts of fixations, each from 0 to
        introductions.
    introductions : int
        Number of introductions per strain, an integer >= 1.
    mutant_sizes : np.ndarray
        Shape (m,); census sizes with each strain fixed, within the ranges of
        the earlier steps.
    selections : np.ndarray
        Shape (m,); selective advantages of the strains, from 0 to 0.05.
    ancestral_size : float
        Census size of the ancestral population, a finite number from 10 to
        1e6.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10.

    Returns
    -------
    generations : float
        The expected duration of the first sweep in generations, as a native
        Python float, accurate to a relative error below 1e-8.

    Raises
    ------
    ValueError
        If the inputs violate the conditions of the earlier steps, which
        propagate unchanged: equal-length one-dimensional arrays, integer
        counts from 0 to introductions, the ranges of the sizes, selections and
        offspring variance, selection * max(ancestral_size, mutant_size) /
        offspring_variance at most 200, and fixation probabilities strictly
        between 0 and 1.
    '''
    return generations  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_first_sweep_duration(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                                 selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> float:
    log_likelihoods = _oracle_response_log_likelihoods(fixations, introductions, mutant_sizes, selections,
                                                       ancestral_size, offspring_variance)
    decided = _response_names()[int(np.argmax(log_likelihoods))]       # the first maximum wins a tie
    table = _oracle_strain_sweep_table(mutant_sizes, selections, ancestral_size, offspring_variance, decided)
    return float(np.dot(table[:, 0], table[:, 1]) / np.sum(table[:, 0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: three strains of an 800-cell ancestral population, 200,000 introductions each ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([888, 771, 809])\n"
                      "sizes = np.array([12800.0, 200.0, 3200.0])\n"
                      "selections = np.array([0.002, 0.004, 0.003])\n"),
            "call": "first_sweep_duration(fixations, 200000, sizes, selections, 800.0, 2.0)",
            "gold_call": "_oracle_first_sweep_duration(fixations, 200000, sizes, selections, 800.0, 2.0)",
            "tol": 1e-7,
        },
        # --- boundary: neutral strains, so every response fits equally and the first listed is decided ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([3, 2, 4])\n"
                      "sizes = np.array([1600.0, 200.0, 800.0])\n"
                      "selections = np.array([0.0, 0.0, 0.0])\n"),
            "call": "first_sweep_duration(fixations, 1500, sizes, selections, 400.0, 1.0)",
            "gold_call": "_oracle_first_sweep_duration(fixations, 1500, sizes, selections, 400.0, 1.0)",
            "tol": 1e-7,
        },
        # --- edge: two strains whose counts favour the harmonic response ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([248, 225])\n"
                      "sizes = np.array([100.0, 6000.0])\n"
                      "selections = np.array([0.006, 0.002])\n"),
            "call": "first_sweep_duration(fixations, 50000, sizes, selections, 500.0, 1.5)",
            "gold_call": "_oracle_first_sweep_duration(fixations, 50000, sizes, selections, 500.0, 1.5)",
            "tol": 1e-7,
        },
    ]
