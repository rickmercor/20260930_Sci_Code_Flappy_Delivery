"""
Joint natural-log likelihood of the fixation counts of several mutant strains under each of the five size responses.

How the size of a population depends on its genetic composition is rarely

measured directly, but it leaves a signature in how often new mutants fix: a strain

that raises the census size weakens drift as it spreads, and one that lowers it

strengthens drift, by amounts that depend on the form of the response. Several

strains introduced into the same ancestral population share that form, so their

fixation counts can be combined into one likelihood for each candidate response.

This step computes the joint log-likelihood of the counts of several strains under

each of the five responses.

Returns
-------
np.ndarray of shape (5,), the joint natural-log likelihood of the strains' fixation counts under the arithmetic, geometric, harmonic, root_mean_square and s_shaped responses, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def response_log_likelihoods(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                             selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> "np.ndarray":
    '''Joint natural-log likelihood of the fixation counts of several mutant strains under each of the five size responses.

    Every strain is introduced separately into the same ancestral population,
    of census size ancestral_size, with the offspring-number variance
    offspring_variance of the previous steps. Strain i has the census size
    mutant_sizes[i] when fixed and the selective advantage selections[i]; each
    of its `introductions` independent introductions starts from a single
    cell, at frequency 1 / ancestral_size, and fixations[i] of them fixed. For
    each response, return the sum over the strains of the log-likelihood of
    the strain's count, with the strain's fixation probability under that
    response from the previous steps.

    Parameters
    ----------
    fixations : np.ndarray
        Shape (m,), m >= 1; integer counts of fixations, each from 0 to
        introductions.
    introductions : int
        Number of introductions per strain, an integer >= 1.
    mutant_sizes : np.ndarray
        Shape (m,); census sizes with each strain fixed, within the ranges of
        the previous steps.
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
    log_likelihoods : np.ndarray
        Shape (5,). Entry j is the joint log-likelihood under the j-th
        response in the order "arithmetic", "geometric", "harmonic",
        "root_mean_square", "s_shaped"; accurate to a relative error below
        1e-9.

    Raises
    ------
    ValueError
        If fixations, mutant_sizes and selections are not one-dimensional
        arrays of equal length >= 1, if fixations holds a value that is not an
        integer from 0 to introductions, if introductions is not an integer
        >= 1, or if ancestral_size is not a finite number from 10 to 1e6.
        Conditions raised by the earlier steps propagate unchanged, including a
        fixation probability that is not strictly between 0 and 1.
    '''
    return log_likelihoods  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_strains(fixations: "np.ndarray", mutant_sizes: "np.ndarray", selections: "np.ndarray") -> tuple:
    arrays = []
    for name, value in (("fixations", fixations), ("mutant_sizes", mutant_sizes), ("selections", selections)):
        try:
            a = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
        arrays.append(a)
    if not arrays[0].size == arrays[1].size == arrays[2].size:
        raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
    return tuple(arrays)


def _oracle_response_log_likelihoods(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                                     selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> "np.ndarray":
    counts, sizes, advantages = _check_strains(fixations, mutant_sizes, selections)
    n = _check_count("introductions", introductions, 1)
    na = _check_size("ancestral_size", ancestral_size, 10.0)
    if na > 1e6:
        raise ValueError("ancestral_size must be a finite number from 10 to 1e6")
    log_likelihoods = np.zeros(len(_response_names()))
    for j, kind in enumerate(_response_names()):
        total = 0.0
        for count, size, advantage in zip(counts, sizes, advantages):
            u = _oracle_fixation_probability(1.0 / na, float(advantage), na, float(size), kind, offspring_variance)
            total += _oracle_introduction_log_likelihood(count, n, u)
        log_likelihoods[j] = total
    return log_likelihoods

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: three strains, a sixteen-fold boost, a four-fold drop and a four-fold boost ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([888, 771, 809])\n"
                      "sizes = np.array([12800.0, 200.0, 3200.0])\n"
                      "selections = np.array([0.002, 0.004, 0.003])\n"),
            "call": "response_log_likelihoods(fixations, 200000, sizes, selections, 800.0, 2.0)",
            "gold_call": "_oracle_response_log_likelihoods(fixations, 200000, sizes, selections, 800.0, 2.0)",
        },
        # --- boundary: one strain that leaves the census size unchanged, so every response gives the same value ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([37])\n"
                      "sizes = np.array([500.0])\n"
                      "selections = np.array([0.004])\n"),
            "call": "response_log_likelihoods(fixations, 10000, sizes, selections, 500.0, 1.0)",
            "gold_call": "_oracle_response_log_likelihoods(fixations, 10000, sizes, selections, 500.0, 1.0)",
        },
        # --- edge: few introductions, a five-fold drop and a thirty-fold boost ---
        {
            "setup": ("import numpy as np\n"
                      "fixations = np.array([12, 40])\n"
                      "sizes = np.array([60.0, 9000.0])\n"
                      "selections = np.array([0.008, 0.001])\n"),
            "call": "response_log_likelihoods(fixations, 2000, sizes, selections, 300.0, 1.2)",
            "gold_call": "_oracle_response_log_likelihoods(fixations, 2000, sizes, selections, 300.0, 1.2)",
        },
    ]
