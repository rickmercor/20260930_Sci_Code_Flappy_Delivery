"""
Fixation probability and conditional mean fixation time of a single cell of each of several mutant strains, under one size response.

Once the form of the size response is fixed, each mutant strain that could arise in

the ancestral population has two numbers that summarise its sweep: how likely a

single introduced cell is to take over, and how long the takeover lasts when it

happens. Strains that raise the census size, and strains that lower it, trade these

two numbers differently. This step tabulates both for a set of strains under one

response, each strain starting from a single cell of the ancestral population.

Returns
-------
np.ndarray of shape (m, 2), the fixation probability and the conditional mean fixation time in generations of a single cell of each strain
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strain_sweep_table(mutant_sizes: "np.ndarray", selections: "np.ndarray", ancestral_size: float,
                       offspring_variance: float, response: str) -> "np.ndarray":
    '''Fixation probability and conditional mean fixation time of a single cell of each of several mutant strains, under one size response.

    Every strain is introduced separately, as a single cell at frequency
    1 / ancestral_size, into the ancestral population of census size
    ancestral_size, with the offspring-number variance offspring_variance and
    the diffusion approximation of the previous steps. Strain i has the census
    size mutant_sizes[i] when fixed and the selective advantage selections[i],
    and the census size follows `response`.

    Parameters
    ----------
    mutant_sizes : np.ndarray
        Shape (m,), m >= 1; census sizes with each strain fixed, within the
        ranges of the previous steps.
    selections : np.ndarray
        Shape (m,); selective advantages of the strains, from 0 to 0.05.
    ancestral_size : float
        Census size of the ancestral population, a finite number from 10 to
        1e6.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped".

    Returns
    -------
    sweep_table : np.ndarray
        Shape (m, 2). sweep_table[i, 0] is the fixation probability of a
        single cell of strain i; sweep_table[i, 1] is its conditional mean
        time to fixation in generations. Entries accurate to a relative error
        below 1e-8.

    Raises
    ------
    ValueError
        If mutant_sizes and selections are not one-dimensional numeric arrays
        of equal length >= 1, or if ancestral_size is not a finite number from
        10 to 1e6. Conditions raised by the earlier steps propagate unchanged.
    '''
    return sweep_table  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_strain_sweep_table(mutant_sizes: "np.ndarray", selections: "np.ndarray", ancestral_size: float,
                               offspring_variance: float, response: str) -> "np.ndarray":
    sizes = None
    try:
        sizes = np.asarray(mutant_sizes, dtype=float)
        advantages = np.asarray(selections, dtype=float)
    except (TypeError, ValueError):
        sizes = None
    if sizes is None or sizes.ndim != 1 or sizes.size < 1 or advantages.ndim != 1 or advantages.size != sizes.size \
            or not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(advantages)):
        raise ValueError("mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
    na = _check_size("ancestral_size", ancestral_size, 10.0)
    if na > 1e6:
        raise ValueError("ancestral_size must be a finite number from 10 to 1e6")
    table = np.zeros((sizes.size, 2))
    for i, (size, advantage) in enumerate(zip(sizes, advantages)):
        table[i, 0] = _oracle_fixation_probability(1.0 / na, float(advantage), na, float(size), response, offspring_variance)
        table[i, 1] = _oracle_conditional_fixation_time(1.0 / na, float(advantage), na, float(size), response,
                                                        offspring_variance)
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: three strains under the S-shaped response ---
        {
            "setup": ("import numpy as np\n"
                      "sizes = np.array([12800.0, 200.0, 3200.0])\n"
                      "selections = np.array([0.002, 0.004, 0.003])\n"),
            "call": "strain_sweep_table(sizes, selections, 800.0, 2.0, 's_shaped')",
            "gold_call": "_oracle_strain_sweep_table(sizes, selections, 800.0, 2.0, 's_shaped')",
            "tol": 1e-7,
        },
        # --- boundary: one strain that leaves the census size unchanged ---
        {
            "setup": ("import numpy as np\n"
                      "sizes = np.array([700.0])\n"
                      "selections = np.array([0.005])\n"),
            "call": "strain_sweep_table(sizes, selections, 700.0, 1.0, 'geometric')",
            "gold_call": "_oracle_strain_sweep_table(sizes, selections, 700.0, 1.0, 'geometric')",
            "tol": 1e-7,
        },
        # --- edge: a five-fold drop and a ten-fold boost under the root-mean-square response ---
        {
            "setup": ("import numpy as np\n"
                      "sizes = np.array([80.0, 4000.0])\n"
                      "selections = np.array([0.006, 0.0015])\n"),
            "call": "strain_sweep_table(sizes, selections, 400.0, 1.5, 'root_mean_square')",
            "gold_call": "_oracle_strain_sweep_table(sizes, selections, 400.0, 1.5, 'root_mean_square')",
            "tol": 1e-7,
        },
    ]
