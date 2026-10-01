"""
Predicted maternal half-sibling pairs per million comparisons between two target classes, after calibrating the score from kin.

A close-kin study that has no animal of known age beyond its newborns can still calibrate its age score: the

newborns fix the score's level at birth and its error, and the half-sibling counts of several contrasting

classes of comparisons fix the rest of the calibration together with the population's mortality, abundance and

within-cohort factor. The calibrated model then predicts how often half-siblings should be found in any other

class of comparisons, for example between old animals of an early survey and young animals of a later one,

which is what a follow-up study would be designed around.

Returns
-------
float, one million times the predicted probability that a comparison of the two target classes is a maternal half-sibling pair, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kin_rate_per_million(newborn_scores: "np.ndarray", contrast_classes: "np.ndarray", contrast_sizes: "np.ndarray", contrast_counts: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, target_classes: "np.ndarray", max_age: int) -> float:
    '''Predicted maternal half-sibling pairs per million comparisons between two target classes, after calibrating the score from kin.

    Scores are gamma distributed with mean alpha + beta * a at age a and
    coefficient of variation sqrt(dispersion). The pipeline:
    1. alpha and dispersion are the maximum-likelihood values from the
       newborns' scores, as in the first step;
    2. beta, mortality, the number of adult females in reference_year and
       the within-cohort factor are the values that make the observed
       half-sibling counts of the contrast class pairs most likely, as in
       the sixth step;
    3. the result is one million times the probability that an animal of
       target_classes[0] and a different animal of target_classes[1] have
       the same mother under those values, as in the fifth step.
    Classes are given as [year, lower, upper, min_age]; births and adult
    females grow at the rate growth; an animal dies at the annual rate
    juvenile_mortality until it matures at maturity_age and at the fitted
    mortality from then on; each adult female breeds once every
    breeding_period years; and each survey samples at random among the
    animals alive in its year that its gear retains. Ages run over the whole years 0, ..., max_age.

    Parameters
    ----------
    newborn_scores : np.ndarray
        Shape (n,) with n >= 2: the unbinned scores of animals of age 0, each
        positive and finite.
    contrast_classes : np.ndarray
        Shape (K, 2, 4) with K >= 1: the two classes of each contrast pair.
    contrast_sizes : np.ndarray
        Shape (K, 2): the number of animals in each class of each pair.
    contrast_counts : np.ndarray
        Shape (K,): the half-sibling pairs found in each pair's comparisons.
    growth : float
        Annual growth rate of births and of adult females, finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    reference_year : int
        The year whose adult females the fitted abundance counts.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    min_mortality : float
        The lowest annual mortality allowed, positive and finite.
    target_classes : np.ndarray
        Shape (2, 4): the two classes whose comparisons are predicted.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    float
        One million times the predicted probability that a comparison of the
        two target classes is a maternal half-sibling pair, as a native
        Python float.

    Raises
    ------
    ValueError
        If target_classes does not have shape (2, 4), or if any argument
        fails the conditions of the step it feeds: the newborn scores, the
        contrast classes, sizes and counts, growth, juvenile_mortality,
        maturity_age, reference_year, breeding_period, min_mortality and
        max_age.
    '''
    return x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kin_rate_per_million(newborn_scores: "np.ndarray", contrast_classes: "np.ndarray", contrast_sizes: "np.ndarray", contrast_counts: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, target_classes: "np.ndarray", max_age: int) -> float:
    targets = np.asarray(target_classes, dtype=float)
    if targets.shape != (2, 4):
        raise ValueError("target_classes must have shape (2, 4)")
    alpha, dispersion = _oracle_newborn_calibration(newborn_scores)
    beta, mortality, females, factor = _oracle_contrast_fit(contrast_classes, contrast_sizes, contrast_counts,
                                                            np.array([alpha, dispersion]), growth, juvenile_mortality,
                                                            maturity_age, reference_year, breeding_period,
                                                            min_mortality, max_age)
    probability = _oracle_class_pair_mhsp_probability(targets[0], targets[1], np.array([alpha, beta, dispersion]),
                                                      np.array([mortality, females, growth, factor,
                                                                juvenile_mortality, maturity_age]),
                                                      reference_year, breeding_period, max_age)
    return 1.0e6 * probability

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped surveys, old animals of 2016 against young animals of 2024 ---
        {
            "setup": "import numpy as np\n"
                     "neo = np.array([1.59, 1.54, 1.51, 1.6, 1.5, 1.69, 1.61, 1.59, 1.35, 1.26, 1.26, 1.36])\n"
                     "A16 = [2016.0, 3.0, 5.0, 0.0]\nA20 = [2020.0, 3.0, 5.0, 0.0]\n"
                     "A24 = [2024.0, 3.0, 5.0, 4.0]\nB16 = [2016.0, 9.0, 11.0, 0.0]\n"
                     "B20 = [2020.0, 9.0, 11.0, 0.0]\n"
                     "classes = np.array([[A16, A16], [A16, A20], [A16, A24], [A16, B16],\n"
                     "                    [A20, A20], [A20, A24], [B16, B16], [B16, B20]])\n"
                     "sizes = np.array([[520, 520], [520, 480], [520, 300], [520, 150],\n"
                     "                  [480, 480], [480, 300], [150, 150], [150, 140]])\n"
                     "counts = np.array([108, 71, 21, 14, 84, 39, 12, 9])\n"
                     "targets = np.array([[2016.0, 17.0, 19.0, 0.0], [2024.0, 7.0, 9.0, 4.0]])\n",
            "call": "kin_rate_per_million(neo, classes, sizes, counts, 0.05, 0.3, 8, 2024, 3, 0.05, targets, 120)",
            "gold_call": "_oracle_kin_rate_per_million(neo, classes, sizes, counts, 0.05, 0.3, 8, 2024, 3, 0.05, targets, 120)",
            "tol": 1e-6,
        },
        # --- boundary: the same six class pairs, predicting two classes the surveys do not pair ---
        {
            "setup": "import numpy as np\n"
                     "neo = np.array([1.5, 1.42, 1.61, 1.55, 1.47, 1.53, 1.6, 1.38])\n"
                     "A12 = [2012.0, 3.0, 5.0, 0.0]\n"
                     "A16 = [2016.0, 3.0, 5.0, 0.0]\n"
                     "B12 = [2012.0, 9.0, 11.0, 0.0]\n"
                     "B16 = [2016.0, 9.0, 11.0, 2.0]\n"
                     "classes = np.array([[A12, A12], [A12, A16], [A12, B12], [A12, B16], [A16, A16], [B12, B12]])\n"
                     "sizes = np.array([[450, 450], [450, 380], [450, 160], [450, 120], [380, 380], [160, 160]])\n"
                     "counts = np.array([63, 49, 11, 19, 43, 9])\n"
                     "targets = np.array([[2012.0, 15.0, 17.0, 0.0], [2016.0, 9.0, 11.0, 2.0]])\n",
            "call": "kin_rate_per_million(neo, classes, sizes, counts, 0.03, 0.42, 5, 2016, 2, 0.05, targets, 100)",
            "gold_call": "_oracle_kin_rate_per_million(neo, classes, sizes, counts, 0.03, 0.42, 5, 2016, 2, 0.05, targets, 100)",
            "tol": 1e-6,
        },
        # --- edge: the same seven class pairs, a four-year cycle and a shrinking population ---
        {
            "setup": "import numpy as np\n"
                     "neo = np.array([1.31, 1.4, 1.36, 1.28, 1.44, 1.33, 1.39, 1.3])\n"
                     "A18 = [2018.0, 3.0, 5.0, 0.0]\n"
                     "A22 = [2022.0, 3.0, 5.0, 3.0]\n"
                     "B18 = [2018.0, 7.0, 9.0, 0.0]\n"
                     "C18 = [2018.0, 11.0, 13.0, 0.0]\n"
                     "classes = np.array([[A18, A18], [A18, A22], [A18, B18], [A18, C18], [A22, A22], [B18, B18],\n"
                     "                    [B18, A22]])\n"
                     "sizes = np.array([[520, 520], [520, 340], [520, 300], [520, 220], [340, 340], [300, 300],\n"
                     "                  [300, 340]])\n"
                     "counts = np.array([217, 48, 44, 9, 152, 62, 19])\n"
                     "targets = np.array([[2018.0, 11.0, 13.0, 0.0], [2022.0, 7.0, 9.0, 3.0]])\n",
            "call": "kin_rate_per_million(neo, classes, sizes, counts, -0.015, 0.28, 9, 2022, 4, 0.05, targets, 110)",
            "gold_call": "_oracle_kin_rate_per_million(neo, classes, sizes, counts, -0.015, 0.28, 9, 2022, 4, 0.05, targets, 110)",
            "tol": 1e-6,
        },
    ]
