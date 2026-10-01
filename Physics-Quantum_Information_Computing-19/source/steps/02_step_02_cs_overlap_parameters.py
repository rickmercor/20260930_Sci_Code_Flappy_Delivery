"""
Compute, for one ordered pair of nominal intensity settings, the photon-number-resolved overlap parameter that quantifies how indistinguishable the two settings remain to an adversary once the finite-memory intensity drift of the source is taken into account.

When the actual intensity of a pulse depends on the settings chosen in a few neighbouring rounds, the setting used in a given round is partially imprinted on the intensities of the rounds that follow it, so an adversary who probes those rounds acquires some information about it. The strength of that leak is governed by how much the emitted photon-number distributions belonging to the two settings overlap, accumulated over the rounds within the memory span, and it is this accumulated overlap that later limits how far the detection statistics belonging to different settings may diverge. The leak weakens as the admissible intensity intervals shrink and strengthens as the memory span grows, so the parameter approaches its largest value, one, when the source prepares its nominal intensities exactly.

The returned array is indexed by photon number from zero up to and including the cut-off. All nominal settings that the transmitter can select, together with the probabilities with which it selects them, enter the calculation for every pair.

Returns
-------
overlaps : np.ndarray — Shape (n_cut + 1,). Entry n is the overlap parameter of the pair at photon number n. Every entry is strictly positive, and lies in (0, 1] whenever every nominal setting is at most one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cs_overlap_parameters(intensity_a: float, intensity_b: float, intensities: "np.ndarray", probabilities: "np.ndarray", delta_max: float, correlation_range: int, n_cut: int) -> "np.ndarray":
    '''Compute the photon-number-resolved overlap parameter for one pair of settings.

    Parameters
    ----------
    intensity_a : float
        First nominal mean photon number of the pair. Strictly positive.
    intensity_b : float
        Second nominal mean photon number of the pair. Strictly positive.
    intensities : np.ndarray
        Shape (A,). Every nominal mean photon number the transmitter can select.
        All entries strictly positive.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each entry of `intensities`.
        Non-negative and summing to one.
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. A non-negative integer.

    Returns
    -------
    overlaps : np.ndarray
        Shape (n_cut + 1,). Entry n is the overlap parameter of the pair at
        photon number n. Every entry is strictly positive, and lies in (0, 1]
        whenever every nominal setting is at most one.

    Raises
    ------
    ValueError
        If either paired intensity is not strictly positive, if `intensities`
        is not a non-empty one-dimensional array of strictly positive values,
        if `probabilities` does not have the shape of `intensities` or is not a
        non-negative vector summing to one within 1e-9, if `delta_max` is
        negative or is not below one, if `correlation_range` is smaller than
        one, or if `n_cut` is negative.
    '''
    return overlaps

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_intensity_family(intensities: "np.ndarray", probabilities: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Validate and normalise a family of nominal settings and their weights."""
    intensities = np.asarray(intensities, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if intensities.ndim != 1 or intensities.size < 1:
        raise ValueError("intensities must be a non-empty 1D array")
    if not np.all(np.isfinite(intensities)) or np.any(intensities <= 0.0):
        raise ValueError("intensities must all be strictly positive")
    if probabilities.shape != intensities.shape:
        raise ValueError("probabilities must have the same shape as intensities")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if abs(float(np.sum(probabilities)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to one")
    return intensities, probabilities


def _oracle_compute_cs_overlap_parameters(intensity_a: float, intensity_b: float, intensities: "np.ndarray", probabilities: "np.ndarray", delta_max: float, correlation_range: int, n_cut: int) -> "np.ndarray":
    for value in (intensity_a, intensity_b, delta_max):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("intensity_a, intensity_b and delta_max must be real scalars")
    for value in (correlation_range, n_cut):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("correlation_range and n_cut must be integers")
    intensity_a = float(intensity_a)
    intensity_b = float(intensity_b)
    delta_max = float(delta_max)
    correlation_range = int(correlation_range)
    n_cut = int(n_cut)
    if not np.isfinite(intensity_a) or intensity_a <= 0.0:
        raise ValueError("intensity_a must be strictly positive")
    if not np.isfinite(intensity_b) or intensity_b <= 0.0:
        raise ValueError("intensity_b must be strictly positive")
    if not np.isfinite(delta_max) or delta_max < 0.0 or delta_max >= 1.0:
        raise ValueError("delta_max must lie in [0, 1)")
    if correlation_range < 1:
        raise ValueError("correlation_range must be at least one")
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")
    intensities, probabilities = _check_intensity_family(intensities, probabilities)

    a_lo, a_hi = intensity_a * (1.0 - delta_max), intensity_a * (1.0 + delta_max)
    b_lo, b_hi = intensity_b * (1.0 - delta_max), intensity_b * (1.0 + delta_max)

    # Weight lost to the spread of every selectable setting's vacuum probability,
    # compounded over the rounds on both sides of the memory span.
    spread = float(np.sum(probabilities * (np.exp(-intensities * (1.0 - delta_max))
                                           - np.exp(-intensities * (1.0 + delta_max)))))
    memory = (1.0 - spread) ** (2 * correlation_range)

    n = np.arange(n_cut + 1, dtype=float)
    overlaps = (np.exp(a_hi + b_hi - (a_lo + b_lo))
                * ((a_lo * b_lo) / (a_hi * b_hi)) ** n * memory)
    overlaps[0] = np.exp(a_lo + b_lo - (a_hi + b_hi)) * memory
    return overlaps

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Retain scientific cases with independent invocation inputs."""
    from textwrap import dedent

    return [
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.48, 0.1, intensities, probabilities, 0.001, 3, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (
                    (0.0001, 0.48, intensities, probabilities, 0.001, 3, 10)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.48, 0.1, intensities, probabilities, 0.0, 3, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
            'tol': 1e-12,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.48, 0.1, intensities, probabilities, 0.001, 1, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.48, 0.1, intensities, probabilities, 0.01, 8, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.1, 0.48, intensities, probabilities, 0.001, 3, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
            'tol': 1e-12,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                args = (0.1, 0.1, intensities, probabilities, 0.001, 3, 6)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.5, 0.05])
                probabilities = np.array([0.8, 0.2])
                args = (0.5, 0.05, intensities, probabilities, 0.005, 2, 0)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.02, 0.02, 0.96])
                args = (0.48, 0.1, intensities, probabilities, 0.01, 3, 10)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (0.48, 0.1, intensities, probabilities, 0.001, 0, 10)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (0.48, 0.1, intensities, np.array([0.7, 0.18, 0.5]),
                    0.001, 3, 10)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (0.48, 0.1, np.array([0.48, 0.0, 0.0001]), probabilities,
                    0.001, 3, 10)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (0.48, 0.1, intensities, probabilities, -0.001, 3, 10)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_cs_overlap_parameters, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_cs_overlap_parameters, *args)
                """
            ),
        },
    ]
