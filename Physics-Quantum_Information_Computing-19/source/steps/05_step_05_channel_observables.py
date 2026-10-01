"""
Compute, for every nominal intensity setting, the sifted detection rate and the sifted error rate that the channel and detection model predicts for phase-randomised weak coherent pulses.

These are the quantities a run of the protocol actually delivers, and they are the only experimentally accessible input to the parameter estimation that follows: everything resolved by photon number has to be inferred from them. Both detectors share one efficiency and one per-gate dark-count probability, and the channel contributes a transmittance and a fixed polarisation misalignment. The detector efficiency is the same in both bases, so the sifted detection rate is basis independent, while the error rate is reported for the basis in which the phase error is later estimated.

Every returned array is indexed in the order of the supplied nominal settings. Each entry already has the basis-selection and intensity-selection probabilities divided out, so it is a rate conditioned on the matching-basis, matching-setting rounds rather than a fraction of all rounds.

Returns
-------
key_basis_rates : np.ndarray — Shape (A,). Sifted detection rate in the basis used to distil the key.; check_basis_rates : np.ndarray — Shape (A,). Sifted detection rate in the basis used to estimate the phase error.; check_basis_error_rates : np.ndarray — Shape (A,). Sifted error rate in the basis used to estimate the phase error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_channel_observables(intensities: "np.ndarray", transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Compute the model's sifted detection and error rates for each nominal setting.

    Parameters
    ----------
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive.
    transmittance : float
        Overall probability that an emitted photon reaches and fires a
        detector, combining channel loss and detector efficiency. In [0, 1].
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).

    Returns
    -------
    key_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate in the basis used to distil the key.
    check_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate in the basis used to estimate the
        phase error.
    check_basis_error_rates : np.ndarray
        Shape (A,). Sifted error rate in the basis used to estimate the phase
        error.

    Raises
    ------
    ValueError
        If `intensities` is not a non-empty one-dimensional array of strictly
        positive values, if `transmittance` is outside [0, 1], if
        `misalignment` is outside [0, pi / 4], or if `dark_count` is negative
        or is not below one.
    '''
    return key_basis_rates, check_basis_rates, check_basis_error_rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_channel_observables(intensities: "np.ndarray", transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    for value in (transmittance, misalignment, dark_count):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("transmittance, misalignment and dark_count must be real scalars")
    transmittance = float(transmittance)
    misalignment = float(misalignment)
    dark_count = float(dark_count)
    if not np.isfinite(transmittance) or transmittance < 0.0 or transmittance > 1.0:
        raise ValueError("transmittance must lie in [0, 1]")
    if not np.isfinite(misalignment) or misalignment < 0.0 or misalignment > 0.25 * np.pi:
        raise ValueError("misalignment must lie in [0, pi / 4]")
    if not np.isfinite(dark_count) or dark_count < 0.0 or dark_count >= 1.0:
        raise ValueError("dark_count must lie in [0, 1)")
    settings = np.asarray(intensities, dtype=float)
    if settings.ndim != 1 or settings.size < 1:
        raise ValueError("intensities must be a non-empty 1D array")
    if not np.all(np.isfinite(settings)) or np.any(settings <= 0.0):
        raise ValueError("intensities must all be strictly positive")

    no_dark = (1.0 - dark_count) ** 2
    detected = 1.0 - no_dark * np.exp(-transmittance * settings)

    aligned = np.exp(-transmittance * settings * np.cos(misalignment) ** 2)
    crossed = np.exp(-transmittance * settings * np.sin(misalignment) ** 2)
    imbalance = 0.5 * (aligned - crossed)

    errors = (0.5 * dark_count ** 2
              + dark_count * (1.0 - dark_count) * (1.0 + imbalance)
              + no_dark * (0.5 + imbalance - 0.5 * np.exp(-transmittance * settings)))
    return detected, np.array(detected, dtype=float, copy=True), errors

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
                args = (
                    (intensities, 10 ** (-0.2 * 35.0 / 10.0) * 0.65, 0.08,
                    7.2e-08)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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
                args = (
                    (intensities, 10 ** (-0.2 * 5.0 / 10.0) * 0.65, 0.08,
                    7.2e-08)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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
                args = (intensities, 0.1296920505, 0.0, 7.2e-08)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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
                args = (intensities, 0.1296920505, 0.08, 0.0)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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


                intensities = np.array([0.48, 0.1])
                args = (intensities, 0.1296920505, 0.25 * np.pi, 0.0)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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
                args = (intensities, 0.0, 0.08, 7.2e-08)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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
                args = (intensities, 1e-06, 0.08, 1e-05)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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


                intensities = np.array([0.9, 0.3, 0.001])
                args = (intensities, 1.0, 0.15, 1e-06)
                """
            ),
            'call': dedent(
                """\
                _isolated(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_compute_channel_observables, *args)
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


                def basis_gap(fn):
                    intensities = np.array([0.48, 0.1, 0.0001])
                    key, check, err = (
                        _isolated(fn, intensities, 0.1296920505, 0.08,
                        7.2e-08)
                    )
                    return np.concatenate([key - check, err])
                """
            ),
            'call': dedent(
                """\
                basis_gap(compute_channel_observables)
                """
            ),
            'gold_call': dedent(
                """\
                basis_gap(_oracle_compute_channel_observables)
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


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.48]), 1.2, 0.08, 7.2e-08)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_channel_observables, *args)
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


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.48]), 0.13, 1.0, 7.2e-08)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_channel_observables, *args)
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


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.48, -0.1]), 0.13, 0.08, 7.2e-08)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_channel_observables, *args)
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


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.48]), 0.13, 0.08, 1.0)
                """
            ),
            'call': dedent(
                """\
                run_model(compute_channel_observables, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_compute_channel_observables, *args)
                """
            ),
        },
    ]
