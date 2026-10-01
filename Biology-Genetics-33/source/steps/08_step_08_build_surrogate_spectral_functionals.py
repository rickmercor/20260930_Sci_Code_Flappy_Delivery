"""
Tabulate the covariance functionals that a candidate parameter triple implies under the working model in which both components are step spectra expressed in one shared eigenbasis.

The fit does not see the covariance matrices, only the functionals a candidate parameter triple would produce, and under a shared-eigenbasis working model those follow from the eigenvalue profiles alone. This is the object the observed functionals are matched against.

Returns
-------
np.ndarray of shape (2 ** (order + 1) - 2,), float: the working model's functional for every word.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_surrogate_spectral_functionals(genetic_level: float,
                                         genetic_fraction: float,
                                         residual_level: float,
                                         residual_fraction: float,
                                         order: int) -> np.ndarray:
    """Tabulate the covariance functionals implied by a candidate parameter triple.

    Under the working model each component has a step eigenvalue profile on the
    unit interval expressed in one shared eigenbasis: the genetic profile equals
    ``genetic_level`` on a leading interval of length ``genetic_fraction`` and
    zero after it, and the residual profile equals ``residual_level`` on a
    leading interval of length ``residual_fraction`` and zero after it, with the
    genetic interval contained in the residual one. The entry for a word is the
    integral over the unit interval of the product of the profiles its letters
    name, letter zero being the genetic profile and letter one the residual
    profile.

    Words are indexed first by length and then lexicographically, so the word
    ``(w_1, ..., w_s)`` sits at position ``2**s - 2 + sum_t w_t * 2**(s - t)``.

    Parameters
    ----------
    genetic_level : float
        Nonzero eigenvalue of the family-level component; finite and at least
        zero.
    genetic_fraction : float
        Fraction of trait directions carrying it, in ``(0, 1]``.
    residual_level : float
        Nonzero eigenvalue of the individual-level component; finite and at
        least zero.
    residual_fraction : float
        Fraction of trait directions carrying it, in ``(0, 1]``.
    order : int
        Longest word length to tabulate; a positive integer.

    Returns
    -------
    functionals : np.ndarray
        Array of shape ``(2 ** (order + 1) - 2,)`` holding the implied
        functional of every word of length one to ``order``, in the index order
        described above.

    Raises
    ------
    ValueError
        If either level is not a finite number of at least zero, if either
        fraction is not a finite number in ``(0, 1]``, or if ``order`` is not an
        integer value of at least one.
    """
    return functionals  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_surrogate_spectral_functionals(genetic_level: float,
                                                 genetic_fraction: float,
                                                 residual_level: float,
                                                 residual_fraction: float,
                                                 order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    for name, value in (("genetic_level", genetic_level),
                        ("residual_level", residual_level)):
        if not _is_number(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite number of at least zero")
    for name, value in (("genetic_fraction", genetic_fraction),
                        ("residual_fraction", residual_fraction)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not _is_number(order) or float(order) != float(int(order)) or int(order) < 1:
        raise ValueError("order must be an integer value of at least one")

    genetic_level = float(genetic_level)
    genetic_fraction = float(genetic_fraction)
    residual_level = float(residual_level)
    residual_fraction = float(residual_fraction)
    order = int(order)

    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            n_genetic = sum(1 for letter in word if letter == 0)
            n_residual = length - n_genetic
            if n_genetic >= 1:
                # The genetic profile vanishes outside its own interval, which
                # lies inside the residual one, so the integral is taken there.
                value = (genetic_fraction * genetic_level ** n_genetic
                         * residual_level ** n_residual)
            else:
                value = residual_fraction * residual_level ** n_residual
            functionals[base + offset] = value

    return functionals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the true parameter triple of the testbed at the order the
        #     pipeline uses (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(1.4, 0.35, 0.6, 0.8, 3), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(1.4, 0.35, 0.6, 0.8, 3), 1.0)",
        },
        # --- Valid: a candidate triple away from the truth, of the kind the
        #     solver visits, with a residual level above the genetic one ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(0.85, 0.22, 1.35, 0.7, 3), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(0.85, 0.22, 1.35, 0.7, 3), 1.0)",
        },
        # --- Valid: fourth order, where words carry three genetic letters and one
        #     residual letter and the two exponents must not be interchanged ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(1.9, 0.15, 0.45, 0.95, 4), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(1.9, 0.15, 0.45, 0.95, 4), 1.0)",
        },
        # --- Boundary: order one, where the two entries are the two profile means
        #     and carry different fractions ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(1.4, 0.35, 0.6, 0.8, 1), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(1.4, 0.35, 0.6, 0.8, 1), 1.0)",
        },
        # --- Boundary: both fractions equal to one, where every word carries the
        #     same fraction and only the levels distinguish the entries ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(1.25, 1.0, 0.8, 1.0, 3), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(1.25, 1.0, 0.8, 1.0, 3), 1.0)",
        },
        # --- Edge: a vanishing genetic level, for which every word containing a
        #     genetic letter collapses to zero while the pure residual words do not ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
""",
            "call": "sig(build_surrogate_spectral_functionals(0.0, 0.35, 0.6, 0.8, 3), 1.0)",
            "gold_call": "sig(_oracle_build_surrogate_spectral_functionals(0.0, 0.35, 0.6, 0.8, 3), 1.0)",
        },
        # --- Invalid: a fraction outside the unit interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_surrogate_spectral_functionals(1.4, 1.5, 0.6, 0.8, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_surrogate_spectral_functionals(1.4, 1.5, 0.6, 0.8, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative eigenvalue level ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_surrogate_spectral_functionals(-1.4, 0.35, 0.6, 0.8, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_surrogate_spectral_functionals(-1.4, 0.35, 0.6, 0.8, 3)
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
