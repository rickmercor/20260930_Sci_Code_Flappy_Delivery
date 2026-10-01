"""
Fixed allele-frequency margins for a binary haplotype panel.

For n phased haplotypes and ALT count m_j at variant j, the fixed-margin

HaploPerturb model uses the Jeffreys-corrected frequency

p_j = (m_j + 1/2) / (n + 1). The latent Gaussian threshold is

tau_j = Phi^{-1}(1 - p_j), so thresholding a standard normal above tau_j

reproduces p_j while keeping thresholds finite even for counts of zero or n.

Inputs

------

alt_counts : one-dimensional ALT counts ordered as lead then partners

n_haplotypes : number of phased haplotypes in the panel

Returns

-------

corrected_frequencies : corrected ALT frequencies

thresholds : standard-normal thresholds

Returns
-------
tuple of two float64 arrays of shape (p,): corrected frequencies and thresholds
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_fixed_margins(
    alt_counts: np.ndarray, n_haplotypes: int
) -> tuple[np.ndarray, np.ndarray]:
    '''Compute fixed ALT-frequency margins and latent Gaussian thresholds.

    Parameters
    ----------
    alt_counts : np.ndarray
        One-dimensional integer ALT counts, ordered as lead then partners.
    n_haplotypes : int
        Positive number of phased haplotypes used to form the counts.

    Returns
    -------
    corrected_frequencies : np.ndarray
        Jeffreys-corrected ALT frequencies as a float64 vector.
    thresholds : np.ndarray
        Standard-normal thresholds as a float64 vector.

    Raises
    ------
    ValueError
        If `alt_counts` is not a finite one-dimensional integer vector with at
        least two entries, if `n_haplotypes` is not a positive integer, or if
        any count lies outside the inclusive interval from zero to
        `n_haplotypes`.
    '''
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from statistics import NormalDist  # noqa: E402

import numpy as np  # noqa: E402, F811


def _oracle_compute_fixed_margins(
    alt_counts: np.ndarray, n_haplotypes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of the fixed-margin transformation."""
    counts = np.asarray(alt_counts, dtype=float)
    if counts.ndim != 1 or counts.size < 2 or not np.isfinite(counts).all():
        raise ValueError("alt_counts must be a finite one-dimensional vector with at least two entries")
    if not isinstance(n_haplotypes, (int, np.integer)) or int(n_haplotypes) < 1:
        raise ValueError("n_haplotypes must be a positive integer")
    if np.any(counts < 0.0) or np.any(counts > int(n_haplotypes)):
        raise ValueError("alt_counts must lie between zero and n_haplotypes")
    if not np.all(counts == np.floor(counts)):
        raise ValueError("alt_counts must contain integer values")

    corrected_frequencies = (counts + 0.5) / (int(n_haplotypes) + 1.0)
    normal = NormalDist()
    thresholds = np.asarray(
        [normal.inv_cdf(1.0 - float(value)) for value in corrected_frequencies],
        dtype=float,
    )
    return corrected_frequencies, thresholds

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """alt_counts = np.array([6, 9, 29, 12, 31, 4, 23, 15])
n_haplotypes = 40
def pack(result):
    return np.stack(result).astype(float, copy=False)
""",
            "call": "pack(compute_fixed_margins(alt_counts, n_haplotypes))",
            "gold_call": "pack(_oracle_compute_fixed_margins(alt_counts, n_haplotypes))",
        },
        {
            "setup": """alt_counts = np.array([0, 1])
n_haplotypes = 1
def pack(result):
    return np.stack(result).astype(float, copy=False)
""",
            "call": "pack(compute_fixed_margins(alt_counts, n_haplotypes))",
            "gold_call": "pack(_oracle_compute_fixed_margins(alt_counts, n_haplotypes))",
        },
        {
            "setup": """alt_counts = np.array([2, 5])
n_haplotypes = 4
def run_model():
    try:
        compute_fixed_margins(alt_counts, n_haplotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_fixed_margins(alt_counts, n_haplotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
