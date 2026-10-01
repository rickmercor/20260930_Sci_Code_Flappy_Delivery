"""
Biallelic polarization changes the represented mutation when the alternate allele is ancestral. For site i, the alternate-carrier indicator A_i is replaced within the observed haplotype indicator C_i by S_i = C_i - A_i; a site whose alternate allele is not ancestral retains A_i. All indicators are binary, and an alternate carrier must also be observed.

Inputs

------

alternate_carriers: Binary array of shape (k, n_samples).

observed_haplotypes: Binary array of shape (k, n_samples).

ancestral_is_alternate: Binary array of shape (k,).

Returns

-------

target_carriers: Binary uint8 array of shape (k, n_samples).

Returns
-------
np.ndarray of shape (k, n_samples), the polarized carrier indicators as uint8
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def polarize_carrier_sets(
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
) -> np.ndarray:
    """Construct carrier sets after biallelic allele polarization.

    Parameters
    ----------
    alternate_carriers : np.ndarray
        Binary alternate-carrier matrix with shape (k, n_samples).
    observed_haplotypes : np.ndarray
        Binary observed-haplotype matrix with the same shape.
    ancestral_is_alternate : np.ndarray
        Binary vector indicating which sites require complementation.

    Raises
    ------
    ValueError
        If the carrier matrices are not matching two-dimensional binary arrays,
        if `ancestral_is_alternate` is not binary with one entry per site, or if
        an alternate carrier is not observed.

    Returns
    -------
    target_carriers : np.ndarray
        Binary uint8 carrier matrix after polarization.
    """
    return target_carriers  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_polarize_carrier_sets(
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    alternate = np.asarray(alternate_carriers)
    observed = np.asarray(observed_haplotypes)
    ancestral = np.asarray(ancestral_is_alternate)
    if alternate.ndim != 2 or observed.shape != alternate.shape:
        raise ValueError("carrier matrices must be two dimensional with matching shapes")
    if ancestral.shape != (alternate.shape[0],):
        raise ValueError("ancestral_is_alternate must have one entry per site")
    if not np.all((alternate == 0) | (alternate == 1)):
        raise ValueError("alternate_carriers must be binary")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("observed_haplotypes must be binary")
    if not np.all((ancestral == 0) | (ancestral == 1)):
        raise ValueError("ancestral_is_alternate must be binary")
    if np.any(alternate > observed):
        raise ValueError("an alternate carrier must be observed")

    complemented = observed.astype(np.uint8) - alternate.astype(np.uint8)
    return np.where(ancestral[:, None] == 1, complemented, alternate).astype(np.uint8)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
alternate = np.array([[0, 0, 1, 1], [1, 0, 1, 0]], dtype=np.uint8)
observed = np.ones((2, 4), dtype=np.uint8)
ancestral = np.array([1, 0], dtype=np.uint8)
""",
            "call": "polarize_carrier_sets(alternate, observed, ancestral).tolist()",
            "gold_call": "_oracle_polarize_carrier_sets(alternate, observed, ancestral).tolist()",
        },
        {
            "setup": """import numpy as np
alternate = np.array([[0]], dtype=np.uint8)
observed = np.array([[0]], dtype=np.uint8)
ancestral = np.array([1], dtype=np.uint8)
""",
            "call": "polarize_carrier_sets(alternate, observed, ancestral).tolist()",
            "gold_call": "_oracle_polarize_carrier_sets(alternate, observed, ancestral).tolist()",
        },
        {
            "setup": """import numpy as np
alternate = np.array([[1, 0]], dtype=np.uint8)
observed = np.array([[0, 1]], dtype=np.uint8)
ancestral = np.array([1], dtype=np.uint8)
def run_model():
    try:
        polarize_carrier_sets(alternate, observed, ancestral)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_polarize_carrier_sets(alternate, observed, ancestral)
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
