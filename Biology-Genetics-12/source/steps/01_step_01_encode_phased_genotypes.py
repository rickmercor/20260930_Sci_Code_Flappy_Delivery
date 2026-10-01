"""
Phased diploid SNP calls retain the order of the two haplotypes. GenoBERT assigns

the four allele pairs ``0|0``, ``0|1``, ``1|0``, and ``1|1`` to states 1, 2, 3,

and 4, respectively, while state 0 denotes a masked genotype. This step stores

the masked model input beside the unmasked target state so phase errors remain

distinguishable during evaluation.

Inputs

------

alleles: binary array of shape (n_samples, n_variants, 2)

masked_sites: integer array of shape (n_masked, 2) containing sample and variant indices

Returns

-------

encoded: integer array of shape (n_samples, n_variants, 2), with observed and target states

Returns
-------
np.ndarray of shape (n_samples, n_variants, 2), the observed and target token states
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def encode_phased_genotypes(
    alleles: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Encode phased allele pairs and replace selected model inputs by MASK.

    Parameters
    ----------
    alleles : np.ndarray
        Binary array with shape ``(n_samples, n_variants, 2)``.
    masked_sites : np.ndarray
        Integer array with shape ``(n_masked, 2)`` whose rows are distinct
        ``(sample_index, variant_index)`` pairs.

    Raises
    ------
    ValueError
        If alleles does not have shape ``(n_samples, n_variants, 2)``, contains
        values other than zero and one, or has no samples or variants; or if
        masked_sites does not have shape ``(n_masked, 2)``, is non-integral,
        contains duplicate rows, or contains an out-of-range index.

    Returns
    -------
    encoded : np.ndarray
        Integer array with observed states in channel 0 and unmasked target
        states in channel 1.
    """
    return encoded  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_encode_phased_genotypes(
    alleles: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    alleles = np.asarray(alleles)
    masked_sites = np.asarray(masked_sites)
    if alleles.ndim != 3 or alleles.shape[2] != 2:
        raise ValueError("alleles must have shape (n_samples, n_variants, 2)")
    if alleles.shape[0] == 0 or alleles.shape[1] == 0:
        raise ValueError("alleles must contain at least one sample and variant")
    if not np.all((alleles == 0) | (alleles == 1)):
        raise ValueError("alleles must be binary")
    if masked_sites.ndim != 2 or masked_sites.shape[1] != 2:
        raise ValueError("masked_sites must have shape (n_masked, 2)")
    if not np.issubdtype(masked_sites.dtype, np.integer) and (
        not np.all(np.isfinite(masked_sites))
        or not np.all(masked_sites == np.floor(masked_sites))
    ):
        raise ValueError("masked_sites must contain integers")
    masked_sites = masked_sites.astype(np.int64, copy=False)
    if len({tuple(row) for row in masked_sites.tolist()}) != len(masked_sites):
        raise ValueError("masked_sites rows must be distinct")
    if masked_sites.size:
        if np.any(masked_sites[:, 0] < 0) or np.any(
            masked_sites[:, 0] >= alleles.shape[0]
        ):
            raise ValueError("masked sample index out of range")
        if np.any(masked_sites[:, 1] < 0) or np.any(
            masked_sites[:, 1] >= alleles.shape[1]
        ):
            raise ValueError("masked variant index out of range")

    target = (
        1 + 2 * alleles[:, :, 0].astype(np.int64) + alleles[:, :, 1].astype(np.int64)
    )
    observed = target.copy()
    if masked_sites.size:
        observed[masked_sites[:, 0], masked_sites[:, 1]] = 0
    return np.stack((observed, target), axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
alleles = np.array([[[0, 0], [0, 1], [1, 0]],
                    [[1, 1], [1, 0], [0, 0]]], dtype=int)
masked_sites = np.array([[0, 1], [1, 0]], dtype=int)
""",
            "call": "encode_phased_genotypes(alleles, masked_sites).tolist()",
            "gold_call": "_oracle_encode_phased_genotypes(alleles, masked_sites).tolist()",
        },
        {
            "setup": """import numpy as np
alleles = np.array([[[1, 0]]], dtype=int)
masked_sites = np.empty((0, 2), dtype=int)
""",
            "call": "encode_phased_genotypes(alleles, masked_sites).tolist()",
            "gold_call": "_oracle_encode_phased_genotypes(alleles, masked_sites).tolist()",
        },
        {
            "setup": """import numpy as np
alleles = np.array([[[0, 2]]], dtype=int)
masked_sites = np.empty((0, 2), dtype=int)
def run_model():
    try:
        encode_phased_genotypes(alleles, masked_sites)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_encode_phased_genotypes(alleles, masked_sites)
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
