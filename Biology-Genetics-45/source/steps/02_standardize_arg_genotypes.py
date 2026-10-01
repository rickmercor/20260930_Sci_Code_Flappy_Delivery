"""
Filter the mutations on minor allele count and return the centred,

frequency-scaled genotype matrix used to build the relatedness matrix.

Under a frequency-dependent architecture the effect size of a variant with

derived allele frequency f has variance proportional to [f (1 - f)]^alpha, so

each genotype column is centred at twice the derived allele frequency and

multiplied by the weight [f (1 - f)]^(alpha / 2), with no factor of 2 inside the

bracket, so that its contribution to X X^T follows that architecture. Mutations

resampled on a genealogy are dominated by very low frequency variants, so a

filter on minor allele count across the 2 n haplotypes is applied before the

scaling.

Returns
-------
np.ndarray of shape (n, p_kept), float64 centred and frequency-scaled genotypes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def standardize_arg_genotypes(
    genotypes: "np.ndarray",
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
) -> "np.ndarray":
    """Filter and frequency-scale a diploid genotype matrix.

    Args:
        genotypes: array of shape (n, p) holding diploid allele counts in
            {0, 1, 2}, one column per mutation.
        alpha: frequency-dependent scaling exponent of the method's genotype
            weight, with f the derived allele frequency across the 2 n
            haplotypes.
        min_minor_allele_count: mutations whose minor allele count across the
            2 n haplotypes is below this value are discarded.

    Returns:
        np.ndarray of shape (n, p_kept) holding the centred, frequency-scaled
        genotypes of the retained mutations, columns in their original order.

    Raises:
        ValueError: if genotypes is not two-dimensional, if any entry lies
            outside {0, 1, 2}, if min_minor_allele_count is below 1, or if no
            mutation survives the filter.
    """
    return x_std

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_standardize_arg_genotypes(
    genotypes: "np.ndarray",
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
) -> "np.ndarray":
    """Reference implementation."""
    genotypes = np.asarray(genotypes, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be a two-dimensional array")
    if np.any(genotypes < 0.0) or np.any(genotypes > 2.0):
        raise ValueError("diploid allele counts must lie in {0, 1, 2}")
    if int(min_minor_allele_count) < 1:
        raise ValueError("min_minor_allele_count must be at least 1")

    n_haplotypes = 2 * genotypes.shape[0]
    allele_count = genotypes.sum(axis=0)
    minor_count = np.minimum(allele_count, n_haplotypes - allele_count)
    keep = minor_count >= int(min_minor_allele_count)
    if not np.any(keep):
        raise ValueError("no mutation passes the minor allele count filter")

    retained = genotypes[:, keep]
    freq = retained.sum(axis=0) / n_haplotypes
    weight = (freq * (1.0 - freq)) ** (float(alpha) / 2.0)
    return (retained - 2.0 * freq) * weight

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _GENOTYPES = [
        [0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 2, 0, 2, 0, 1, 0, 2, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0],
        [0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 0, 0],
    ]
    return [
        {
            "setup": """import numpy as np
genotypes = np.array(%r, dtype=float)
""" % (_GENOTYPES,),
            "call": "np.round(standardize_arg_genotypes(genotypes, -0.25, 2), 10).tolist()",
            "gold_call": "np.round(_oracle_standardize_arg_genotypes(genotypes, -0.25, 2), 10).tolist()",
        },
        {
            "setup": """import numpy as np
genotypes = np.array(%r, dtype=float)
""" % (_GENOTYPES,),
            "call": "np.round(standardize_arg_genotypes(genotypes, 0.0, 1), 10).tolist()",
            "gold_call": "np.round(_oracle_standardize_arg_genotypes(genotypes, 0.0, 1), 10).tolist()",
        },
        {
            "setup": """import numpy as np
genotypes = np.array(%r, dtype=float)
""" % (_GENOTYPES,),
            "call": "np.round(standardize_arg_genotypes(genotypes, -0.5, 4), 10).tolist()",
            "gold_call": "np.round(_oracle_standardize_arg_genotypes(genotypes, -0.5, 4), 10).tolist()",
        },
        {
            "setup": """import numpy as np
genotypes = np.array(%r, dtype=float)

def run_model():
    try:
        standardize_arg_genotypes(genotypes, -0.25, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_standardize_arg_genotypes(genotypes, -0.25, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (_GENOTYPES,),
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # Derived-majority and fixed sites. Their derived counts (19 and 20) pass a filter on the
        # derived allele but their minor allele counts (1 and 0) do not; the fixed site also has f (1 - f) = 0.
        {
            "setup": """import numpy as np
genotypes = np.array(%r, dtype=float)
extra = np.full((10, 3), 2.0)
extra[3, 0] = 1.0
extra[0, 2] = 0.0
extra[1, 2] = 0.0
genotypes = np.hstack([genotypes, extra])
""" % (_GENOTYPES,),
            "call": "np.round(standardize_arg_genotypes(genotypes, -0.25, 2), 10).tolist()",
            "gold_call": "np.round(_oracle_standardize_arg_genotypes(genotypes, -0.25, 2), 10).tolist()",
        },
    ]
