"""
Enumerate, in a canonical order, every haplotype phasing of a short run of heterozygous biallelic SNPs that is consistent with the called polyploid genotypes.

A phasing of P positions in a K-ploid genome is a K-by-P allele matrix whose column sums must reproduce the called alternate-allele counts, and because haplotypes carry no intrinsic order only the multiset of its rows is meaningful. Imposing the genotype constraint collapses the 2^(K*P) unconstrained matrices to a small set that grows linearly rather than exponentially in the ploidy.

Returns
-------
np.ndarray of shape (M, ploidy, P), int: the canonical genotype-consistent phasings in ascending lexicographic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def enumerate_valid_phasings(ploidy: int, genotypes: np.ndarray) -> np.ndarray:
    """Enumerate the genotype-consistent phasings of a short run of SNPs.

    A phasing is returned in canonical form: its rows are sorted in ascending
    lexicographic order, so that two phasings differing only by a permutation
    of haplotype labels are represented by the same matrix. The enumeration is
    ordered by ascending lexicographic order of the row-major flattened matrix.

    Parameters
    ----------
    ploidy : int
        Number of haplotypes K carried by the organism (ploidy >= 1).
    genotypes : np.ndarray
        One-dimensional integer array of length P holding, for each of the P
        positions, the called count of alternate alleles across the K
        haplotypes. Every entry lies between 0 and ``ploidy`` inclusive and P
        is between 1 and 4 inclusive.

    Returns
    -------
    phasings : np.ndarray
        Integer array of shape (M, ploidy, P) with entries in {0, 1}. Entry
        ``phasings[m, k, p]`` is the allele that haplotype k carries at the
        p-th position under the m-th phasing, and every phasing satisfies
        ``phasings[m].sum(axis=0) == genotypes``.

    Raises
    ------
    ValueError
        If ``ploidy`` is not an integer greater than zero, if ``genotypes`` is
        not a one-dimensional integer-valued array whose length is between one
        and four inclusive, or if any entry of ``genotypes`` lies outside the
        range from zero to ``ploidy`` inclusive.
    """
    return phasings  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_enumerate_valid_phasings(ploidy: int, genotypes: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    if not (isinstance(ploidy, (int, np.integer)) and not isinstance(ploidy, bool)
            and int(ploidy) >= 1):
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)

    counts = np.asarray(genotypes, dtype=float)
    if counts.ndim != 1 or counts.size < 1 or counts.size > 4:
        raise ValueError("genotypes must be a one-dimensional array of one to four entries")
    if not np.all(np.isfinite(counts)):
        raise ValueError("genotypes must contain only finite entries")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    counts = np.round(counts).astype(int)
    if np.any(counts < 0) or np.any(counts > ploidy):
        raise ValueError("genotypes entries must lie between zero and ploidy inclusive")

    n_positions = int(counts.size)
    # The allele patterns a single haplotype can carry over the P positions,
    # listed in ascending lexicographic order.
    patterns = [np.array(p, dtype=int)
                for p in itertools.product((0, 1), repeat=n_positions)]

    # Choosing a multiset of K patterns with repetition already produces rows in
    # ascending lexicographic order, which is the canonical form, and the
    # combinations are generated in ascending lexicographic order themselves.
    phasings = []
    for choice in itertools.combinations_with_replacement(range(len(patterns)), ploidy):
        matrix = np.array([patterns[i] for i in choice], dtype=int)
        if np.array_equal(matrix.sum(axis=0), counts):
            phasings.append(matrix)

    return np.array(phasings, dtype=int).reshape(-1, ploidy, n_positions)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the tetraploid pair of the worked appendix example, whose
        #     three phasings are the parameterised family of that derivation ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 4
genotypes = np.array([2, 2])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 10.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 10.0)",
        },
        # --- Valid: an asymmetric tetraploid pair, where the feasible family is
        #     shorter than the symmetric one ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 4
genotypes = np.array([3, 1])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 10.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 10.0)",
        },
        # --- Valid: a hexaploid triple, the size the three-SNP factors need ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 6
genotypes = np.array([2, 3, 4])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 100.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 100.0)",
        },
        # --- Boundary: a homozygous alternate position, which admits exactly one
        #     phasing however large the ploidy ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 5
genotypes = np.array([5, 5])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 1.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 1.0)",
        },
        # --- Edge: a diploid triple, the smallest ploidy for which the genotype
        #     constraint still leaves a choice at every position ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 2
genotypes = np.array([1, 1, 1])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 1.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 1.0)",
        },
        # --- Edge: a single position, where the phasing family is a single
        #     canonical column whatever the genotype ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
ploidy = 4
genotypes = np.array([3])
""",
            "call": "sig(enumerate_valid_phasings(ploidy, genotypes), 1.0)",
            "gold_call": "sig(_oracle_enumerate_valid_phasings(ploidy, genotypes), 1.0)",
        },
        # --- Invalid: an alternate-allele count exceeding the ploidy ---
        {
            "setup": """import numpy as np
ploidy = 3
genotypes = np.array([2, 4])
def run_model():
    try:
        enumerate_valid_phasings(ploidy, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_valid_phasings(ploidy, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a fractional alternate-allele count ---
        {
            "setup": """import numpy as np
ploidy = 4
genotypes = np.array([1.5, 2.0])
def run_model():
    try:
        enumerate_valid_phasings(ploidy, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_valid_phasings(ploidy, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive ploidy ---
        {
            "setup": """import numpy as np
genotypes = np.array([0, 0])
def run_model():
    try:
        enumerate_valid_phasings(0, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_valid_phasings(0, genotypes)
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
