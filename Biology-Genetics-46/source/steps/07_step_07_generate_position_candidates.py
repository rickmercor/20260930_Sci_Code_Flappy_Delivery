"""
Enumerate every haplotype matrix over the SNP window spanned by the frontier vertices that both reproduces the alleles already assigned and respects the called genotypes.

A newly phased variant may permute haplotype labels only in ways that leave the already-assigned columns untouched, so the already-assigned columns of the window are frozen and only the still-unassigned ones are free. Each free column is any assignment of its called alternate-allele count to the haplotype rows, which keeps the candidate set small and exhaustive.

Returns
-------
np.ndarray of shape (n_candidates, K, P), int: the admissible haplotype matrices over the window.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def generate_position_candidates(positions: np.ndarray, haplotypes: np.ndarray,
                                 genotypes: np.ndarray) -> np.ndarray:
    """Enumerate the admissible haplotype matrices over a window of SNPs.

    Candidates are listed in ascending lexicographic order of the sequence of
    their free columns, each free column being read from the first haplotype
    row to the last and the earlier free columns varying more slowly than the
    later ones.

    Parameters
    ----------
    positions : np.ndarray
        One-dimensional integer array of P strictly increasing SNP indices, the
        window the candidates span.
    haplotypes : np.ndarray
        Integer array of shape (K, n_snps) holding the global haplotype matrix
        built so far, with 0 and 1 for assigned alleles and -1 for a SNP that
        has not been assigned yet. A column is treated as assigned only when
        every one of its K entries is non-negative.
    genotypes : np.ndarray
        One-dimensional integer array of length n_snps holding the called count
        of alternate alleles at each SNP.

    Returns
    -------
    candidates : np.ndarray
        Integer array of shape (n_candidates, K, P) with entries in {0, 1}.
        Every candidate reproduces ``haplotypes`` on the assigned columns of
        the window and has column sums equal to the genotypes of the window.

    Raises
    ------
    ValueError
        If ``positions`` is not a one-dimensional strictly increasing array of
        valid SNP indices, if ``haplotypes`` is not a two-dimensional
        integer-valued array with entries in {-1, 0, 1}, if ``genotypes`` is not
        a one-dimensional integer-valued array of length n_snps whose entries
        lie between zero and K inclusive, or if an already-assigned column of
        the window contradicts its called genotype.
    """
    return candidates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_position_candidates(positions: np.ndarray, haplotypes: np.ndarray,
                                         genotypes: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    matrix = np.asarray(haplotypes, dtype=float)
    if matrix.ndim != 2 or matrix.size < 1:
        raise ValueError("haplotypes must be a non-empty two-dimensional array")
    if not np.all(np.isin(matrix, (-1.0, 0.0, 1.0))):
        raise ValueError("haplotypes entries must be -1, 0 or 1")
    matrix = matrix.astype(int)
    ploidy, n_snps = matrix.shape

    counts = np.asarray(genotypes, dtype=float)
    if counts.ndim != 1 or counts.size != n_snps:
        raise ValueError("genotypes must be one-dimensional of length n_snps")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    counts = np.round(counts).astype(int)
    if np.any(counts < 0) or np.any(counts > ploidy):
        raise ValueError("genotypes entries must lie between zero and the ploidy inclusive")

    window = np.asarray(positions, dtype=float)
    if window.ndim != 1 or window.size < 1:
        raise ValueError("positions must be a non-empty one-dimensional array")
    if not np.allclose(window, np.round(window), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    window = np.round(window).astype(int)
    if np.any(np.diff(window) <= 0):
        raise ValueError("positions entries must be strictly increasing")
    if np.any(window < 0) or np.any(window >= n_snps):
        raise ValueError("positions entries must be valid SNP indices")

    # Each window column is either frozen by the assembly so far or free, in
    # which case it is any placement of its alternate-allele count on the rows.
    column_options = []
    for snp in window.tolist():
        column = matrix[:, snp]
        if np.all(column >= 0):
            if int(column.sum()) != int(counts[snp]):
                raise ValueError("an assigned column contradicts its called genotype")
            column_options.append([column.copy()])
        else:
            column_options.append([np.array(pattern, dtype=int)
                                   for pattern in itertools.product((0, 1), repeat=ploidy)
                                   if sum(pattern) == int(counts[snp])])

    candidates = [np.stack(choice, axis=1)
                  for choice in itertools.product(*column_options)]

    return np.array(candidates, dtype=int).reshape(-1, ploidy, int(window.size))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a tetraploid window with two frozen columns and one free
        #     column, the usual situation mid-assembly (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, 0, -1, -1, -1],
                       [0, 1, -1, -1, -1],
                       [1, 0, -1, -1, -1],
                       [1, 1, -1, -1, -1]], dtype=int)
genotypes = np.array([2, 2, 2, 1, 3])
positions = np.array([0, 1, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 100.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 100.0)",
        },
        # --- Valid: two free columns at once, the case of a frontier vertex
        #     that reaches beyond the selected variant ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, -1, -1, -1, -1],
                       [0, -1, -1, -1, -1],
                       [1, -1, -1, -1, -1],
                       [1, -1, -1, -1, -1]], dtype=int)
genotypes = np.array([2, 1, 3, 2, 2])
positions = np.array([0, 1, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 1000.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 1000.0)",
        },
        # --- Valid: a window whose free column sits between two frozen ones ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, -1, 1, -1, -1],
                       [1, -1, 0, -1, -1],
                       [1, -1, 1, -1, -1],
                       [0, -1, 0, -1, -1]], dtype=int)
genotypes = np.array([2, 3, 2, 1, 1])
positions = np.array([0, 1, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 100.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 100.0)",
        },
        # --- Valid: a column in which some but not all haplotypes carry an
        #     allele, which counts as unassigned and is therefore free ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, 1, -1],
                       [1, -1, -1],
                       [1, 0, -1],
                       [0, -1, -1]], dtype=int)
genotypes = np.array([2, 2, 2])
positions = np.array([0, 1, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 1000.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 1000.0)",
        },
        # --- Boundary: a window that is already fully assigned, which admits
        #     exactly the one candidate the assembly already holds ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 1]], dtype=int)
genotypes = np.array([2, 2, 2])
positions = np.array([0, 1, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
        },
        # --- Boundary: a free column at a homozygous alternate call, which has
        #     only one admissible placement whatever the ploidy ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, -1],
                       [0, -1],
                       [1, -1],
                       [1, -1]], dtype=int)
genotypes = np.array([2, 4])
positions = np.array([0, 1])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
        },
        # --- Edge: a diploid assembly, where a free column admits only the two
        #     complementary placements ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
haplotypes = np.array([[0, -1, -1],
                       [1, -1, -1]], dtype=int)
genotypes = np.array([1, 1, 1])
positions = np.array([0, 2])
""",
            "call": "sig(generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
            "gold_call": "sig(_oracle_generate_position_candidates(positions, haplotypes, genotypes), 1.0)",
        },
        # --- Invalid: an assigned column that contradicts its called genotype ---
        {
            "setup": """import numpy as np
haplotypes = np.array([[0, -1],
                       [0, -1],
                       [1, -1],
                       [1, -1]], dtype=int)
genotypes = np.array([3, 2])
positions = np.array([0, 1])
def run_model():
    try:
        generate_position_candidates(positions, haplotypes, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_position_candidates(positions, haplotypes, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a window whose SNP indices are not strictly increasing ---
        {
            "setup": """import numpy as np
haplotypes = np.array([[0, -1, -1],
                       [1, -1, -1]], dtype=int)
genotypes = np.array([1, 1, 1])
positions = np.array([2, 0])
def run_model():
    try:
        generate_position_candidates(positions, haplotypes, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_position_candidates(positions, haplotypes, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a called alternate-allele count exceeding the ploidy ---
        {
            "setup": """import numpy as np
haplotypes = np.array([[0, -1],
                       [1, -1]], dtype=int)
genotypes = np.array([1, 3])
positions = np.array([0, 1])
def run_model():
    try:
        generate_position_candidates(positions, haplotypes, genotypes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_position_candidates(positions, haplotypes, genotypes)
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
