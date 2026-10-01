"""
Score a finished, possibly fragmented and partially phased assembly by the minimum number of read allele corrections it needs, charged for every extra block a fragment has to span.

Minimum error correction counts, for every fragment, the flips needed to make it match its best-fitting reconstructed haplotype, but that count collapses to zero if the assembly is cut into enough short blocks. Charging each fragment for every block boundary it crosses, and treating an unresolved allele as a mismatch, restores the comparability the raw count loses.

Returns
-------
float: the block-adjusted minimum error correction of the assembly, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_block_adjusted_mec(reads: np.ndarray, haplotypes: np.ndarray,
                               blocks: np.ndarray, ploidy: int) -> float:
    """Score an assembly by its block-adjusted minimum error correction.

    A SNP is resolved when every one of the K haplotypes carries an allele at
    it. Each called allele of a fragment at an unresolved SNP costs one,
    whichever haplotype it is compared against. The resolved SNPs a fragment
    calls are grouped by the phasing block they belong to, and each group
    contributes the smallest number of mismatches any one of the K haplotypes
    leaves on it; the groups are minimised separately because the haplotype
    labels of different blocks are unrelated. The fragment is then charged one
    minus the reciprocal of the ploidy for every block it spans beyond the
    first, a fragment spanning no block being treated as spanning one. The
    total over fragments is divided by the number of called alleles in the
    whole fragment set.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0, 1 and -1 for reference, alternate and uncalled.
    haplotypes : np.ndarray
        Integer array of shape (K, n_snps) holding the reconstructed haplotypes,
        with -1 where a SNP was left unresolved.
    blocks : np.ndarray
        One-dimensional integer array of length n_snps holding the phasing
        block each SNP belongs to, numbered from one, with 0 for a SNP that
        belongs to no block. A SNP carries a positive label exactly when it is
        resolved.
    ploidy : int
        Number of haplotypes K, which must equal the first axis of
        ``haplotypes`` and be greater than zero.

    Returns
    -------
    mec : float
        The block-adjusted minimum error correction of the assembly, as a
        native Python float.

    Raises
    ------
    ValueError
        If ``reads`` or ``haplotypes`` is not a two-dimensional integer-valued
        array with entries in {-1, 0, 1}, if the two disagree on the number of
        SNPs, if ``blocks`` is not a one-dimensional array of non-negative
        integers of that same length, if some SNP carries a positive block label
        without being resolved or is resolved without carrying one, if
        ``ploidy`` is not an integer greater than zero matching the first axis
        of ``haplotypes``, or if no fragment calls any allele at all.
    """
    return mec  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_block_adjusted_mec(reads: np.ndarray, haplotypes: np.ndarray,
                                       blocks: np.ndarray, ploidy: int) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    assembly = np.asarray(haplotypes, dtype=float)
    for name, array in (("reads", fragments), ("haplotypes", assembly)):
        if array.ndim != 2 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional array")
        if not np.all(np.isin(array, (-1.0, 0.0, 1.0))):
            raise ValueError(f"{name} entries must be -1, 0 or 1")
    fragments = fragments.astype(int)
    assembly = assembly.astype(int)
    if fragments.shape[1] != assembly.shape[1]:
        raise ValueError("reads and haplotypes must agree on the number of SNPs")

    labels = np.asarray(blocks, dtype=float)
    if labels.ndim != 1 or labels.size != fragments.shape[1]:
        raise ValueError("blocks must be one-dimensional of length n_snps")
    if not np.allclose(labels, np.round(labels), rtol=0.0, atol=1e-12):
        raise ValueError("blocks entries must be integer valued")
    labels = np.round(labels).astype(int)
    if np.any(labels < 0):
        raise ValueError("blocks entries must be non-negative")
    if np.any(np.all(assembly >= 0, axis=0) != (labels > 0)):
        raise ValueError("a SNP must carry a positive block label exactly when it is resolved")

    if not (isinstance(ploidy, (int, np.integer)) and not isinstance(ploidy, bool)
            and int(ploidy) >= 1):
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)
    if ploidy != assembly.shape[0]:
        raise ValueError("ploidy must match the first axis of haplotypes")

    penalty = 1.0 - 1.0 / float(ploidy)
    resolved_snp = np.all(assembly >= 0, axis=0)
    total = 0.0
    called = 0
    for fragment in fragments:
        covered = np.flatnonzero(fragment >= 0)
        if covered.size == 0:
            continue
        called += int(covered.size)

        # An allele at an unresolved SNP costs one against every haplotype, so
        # it is charged outright and takes no part in the matching.
        unresolved = covered[~resolved_snp[covered]]
        total += float(unresolved.size)

        # Blocks carry independent haplotype labels, so the best-fitting
        # haplotype is chosen inside each block on its own.
        resolved = covered[resolved_snp[covered]]
        spanned = sorted({int(b) for b in labels[resolved]})
        for block in spanned:
            group = resolved[labels[resolved] == block]
            window = assembly[:, group]
            mismatch = (window != fragment[group][None, :]).astype(int)
            total += float(mismatch.sum(axis=1).min())
        total += (max(len(spanned), 1) - 1) * penalty

    if called == 0:
        raise ValueError("the fragment set must call at least one allele")

    return float(total / called)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a single-block tetraploid assembly with two fragments that
        #     each need one correction (normal scenario) ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, 1, -1],
                  [1, 0, 1, 0, -1],
                  [-1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [1, 1, 0, 1, -1]], dtype=int)
haplotypes = np.array([[0, 1, 0, 1, 1],
                       [1, 0, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0]], dtype=int)
blocks = np.array([1, 1, 1, 1, 1])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Valid: the same assembly cut into two blocks, so that fragments
        #     crossing the cut are charged for it ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, 1, -1],
                  [1, 0, 1, 0, -1],
                  [-1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [1, 1, 0, 1, -1]], dtype=int)
haplotypes = np.array([[0, 1, 0, 1, 1],
                       [1, 0, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0]], dtype=int)
blocks = np.array([1, 1, 2, 2, 2])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Valid: an assembly leaving two SNPs unresolved, which every
        #     fragment covering them must be charged for ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, 1, -1],
                  [1, 0, 1, 0, -1],
                  [-1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, 1]], dtype=int)
haplotypes = np.array([[0, 1, -1, 1, -1],
                       [1, 0, -1, 0, -1],
                       [1, 1, -1, 1, -1],
                       [0, 0, -1, 0, -1]], dtype=int)
blocks = np.array([1, 1, 0, 1, 0])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Valid: a diploid fragment crossing one boundary whose two halves
        #     match different haplotype rows, which costs only the boundary
        #     charge because the labels of the two blocks are unrelated ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
haplotypes = np.array([[0, 0],
                       [1, 1]], dtype=int)
blocks = np.array([1, 2])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
        },
        # --- Valid: the same two fragments with the cut removed, where the
        #     labels are now shared and each fragment costs a real correction ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
haplotypes = np.array([[0, 0],
                       [1, 1]], dtype=int)
blocks = np.array([1, 1])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
        },
        # --- Boundary: a perfect assembly, for which the score must vanish ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, -1, -1],
                  [1, 0, 1, -1, -1],
                  [-1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0]], dtype=int)
haplotypes = np.array([[0, 1, 0, 1, 1],
                       [1, 0, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0]], dtype=int)
blocks = np.array([1, 1, 1, 1, 1])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Boundary: a diploid assembly, where the per-boundary charge takes
        #     its largest relative value ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 1, 0]], dtype=int)
haplotypes = np.array([[0, 1, 0, 1],
                       [1, 0, 1, 0]], dtype=int)
blocks = np.array([1, 1, 2, 3])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 2), 12)",
        },
        # --- Edge: a fragment calling a single allele at a SNP that belongs to
        #     no block, which must be charged one and no boundary ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, -1, -1],
                  [-1, -1, -1, 1],
                  [-1, -1, -1, 0],
                  [1, 0, 1, -1]], dtype=int)
haplotypes = np.array([[0, 1, 0, -1],
                       [1, 0, 1, -1],
                       [1, 1, 1, -1],
                       [0, 0, 0, -1]], dtype=int)
blocks = np.array([1, 1, 1, 0])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Edge: a fragment spanning three blocks, which is charged twice ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1],
                  [-1, -1, 1, 1, -1, -1]], dtype=int)
haplotypes = np.array([[0, 1, 0, 1, 1, 0],
                       [1, 0, 1, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0]], dtype=int)
blocks = np.array([1, 1, 2, 2, 3, 3])
""",
            "call": "round(compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
            "gold_call": "round(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4), 12)",
        },
        # --- Invalid: a ploidy that disagrees with the assembly ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
haplotypes = np.array([[0, 1], [1, 0]], dtype=int)
blocks = np.array([1, 1])
def run_model():
    try:
        compute_block_adjusted_mec(reads, haplotypes, blocks, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative block label ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
haplotypes = np.array([[0, 1], [1, 0]], dtype=int)
blocks = np.array([1, -1])
def run_model():
    try:
        compute_block_adjusted_mec(reads, haplotypes, blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a fragment set in which no allele is called at all ---
        {
            "setup": """import numpy as np
reads = np.array([[-1, -1], [-1, -1]], dtype=int)
haplotypes = np.array([[0, 1], [1, 0]], dtype=int)
blocks = np.array([1, 1])
def run_model():
    try:
        compute_block_adjusted_mec(reads, haplotypes, blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, 2)
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
