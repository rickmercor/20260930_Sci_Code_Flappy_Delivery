"""
Score every candidate phasing of a short run of SNPs against the aligned fragments that cover the whole run, producing the factor potential the phasing carries in the graphical model.

A fragment is an independent noisy observation of one unknown haplotype, so its unnormalised likelihood under a phasing is the sum over the K haplotypes of a per-base error model whose exponent is the Hamming distance between the fragment and that haplotype. Only fragments covering every position of the run carry information about the joint phase of that run.

Returns
-------
np.ndarray of shape (M,), float: the read-evidence potential of each supplied phasing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_phasing_potentials(reads: np.ndarray, phasings: np.ndarray,
                               positions: np.ndarray,
                               error_rate: float) -> np.ndarray:
    """Compute the read-evidence potential of each candidate phasing.

    A fragment contributes only if it has a called allele at every position of
    ``positions``; if no fragment does, the run carries no read evidence and
    every phasing receives the same potential of one.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0 for the reference allele, 1 for the alternate allele and
        -1 where the fragment has no called allele.
    phasings : np.ndarray
        Integer array of shape (M, K, P) with entries in {0, 1} holding the
        candidate phasings of the run.
    positions : np.ndarray
        One-dimensional integer array of length P holding the distinct SNP
        indices of the run, each a valid column index of ``reads``.
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.

    Returns
    -------
    potentials : np.ndarray
        Array of shape (M,) of non-negative floats, the potential of each
        phasing in the order the phasings were supplied.

    Raises
    ------
    ValueError
        If ``reads`` is not a two-dimensional integer-valued array with entries
        in {-1, 0, 1}, if ``phasings`` is not a three-dimensional
        integer-valued array with entries in {0, 1}, if ``positions`` is not a
        one-dimensional array of distinct valid column indices of ``reads``
        whose length equals the last axis of ``phasings``, or if ``error_rate``
        is not a finite number strictly between zero and one.
    """
    return potentials  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_phasing_potentials(reads: np.ndarray, phasings: np.ndarray,
                                       positions: np.ndarray,
                                       error_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    states = np.asarray(phasings, dtype=float)
    if states.ndim != 3 or states.size < 1:
        raise ValueError("phasings must be a non-empty three-dimensional array")
    if not np.all(np.isin(states, (0.0, 1.0))):
        raise ValueError("phasings entries must be 0 or 1")
    states = states.astype(int)

    columns = np.asarray(positions, dtype=float)
    if columns.ndim != 1 or columns.size != states.shape[2]:
        raise ValueError("positions must be one-dimensional and match the phasing width")
    if not np.allclose(columns, np.round(columns), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    columns = np.round(columns).astype(int)
    if np.unique(columns).size != columns.size:
        raise ValueError("positions entries must be distinct")
    if np.any(columns < 0) or np.any(columns >= fragments.shape[1]):
        raise ValueError("positions entries must be valid column indices of reads")

    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    epsilon = float(error_rate)

    width = int(columns.size)
    observed = fragments[:, columns]
    # Only fragments with a called allele at every position of the run inform
    # the joint phase of that run.
    informative = observed[np.all(observed >= 0, axis=1)]
    if informative.shape[0] == 0:
        return np.ones(states.shape[0], dtype=float)

    potentials = np.empty(states.shape[0], dtype=float)
    for m in range(states.shape[0]):
        # distance[r, k] is the Hamming distance between fragment r and
        # haplotype k of this phasing over the run.
        distance = (states[m][None, :, :] != informative[:, None, :]).sum(axis=2)
        likelihood = (epsilon ** distance) * ((1.0 - epsilon) ** (width - distance))
        potentials[m] = float(likelihood.sum())

    return potentials

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a tetraploid SNP pair with four covering fragments, one of
        #     which carries a sequencing error (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1, -1],
                  [1, 0, -1, -1],
                  [1, 1, 0, -1],
                  [0, 0, 1, 1],
                  [1, 1, -1, -1]], dtype=int)
phasings = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]],
                     [[0, 0], [0, 1], [1, 0], [1, 1]],
                     [[0, 0], [0, 0], [1, 1], [1, 1]]], dtype=int)
positions = np.array([0, 1])
error_rate = 0.02
""",
            "call": "sig(compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
            "gold_call": "sig(_oracle_compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
        },
        # --- Valid: the same pair at a much larger error rate, which flattens
        #     the potential across phasings ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1, -1],
                  [1, 0, -1, -1],
                  [1, 1, 0, -1],
                  [0, 0, 1, 1],
                  [1, 1, -1, -1]], dtype=int)
phasings = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]],
                     [[0, 0], [0, 1], [1, 0], [1, 1]],
                     [[0, 0], [0, 0], [1, 1], [1, 1]]], dtype=int)
positions = np.array([0, 1])
error_rate = 0.3
""",
            "call": "sig(compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
            "gold_call": "sig(_oracle_compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
        },
        # --- Valid: a three-position run, where only the single fragment that
        #     spans all three positions contributes ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1, -1],
                  [1, 0, -1, -1],
                  [1, 1, 0, -1],
                  [0, 0, 1, 1],
                  [1, 1, -1, -1]], dtype=int)
phasings = np.array([[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                     [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]], dtype=int)
positions = np.array([0, 1, 2])
error_rate = 0.02
""",
            "call": "sig(compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
            "gold_call": "sig(_oracle_compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
        },
        # --- Boundary: a run that no fragment covers completely, for which the
        #     potential must fall back to a flat one rather than to zero ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1, -1],
                  [1, 0, -1, -1],
                  [1, 1, 0, -1]], dtype=int)
phasings = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                     [[0, 0], [0, 0], [1, 1], [1, 1]]], dtype=int)
positions = np.array([2, 3])
error_rate = 0.02
""",
            "call": "sig(compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
            "gold_call": "sig(_oracle_compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
        },
        # --- Edge: a non-adjacent pair read in descending index order, which
        #     must be scored against the phasing columns as supplied ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1]], dtype=int)
phasings = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]],
                     [[0, 0], [0, 1], [1, 0], [1, 1]],
                     [[0, 0], [0, 0], [1, 1], [1, 1]]], dtype=int)
positions = np.array([3, 0])
error_rate = 0.05
""",
            "call": "sig(compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
            "gold_call": "sig(_oracle_compute_phasing_potentials(reads, phasings, positions, error_rate), 1.0)",
        },
        # --- Invalid: an error rate of exactly one ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 1])
def run_model():
    try:
        compute_phasing_potentials(reads, phasings, positions, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_phasing_potentials(reads, phasings, positions, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a position index outside the fragment matrix ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 1], [1, 0]], dtype=int)
phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 5])
def run_model():
    try:
        compute_phasing_potentials(reads, phasings, positions, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_phasing_potentials(reads, phasings, positions, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a fragment matrix carrying an allele code of 2 ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 2], [1, 0]], dtype=int)
phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 1])
def run_model():
    try:
        compute_phasing_potentials(reads, phasings, positions, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_phasing_potentials(reads, phasings, positions, 0.02)
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
