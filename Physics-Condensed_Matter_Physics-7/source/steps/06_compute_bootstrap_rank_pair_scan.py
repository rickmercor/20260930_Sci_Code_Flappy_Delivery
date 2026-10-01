"""
Evaluate an ordered list of retained-rank pairs with one common nonparametric block-bootstrap table. Each row records support, logarithmic robust spread, the full-record intersector gap, its valid-bootstrap mean, and the additive first-order bootstrap bias correction.

For every sector, retain the requested number of largest density eigenmodes and use those same eigenvectors to reduce the matching shifted-energy matrix. A bootstrap replicate is valid only when every retained density eigenvalue is strictly positive and strictly larger than its sector cutoff times the largest eigenvalue, and when its lowest Q=-1 energy minus its lowest Q=+1 energy is finite and strictly positive. For every nonempty set of valid gaps ``Delta``, including sets of size one or two, use ``1.4826*median(|log(Delta)-median(log(Delta))|)`$and $\Delta_{\mathrm{bc}}=2\Delta_{\mathrm{full}}-\operatorname{mean}(\Delta_{\mathrm{boot}})$. A single valid gap therefore has zero log-MAD. If no replicate is valid, set$$log_mad$`, `$bootstrap_mean$`, and `$bias_corrected_gap$` to the largest finite binary64 value; `$central_gap$$retains its full-record value and$$valid_fraction$` is zero. The identical seeded resampling table is used for every candidate pair. The full-record central spectrum requires strictly positive retained density eigenvalues and a finite positive intersector gap; the relative cutoffs apply only to bootstrap replicates.

Returns
-------
np.ndarray, a float array with seven columns and one row per candidate pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_bootstrap_rank_pair_scan(
    block_matrices: np.ndarray,
    energy_shift: float,
    plus_dimension: int,
    candidate_pairs: np.ndarray,
    relative_cutoffs: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> np.ndarray:
    """Compute a shared-bootstrap stability scan over sector-rank pairs.

    Parameters
    ----------
    block_matrices : np.ndarray
        Finite array of shape ``(2, 2, blocks, d, d)``. Sector 0 occupies
        the leading ``plus_dimension`` coordinates; sector 1 occupies all
        ``d`` coordinates. Matrix channel 0 is density and channel 1 is
        shifted energy. There must be at least three blocks and d >= 3.
    energy_shift : float
        Finite additive energy shift applied to both generalized spectra.
    plus_dimension : int
        Integer dimension of the leading Q=+1 matrix block, in [2, d].
    candidate_pairs : np.ndarray
        Nonempty integer array of shape ``(candidates, 2)`` containing
        ``(r_plus, r_minus)`` in the exact order to be assessed, with no
        duplicate pair. Require 2 <= r_plus <= plus_dimension and
        2 <= r_minus <= d.
    relative_cutoffs : np.ndarray
        Two finite values in ``[0, 1)`` for Q=+1 and Q=-1 bootstrap spectra.
        A retained bootstrap mode must obey both ``lambda > 0`` and
        ``lambda > cutoff*lambda_max``. For the full-record central spectra,
        retained modes must be strictly positive; relative cutoffs do not apply.
    n_bootstrap : int
        Number of common block-bootstrap replicates, at least five.
    bootstrap_seed : int
        Nonnegative seed for ``np.random.default_rng``.

    Returns
    -------
    scan : np.ndarray
        Float array of shape ``(candidates, 7)`` with columns
        ``r_plus, r_minus, valid_fraction, log_mad, central_gap,
        bootstrap_mean, bias_corrected_gap``. For every nonempty set of
        valid bootstrap gaps, including one or two gaps, compute log_mad
        with the stated median formula. If no replicate is valid, log_mad,
        bootstrap_mean, and bias_corrected_gap use the largest finite
        binary64 value as a sentinel; valid_fraction is zero and central_gap
        retains its full-record value.

    Raises
    ------
    ValueError
        If an input is invalid, a candidate is duplicated or out of range,
        or a central candidate spectrum does not yield a positive finite
        intersector gap.
    """
    return scan

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_compute_bootstrap_rank_pair_scan(
    block_matrices: np.ndarray,
    energy_shift: float,
    plus_dimension: int,
    candidate_pairs: np.ndarray,
    relative_cutoffs: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    blocks = np.asarray(block_matrices, dtype=float)
    if (
        blocks.ndim != 5
        or blocks.shape[0] != 2
        or blocks.shape[1] != 2
        or blocks.shape[2] < 3
        or blocks.shape[3] < 3
        or blocks.shape[3] != blocks.shape[4]
        or not np.all(np.isfinite(blocks))
    ):
        raise ValueError("block_matrices must have finite shape (2,2,blocks>=3,d,d), d>=3")
    d = int(blocks.shape[3])
    if (
        isinstance(plus_dimension, bool)
        or not isinstance(plus_dimension, Integral)
        or not (2 <= int(plus_dimension) <= d)
    ):
        raise ValueError("plus_dimension must be an integer in [2,d]")
    p_dim = int(plus_dimension)

    raw_pairs = np.asarray(candidate_pairs)
    if (
        raw_pairs.ndim != 2
        or raw_pairs.shape[0] < 1
        or raw_pairs.shape[1] != 2
        or not np.issubdtype(raw_pairs.dtype, np.integer)
        or np.issubdtype(raw_pairs.dtype, np.bool_)
    ):
        raise ValueError("candidate_pairs must have nonempty integer shape (candidates,2)")
    pairs = raw_pairs.astype(int, copy=False)
    if len({(int(a), int(b)) for a, b in pairs}) != pairs.shape[0]:
        raise ValueError("candidate_pairs must not contain duplicates")
    if np.any(pairs[:, 0] < 2) or np.any(pairs[:, 0] > p_dim):
        raise ValueError("a Q=+1 candidate rank is out of range")
    if np.any(pairs[:, 1] < 2) or np.any(pairs[:, 1] > d):
        raise ValueError("a Q=-1 candidate rank is out of range")

    cuts = np.asarray(relative_cutoffs, dtype=float)
    if cuts.shape != (2,) or not np.all(np.isfinite(cuts)) or np.any(cuts < 0.0) or np.any(cuts >= 1.0):
        raise ValueError("relative_cutoffs must contain two finite values in [0,1)")
    if isinstance(energy_shift, bool) or not isinstance(energy_shift, Real) or not math.isfinite(float(energy_shift)):
        raise ValueError("energy_shift must be a finite real scalar")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or int(n_bootstrap) < 5:
        raise ValueError("n_bootstrap must be an integer >= 5")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral) or int(bootstrap_seed) < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")

    def _retained_spectrum(z_matrix, e_matrix, rank, cutoff, enforce_cutoff):
        z = 0.5 * (z_matrix + z_matrix.T)
        e = 0.5 * (e_matrix + e_matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(z)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
            return None
        retained = eigenvalues[-int(rank):]
        if np.any(retained <= 0.0):
            return None
        if enforce_cutoff and np.any(retained <= float(cutoff) * eigenvalues[-1]):
            return None
        u = eigenvectors[:, -int(rank):]
        scales = np.sqrt(retained)
        h_eff = (u.T @ e @ u) / (scales[:, None] * scales[None, :])
        h_eff = 0.5 * (h_eff + h_eff.T) + float(energy_shift) * np.eye(int(rank))
        energies = np.linalg.eigvalsh(h_eff)
        if not np.all(np.isfinite(energies)):
            return None
        return energies

    block_count = int(blocks.shape[2])
    indices = np.random.default_rng(int(bootstrap_seed)).integers(
        0, block_count, size=(int(n_bootstrap), block_count)
    )
    central_plus_z = np.mean(blocks[0, 0, :, :p_dim, :p_dim], axis=0)
    central_plus_e = np.mean(blocks[0, 1, :, :p_dim, :p_dim], axis=0)
    central_minus_z = np.mean(blocks[1, 0], axis=0)
    central_minus_e = np.mean(blocks[1, 1], axis=0)
    max_float = np.finfo(float).max
    scan = np.empty((pairs.shape[0], 7), dtype=float)

    for row_index, (rank_plus, rank_minus) in enumerate(pairs):
        central_plus = _retained_spectrum(
            central_plus_z, central_plus_e, int(rank_plus), cuts[0], False
        )
        central_minus = _retained_spectrum(
            central_minus_z, central_minus_e, int(rank_minus), cuts[1], False
        )
        if central_plus is None or central_minus is None:
            raise ValueError("a central retained density spectrum is not positive")
        central_gap = float(central_minus[0] - central_plus[0])
        if not math.isfinite(central_gap) or central_gap <= 0.0:
            raise ValueError("a central candidate does not have a positive finite intersector gap")

        valid_values = []
        for bootstrap_row in indices:
            plus_z = np.mean(blocks[0, 0, bootstrap_row, :p_dim, :p_dim], axis=0)
            plus_e = np.mean(blocks[0, 1, bootstrap_row, :p_dim, :p_dim], axis=0)
            minus_z = np.mean(blocks[1, 0, bootstrap_row], axis=0)
            minus_e = np.mean(blocks[1, 1, bootstrap_row], axis=0)
            plus_energies = _retained_spectrum(
                plus_z, plus_e, int(rank_plus), cuts[0], True
            )
            minus_energies = _retained_spectrum(
                minus_z, minus_e, int(rank_minus), cuts[1], True
            )
            if plus_energies is None or minus_energies is None:
                continue
            gap = float(minus_energies[0] - plus_energies[0])
            if math.isfinite(gap) and gap > 0.0:
                valid_values.append(gap)

        valid = np.asarray(valid_values, dtype=float)
        fraction = float(valid.size) / float(int(n_bootstrap))
        if valid.size == 0:
            spread = max_float
            bootstrap_mean = max_float
            corrected = max_float
        else:
            bootstrap_mean = float(np.mean(valid))
            corrected = float(2.0 * central_gap - bootstrap_mean)
            logs = np.log(valid)
            center = np.median(logs)
            spread = float(1.4826 * np.median(np.abs(logs - center)))
        scan[row_index] = (
            float(rank_plus),
            float(rank_minus),
            fraction,
            spread,
            central_gap,
            bootstrap_mean,
            corrected,
        )
    return scan

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
B=5; D=4; E0=10.0
blocks=np.zeros((2,2,B,D,D),dtype=float)
for b in range(B):
 dp=np.array([1.0+0.02*b,2.0-0.01*b])
 dm=np.array([0.4+0.01*b,0.6-0.005*b,0.8+0.01*b,1.0-0.01*b])
 blocks[0,0,b,:2,:2]=np.diag(dp)
 blocks[0,1,b,:2,:2]=np.diag(dp*np.array([1.0-E0,2.0-E0]))
 blocks[1,0,b]=np.diag(dm)
 blocks[1,1,b]=np.diag(dm*(np.array([3.0,4.0,5.0,6.0])-E0))
pairs=np.array([[2,4],[2,3]],dtype=int)
cuts=np.array([0.2,0.2],dtype=float)"""
    return [
        {
            "setup": base,
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),17,7)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),17,7)",
        },
        {
            "setup": base + "\ncuts=np.array([0.6,0.7])",
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),19,3)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),19,3)",
        },
        {
            "setup": base + "\nblocks[1,1,2,0,1]=0.03; blocks[1,1,2,1,0]=0.03",
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),-2.0,2,pairs.copy(),cuts.copy(),23,1)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),-2.0,2,pairs.copy(),cuts.copy(),23,1)",
        },
        {
            "setup": base,
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),17,8)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),17,8)",
        },
        {
            "setup": """import numpy as np
blocks=np.zeros((2,2,2,4,4)); pairs=np.array([[2,3]]); cuts=np.array([0.2,0.2])
def run_model():
 try:
  compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": base + """
pairs=np.array([[2,3],[2,3]],dtype=int)
def run_model():
 try:
  compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": base + """
blocks[1,1,:,:,:]=blocks[1,0,:,:,:]*(-20.0)
def run_model():
 try:
  compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),E0,2,pairs.copy(),cuts.copy(),7,0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\nblocks=np.zeros((2,2,3,3,3))\nfor b,v in enumerate([0.9,0.05,0.05]):\n    blocks[0,0,b,:2,:2]=np.eye(2)\n    blocks[0,1,b,:2,:2]=np.diag([1.0,2.0])\n    blocks[1,0,b]=np.diag([0.01,v,1.0])\n    blocks[1,1,b]=np.diag([0.03,3*v,4.0])\npairs=np.array([[2,2]])\ncuts=np.array([0.2,0.5])\n",
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),5,2)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),5,2)"
        },
        {
            "setup": "import numpy as np\nblocks=np.zeros((2,2,3,3,3))\nfor b,v in enumerate([0.9,0.05,0.05]):\n    blocks[0,0,b,:2,:2]=np.eye(2)\n    blocks[0,1,b,:2,:2]=np.diag([1.0,2.0])\n    blocks[1,0,b]=np.diag([0.01,v,1.0])\n    blocks[1,1,b]=np.diag([0.03,3*v,4.0])\npairs=np.array([[2,2]])\ncuts=np.array([0.2,0.5])\n",
            "call": "compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),5,0)",
            "gold_call": "_oracle_compute_bootstrap_rank_pair_scan(blocks.copy(),0.0,2,pairs.copy(),cuts.copy(),5,0)"
        },
    ]
