"""
Compute the Jordan-Wigner pair string phases alpha[m, n, q] that dress the fermion hopping
between sites m and n in the cluster described by cfg.

cfg describes an open nx x ny cluster of N = nx ny spin-1/2 sites (x, y), numbered p = (x if y
is even else nx - 1 - x) + y nx. The spins are mapped to spinless fermions, one orbital per site
with orbital index p, by the Jordan-Wigner transformation s^+_p = c^dag_p exp(i pi sum_{q<p}
n_q), s^z_p = n_p - 1/2, n_p = c^dag_p c_p. The sector is M = 0: N is even and every determinant
holds N_f = N/2 fermions.

Write the string of site p as exp(i sum_q theta_pq n_q) with theta_pq = pi for q < p and 0
otherwise (theta_pp = 0). In s^+_m s^-_n = c^dag_m c_n exp(i sum_q (theta_mq - theta_nq) n_q)
the terms q = m and q = n drop out next to c^dag_m and c_n, so the pair string runs over q not
in {m, n} only.

alpha[m, n, q] = theta_mq - theta_nq for q not in {m, n}, exactly as that difference (not
reduced modulo 2 pi), and alpha[m, n, q] = 0 for q = m or q = n; the array covers every ordered
pair (m, n), m = n included.

Returns
-------
alpha : np.ndarray -- Real array of shape (N, N, N), indexed [m, n, q].
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def jw_pair_phases(cfg: dict) -> "np.ndarray":
    '''Compute the Jordan-Wigner pair string phases alpha[m, n, q] that dress the fermion
    hopping between sites m and n in the cluster described by cfg.

    Parameters
    ----------
    cfg : dict
        Cluster parameters. Keys read:
            nx : number of columns, x = 0, ..., nx-1
            ny : number of rows, y = 0, ..., ny-1 (ny = 1 is a chain)

    Returns
    -------
    alpha : np.ndarray
        Real array of shape (N, N, N), indexed [m, n, q].
    '''
    return alpha

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_jw_pair_phases(cfg: dict) -> "np.ndarray":
    N = cfg["nx"] * cfg["ny"]
    theta = np.pi * np.tril(np.ones((N, N)), -1)
    alpha = theta[:, None, :] - theta[None, :, :]
    k = np.arange(N)
    alpha[k, :, k] = 0.0
    alpha[:, k, k] = 0.0
    return alpha

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: a four-site chain, full array
        {
            "setup": 'cfg = dict(nx=4, ny=1, J1=1.0, J2=0.0, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'jw_pair_phases(cfg) + 4.0',
            "gold_call": '_oracle_jw_pair_phases(cfg_ref) + 4.0',
        },
        # normal: a 3x2 ladder, full array
        {
            "setup": 'cfg = dict(nx=3, ny=2, J1=1.0, J2=0.6, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'jw_pair_phases(cfg) + 4.0',
            "gold_call": '_oracle_jw_pair_phases(cfg_ref) + 4.0',
        },
        # normal: a 5x4 cluster, index-weighted sum and one pair string
        {
            "setup": 'import numpy as np\ncfg = dict(nx=5, ny=4, J1=1.0, J2=0.6, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": '(lambda a: np.concatenate([[np.sum(a * np.arange(a.size).reshape(a.shape))], a[17, 3] + 4.0, a[2, 19] + 4.0]))(jw_pair_phases(cfg))',
            "gold_call": '(lambda a: np.concatenate([[np.sum(a * np.arange(a.size).reshape(a.shape))], a[17, 3] + 4.0, a[2, 19] + 4.0]))(_oracle_jw_pair_phases(cfg_ref))',
        },
        # edge: two sites, where no string survives
        {
            "setup": 'cfg = dict(nx=2, ny=1, J1=1.0, J2=0.0, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'jw_pair_phases(cfg) + 4.0',
            "gold_call": '_oracle_jw_pair_phases(cfg_ref) + 4.0',
        },
    ]
