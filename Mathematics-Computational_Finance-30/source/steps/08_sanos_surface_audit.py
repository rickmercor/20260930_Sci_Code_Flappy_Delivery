"""
Run the whole pipeline for the given scenario index and report the audit table. Build the scenario's quoted mids exactly as the problem statement specifies, build the three maps with steps 2, 3 and 4, hand them to step 5 to fit the marginals, then for each of the four probe points listed in the problem statement report the model value from step 6 and the local variance from step 7, in that column order and in the probe order given. Report only computed quantities; the probe coordinates are inputs. This step must call the earlier step functions; do not re-implement any of them here.

This is the integration step. Every convention the earlier steps fix feeds the reported table, so a single wrong reading anywhere moves the reported numbers.

Returns
-------
A float64 matrix with one row per probe point and two columns: model value, local variance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sanos_surface_audit(scenario: int) -> "np.ndarray":
    """Run the whole pipeline for one scenario index and report the audit table at the four probe points.

    Parameters
    ----------
    scenario : int
        Non-negative scenario index; only scenario mod 5 affects the quoted dispersions, as the problem statement specifies.

    Returns
    -------
    audit : numpy.ndarray
        float64 array of shape (4, 2): row r holds the model value and the local variance at probe r, in the problem statement's probe order.

    Raises
    ------
    ValueError
        If scenario is negative.
    """
    return audit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def _oracle_sanos_surface_audit(scenario: int) -> "np.ndarray":
    """Chain steps 1-7 and return a (n_probe, 2) matrix of computed quantities:
    [smooth model value, local variance] per probe point."""
    s = int(scenario)
    if s < 0:
        raise ValueError("scenario must be a non-negative integer")

    K = np.array([0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30])
    T = np.array([0.20, 0.50, 1.00])
    V = np.array([0.040, 0.075, 0.140])
    eta = 0.25

    shift = 0.002 * ((s % 5) - 2)
    Vq = np.array([V[j] + 0.030 * (1.0 - K) for j in range(T.size)])
    Vq[1] -= (0.022 + shift) * np.exp(-((K - 1.00) / 0.16) ** 2)
    Vq[2] -= (0.030 + shift) * np.exp(-((K - 1.05) / 0.14) ** 2)
    mids = np.vstack([_oracle_bs_call_price(np.ones_like(K), K, Vq[j])
                      for j in range(T.size)])
    W = np.ones_like(mids)

    # chain the maps explicitly so every earlier step is reached from here
    C = np.stack([_oracle_model_price_matrix(K, V[j], eta) for j in range(T.size)])
    U = _oracle_payoff_matrix(K)
    blk = _oracle_marginal_constraints(K)
    q = _oracle_solve_marginals(C, U, blk, mids, W)

    probes = [(0.75, 1.075), (0.40, 0.925), (0.90, 1.150), (0.30, 0.850)]
    out = np.zeros((len(probes), 2), dtype=np.float64)
    for r, (tq, kq) in enumerate(probes):
        out[r, 0] = _oracle_smooth_surface_price(K, T, V, q, tq, kq, eta)
        out[r, 1] = _oracle_discrete_local_variance(K, T, V, q, tq, kq, eta)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np',
            'call': 'sanos_surface_audit(0)',
            'gold_call': '_oracle_sanos_surface_audit(0)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'sanos_surface_audit(3)',
            'gold_call': '_oracle_sanos_surface_audit(3)',
        },
        {
            'setup': 'import numpy as np\ndef colsums(m):\n    return (float(np.sum(m[:, 0])), float(np.sum(m[:, 1])))\n',
            'call': 'colsums(sanos_surface_audit(2))',
            'gold_call': 'colsums(_oracle_sanos_surface_audit(2))',
        },
        {
            'setup': 'import numpy as np',
            'call': 'sanos_surface_audit(7)',
            'gold_call': '_oracle_sanos_surface_audit(7)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:\n        sanos_surface_audit(-1)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_sanos_surface_audit(-1)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
