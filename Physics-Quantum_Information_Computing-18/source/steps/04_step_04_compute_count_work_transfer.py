"""
Compute joint physical count, terminal-state and work coefficients for every initial cycle.

The work counter counts every attempted elementary edge proposal, including

rejected proposals and the first trial of an immediately returning update.

For arrival state $j$ from state $i$, coefficient $r$ means

$\mathbb E[\binom{W}{r}\mathbf1_{\{j\}}\mid i]$ for $r=0,1,2,3$.

These are Taylor coefficients at $z=1$ of the probability generating function;

they are not raw moments or factorial derivatives. Costs add along a path.



A transition precedes each observation. For $N$ records, retain the joint law

of accumulated physical edge counts $c$, terminal cycle $j$, and total work,

conditional on each possible initial cycle $i$. Do not discard $j$ or the

work-count correlation. Return rows ordered lexicographically by $(c,j)$;

include a row exactly when at least one initial state gives it positive mass.

Within a row the coefficient block is flattened with initial-state index first

and order index last.

Returns
-------
Return the joint count and terminal-state table with four work coefficients for every initial cycle.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_count_work_transfer(
    recording_kernel: "np.ndarray", physical_errors: "np.ndarray", n_samples: int
) -> "np.ndarray":
    r"""Compute joint physical count, terminal-state and work coefficients for every initial cycle.

    Parameters
    ----------
    recording_kernel : np.ndarray
        Nonnegative finite array $(k,k,4)$ of binomial work coefficients;
        its order-zero rows sum to one.
    physical_errors : np.ndarray
        Distinct binary physical rows $x_i$ of shape $(k,E)$, $1\le k,E\le8$.
        Row order agrees with the recording kernel.
    n_samples : int
        Number of recorded samples, $1\le N\le3$.

    Returns
    -------
    transfer : np.ndarray
        Table of shape $(J,E+1+4k)$. Columns are the $E$ counts, terminal
        index $j$, then $\mathbb E[\binom{W}{r}\mathbf1_{\{c,j\}}\mid i]$
        for $i=0,\ldots,k-1$ and $r=0,1,2,3$, in that nested order.

    Raises
    ------
    ValueError
        If physical rows, coefficient dimensions/nonnegativity, zeroth-order
        stochasticity or the sample count violate the stated contract.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _work_kernel(kernel, n_states):
    kernel = np.asarray(kernel, dtype=float)
    if (
        kernel.shape != (n_states, n_states, 4)
        or not np.all(np.isfinite(kernel))
        or np.any(kernel < 0)
    ):
        raise ValueError("work kernel must be a finite nonnegative (k,k,4) array")
    if not np.allclose(kernel[..., 0].sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("zeroth kernel coefficient must be row stochastic")
    return kernel


def _observations(physical):
    physical = _binary(physical, 2, "physical_errors")
    if not 1 <= len(physical) <= 8 or not 1 <= physical.shape[1] <= 8:
        raise ValueError("physical observation dimensions exceed the contract")
    if len(np.unique(physical, axis=0)) != len(physical):
        raise ValueError("physical rows must be distinct")
    return physical


def _history_work(kernel, observations, samples):
    k = len(kernel)
    initial = np.zeros_like(kernel)
    initial[..., 0] = np.eye(k)
    history = {(0,) * observations.shape[1]: initial}
    for _ in range(samples):
        updated = {}
        for counts, mass in history.items():
            arrival = _matrix_jet_product(mass, kernel)
            for terminal in range(k):
                if not np.any(arrival[:, terminal, 0] > 0):
                    continue
                key = tuple(np.asarray(counts) + observations[terminal])
                if key not in updated:
                    updated[key] = np.zeros_like(kernel)
                updated[key][:, terminal] += arrival[:, terminal]
        history = updated
    return history


def _oracle_compute_count_work_transfer(
    recording_kernel: "np.ndarray", physical_errors: "np.ndarray", n_samples: int
) -> "np.ndarray":
    physical = _observations(physical_errors)
    kernel = _work_kernel(recording_kernel, len(physical))
    samples = _integer(n_samples, 1, 3, "n_samples")
    history = _history_work(kernel, physical, samples)
    rows = []
    for counts, mass in sorted(history.items()):
        for terminal in range(len(physical)):
            if np.any(mass[:, terminal, 0] > 0):
                rows.append([*counts, terminal, *mass[:, terminal].ravel()])
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return distinct normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.7,0.3],[0.2,0.8]])
costs=np.array([[1,2],[3,1]])
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,1,0],[1,0,1]])
""",
            "call": "compute_count_work_transfer(K.copy(), X.copy(), 3)",
            "gold_call": "_oracle_compute_count_work_transfer(K.copy(), X.copy(), 3)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
K=np.array([[[1.,2.,1.,0.]]])
X=np.array([[1,0]])
""",
            "call": "compute_count_work_transfer(K.copy(), X.copy(), 3)",
            "gold_call": "_oracle_compute_count_work_transfer(K.copy(), X.copy(), 3)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.1,0.8,0.1],[0.4,0.2,0.4],[0.3,0.2,0.5]])
costs=np.array([[1,3,2],[4,2,1],[1,2,3]])
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,0],[1,0],[0,1]])
""",
            "call": "compute_count_work_transfer(K.copy(), X.copy(), 2)",
            "gold_call": "_oracle_compute_count_work_transfer(K.copy(), X.copy(), 2)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.7,0.3],[0.2,0.8]])
costs=np.array([[1,2],[3,1]])
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,1,0],[1,0,1]])
K[0,0,0]=0.9

def _check_error(fn):
    try:
        fn(K.copy(), X.copy(), 2)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_count_work_transfer)",
            "gold_call": "_check_error(_oracle_compute_count_work_transfer)",
        },
    ]
