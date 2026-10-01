"""
Compute joint physical plurality-report and work coefficients from every initial cycle.

The work counter counts every attempted elementary edge proposal, including

rejected proposals and the first trial of an immediately returning update.

For arrival state $j$ from state $i$, coefficient $r$ means

$\mathbb E[\binom{W}{r}\mathbf1_{\{j\}}\mid i]$ for $r=0,1,2,3$.

These are Taylor coefficients at $z=1$ of the probability generating function;

they are not raw moments or factorial derivatives. Costs add along a path.



Physical label bits are $Lx\pmod2$, with row zero the least significant bit.

Record after each transition and choose the largest realized logical-class

count, breaking ties by the smallest physical integer label. Preserve the

joint dependence of work, terminal state and class histogram until the last

record. The initial state is not an observation. Labels need not be distinct

across physical error rows, and some label probabilities can be zero.

Returns
-------
Return all physical reported-label work coefficients conditional on each initial cycle.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_report_work_transfer(
    recording_kernel: "np.ndarray",
    physical_errors: "np.ndarray",
    logical: "np.ndarray",
    n_samples: int,
) -> "np.ndarray":
    r"""Compute joint physical plurality-report and work coefficients from every initial cycle.

    Parameters
    ----------
    recording_kernel : np.ndarray
        Finite nonnegative binomial-work kernel of shape $(k,k,4)$,
        with a row-stochastic zeroth coefficient.
    physical_errors : np.ndarray
        Distinct binary physical configurations $(k,E)$, $1\le k,E\le8$,
        ordered consistently with the kernel.
    logical : np.ndarray
        Binary map $(r,E)$, $1\le r\le2$, with least significant bit first.
    n_samples : int
        Number of correlated records, $1\le N\le5$.

    Returns
    -------
    report_coefficients : np.ndarray
        Array $(k,2^r,4)$ with entry $(i,g,s)$ equal to
        $\mathbb E[\binom{W}{s}\mathbf1_{\{\widehat g=g\}}\mid i]$.
        Keep the full label range, including unattainable labels.

    Raises
    ------
    ValueError
        If physical rows, logical map, coefficient kernel, stochasticity
        or sample count violates its stated contract.

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


def _logical_classes(physical, logical):
    logical = _binary(logical, 2, "logical")
    if logical.shape[1] != physical.shape[1] or not 1 <= len(logical) <= 2:
        raise ValueError("logical must have one or two rows and E columns")
    labels = ((physical @ logical.T) % 2) @ (1 << np.arange(len(logical)))
    return labels, 1 << len(logical)


def _oracle_compute_report_work_transfer(
    recording_kernel: "np.ndarray",
    physical_errors: "np.ndarray",
    logical: "np.ndarray",
    n_samples: int,
) -> "np.ndarray":
    physical = _observations(physical_errors)
    kernel = _work_kernel(recording_kernel, len(physical))
    samples = _integer(n_samples, 1, 5, "n_samples")
    labels, n_classes = _logical_classes(physical, logical)
    observations = np.eye(n_classes, dtype=int)[labels]
    history = _history_work(kernel, observations, samples)
    result = np.zeros((len(physical), n_classes, 4))
    for counts, mass in history.items():
        result[:, int(np.argmax(counts))] += mass.sum(axis=1)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return distinct normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.5,0.2,0.2,0.1],[0.1,0.6,0.1,0.2],[0.2,0.1,0.5,0.2],[0.3,0.2,0.1,0.4]])
costs=1+(np.arange(4)[:,None]+2*np.arange(4)[None,:])%4
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,0],[1,0],[0,1],[1,1]])
L=np.eye(2,dtype=int)
""",
            "call": "compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 5)",
            "gold_call": "_oracle_compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 5)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
K=np.array([[[1.,2.,1.,0.]]])
X=np.array([[1,0]])
L=np.array([[1,0]])
""",
            "call": "compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 1)",
            "gold_call": "_oracle_compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 1)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.5,0.2,0.2,0.1],[0.1,0.6,0.1,0.2],[0.2,0.1,0.5,0.2],[0.3,0.2,0.1,0.4]])
costs=1+(np.arange(4)[:,None]+2*np.arange(4)[None,:])%4
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,0],[1,0],[0,1],[1,1]])
L=np.eye(2,dtype=int)
L=np.array([[1,0],[0,0]])
""",
            "call": "compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 2)",
            "gold_call": "_oracle_compute_report_work_transfer(K.copy(), X.copy(), L.copy(), 2)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from math import comb
P=np.array([[0.5,0.2,0.2,0.1],[0.1,0.6,0.1,0.2],[0.2,0.1,0.5,0.2],[0.3,0.2,0.1,0.4]])
costs=1+(np.arange(4)[:,None]+2*np.arange(4)[None,:])%4
K=np.stack([P*np.vectorize(lambda w: comb(int(w),r))(costs) for r in range(4)],axis=-1)
X=np.array([[0,0],[1,0],[0,1],[1,1]])
L=np.eye(2,dtype=int)
L[0,0]=2

def _check_error(fn):
    try:
        fn(K.copy(), X.copy(), L.copy(), 2)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_report_work_transfer)",
            "gold_call": "_check_error(_oracle_compute_report_work_transfer)",
        },
    ]
