"""
Compute the completed-update recording kernel with work coefficients through order three.

The work counter counts every attempted elementary edge proposal, including

rejected proposals and the first trial of an immediately returning update.

For arrival state $j$ from state $i$, coefficient $r$ means

$\mathbb E[\binom{W}{r}\mathbf1_{\{j\}}\mid i]$ for $r=0,1,2,3$.

These are Taylor coefficients at $z=1$ of the probability generating function;

they are not raw moments or factorial derivatives. Costs add along a path.



Keep the tail fixed throughout each excursion and reselect it uniformly for

each completed update. The boundary vertex is included. Eliminate every open

excursion length exactly before composing the prescribed recording spacing.

Initial rejection completes a one-trial update; rejection at an open state

adds work without completing that update. The returned coefficients retain

correlation between work and the arrival relative cycle.

Returns
-------
Return the arrival-state recording kernel with binomial work coefficients of orders zero through three.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_work_recording_kernel(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    spacing: int,
) -> "np.ndarray":
    r"""Compute the completed-update recording kernel with work coefficients through order three.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.
    probabilities : np.ndarray
        Physical priors of shape $(E,)$ with finite $0<q_e<1$.
    spacing : int
        Number of completed updates before recording, $1\le a\le3$.

    Returns
    -------
    coefficients : np.ndarray
        Nonnegative array of shape $(k,k,4)$ in relative-mask order, where
        $k=2^{E-n+1}$ and the last index is binomial work order $r$.
        The order-zero matrix is row stochastic.

    Raises
    ------
    ValueError
        If graph/reference constraints, probabilities or spacing are invalid,
        or the first-return systems have no numerically finite moments.

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


def _jet_product(left, right):
    left, right = np.broadcast_arrays(left, right)
    out = np.zeros_like(left, dtype=float)
    for degree in range(4):
        for split in range(degree + 1):
            out[..., degree] += left[..., split] * right[..., degree - split]
    return out


def _matrix_jet_product(left, right):
    out = np.zeros((left.shape[0], right.shape[1], 4))
    for degree in range(4):
        for split in range(degree + 1):
            out[..., degree] += left[..., split] @ right[..., degree - split]
    return out


def _oracle_compute_work_recording_kernel(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    spacing: int,
) -> "np.ndarray":
    incidence = _graph(detectors)
    k = len(_oracle_construct_physical_cycle_ensemble(detectors, reference))
    spacing = _integer(spacing, 1, 3, "spacing")
    jets = np.zeros((k, k, 4))
    for tail in range(len(incidence)):
        transition = _oracle_build_oriented_transition(
            detectors, probabilities, reference, tail
        )
        direct = transition[:k, :k]
        departure = transition[:k, k:]
        arrival = transition[k:, :k]
        transient = transition[k:, k:]
        system = np.eye(len(transient)) - transient
        try:
            resolvents = [np.linalg.solve(system, arrival)]
            for _ in range(3):
                resolvents.append(np.linalg.solve(system, transient @ resolvents[-1]))
        except np.linalg.LinAlgError as error:
            raise ValueError("excursions must have finite moments") from error
        for degree in range(4):
            term = resolvents[degree].copy()
            if degree >= 1:
                term += 2.0 * resolvents[degree - 1]
            if degree >= 2:
                term += resolvents[degree - 2]
            jets[..., degree] += departure @ term
            if degree <= 1:
                jets[..., degree] += direct
    jets /= len(incidence)
    if not np.all(np.isfinite(jets)) or np.min(jets) < -1e-8:
        raise ValueError("first-return moments are not numerically finite")
    jets = np.maximum(jets, 0.0)
    result = np.zeros_like(jets)
    result[..., 0] = np.eye(k)
    for _ in range(spacing):
        result = _matrix_jet_product(result, jets)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return distinct normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
HZ = np.array([[1,1,1,0,0,0,1],[1,0,0,1,1,0,0],
               [0,1,0,1,0,1,0],[0,0,1,0,1,1,0]])
HX = np.array([[1,1,0,1,0,0,0],[1,0,1,0,1,1,0],
               [0,1,1,0,0,0,1],[0,0,0,1,1,0,0]])
MZ = np.array([0,1,0,0,0,0,1])
MX = np.array([1,0,0,0,0,1,0])
SZ = np.array([0,0,1,0])
SX = np.array([1,0,0,0])
LX = np.array([[1,0,0,0,0,0,0],[0,0,1,0,0,0,0]])
mapping = np.array([3,6,1,5,0,4,2])
samples = np.array([2,2,2,5])
spacings = np.array([2,2,1,2])
p = 0.24
""",
            "call": "compute_work_recording_kernel(HZ.copy(), np.full(7,2*p/3), MZ.copy(), 2)",
            "gold_call": "_oracle_compute_work_recording_kernel(HZ.copy(), np.full(7,2*p/3), MZ.copy(), 2)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
H=np.array([[1]])
M=np.array([0])
""",
            "call": "compute_work_recording_kernel(H.copy(), np.array([0.5]), M.copy(), 2)",
            "gold_call": "_oracle_compute_work_recording_kernel(H.copy(), np.array([0.5]), M.copy(), 2)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ = np.array([[1,0,1],[1,1,0]])
HX = HZ.copy()
MZ = np.array([0,0,0])
MX = np.array([1,0,0])
SZ = HZ @ MZ % 2
SX = HX @ MX % 2
LX = np.array([[1,0,0]])
mapping = np.array([2,0,1])
samples = np.array([2,1,2,3])
spacings = np.array([1,2,1,1])
p = 0.36
""",
            "call": "compute_work_recording_kernel(HX.copy(), np.array([0.08,0.7,0.33]), MX.copy(), 3)",
            "gold_call": "_oracle_compute_work_recording_kernel(HX.copy(), np.array([0.08,0.7,0.33]), MX.copy(), 3)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ = np.array([[1,0,1],[1,1,0]])
HX = HZ.copy()
MZ = np.array([0,0,0])
MX = np.array([1,0,0])
SZ = HZ @ MZ % 2
SX = HX @ MX % 2
LX = np.array([[1,0,0]])
mapping = np.array([2,0,1])
samples = np.array([2,1,2,3])
spacings = np.array([1,2,1,1])
p = 0.36

def _check_error(fn):
    try:
        fn(HX.copy(), np.full(3,0.2), MX.copy(), 0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_work_recording_kernel)",
            "gold_call": "_check_error(_oracle_compute_work_recording_kernel)",
        },
    ]
