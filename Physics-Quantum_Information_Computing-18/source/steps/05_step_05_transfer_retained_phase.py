"""
Transfer a joint retained-state and accumulated-work law through one empirical-feedback phase.

The work counter counts every attempted elementary edge proposal, including

rejected proposals and the first trial of an immediately returning update.

For arrival state $j$ from state $i$, coefficient $r$ means

$\mathbb E[\binom{W}{r}\mathbf1_{\{j\}}\mid i]$ for $r=0,1,2,3$.

These are Taylor coefficients at $z=1$ of the probability generating function;

they are not raw moments or factorial derivatives. Costs add along a path.



The input distinguishes the just-sampled inactive chain state $b$, the active

chain's retained starting state $i$, and the inactive phase's realized physical

counts $c$. Infer the active priors from the paired depolarizing channel and

$c/N$, freeze them, and resume the active chain at $i$. The four coefficients

already include all earlier work and its correlation with $(b,i,c)$.

After sampling, exchange roles: the output keys are $(j,b,d)$, where $j$ is

the newly sampled terminal cycle and $d$ its physical counts. Accumulated work

includes both the incoming work and the new phase. The paired channel assigns

$1-p,p/3,p/3,p/3$ to $I,X,Y,Z$; use source-edge $e$ to target-edge $f_e$.

Returns
-------
Return the role-exchanged joint retained-state, empirical-count and total-work coefficient table.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transfer_retained_phase(
    incoming: "np.ndarray",
    source_samples: int,
    channel_probability: float,
    edge_map: "np.ndarray",
    target_detectors: "np.ndarray",
    target_reference: "np.ndarray",
    passive_states: int,
    target_samples: int,
    target_spacing: int,
) -> "np.ndarray":
    r"""Transfer a joint retained-state and accumulated-work law through one empirical-feedback phase.

    Parameters
    ----------
    incoming : np.ndarray
        Nonnegative finite table $(J,E+6)$ with unique integer keys
        $(b,i,c_0,\ldots,c_{E-1})$, then four binomial work coefficients.
        State bounds are $0\le b<B$, $0\le i<k$, and $0\le c_e\le N$.
        Zeroth coefficients sum to one; rows may be in any order.
    source_samples : int
        Previous phase's sample count $1\le N\le3$.
    channel_probability : float
        Dimensionless depolarizing strength $0<p<1$.
    edge_map : np.ndarray
        Integer permutation of length $E$ mapping source edge to target edge.
    target_detectors : np.ndarray
        Binary matchable detector matrix with $E\le8$, whose augmented graph
        is simple and connected, has two to six vertices and cycle rank at most three.
    target_reference : np.ndarray
        Binary target reference chain of length $E$; cycle rows use relative-mask order.
    passive_states : int
        Number $B$ of inactive-chain cycles, $1\le B\le8$.
    target_samples : int
        Number of new physical observations, between one and three.
    target_spacing : int
        Completed updates preceding each new record, between one and three.

    Returns
    -------
    outgoing : np.ndarray
        Nonnegative table $(J',E+6)$ sorted by the keys $(j,b,d)$.
        Its last four columns contain total-work coefficients, without
        normalizing higher orders or resetting either retained state.

    Raises
    ------
    ValueError
        If any incoming key, dimension, coefficient, probability normalization,
        graph, channel, permutation or sampling parameter is invalid.

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


def _channel_probability(value):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("channel probability must be finite and in (0,1)")
    return float(value)


def _paired_map(value, edges):
    value = np.asarray(value)
    if (
        value.shape != (edges,)
        or not np.issubdtype(value.dtype, np.integer)
        or not np.array_equal(np.sort(value), np.arange(edges))
    ):
        raise ValueError("edge_map must be an edge permutation")
    return value.astype(int, copy=True)


def _feedback_prior(p, counts, samples, mapping):
    alpha = np.asarray(counts, dtype=float) / samples
    q = np.empty_like(alpha)
    q[mapping] = alpha / 2.0 + (1.0 - alpha) * p / (3.0 - 2.0 * p)
    return q


def _incoming_groups(incoming, edges, active_states, passive_states, samples):
    incoming = np.asarray(incoming, dtype=float)
    if (
        incoming.ndim != 2
        or incoming.shape[1] != edges + 6
        or len(incoming) == 0
        or not np.all(np.isfinite(incoming))
        or np.any(incoming < 0)
    ):
        raise ValueError("incoming law has invalid dimensions or values")
    keys = incoming[:, : edges + 2]
    if (
        np.any(keys != np.floor(keys))
        or np.any(keys[:, 0] >= passive_states)
        or np.any(keys[:, 1] >= active_states)
        or np.any(keys[:, 2:] > samples)
    ):
        raise ValueError("state indices and empirical counts must be in range")
    if not np.isclose(incoming[:, edges + 2].sum(), 1.0, atol=1e-9, rtol=0):
        raise ValueError("incoming zeroth coefficients must sum to one")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("incoming keys must be unique")
    groups = {}
    for row in incoming:
        passive, active = row[:2].astype(int)
        key = tuple(row[2 : edges + 2].astype(int))
        if key not in groups:
            groups[key] = np.zeros((passive_states, active_states, 4))
        groups[key][passive, active] = row[-4:]
    return groups


def _oracle_transfer_retained_phase(
    incoming: "np.ndarray",
    source_samples: int,
    channel_probability: float,
    edge_map: "np.ndarray",
    target_detectors: "np.ndarray",
    target_reference: "np.ndarray",
    passive_states: int,
    target_samples: int,
    target_spacing: int,
) -> "np.ndarray":
    physical = _oracle_construct_physical_cycle_ensemble(
        target_detectors, target_reference
    )
    k, edges = physical.shape
    passive_states = _integer(passive_states, 1, 8, "passive_states")
    source_samples = _integer(source_samples, 1, 3, "source_samples")
    target_samples = _integer(target_samples, 1, 3, "target_samples")
    target_spacing = _integer(target_spacing, 1, 3, "target_spacing")
    p = _channel_probability(channel_probability)
    mapping = _paired_map(edge_map, edges)
    groups = _incoming_groups(incoming, edges, k, passive_states, source_samples)
    accumulated = {}
    for source_counts, incoming_mass in groups.items():
        q = _feedback_prior(p, source_counts, source_samples, mapping)
        kernel = _oracle_compute_work_recording_kernel(
            target_detectors, q, target_reference, target_spacing
        )
        table = _oracle_compute_count_work_transfer(kernel, physical, target_samples)
        for row in table:
            counts = tuple(row[:edges].astype(int))
            terminal = int(row[edges])
            transfer = row[edges + 1 :].reshape(k, 1, 4)
            mass = _matrix_jet_product(incoming_mass, transfer)[:, 0]
            for passive in range(passive_states):
                if mass[passive, 0] <= 0:
                    continue
                key = (terminal, passive, *counts)
                if key not in accumulated:
                    accumulated[key] = np.zeros(4)
                accumulated[key] += mass[passive]
    return np.asarray([(*key, *mass) for key, mass in sorted(accumulated.items())])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return distinct normal, boundary, edge, and invalid-input cases."""
    return [
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
incoming=np.array([[0,1,0,2,0,0.4,1.2,1.2,0.4],
                   [1,0,2,0,2,0.6,3.0,6.0,6.0]])
""",
            "call": "transfer_retained_phase(incoming.copy(), 2, p, mapping.copy(), HX.copy(), MX.copy(), 2, 2, 1)",
            "gold_call": "_oracle_transfer_retained_phase(incoming.copy(), 2, p, mapping.copy(), HX.copy(), MX.copy(), 2, 2, 1)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
H=np.array([[1]])
M=np.array([0])
A=np.array([[0.,0.,0.,1.,2.,1.,0.]])
""",
            "call": "transfer_retained_phase(A.copy(), 1, 0.24, np.array([0]), H.copy(), M.copy(), 1, 3, 2)",
            "gold_call": "_oracle_transfer_retained_phase(A.copy(), 1, 0.24, np.array([0]), H.copy(), M.copy(), 1, 3, 2)",
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
incoming=np.array([[0,1,0,2,0,0.4,1.2,1.2,0.4],
                   [1,0,2,0,2,0.6,3.0,6.0,6.0]])
incoming=incoming[::-1].copy()
mapping=np.array([1,2,0])
p=0.82
""",
            "call": "transfer_retained_phase(incoming.copy(), 2, p, mapping.copy(), HX.copy(), MX.copy(), 2, 3, 2)",
            "gold_call": "_oracle_transfer_retained_phase(incoming.copy(), 2, p, mapping.copy(), HX.copy(), MX.copy(), 2, 3, 2)",
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
incoming=np.array([[0,1,0,2,0,0.4,1.2,1.2,0.4],
                   [1,0,2,0,2,0.6,3.0,6.0,6.0]])
incoming[0,2]=3

def _check_error(fn):
    try:
        fn(incoming.copy(), 2, p, mapping.copy(), HX.copy(), MX.copy(), 2, 2, 1)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(transfer_retained_phase)",
            "gold_call": "_check_error(_oracle_transfer_retained_phase)",
        },
    ]
