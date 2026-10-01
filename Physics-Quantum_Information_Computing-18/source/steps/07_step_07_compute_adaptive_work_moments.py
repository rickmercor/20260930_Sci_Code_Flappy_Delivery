"""
Compute the physical report law and work coefficients of the retained-state feedback decoder.

The work counter counts every attempted elementary edge proposal, including

rejected proposals and the first trial of an immediately returning update.

For arrival state $j$ from state $i$, coefficient $r$ means

$\mathbb E[\binom{W}{r}\mathbf1_{\{j\}}\mid i]$ for $r=0,1,2,3$.

These are Taylor coefficients at $z=1$ of the probability generating function;

they are not raw moments or factorial derivatives. Costs add along a path.



Run $Z_0\to X_1\to Z_2\to X_3$. Both relative cycles initially equal zero,

but each sector resumes its last terminal cycle on its second visit.

Counters of empirical physical edge occupancy restart in each phase; work

accumulates across all phases. Initial Z priors are the depolarizing marginal;

subsequent priors use the previous phase's realized marginals with forward or

inverse pairing, and remain frozen during that phase. The final X records

produce a physical plurality label with the smallest-label tie rule.

Retained cycle states, empirical counts and work must remain jointly weighted

where they affect future transitions.

Returns
-------
Return the four unnormalized binomial work coefficients for every final physical reported label.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_adaptive_work_moments(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
) -> "np.ndarray":
    r"""Compute the physical report law and work coefficients of the retained-state feedback decoder.

    Parameters
    ----------
    detectors_z, detectors_x : np.ndarray
        Binary paired detector matrices with the same edge count $1\le E\le8$.
        Each augmented graph is simple, connected, has two to six vertices
        and cycle rank zero through three; columns have support one or two.
    reference_z, reference_x : np.ndarray
        Binary length-$E$ reference chains in the corresponding edge orders.
    syndrome_z, syndrome_x : np.ndarray
        Binary detector vectors satisfying $H_ZM_Z=s_Z$ and $H_XM_X=s_X$
        modulo two; their lengths match the corresponding detector row counts.
    logical_x : np.ndarray
        Binary final-X logical map $(r,E)$, $1\le r\le2$; row zero is the
        least significant physical-label bit.
    channel_probability : float
        Finite dimensionless depolarizing strength $0<p<1$.
    edge_map : np.ndarray
        Integer permutation $f$ from Z-edge indices to paired X-edge indices.
    sample_counts : np.ndarray
        Four positive integers for $Z_0,X_1,Z_2,X_3$; the first three are
        at most three and the last is at most five.
    spacings : np.ndarray
        Four integers in $[1,3]$, counting completed updates before each record.

    Returns
    -------
    report_coefficients : np.ndarray
        Array $(2^r,4)$ whose entry $(g,s)$ is
        $\mathbb E[\binom{W}{s}\mathbf1_{\{\widehat g=g\}}]$ for the
        total attempted-proposal count $W$. Order-zero entries sum to one.

    Raises
    ------
    ValueError
        If any model, syndrome/reference relation, channel, permutation,
        schedule, or intermediate transition/moment system is invalid.

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


def _phase_schedule(value, last_high, name):
    value = np.asarray(value)
    high = np.array([3, 3, 3, last_high])
    if (
        value.shape != (4,)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any(value < 1)
        or np.any(value > high)
    ):
        raise ValueError(f"{name} must contain four supported positive integers")
    return value.astype(int, copy=True)


def _model_inputs(hz, hx, mz, mx, sz, sx, logical, p, mapping, samples, spacings):
    z = _oracle_construct_physical_cycle_ensemble(hz, mz)
    x = _oracle_construct_physical_cycle_ensemble(hx, mx)
    if z.shape[1] != x.shape[1]:
        raise ValueError("paired graphs must have the same edge count")
    for h, m, s in [(hz, mz, sz), (hx, mx, sx)]:
        s = _binary(s, 1, "syndrome")
        h = np.asarray(h)
        if s.shape != (len(h),) or not np.array_equal(h @ m % 2, s):
            raise ValueError("reference must realize the syndrome")
    _logical_classes(x, logical)
    return (
        z,
        x,
        _channel_probability(p),
        _paired_map(mapping, z.shape[1]),
        _phase_schedule(samples, 5, "sample_counts"),
        _phase_schedule(spacings, 3, "spacings"),
    )


def _oracle_compute_adaptive_work_moments(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
) -> "np.ndarray":
    z, x, p, mapping, samples, spacings = _model_inputs(
        detectors_z,
        detectors_x,
        reference_z,
        reference_x,
        syndrome_z,
        syndrome_x,
        logical_x,
        channel_probability,
        edge_map,
        sample_counts,
        spacings,
    )
    edges = z.shape[1]
    kernel = _oracle_compute_work_recording_kernel(
        detectors_z, np.full(edges, 2 * p / 3), reference_z, int(spacings[0])
    )
    table = _oracle_compute_count_work_transfer(kernel, z, int(samples[0]))
    # The inactive X chain starts at relative state zero.
    incoming = np.array(
        [
            (int(row[edges]), 0, *row[:edges], *row[edges + 1 : edges + 5])
            for row in table
            if row[edges + 1] > 0
        ]
    )
    incoming = _oracle_transfer_retained_phase(
        incoming,
        int(samples[0]),
        p,
        mapping,
        detectors_x,
        reference_x,
        len(z),
        int(samples[1]),
        int(spacings[1]),
    )
    incoming = _oracle_transfer_retained_phase(
        incoming,
        int(samples[1]),
        p,
        np.argsort(mapping),
        detectors_z,
        reference_z,
        len(x),
        int(samples[2]),
        int(spacings[2]),
    )
    _, classes = _logical_classes(x, logical_x)
    result = np.zeros((classes, 4))
    # Marginalize the last Z terminal state only after its last use.
    groups = _incoming_groups(incoming, edges, len(x), len(z), int(samples[2]))
    for counts, mass in groups.items():
        q = _feedback_prior(p, counts, int(samples[2]), mapping)
        kernel = _oracle_compute_work_recording_kernel(
            detectors_x, q, reference_x, int(spacings[3])
        )
        report = _oracle_compute_report_work_transfer(
            kernel, x, logical_x, int(samples[3])
        )
        result += _matrix_jet_product(mass.sum(axis=0)[None, ...], report)[0]
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
            "call": "compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
            "gold_call": "_oracle_compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
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
            "call": "compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
            "gold_call": "_oracle_compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
            "tol": 1e-10,
        },
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
p=0.37
samples=np.array([1,2,1,3])
spacings=np.array([1,3,2,1])
mapping=np.array([2,0,5,1,6,4,3])
""",
            "call": "compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
            "gold_call": "_oracle_compute_adaptive_work_moments(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())",
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
SX[0]^=1

def _check_error(fn):
    try:
        fn(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy())
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_adaptive_work_moments)",
            "gold_call": "_check_error(_oracle_compute_adaptive_work_moments)",
        },
    ]
