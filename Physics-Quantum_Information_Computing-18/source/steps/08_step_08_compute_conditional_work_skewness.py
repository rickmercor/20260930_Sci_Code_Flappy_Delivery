"""
Compute conditional work skewness for a reported physical logical class.

Work is the total number of attempted elementary proposals across the

four retained-state empirical-feedback phases. For the event

$G=\{\widehat g=g\}$, the requested dimensionless statistic is



$$

\gamma_g=\frac{\mathbb E[(W-\mu_g)^3\mid G]}

{\operatorname{Var}(W\mid G)^{3/2}},\qquad

\mu_g=\mathbb E[W\mid G].

$$



Condition the work moments on the reported event before centering. Binomial

coefficients of the joint work generating function must be converted to

ordinary moments. This statistic is undefined for a zero-probability event

or zero conditional variance; those inputs raise ValueError.

Returns
-------
Return one finite conditional skewness of total attempted-proposal work for the specified physical report.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_conditional_work_skewness(
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
    target_label: int,
) -> float:
    r"""Compute conditional work skewness for a reported physical logical class.

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
    target_label : int
        Physical integer label in $[0,2^r)$ on which work is conditioned.

    Returns
    -------
    skewness : float
        One finite dimensionless conditional third central moment divided by
        conditional variance to the power three halves.

    Raises
    ------
    ValueError
        If any model or schedule condition fails, the target label is invalid,
        its event has zero probability, or the conditional variance is not
        positive and finite. A nonfinite final statistic also raises ValueError.

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


def _conditional_skewness(coefficients, target):
    probability, b1, b2, b3 = coefficients[target]
    if probability <= 0:
        raise ValueError("conditioning event must have positive probability")
    mean = b1 / probability
    second = (2 * b2 + b1) / probability
    third = (6 * b3 + 6 * b2 + b1) / probability
    variance = second - mean * mean
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("conditional work must have finite positive variance")
    value = (third - 3 * mean * second + 2 * mean**3) / variance**1.5
    if not np.isfinite(value):
        raise ValueError("conditional skewness must be finite")
    return float(value)


def _oracle_compute_conditional_work_skewness(
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
    target_label: int,
) -> float:
    logical = _binary(logical_x, 2, "logical_x")
    if not 1 <= len(logical) <= 2:
        raise ValueError("logical_x must have one or two rows")
    target = _integer(target_label, 0, (1 << len(logical)) - 1, "target_label")
    coefficients = _oracle_compute_adaptive_work_moments(
        detectors_z,
        detectors_x,
        reference_z,
        reference_x,
        syndrome_z,
        syndrome_x,
        logical,
        channel_probability,
        edge_map,
        sample_counts,
        spacings,
    )
    return _conditional_skewness(coefficients, target)

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
            "call": "compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 2)",
            "gold_call": "_oracle_compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 2)",
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
            "call": "compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 1)",
            "gold_call": "_oracle_compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 1)",
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
            "call": "compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 0)",
            "gold_call": "_oracle_compute_conditional_work_skewness(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 0)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ=np.array([[1]])
HX=HZ.copy()
MZ=np.array([0])
MX=MZ.copy()
SZ=np.array([0])
SX=SZ.copy()
LX=np.array([[1]])
mapping=np.array([0])
samples=np.array([2,1,2,3])
spacings=np.array([1,2,1,1])
p=0.24
p=0.75

def _check_error(fn):
    try:
        fn(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_conditional_work_skewness)",
            "gold_call": "_check_error(_oracle_compute_conditional_work_skewness)",
        },
        {
            "setup": """import numpy as np
HZ=np.array([[1]])
HX=HZ.copy()
MZ=np.array([0])
MX=MZ.copy()
SZ=np.array([0])
SX=SZ.copy()
LX=np.array([[1]])
mapping=np.array([0])
samples=np.array([2,1,2,3])
spacings=np.array([1,2,1,1])
p=0.24

def _check_error(fn):
    try:
        fn(HZ.copy(), HX.copy(), MZ.copy(), MX.copy(), SZ.copy(), SX.copy(), LX.copy(), p, mapping.copy(), samples.copy(), spacings.copy(), 1)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(compute_conditional_work_skewness)",
            "gold_call": "_check_error(_oracle_compute_conditional_work_skewness)",
        },
    ]
