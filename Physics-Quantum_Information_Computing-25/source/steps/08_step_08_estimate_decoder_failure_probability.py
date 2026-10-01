"""
Compose all earlier steps to evaluate calibration-averaged failure for the ring-code hypergraph product. Orchestrator: yes.

A gain uncertainty changes the decoder weights without changing the physical distribution of fault supports. Exact gain integration separates those two roles.

Returns
-------
float: the unconditional decoder failure probability truncated at max_faults and averaged over the uniform calibration gain interval, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_decoder_failure_probability(
    ring_lengths: tuple = (3, 4),
    data_prob: tuple = (0.010, 0.001),
    meas_prob: tuple = (0.006, 0.0005),
    max_iter: int = 3,
    max_faults: int = 3,
    gain_interval: tuple = ((3, 4), (5, 4)),
    weight_scale: int = 1000000,
) -> float:
    r"""Return the truncated failure probability averaged over calibration gain.

    Build the length-$L$ ring checks, with ones at $(i, i)$ and
    $(i, (i+1) \bmod L)$, and their phenomenological hypergraph product using
    steps 01–03. With $(d_0, d_1)$ = ``data_prob`` and $(e_0, e_1)$ =
    ``meas_prob``, the actual fault probabilities are $p_j = d_0 + j d_1$ on
    data columns and $q_i = e_0 + i e_1$ on outcome-flip columns. For each
    column let $W$ be nearest-integer rounding, ties to even, of
    $\sigma \ln\frac{1-p}{p}$ with $\sigma$ = ``weight_scale`` and $p$ that
    column's probability. Data-column decoder coefficients are $(0, W)$ and
    outcome-column coefficients are $(W, 0)$: gain multiplies only the
    data-column weights. This fixed-point calibration model is an explicit
    task extension; the underlying monotone hierarchical decoder is unchanged.

    Compose the exact hierarchical correction profiles and the gain-averaged
    fault enumeration, and sum its contributions for orders
    $0, \dots,$ ``max_faults``. The gain is uniform on ``gain_interval`` and
    independent of the actual faults. Every default is specified in the
    problem statement.

    Parameters
    ----------
    ring_lengths : tuple
        Two integers at least 2.
    data_prob : tuple
        Two finite numbers $(d_0, d_1)$ defining data probabilities strictly between 0 and 1.
    meas_prob : tuple
        Two finite numbers $(e_0, e_1)$ defining outcome probabilities strictly between 0 and 1.
    max_iter : int
        Non-negative cap on rounds at both levels.
    max_faults : int
        Integer from 0 to the total number of fault columns.
    gain_interval : tuple
        ``((lo_num, lo_den), (hi_num, hi_den))``, with $0 < \mathrm{lo} < \mathrm{hi}$
        and positive denominators.
    weight_scale : int
        Positive integer $\sigma$ multiplying the log-odds before rounding.

    Returns
    -------
    float
        Unconditional gain-averaged truncated failure probability.

    Raises
    ------
    ValueError
        If an argument violates the stated domain or a composed step rejects its input.

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_estimate_decoder_failure_probability(
    ring_lengths: tuple = (3, 4),
    data_prob: tuple = (0.010, 0.001),
    meas_prob: tuple = (0.006, 0.0005),
    max_iter: int = 3,
    max_faults: int = 3,
    gain_interval: tuple = ((3, 4), (5, 4)),
    weight_scale: int = 1000000,
) -> float:
    import math
    try:
        l1,l2=ring_lengths;p0,dp=data_prob;q0,dq=meas_prob
    except (TypeError,ValueError):raise ValueError("lengths and probability parameters must be pairs") from None
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<2 for v in (l1,l2)):
        raise ValueError("ring lengths must be integers at least 2")
    if isinstance(weight_scale,(bool,np.bool_)) or not isinstance(weight_scale,(int,np.integer)) or weight_scale<=0:
        raise ValueError("weight_scale must be a positive integer")
    _gain_bounds(gain_interval)
    def ring(length):
        h=np.eye(length,dtype=np.int64);h[np.arange(length),(np.arange(length)+1)%length]=1
        return h
    h1,h2=ring(int(l1)),ring(int(l2))
    d,hz=_oracle_build_phenomenological_check_matrix(h1,h2);m,n=d.shape;nd=hz.shape[1]
    p=np.concatenate((p0+dp*np.arange(nd),q0+dq*np.arange(m)))
    if not np.all((p>0)&(p<1)):raise ValueError("all fault probabilities must lie in (0,1)")
    nominal=np.array([round(int(weight_scale)*math.log((1-float(x))/float(x))) for x in p],dtype=np.int64)
    w=np.zeros((n,2),dtype=np.int64);w[:nd,1]=nominal[:nd];w[nd:,0]=nominal[nd:]
    perm=_oracle_decouple_hypergraph_product(h1,h2);k=int(l1);width=2*int(l2)
    t=_oracle_compute_decoupling_transform(d,perm,k,width)
    def decode(s):return _oracle_hierarchical_decode(s,d,perm,t,k,width,w,max_iter,gain_interval)
    counts,contributions=_oracle_truncated_failure_probability(d,hz,p,decode,max_faults,gain_interval)
    return float(math.fsum(contributions))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return explicit differential cases for Studio contract inspection."""
    return [{'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((2,3),(.03,.001),(.008,.001),2,2,((1,2),(3,2)),1000000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((2,3),(.03,.001),(.008,.001),2,2,((1,2),(3,2)),1000000)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((3,2),(.015,.002),(.007,.0005),3,2,((3,4),(5,4)),1000000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((3,2),(.015,.002),(.007,.0005),3,2,((3,4),(5,4)),1000000)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((2,2),(.08,-.004),(.02,.002),1,3,((1,2),(2,1)),10000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((2,2),(.08,-.004),(.02,.002),1,3,((1,2),(2,1)),10000)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((2,3),(.03,.001),(.008,.001),0,2,((3,4),(5,4)),1000000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((2,3),(.03,.001),(.008,.001),0,2,((3,4),(5,4)),1000000)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((3,4),(.010,.001),(.006,.0005),3,0,((3,4),(5,4)),1000000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((3,4),(.010,.001),(.006,.0005),3,0,((3,4),(5,4)),1000000)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((2,2),(.10,.01),(.03,.004),2,2,((1,1),(2,1)),10)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((2,2),(.10,.01),(.03,.004),2,2,((1,1),(2,1)),10)',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n',
      'call': 'estimate_decoder_failure_probability((3,4),(.010,.001),(.006,.0005),3,3,((3,4),(5,4)),1000000)',
      'gold_call': '_oracle_estimate_decoder_failure_probability((3,4),(.010,.001),(.006,.0005),3,3,((3,4),(5,4)),1000000)',
      'tol': 1e-11}]
