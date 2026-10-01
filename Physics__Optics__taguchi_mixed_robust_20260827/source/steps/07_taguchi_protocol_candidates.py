"""
Evaluate four archived factor-level selection protocols.

Each row contracts the same balanced experiment around its selected levels.

Returns
-------
np.ndarray of shape (4,K,s+3), float: protocol, factor, and packed update columns.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def taguchi_protocol_candidates(levels: np.ndarray, oa: np.ndarray, responses: np.ndarray, level_differences: np.ndarray, reduction_rate: float) -> np.ndarray:
    """Return contracted updates for four candidate protocols.

Parameters
----------
levels : numpy.ndarray
    Float array of shape `(K,s)` in ascending level order.
oa : numpy.ndarray
    Integer array of shape `(M,K)` with zero-based levels.
responses : numpy.ndarray
    Length-`M` finite nonnegative smaller-is-better responses.
level_differences : numpy.ndarray
    Length-`K` positive current spacings.
reduction_rate : float
    Finite contraction rate in `(0,1)`.

Returns
-------
updates : numpy.ndarray
    Float array of shape `(4,K,s+3)`. Rows follow the task's candidate
    order; for each factor, columns are
    `[selected_index,center,new_spacing,new_levels...]`.

Conventions
-----------
Require K>=1, s>=2, M>=1, strictly ascending finite levels, valid integer OA indices with every factor level represented, finite nonnegative responses and finite positive spacings. Use candidate order: single-run minimum response, maximum level-mean SNR, minimum level-mean response, maximum level-median SNR. SNR=-20*log10(f) for f>0 and positive infinity for f=0. A mean containing positive infinity is positive infinity; use the usual ordered median, averaging the two middle values for even counts, with finite plus positive infinity equal to positive infinity. Exact single-run ties choose the lowest OA row index; exact factorwise ties choose the lowest level index. Recenter each factor on its selected old level, multiply its spacing by reduction_rate, and use offsets arange(s)-(s-1)/2. Do not clip new levels to initialization intervals. Return finite float64 updates and strictly positive representable contracted spacings.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, np.asarray(levels).shape[0], np.asarray(levels).shape[1] + 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_taguchi_protocol_candidates(levels, oa, responses, level_differences, reduction_rate):
    values = np.asarray(levels, dtype=float)
    design = np.asarray(oa)
    response = np.asarray(responses, dtype=float)
    spacing = np.asarray(level_differences, dtype=float)
    rate = float(reduction_rate)
    if values.ndim != 2 or design.ndim != 2:
        raise ValueError("Levels and OA must be two-dimensional")
    factors, count = values.shape
    if factors < 1 or count < 2 or design.shape[0] < 1 or design.shape[1] != factors:
        raise ValueError("Invalid level table or OA shape")
    if response.shape != (design.shape[0],) or spacing.shape != (factors,):
        raise ValueError("Incompatible response or spacing shape")
    if not np.issubdtype(design.dtype, np.integer) or np.any(design < 0) or np.any(design >= count):
        raise ValueError("OA entries must be valid zero-based integer levels")
    if not all(np.isfinite(array).all() for array in (values, response, spacing)):
        raise ValueError("Inputs must be finite")
    if np.any(values[:, 1:] <= values[:, :-1]) or np.any(response < 0) or np.any(spacing <= 0) or not 0 < rate < 1:
        raise ValueError("Require ascending levels, nonnegative responses, positive spacing and 0<rate<1")
    snr = np.full(response.shape, np.inf)
    positive = response > 0
    snr[positive] = -20 * np.log10(response[positive])
    mean_snr = np.empty((factors, count))
    mean_response = np.empty((factors, count))
    median_snr = np.empty((factors, count))
    for factor in range(factors):
        for level in range(count):
            mask = design[:, factor] == level
            if not np.any(mask):
                raise ValueError("Every factor level must occur")
            selected = response[mask]
            scale = float(np.max(selected))
            mean_response[factor, level] = 0 if scale == 0 else scale * np.mean(selected / scale)
            mean_snr[factor, level] = np.mean(snr[mask])
            median_snr[factor, level] = np.median(snr[mask])
    indices = np.vstack((design[np.argmin(response)], np.argmax(mean_snr, axis=1),
                         np.argmin(mean_response, axis=1), np.argmax(median_snr, axis=1)))
    output = np.empty((4, factors, count + 3))
    contracted = spacing * rate
    offsets = np.arange(count) - (count - 1) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        for mode in range(4):
            centers = values[np.arange(factors), indices[mode]]
            next_levels = centers[:, None] + contracted[:, None] * offsets
            output[mode] = np.column_stack((indices[mode], centers, contracted, next_levels))
    if not np.isfinite(output).all() or np.any(contracted <= 0):
        raise ValueError("Update or positive contracted spacing is not representable")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight designs distinguish balance, log averaging, ties, coupling, scaling, and dimensionality."""
    return [{'setup': 'l=np.array([[1.,2.],[10.,20.]]);o=np.array([[0,0],[0,1],[1,0],[1,1]]);f=np.array([1.,.1,.01,.001]);d=np.array([1.,10.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.5)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.5)'},
        {'setup': 'l=np.array([[0.,2.,4.]]);o=np.array([[0],[1],[2],[0],[1],[2]]);f=np.array([2.,4.,8.,1.,2.,4.]);d=np.array([2.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.25)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.25)'},
        {'setup': 'l=np.array([[1.,2.,3.],[4.,5.,6.]]);o=np.array([[a,b] for a in range(3) for b in range(3)]);f=np.linspace(.2,1.,9);d=np.array([1.,1.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.8)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.8)'},
        {'setup': 'l=np.array([[-1.,0.,1.],[2.,3.,4.]]);o=np.array([[a,b] for a in range(3) for b in range(3)]);f=np.array([1e-8,1.,2.,3.,4.,5.,6.,7.,8.]);d=np.array([1.,1.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.4)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.4)'},
        {'setup': 'l=np.array([[1.,2.,3.]]);o=np.array([[0],[1],[2]]);f=np.ones(3);d=np.array([1.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.5)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.5)'},
        {'setup': 'l=np.tile(np.array([1.,2.,3.]),(4,1));o=np.array([[a,b,(a+b)%3,(a+2*b)%3] for a in range(3) for b in range(3)]);f=np.array([.7,.4,.9,.2,.8,.6,.5,.3,1.]);d=np.ones(4)', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.6)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.6)'},
        {'setup': 'l=np.array([[-8.,-2.,4.],[.1,.2,.3],[10.,20.,30.]]);o=np.array([[a,b,(a+b)%3] for a in range(3) for b in range(3)]);f=np.array([9.,1.,7.,3.,2.,8.,6.,4.,5.]);d=np.array([6.,.1,10.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.2)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.2)'},
        {'setup': 'l=np.array([[.25,.5,.75],[2.,4.,8.]]);o=np.array([[a,b] for a in range(3) for b in range(3)]);f=np.array([.4,.6,.3,.9,.2,.7,.8,.5,.1]);d=np.array([.25,2.])', 'call': 'taguchi_protocol_candidates(l,o,f,d,0.95)', 'gold_call': '_oracle_taguchi_protocol_candidates(l,o,f,d,0.95)'},
        {'setup': '', 'call': 'taguchi_protocol_candidates(np.array([[1.,2.,3.]]),np.arange(3)[:,None],np.zeros(3),np.ones(1),.5)', 'gold_call': '_oracle_taguchi_protocol_candidates(np.array([[1.,2.,3.]]),np.arange(3)[:,None],np.zeros(3),np.ones(1),.5)'},
        {'setup': '', 'call': 'taguchi_protocol_candidates(np.array([[1.,2.,3.]]),np.arange(3)[:,None],np.array([1.,0.,2.]),np.ones(1),.5)', 'gold_call': '_oracle_taguchi_protocol_candidates(np.array([[1.,2.,3.]]),np.arange(3)[:,None],np.array([1.,0.,2.]),np.ones(1),.5)'}]
