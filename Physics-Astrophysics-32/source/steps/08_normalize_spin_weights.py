"""
Marginalize remnant mass and normalize the spin posterior.

Normalize the supplied mass log weights separately within each spin. For state score $\ell_{ka}$ form $L_k=\operatorname{LSE}_a(\log w_a+\ell_{ka})$ and use a uniform discrete spin prior, $W_k=e^{L_k}/\sum_j e^{L_j}$. Conditional summaries use the same state weights.

Returns
-------
`numpy.ndarray` of shape `(s,7)` with `(spin,log_marginal,weight,E[m],E[parent_score],E[child_score],E[child_mahalanobis])`, compared at `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def normalize_spin_weights(
    predictive_scores: np.ndarray,
    mass_log_weights: np.ndarray,
) -> np.ndarray:
    """Marginalize the mass nodes and return normalized spin weights.

    Parameters
    ----------
    predictive_scores : numpy.ndarray, shape (n, 8)
        Step-07 rows ordered with spin outer and mass inner.
    mass_log_weights : numpy.ndarray, shape (n,)
        Finite log quadrature weights paired with the rows. They are
        normalized separately inside every spin.

    Returns
    -------
    numpy.ndarray, shape (s, 7)
        `(spin,log_marginal,weight,E[m],E[parent_score],E[child_score],
        E[child_mahalanobis])`, compared at tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or nonfinite entry; masses are
        not positive; a spin has duplicate masses; or spin groups have unequal
        counts.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""Marginalize remnant mass and normalize the spin posterior.

For state scores $\ell_{ka}$ at spin $k$ and mass node $a$, normalize the
provided mass quadrature weights inside each spin and form

$$L_k=\operatorname{LSE}_a(\log w_a+\ell_{ka}),\qquad
W_k=\frac{e^{L_k}}{\sum_j e^{L_j}}.$$

The task uses a uniform prior over its discrete spin grid. Conditional scalar
summaries use the same normalized mass-node weights.

Returns
-------
`numpy.ndarray` of shape `(s, 7)` with spin, marginalized log score, normalized
spin weight, conditional mean mass, parent score, child score, and child
Mahalanobis residual. Components are compared at `1e-9`.
"""
import numpy as np
import numpy as np
def _spin_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    total = float(np.sum(np.exp(values - maximum)))
    if not np.isfinite(maximum) or not np.isfinite(total) or total <= 0.0:
        raise ValueError("log weights cannot be normalized")
    return maximum + np.log(total)
def _oracle_normalize_spin_weights(
    predictive_scores: np.ndarray,
    mass_log_weights: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(predictive_scores, dtype=float)
    mass_prior = np.asarray(mass_log_weights, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 8 or scores.shape[0] == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("predictive_scores must have nonempty finite shape (n, 8)")
    if mass_prior.shape != (scores.shape[0],) or not np.all(np.isfinite(mass_prior)):
        raise ValueError("mass_log_weights must have finite shape (n,)")
    if np.any(scores[:, 1] <= 0.0):
        raise ValueError("mass ratios must be positive")
    spins = np.unique(scores[:, 0])
    groups = [np.flatnonzero(scores[:, 0] == spin) for spin in spins]
    counts = np.array([group.size for group in groups])
    if np.any(counts == 0) or np.any(counts != counts[0]):
        raise ValueError("every spin must have the same positive mass-node count")
    output = np.empty((spins.size, 7), dtype=float)
    log_marginals = np.empty(spins.size, dtype=float)
    for spin_index, (spin, indices) in enumerate(zip(spins, groups)):
        masses = scores[indices, 1]
        if np.unique(masses).size != masses.size:
            raise ValueError("mass nodes must be unique within each spin")
        local_log_prior = mass_prior[indices]
        local_log_prior -= _spin_logsumexp(local_log_prior)
        state_log_weights = local_log_prior + scores[indices, 4]
        log_marginal = _spin_logsumexp(state_log_weights)
        local_weights = np.exp(state_log_weights - log_marginal)
        local_weights /= local_weights.sum()
        log_marginals[spin_index] = log_marginal
        output[spin_index] = (
            spin, log_marginal, 0.0,
            local_weights @ masses,
            local_weights @ scores[indices, 2],
            local_weights @ scores[indices, 3],
            local_weights @ scores[indices, 5],
        )
    normalization = _spin_logsumexp(log_marginals)
    output[:, 2] = np.exp(log_marginals - normalization)
    output[:, 2] /= output[:, 2].sum()
    if (
        not np.all(np.isfinite(output))
        or np.any(output[:, 2] < 0.0)
        or not np.isclose(output[:, 2].sum(), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise ValueError("spin posterior failed normalization")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Irregular weights, boundary scores, and invalid state layouts."""
    return [
        {
            "setup": "import numpy as np\nspins=np.repeat([.72,.78,.84],3); masses=np.tile([.97,1.,1.03],3); score=np.zeros((9,8)); score[:,0]=spins; score[:,1]=masses; score[:,2]=np.linspace(2.,4.,9); score[:,3]=np.linspace(7.,5.,9); score[:,4]=score[:,2]+score[:,3]; score[:,5]=np.linspace(.2,2.,9); score[:,6:]=np.arange(18).reshape(9,2)/10; logw=np.tile(np.log([.2,.5,.3]),3)",
            "call": "normalize_spin_weights(score,logw)",
            "gold_call": "_oracle_normalize_spin_weights(score,logw)",
        },
        {
            "setup": "import numpy as np\nspins=np.repeat([.7,.86],2); masses=np.tile([.96,1.04],2); score=np.zeros((4,8)); score[:,0]=spins; score[:,1]=masses; score[:,2:6]=np.array([[100,-99,1,.1],[99,-98,1,.2],[-100,101,1,.3],[-99,100,1,.4]]); logw=np.log([.5,.5,.5,.5])",
            "call": "normalize_spin_weights(score,logw)",
            "gold_call": "_oracle_normalize_spin_weights(score,logw)",
        },
        {
            "setup": "import numpy as np\nscore=np.zeros((3,8)); score[:,0]=[.72,.78,.84]; score[:,1]=1.; score[:,4]=[0.,-800.,-1600.]; score[:,2]=[2.,3.,4.]; logw=np.zeros(3)",
            "call": "normalize_spin_weights(score,logw)",
            "gold_call": "_oracle_normalize_spin_weights(score,logw)",
        },
        {
            "setup": "import numpy as np\ndef catches(fn):\n try: fn()\n except ValueError: return 1\n except Exception: return 2\n return 0\ngood=np.zeros((4,8)); good[:,0]=np.repeat([.74,.8],2); good[:,1]=np.tile([.98,1.02],2); duplicate=good.copy(); duplicate[1,1]=duplicate[0,1]; unequal=good[:-1]; nonfinite=good.copy(); nonfinite[0,4]=np.nan; nonpositive=good.copy(); nonpositive[0,1]=0.",
            "call": "np.array([catches(lambda: normalize_spin_weights(duplicate,np.zeros(4))),catches(lambda: normalize_spin_weights(unequal,np.zeros(3))),catches(lambda: normalize_spin_weights(nonfinite,np.zeros(4))),catches(lambda: normalize_spin_weights(nonpositive,np.zeros(4))),catches(lambda: normalize_spin_weights(good,np.zeros(3))),catches(lambda: normalize_spin_weights(np.zeros((0,8)),np.zeros(0)))])",
            "gold_call": "np.array([catches(lambda: _oracle_normalize_spin_weights(duplicate,np.zeros(4))),catches(lambda: _oracle_normalize_spin_weights(unequal,np.zeros(3))),catches(lambda: _oracle_normalize_spin_weights(nonfinite,np.zeros(4))),catches(lambda: _oracle_normalize_spin_weights(nonpositive,np.zeros(4))),catches(lambda: _oracle_normalize_spin_weights(good,np.zeros(3))),catches(lambda: _oracle_normalize_spin_weights(np.zeros((0,8)),np.zeros(0)))])",
        },
    ]
