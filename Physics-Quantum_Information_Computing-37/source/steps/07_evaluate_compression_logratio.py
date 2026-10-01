"""
Use the six preceding public operations to select a certified retained dimension and evaluate the reduced-to-original log-likelihood ratio.

The compression reference weights typical histories but does not restrict deployment stimuli. A complete calculation first obtains the original stationary memory, then searches retained dimensions $r=2,\ldots,N$ using the purified-history certificate after stimulus-wise repair. Each accepted subspace defines a new conditional process, whose initial memory is the normalized projection of the original stationary marginal. The final comparison uses two independent sequential filters on the same forced record. Compose build_clock_instrument, solve_routed_stationary, select_memory_subspace, repair_controlled_instrument, compute_history_rate, and compute_record_loglikelihood; do not replace them by a separately reimplemented pipeline.

Returns
-------
float, unrounded reduced-minus-original conditional log likelihood in nats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import numpy as np

def evaluate_compression_logratio(
    n: int = 6,
    reference: np.ndarray = (((.86,.08),(.01,.05)),((.10,.51),(.19,.20))),
    rate_limit: float = .007,
    stimuli: np.ndarray = (0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0),
    actions: np.ndarray = (0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1)
) -> float:
    """Select a certified clock-memory rank and compare record probabilities.

    Parameters
    ----------
    n : int
        Clock size, at least 2; the default is the six-age prompt instance.
    reference : np.ndarray
        Stochastic array (C,2,C) in old-source/stimulus/new-source order.
        The default is the two-state correlated source in the problem.
    rate_limit : float
        Nonnegative finite real certificate budget in bits per timestep,
        excluding booleans. The default is 0.007. Accept equality.
    stimuli : np.ndarray
        One-dimensional integer stimulus record in {0,1}; the default is
        the 24-observation prompt record. Inputs are externally forced.
    actions : np.ndarray
        Matching integer action record in {0,1}; the default is the prompt
        record. Empty paired records are allowed.

    Returns
    -------
    log_ratio : float
        Unrounded finite native Python float in nats: reduced log likelihood
        minus original log likelihood. Select the first rank in 2,...,n whose
        purified-history rate is at most rate_limit. The initial original
        memory is its reference-stationary marginal; the initial reduced
        state is its normalized projection, not the repaired stationary state.
        Use support cutoff 1e-12 and spectral-cut gap 1e-10. Do not mutate inputs.

    Raises
    ------
    ValueError
        If n or rate_limit is invalid; the reference/instrument lacks a unique
        stationary state; a visited interior spectral cut is ambiguous; record
        shapes, ranges or density conditions fail; a conditional probability
        vanishes; or no rank has a defined admissible certificate.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_compression_logratio(
    n: int = 6,
    reference: "np.ndarray" = (((.86,.08),(.01,.05)),((.10,.51),(.19,.20))),
    rate_limit: float = .007,
    stimuli: "np.ndarray" = (0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0),
    actions: "np.ndarray" = (0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1)
) -> float:
    import numpy as np
    from numbers import Real

    if isinstance(rate_limit, (bool, np.bool_)) or not isinstance(rate_limit, Real) or not np.isfinite(rate_limit) or rate_limit < 0:
        raise ValueError("rate_limit must be a nonnegative finite real")
    K = _oracle_build_clock_instrument(n)
    blocks = _oracle_solve_routed_stationary(K, reference)
    rho = blocks.sum(axis=0)
    selected_P = None
    selected_B = None
    for rank in range(2, int(n)+1):
        P = _oracle_select_memory_subspace(blocks, rank)
        B = _oracle_repair_controlled_instrument(K, P, 1e-12)
        rate = _oracle_compute_history_rate(K, B, reference)
        if rate <= rate_limit:
            selected_P, selected_B = P, B
            break
    if selected_P is None:
        raise ValueError("no admissible retained rank")
    initial_reduced = selected_P @ rho @ selected_P
    initial_reduced /= np.trace(initial_reduced).real
    original_log = _oracle_compute_record_loglikelihood(K, rho, stimuli, actions)
    reduced_log = _oracle_compute_record_loglikelihood(selected_B, initial_reduced, stimuli, actions)
    return float(reduced_log-original_log)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
""",
            "call": 'evaluate_compression_logratio(5, R, .007, x, y)',
            "gold_call": '_oracle_evaluate_compression_logratio(5, R, .007, x, y)',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
""",
            "call": 'evaluate_compression_logratio(6, R, .004, x, y)',
            "gold_call": '_oracle_evaluate_compression_logratio(6, R, .004, x, y)',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
""",
            "call": 'evaluate_compression_logratio(6, R, .010, x, y)',
            "gold_call": '_oracle_evaluate_compression_logratio(6, R, .010, x, y)',
        },
        {
            "setup": """import numpy as np
x=np.array([],dtype=int)
R=np.array([[[.8],[.2]]])
""",
            "call": 'evaluate_compression_logratio(2, R, 0.0, x, x)',
            "gold_call": '_oracle_evaluate_compression_logratio(2, R, 0.0, x, x)',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
x=np.zeros(600,dtype=int);y=np.ones(600,dtype=int)
""",
            "call": 'evaluate_compression_logratio(4, R, .02, x, y)',
            "gold_call": '_oracle_evaluate_compression_logratio(4, R, .02, x, y)',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])

def _run_model():
    try:
        evaluate_compression_logratio(5, R, -1.0, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_evaluate_compression_logratio(5, R, -1.0, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
R[0,0,0] += .1

def _run_model():
    try:
        evaluate_compression_logratio(5, R, .007, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_evaluate_compression_logratio(5, R, .007, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
x=np.array([0,0,1,0,0,0,0,1,0,0])
y=np.array([0,1,0,0,0,1,0,0,0,1])
x=np.array([1]); y=np.array([1])

def _run_model():
    try:
        evaluate_compression_logratio(4, R, .007, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_evaluate_compression_logratio(4, R, .007, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
    ]
