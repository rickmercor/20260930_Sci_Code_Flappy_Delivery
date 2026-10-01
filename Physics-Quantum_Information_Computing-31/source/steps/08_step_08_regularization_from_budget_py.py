"""
Invert the conditional landscape envelope to choose a regularization budget.

The noise hinge separates bias-dominated and noise-dominated branches. A source-derived bound, not a generic tuning heuristic, selects the largest admissible coefficient.

Returns
-------
float, the largest regularization satisfying the conditional error budget.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularization_from_budget(op_norm: float, r_star: int, r: int, alpha: float, beta: float, budget: float) -> float:
    """Choose the largest regularization satisfying the conditional error budget.

    Parameters
    ----------
    op_norm : float
        Nonnegative realized adjoint-noise operator norm eta.
    r_star : int
        Positive true rank.
    r : int
        Search rank with r_star <= r <= 10*r_star. In this domain the source
        bound is strictly increasing with nonnegative regularization.
    alpha : float
        Positive lower squared-norm isometry constant.
    beta : float
        Tangent upper constant, at least alpha, satisfying the source threshold.
    budget : float
        Finite nonnegative upper limit on the conditional Frobenius-error bound.

    Returns
    -------
    result : float
        Largest lambda >= 0 for which the landscape bound is at most budget,
        as a native float. Both branches of its positive-part term are included.

    Raises
    ------
    ValueError
        If budget is below the zero-regularization bound, or the source
        isometry applicability conditions fail.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_regularization_from_budget(op_norm: float, r_star: int, r: int, alpha: float, beta: float, budget: float) -> float:
    base=_oracle_certification_bound(op_norm,r_star,r,alpha,beta,0.0)
    if budget<base:
        raise ValueError('conditional error budget is infeasible')
    knot=_oracle_certification_bound(op_norm,r_star,r,alpha,beta,op_norm)
    denominator=6*alpha-(np.sqrt(5)+2)*beta
    if budget<=knot:
        slope=(36*np.sqrt(r_star)-10*np.sqrt(r+r_star))/denominator
        return float((budget-base)/slope)
    slope=36*np.sqrt(r_star)/denominator
    return float(op_norm+(budget-knot)/slope)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return separate normal, boundary, and edge comparison cases."""
    return [{'setup': 'import numpy as np\n'
               'eta=0.02; r_star=2; r=2; alpha=1.0; beta=1.2\n'
               'den=6*alpha-(np.sqrt(5)+2)*beta\n'
               'budget=((36*0.007+12*eta)*np.sqrt(2)+20*(eta-0.007))/den',
      'call': 'regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'gold_call': '_oracle_regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'eta=0.02; r_star=2; r=2; alpha=1.0; beta=1.2\n'
               'den=6*alpha-(np.sqrt(5)+2)*beta\n'
               'budget=48*eta*np.sqrt(2)/den',
      'call': 'regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'gold_call': '_oracle_regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'eta=0.02; r_star=2; r=2; alpha=1.0; beta=1.2\n'
               'den=6*alpha-(np.sqrt(5)+2)*beta\n'
               'budget=(36*0.05+12*eta)*np.sqrt(2)/den',
      'call': 'regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'gold_call': '_oracle_regularization_from_budget(eta,r_star,r,alpha,beta,budget)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'eta=0.02; r_star=2; r=2; alpha=1.0; beta=1.2\n'
               'den=6*alpha-(np.sqrt(5)+2)*beta\n'
               'budget=0.0\n'
               'def catches(fn):\n'
               '    try:\n'
               '        fn(eta,r_star,r,alpha,beta,budget)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'catches(regularization_from_budget)',
      'gold_call': 'catches(_oracle_regularization_from_budget)'}]
