"""
Construct the prescribed sinh-clustered log-price grid, including the uniform-grid limit and exact endpoints. Follow the complete domain and return-shape contract in the function docstring.

The transformed state is the logarithm of the spot-to-strike ratio. The Section 6 sinh map clusters spatial nodes around the interval midpoint. The task treats count as the number of nodes and explicitly includes the zero-clustering limit. The exact formulas are preserved in the raw function docstring.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def clustered_grid(left: float, right: float, count: int, alpha: float = 0.4) -> "np.ndarray":
    r"""Return the prescribed sinh-clustered log-price grid.

    With $z_i=-1+2i/(M-1)$, midpoint $c=(a+b)/2$ and half-width
    $h=(b-a)/2$, return $x_i=c+h\sinh(\alpha z_i)/\sinh(\alpha)$.
    At $\alpha=0$, use the continuous limit $x_i=c+hz_i$. Set the
    first and last entries to $a$ and $b$ exactly. The count $M$ denotes
    nodes, not intervals; this task fixes the indexing convention in Section 6.

    Parameters
    ----------
    left : float
        Finite left endpoint $a$.
    right : float
        Finite right endpoint $b>a$.
    count : int
        Number of nodes $M\geq2$.
    alpha : float, default 0.4
        Finite clustering strength $\alpha\geq0$; zero gives uniform spacing.

    Returns
    -------
    numpy.ndarray, shape $(\mathrm{count},)$
        Increasing binary64 log-price coordinates including both endpoints.

    Raises
    ------
    ValueError
        If the numeric arguments violate the stated ordering, finiteness,
        integer-count or nonnegative-clustering conditions.

    Inputs must keep the displayed expressions finite and the resulting
    nodes distinct in binary64 arithmetic; behavior beyond this domain is
    otherwise unspecified.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_clustered_grid(left: float, right: float, count: int, alpha: float = 0.4) -> np.ndarray:
    r"""Return the prescribed sinh-clustered log-price grid.

    With $z_i=-1+2i/(M-1)$, midpoint $c=(a+b)/2$ and half-width
    $h=(b-a)/2$, return $x_i=c+h\sinh(\alpha z_i)/\sinh(\alpha)$.
    At $\alpha=0$, use the continuous limit $x_i=c+hz_i$. Set the
    first and last entries to $a$ and $b$ exactly. The count $M$ denotes
    nodes, not intervals; this task fixes the indexing convention in Section 6.

    Parameters
    ----------
    left : float
        Finite left endpoint $a$.
    right : float
        Finite right endpoint $b>a$.
    count : int
        Number of nodes $M\geq2$.
    alpha : float, default 0.4
        Finite clustering strength $\alpha\geq0$; zero gives uniform spacing.

    Returns
    -------
    numpy.ndarray, shape $(\mathrm{count},)$
        Increasing binary64 log-price coordinates including both endpoints.

    Raises
    ------
    ValueError
        If the numeric arguments violate the stated ordering, finiteness,
        integer-count or nonnegative-clustering conditions.

    Inputs must keep the displayed expressions finite and the resulting
    nodes distinct in binary64 arithmetic; behavior beyond this domain is
    otherwise unspecified.
    """
    left, right, alpha = float(left), float(right), float(alpha)
    if (not np.isfinite([left, right, alpha]).all() or left >= right or alpha < 0
            or isinstance(count, (bool, np.bool_)) or not np.isscalar(count)
            or not np.isfinite(count) or int(count) != count or count < 2):
        raise ValueError("finite ordered endpoints, nonnegative alpha, and integer count>=2 required")
    count = int(count)
    z = np.linspace(-1.0, 1.0, count)
    mapped = z if alpha == 0.0 else np.sinh(alpha*z)/np.sinh(alpha)
    x = (left+right)/2.0 + (right-left)/2.0*mapped
    x[0], x[-1] = left, right
    return x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'clustered_grid(-1.3, 2.7, 6, alpha=0.0).tolist()', 'gold_call': '_oracle_clustered_grid(-1.3, 2.7, 6, alpha=0.0).tolist()', 'tol': 2e-13}, {'setup': '', 'call': 'clustered_grid(-2.2, 0.8, 9, alpha=1.4).tolist()', 'gold_call': '_oracle_clustered_grid(-2.2, 0.8, 9, alpha=1.4).tolist()', 'tol': 2e-13}, {'setup': '', 'call': 'clustered_grid(-0.7, 1.9, 2, alpha=0.65).tolist()', 'gold_call': '_oracle_clustered_grid(-0.7, 1.9, 2, alpha=0.65).tolist()', 'tol': 2e-13}]
