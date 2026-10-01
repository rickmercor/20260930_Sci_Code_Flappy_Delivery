"""
Construct normalized Floater--Hormann barycentric weights on the supplied increasing nodes. Preserve the specified degree convention, common-sign normalization and input validation.

Section 3 constructs a pole-free Floater--Hormann rational interpolant by blending local polynomial interpolants of the specified degree. Multiplying every barycentric weight by a common nonzero constant does not change the interpolant; the task fixes the first weight positive and maximum-absolute-value normalization for a unique returned array.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def floater_hormann_weights(nodes: "np.ndarray", degree: int) -> "np.ndarray":
    r"""Return normalized Floater--Hormann rational interpolation weights.

    For $M$ increasing nodes and degree $d$, define
    $$\widehat w_i=(-1)^i\sum_{k=\max(0,i-d)}^{\min(i,M-d-1)}
    \prod_{\substack{j=k\\j\ne i}}^{k+d}|x_i-x_j|^{-1},\qquad
    w_i=\widehat w_i/\max_j|\widehat w_j|.$$
    An empty product is one. The first weight is positive and subsequent
    signs alternate. This fixes the immaterial common sign in Section 3.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    degree : int
        Local interpolation degree $0\leq d<M$.

    Returns
    -------
    numpy.ndarray, shape $(M,)$
        Weights in node order, with maximum absolute value one.

    Raises
    ------
    ValueError
        If the node vector or numeric degree violates its stated domain.

    Products and sums must remain finite and nonzero in binary64 arithmetic;
    behavior outside this representable domain is otherwise unspecified.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_floater_hormann_weights(nodes: np.ndarray, degree: int) -> np.ndarray:
    r"""Return normalized Floater--Hormann rational interpolation weights.

    For $M$ increasing nodes and degree $d$, define
    $$\widehat w_i=(-1)^i\sum_{k=\max(0,i-d)}^{\min(i,M-d-1)}
    \prod_{\substack{j=k\\j\ne i}}^{k+d}|x_i-x_j|^{-1},\qquad
    w_i=\widehat w_i/\max_j|\widehat w_j|.$$
    An empty product is one. The first weight is positive and subsequent
    signs alternate. This fixes the immaterial common sign in Section 3.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    degree : int
        Local interpolation degree $0\leq d<M$.

    Returns
    -------
    numpy.ndarray, shape $(M,)$
        Weights in node order, with maximum absolute value one.

    Raises
    ------
    ValueError
        If the node vector or numeric degree violates its stated domain.

    Products and sums must remain finite and nonzero in binary64 arithmetic;
    behavior outside this representable domain is otherwise unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    if (isinstance(degree, (bool, np.bool_)) or not np.isscalar(degree)
            or not np.isfinite(degree) or int(degree) != degree
            or degree < 0 or degree >= x.size):
        raise ValueError("degree must be less than the node count")
    d = int(degree)
    w = np.empty(x.size)
    for i in range(x.size):
        total = 0.0
        for k in range(max(0, i-d), min(i, x.size-d-1)+1):
            others = [j for j in range(k, k+d+1) if j != i]
            total += 1.0/np.prod(np.abs(x[i]-x[others]))
        w[i] = (-1.0 if i % 2 else 1.0)*total
    return w/np.max(np.abs(w))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'nodes=[-1.4,-0.2,0.1,0.65,1.7]', 'call': 'floater_hormann_weights(nodes, 0).tolist()', 'gold_call': '_oracle_floater_hormann_weights(nodes, 0).tolist()', 'tol': 2e-13}, {'setup': 'nodes=[-1.4,-0.2,0.1,0.65,1.7]', 'call': 'floater_hormann_weights(nodes, 4).tolist()', 'gold_call': '_oracle_floater_hormann_weights(nodes, 4).tolist()', 'tol': 2e-13}, {'setup': 'nodes=[-1.6,-0.9,-0.35,0.0,0.22,0.8,1.3]', 'call': 'floater_hormann_weights(nodes, 3).tolist()', 'gold_call': '_oracle_floater_hormann_weights(nodes, 3).tolist()', 'tol': 2e-13}]
