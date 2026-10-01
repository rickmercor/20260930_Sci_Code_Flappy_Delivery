"""
Evaluate rational cardinal functions and their first two analytic derivatives at arbitrary query points, including exact and near-node limits. Preserve evaluation-row and interpolation-column ordering.

The BRI spatial scheme needs rational cardinal functions at quadrature points and their first two derivatives at collocation nodes. The same analytic interpolation machinery also supplies price sensitivities at a requested spot. Row-point/column-node orientation removes the transposed index notation in the source. A rational second derivative need not equal the square of the nodal first-derivative matrix.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rational_evaluation(nodes: "np.ndarray", weights: "np.ndarray", points: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Evaluate rational cardinal bases and their first two derivatives.

    The rational cardinal functions are
    $$\ell_j(p)=\frac{w_j/(p-x_j)}{\sum_k w_k/(p-x_k)}.$$
    Return $E_{ij}=\ell_j(p_i)$, $(D_1)_{ij}=\ell'_j(p_i)$ and
    $(D_2)_{ij}=\ell''_j(p_i)$, with derivatives taken with respect to $p$.
    At an exact node, use the analytic removable limits; the corresponding
    row of $E$ is a unit vector. Evaluate derivatives accurately near nodes.
    The second derivative belongs to this rational interpolant: do not use
    finite differences or square a nodal first-derivative matrix.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero barycentric weights. Common nonzero rescaling has no
        effect on the result.
    points : float or array_like, shape $(Q,)$
        Finite evaluation coordinates; a scalar produces one output row.
        An empty vector produces three arrays with shape $(0,M)$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Exactly $(E,D_1,D_2)$, each binary64 with shape $(Q,M)$; rows are
        evaluation points and columns are interpolation nodes.

    Raises
    ------
    ValueError
        If the shapes, finiteness, node ordering or nonzero weights are invalid.
    ArithmeticError
        If the evaluated rational denominator is zero or nonfinite.

    Inputs must otherwise allow finite binary64 interpolation and derivative
    values. The analytic denominator is assumed nonzero away from nodes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rational_evaluation(nodes: np.ndarray, weights: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate rational cardinal bases and their first two derivatives.

    The rational cardinal functions are
    $$\ell_j(p)=\frac{w_j/(p-x_j)}{\sum_k w_k/(p-x_k)}.$$
    Return $E_{ij}=\ell_j(p_i)$, $(D_1)_{ij}=\ell'_j(p_i)$ and
    $(D_2)_{ij}=\ell''_j(p_i)$, with derivatives taken with respect to $p$.
    At an exact node, use the analytic removable limits; the corresponding
    row of $E$ is a unit vector. Evaluate derivatives accurately near nodes.
    The second derivative belongs to this rational interpolant: do not use
    finite differences or square a nodal first-derivative matrix.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero barycentric weights. Common nonzero rescaling has no
        effect on the result.
    points : float or array_like, shape $(Q,)$
        Finite evaluation coordinates; a scalar produces one output row.
        An empty vector produces three arrays with shape $(0,M)$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Exactly $(E,D_1,D_2)$, each binary64 with shape $(Q,M)$; rows are
        evaluation points and columns are interpolation nodes.

    Raises
    ------
    ValueError
        If the shapes, finiteness, node ordering or nonzero weights are invalid.
    ArithmeticError
        If the evaluated rational denominator is zero or nonfinite.

    Inputs must otherwise allow finite binary64 interpolation and derivative
    values. The analytic denominator is assumed nonzero away from nodes.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    w = np.asarray(weights, dtype=float)
    p = np.atleast_1d(np.asarray(points, dtype=float))
    if w.shape != x.shape or not np.isfinite(w).all() or np.any(w == 0):
        raise ValueError("weights must be a finite nonzero vector matching nodes")
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("points must be a finite scalar or one-dimensional array")
    w = w/np.max(np.abs(w))
    E = np.empty((p.size, x.size))
    D1, D2 = np.empty_like(E), np.empty_like(E)
    for row, point in enumerate(p):
        k = int(np.argmin(np.abs(point-x)))
        other = np.arange(x.size) != k
        diff = point-x[other]
        # Factoring point-x[k] removes the nearest pole before differentiation.
        # This formula is defined at an exact node and stable very close to it.
        b, db, ddb = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
        b[k] = w[k]
        b[other] = w[other]*(point-x[k])/diff
        db[other] = w[other]*(x[k]-x[other])/diff**2
        ddb[other] = -2.0*w[other]*(x[k]-x[other])/diff**3
        total, first, second = np.sum(b), np.sum(db), np.sum(ddb)
        if total == 0.0 or not np.isfinite(total):
            raise ArithmeticError("the rational denominator vanishes or is nonfinite")
        E[row] = b/total
        D1[row] = (db-E[row]*first)/total
        D2[row] = (ddb-E[row]*second-2.0*D1[row]*first)/total
    return E, D1, D2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nnodes=np.array([-1.2,-0.35,0.15,0.8])\nweights=_oracle_floater_hormann_weights(nodes,3)', 'call': '[part.tolist() for part in rational_evaluation(nodes,weights,nodes)]', 'gold_call': '[part.tolist() for part in _oracle_rational_evaluation(nodes,weights,nodes)]', 'tol': 2e-10}, {'setup': 'import numpy as np\nnodes=np.array([-1.2,-0.35,0.15,0.8])\nweights=_oracle_floater_hormann_weights(nodes,3)\npoints=np.array([-0.91,-0.12,0.47])\nvalues=1.3-0.7*nodes+0.8*nodes**2-0.2*nodes**3', 'call': '[(part@values).tolist() for part in rational_evaluation(nodes,weights,points)]', 'gold_call': '[(part@values).tolist() for part in _oracle_rational_evaluation(nodes,weights,points)]', 'tol': 2e-10}, {'setup': 'import numpy as np\nnodes=np.array([-1.2,-0.35,0.15,0.8])\nweights=_oracle_floater_hormann_weights(nodes,3)\npoints=np.r_[np.nextafter(nodes, np.inf),nodes[1]+1e-12,nodes[2]-1e-12]', 'call': '[part.tolist() for part in rational_evaluation(nodes,weights,points)]', 'gold_call': '[part.tolist() for part in _oracle_rational_evaluation(nodes,weights,points)]', 'tol': 2e-09}, {'setup': 'import numpy as np\nnodes=np.array([-1.4,-0.6,-0.1,0.45,1.2])\nweights=37.0*_oracle_floater_hormann_weights(nodes,1)\npoints=np.array([-1.1,-0.6,0.0,0.45,0.93])', 'call': '[part.tolist() for part in rational_evaluation(nodes,weights,points)]', 'gold_call': '[part.tolist() for part in _oracle_rational_evaluation(nodes,weights,points)]', 'tol': 2e-10}]
