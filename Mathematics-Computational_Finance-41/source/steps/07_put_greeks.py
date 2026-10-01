"""
Compute price, delta and gamma from analytic derivatives of the same rational interpolant at the supplied spot. Use the fixed-strike log-price chain rule and preserve full precision.

The interpolant uses a log-price coordinate, but financial delta and gamma differentiate with respect to spot at fixed strike. The raw function docstring specifies the exact chain-rule conversion. Analytic derivatives of the same rational interpolant keep the requested output consistent with the BRI scheme.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def put_greeks(nodes: "np.ndarray", weights: "np.ndarray", values: "np.ndarray",
               spot: float, strike: float) -> tuple[float, float, float]:
    r"""Evaluate put price, delta and gamma from the same rational interpolant.

    At $x_*=\log(S/K)$, obtain $(E,D_1,D_2)$ by calling
    rational_evaluation with the supplied nodes and weights. With nodal
    values $U$, compute $u=EU$, $u_x=D_1U$, and $u_{xx}=D_2U$.
    Return
    $$(V,\Delta,\Gamma)=\left(u,\frac{u_x}{S},
       \frac{u_{xx}-u_x}{S^2}\right).$$
    The derivatives are with respect to the underlying price $S$, holding
    the strike fixed. Use analytic rational derivatives, including exact-node
    limits, without finite differencing, interpolation substitution or rounding.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero rational interpolation weights matching the nodes.
    values : array_like, shape $(M,)$
        Finite nodal option values on this same grid.
    spot : float
        Finite strictly positive underlying price $S$ such that
        $\log(S/K)\in[x_0,x_{M-1}]$.
    strike : float
        Finite strictly positive strike $K$ defining the log-price coordinate.

    Returns
    -------
    tuple[float, float, float]
        Exactly the price, delta and gamma in that order, as unrounded scalars.

    Raises
    ------
    ValueError
        If the arrays or prices violate their stated shape, finiteness,
        ordering, sign or evaluation-interval conditions.
    ArithmeticError
        If rational_evaluation encounters an invalid denominator.

    Inputs must allow finite binary64 outputs; behavior outside the stated
    representable domain is otherwise unspecified.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_put_greeks(nodes: np.ndarray, weights: np.ndarray, values: np.ndarray,
               spot: float, strike: float) -> tuple[float, float, float]:
    r"""Evaluate put price, delta and gamma from the same rational interpolant.

    At $x_*=\log(S/K)$, obtain $(E,D_1,D_2)$ by calling
    _oracle_rational_evaluation with the supplied nodes and weights. With nodal
    values $U$, compute $u=EU$, $u_x=D_1U$, and $u_{xx}=D_2U$.
    Return
    $$(V,\Delta,\Gamma)=\left(u,\frac{u_x}{S},
       \frac{u_{xx}-u_x}{S^2}\right).$$
    The derivatives are with respect to the underlying price $S$, holding
    the strike fixed. Use analytic rational derivatives, including exact-node
    limits, without finite differencing, interpolation substitution or rounding.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero rational interpolation weights matching the nodes.
    values : array_like, shape $(M,)$
        Finite nodal option values on this same grid.
    spot : float
        Finite strictly positive underlying price $S$ such that
        $\log(S/K)\in[x_0,x_{M-1}]$.
    strike : float
        Finite strictly positive strike $K$ defining the log-price coordinate.

    Returns
    -------
    tuple[float, float, float]
        Exactly the price, delta and gamma in that order, as unrounded scalars.

    Raises
    ------
    ValueError
        If the arrays or prices violate their stated shape, finiteness,
        ordering, sign or evaluation-interval conditions.
    ArithmeticError
        If _oracle_rational_evaluation encounters an invalid denominator.

    Inputs must allow finite binary64 outputs; behavior outside the stated
    representable domain is otherwise unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    spot, strike = float(spot), float(strike)
    if not np.isfinite([spot, strike]).all() or spot <= 0 or strike <= 0:
        raise ValueError("positive finite spot and strike required")
    values = np.asarray(values, dtype=float)
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector matching nodes")
    point = np.log(spot/strike)
    if point < x[0] or point > x[-1]:
        raise ValueError("log(spot/strike) must lie in the grid interval")
    E, D1, D2 = _oracle_rational_evaluation(x, weights, [point])
    price, ux, uxx = float(E[0]@values), float(D1[0]@values), float(D2[0]@values)
    return price, ux/spot, (uxx-ux)/spot**2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nnodes=np.array([-1.2,-0.35,0.15,0.8])\nweights=_oracle_floater_hormann_weights(nodes,3)\nvalues=2.5-1.7*nodes+0.8*nodes**2-0.2*nodes**3\nstrike=113.0\nspot=strike*np.exp(0.27)', 'call': 'put_greeks(nodes,weights,values,spot,strike)', 'gold_call': '_oracle_put_greeks(nodes,weights,values,spot,strike)', 'tol': 2e-10}, {'setup': 'import numpy as np\nnodes=np.array([-1.1,-0.5,0.0,0.45,1.2])\nweights=_oracle_floater_hormann_weights(nodes,4)\nvalues=1.2+0.9*nodes+0.4*nodes**2-0.15*nodes**3+0.08*nodes**4\nstrike=90.0\nspot=strike*np.exp(nodes[1])', 'call': 'put_greeks(nodes,weights,values,spot,strike)', 'gold_call': '_oracle_put_greeks(nodes,weights,values,spot,strike)', 'tol': 2e-10}, {'setup': 'import numpy as np\nnodes=np.array([-1.4,-0.6,-0.1,0.45,1.2])\nweights=_oracle_floater_hormann_weights(nodes,1)\nvalues=np.full(nodes.size,7.3)', 'call': 'put_greeks(nodes,weights,values,102.0,100.0)', 'gold_call': '_oracle_put_greeks(nodes,weights,values,102.0,100.0)', 'tol': 2e-10}]
