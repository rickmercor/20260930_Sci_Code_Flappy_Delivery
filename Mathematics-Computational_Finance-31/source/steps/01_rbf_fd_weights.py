"""
Return the specified three-point RBF-FD weight triples.



The nodes are $X_-=X_0-h$, $X_0$, and $X_+=X_0+oh$.

The first-derivative weights are

$$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad

a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad

a_+=\frac{h^2/c^2+2/o}{2h(o+1)}.$$

The second-derivative weights are

$$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad

b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad

b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$

Treat all geometry inputs independently. Do not impose a shape rule or

specialize to a uniform stencil. Use binary64 without intermediate rounding.

RBF-FD approximates derivatives at $X_0$ by weighted samples at $X_-$, $X_0$, and $X_+$. The left gap is $h$ and the right gap is $oh$. The finite-shape corrections in the specified coefficients are part of this task; dropping the terms involving $c$ changes the model. Both weight triples use the order $(-,0,+)$.

Returns
-------
return (first_weights, second_weights)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rbf_fd_weights(
    h: float, o: float, c: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    r"""Return the specified three-point RBF-FD weight triples.

    The nodes are $X_-=X_0-h$, $X_0$, and $X_+=X_0+oh$.
    The first-derivative weights are
    $$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad
    a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad
    a_+=\frac{h^2/c^2+2/o}{2h(o+1)}.$$
    The second-derivative weights are
    $$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad
    b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad
    b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$
    Treat all geometry inputs independently. Do not impose a shape rule or
    specialize to a uniform stencil. Use binary64 without intermediate rounding.

    Parameters
    ----------
    h : float
        Finite left-node spacing $h>0$.
    o : float
        Finite right-to-left spacing ratio $o=(X_+-X_0)/(X_0-X_-)>0$.
    c : float
        Finite independent RBF shape parameter $c>0$.
        Inputs must keep the displayed arithmetic finite and representable.

    Returns
    -------
    tuple[tuple[float, float, float], tuple[float, float, float]]
        Exactly $((a_-,a_0,a_+),(b_-,b_0,b_+))$: first-derivative weights,
        then second-derivative weights, both in node order $(-,0,+)$.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    first_weights = (0.0, 0.0, 0.0)
    second_weights = (0.0, 0.0, 0.0)
    return (first_weights, second_weights)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rbf_fd_weights(
    h: float, o: float, c: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a_minus = -o * (2 * c * c + h * h * o) / (2 * c * c * h * (o + 1))
    a_zero = h * (o - 1) / (2 * c * c) + (o - 1) / (h * o)
    a_plus = (h * h / (c * c) + 2 / o) / (2 * h * (o + 1))
    b_minus = (2 / (h * h) - (o - 3) * o / (c * c)) / (o + 1)
    b_zero = ((o * o - 4 * o + 1) / (c * c) - 2 / (h * h)) / o
    b_plus = (2 * c * c + h * h * (3 * o - 1)) / (c * c * h * h * o * (o + 1))
    return (a_minus, a_zero, a_plus), (b_minus, b_zero, b_plus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "rbf_fd_weights(10.0, 1.0, 30.0)",
            "gold_call": "_oracle_rbf_fd_weights(10.0, 1.0, 30.0)",
        },
        {
            "setup": "",
            "call": "rbf_fd_weights(5.0, 1.5, 20.0)",
            "gold_call": "_oracle_rbf_fd_weights(5.0, 1.5, 20.0)",
        },
        {
            "setup": "",
            "call": "rbf_fd_weights(8.0, 0.8, 24.0)",
            "gold_call": "_oracle_rbf_fd_weights(8.0, 0.8, 24.0)",
        },
    ]
