"""
Convert raw source multipoles to factorial-normalized local coefficients.

The interaction uses the two-sided Cartesian expansion of the softened kernel

Returns
-------
np.ndarray, shape ((p+1)(p+2)(p+3)/6,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def multipole_to_local(moments: "np.ndarray", displacement: "np.ndarray", epsilon: float, p: int, G: float) -> "np.ndarray":
    """Evaluate a Cartesian multipole-to-local interaction.

    Parameters
    ----------
    moments : np.ndarray
        Raw source multipoles through p, in kernel_derivatives order.
    displacement : np.ndarray
        Source center minus destination center, shape (3,).
    epsilon : float
        Positive Plummer softening.
    p : int
        Total interaction order, 0 <= p <= 6. Retain |n|+|k| <= p.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Normalized local coefficients L_k = (1/k!) d^k phi/dx^k at the
        destination center, where phi is G times the potential of the raw
        source moments under the two-sided Cartesian expansion of the kernel
        about the source and destination centers, keeping only terms with
        |n| + |k| <= p. Derivatives are with respect to destination position;
        ordered as kernel_derivatives.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_multipole_to_local(moments: "np.ndarray", displacement: "np.ndarray", epsilon: float, p: int, G: float) -> "np.ndarray":
    indices = _indices(p)
    lookup = {n:i for i,n in enumerate(indices)}
    derivatives = _oracle_kernel_derivatives(displacement, epsilon, p)
    result = np.zeros(len(indices))
    for i,k in enumerate(indices):
        for j,n in enumerate(indices):
            if sum(k)+sum(n) <= p:
                result[i] += derivatives[lookup[tuple(np.add(k,n))]]*moments[j]/_factorial(n)
        result[i] *= G*(-1)**sum(k)/_factorial(k)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nq=np.linspace(-.1,.4,35)\nd=np.array([2.,.1,-.2])', 'call': 'multipole_to_local(q.copy(), d.copy(), .18, 4, .7)', 'gold_call': '_oracle_multipole_to_local(q.copy(), d.copy(), .18, 4, .7)', 'tol': 1e-10}, {'setup': 'import numpy as np\nq=np.array([2.])\nd=np.zeros(3)', 'call': 'multipole_to_local(q.copy(), d.copy(), .2, 0, 1.)', 'gold_call': '_oracle_multipole_to_local(q.copy(), d.copy(), .2, 0, 1.)', 'tol': 1e-10}, {'setup': 'import numpy as np\nq=np.array([0.,.3,-.2,.1,0.,0.,0.,0.,0.,0.])\nd=np.array([-.9,.2,.4])', 'call': 'multipole_to_local(q.copy(), d.copy(), .1, 2, .8)', 'gold_call': '_oracle_multipole_to_local(q.copy(), d.copy(), .1, 2, .8)', 'tol': 1e-10}]
