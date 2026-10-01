"""
Given a resolution, a spacetime dimension, a multipole number and a spin label, assemble the two matrices of the total transmission mode pencil on the Chebyshev-Lobatto grid, using the compactification weight, its derivative and the reduced potential of step 1 together with the grid operators of step 2, in units where the horizon radius is one. The second-order term is expanded by the product rule rather than left in divergence form, so that the operator acts on the interpolating polynomial directly. Return both matrices together with the grid, the differentiation matrix and the smallest singular value of the second matrix, which records that it is singular.

Quasinormal modes are ingoing at the horizon and outgoing at infinity. Total transmission modes carry the same asymptotic behaviour at both ends instead, so that a wave crosses the barrier leaving nothing reflected, and the left family is the one whose master field behaves as exp(+i omega x) as the tortoise coordinate runs to either infinity. The construction that turns this into a matrix problem is to adopt the Eddington-Finkelstein time adapted to that family, which is the outgoing combination (t - x) divided by the horizon radius, and then to strip the common asymptotic factor from the field by rescaling with exp(-i omega x). The rescaled field is then finite and analytic at both ends precisely when the frequency is a total transmission mode, so the eigenvalue condition becomes regularity alone and no boundary condition is imposed by hand.

Carrying that rescaling through the frequency-domain master equation removes the term in omega squared, because the free-propagation piece of the potential cancels against the exponential, and leaves a wave operator that is first order in the frequency. Written in the compactified coordinate, with the weight p and the reduced potential q of step 1, the equation becomes a generalised eigenvalue problem for the pencil of two operators: the second-order operator d/dsigma of p d/dsigma minus q, and the first-order operator twice d/dsigma, with the frequency entering linearly through the product of i and omega. The second operator is fixed to be exactly twice the derivative and not some multiple of it, because multiplying both operators by a common factor leaves the spectrum untouched while dividing every condition number by that factor; the condition numbers only become well-defined numbers once this scale is pinned.

Two features of the resulting pencil are worth naming, because they decide how the problem must be solved. First, the second operator is singular, since it annihilates constants, so the pencil carries an infinite eigenvalue and cannot be converted into a standard eigenvalue problem by inverting it. Second, the weight p vanishes at both ends of the interval, so the collocation rows at the two endpoints degenerate: at infinity the second-order operator contributes nothing and the row reduces to a relation between the value and the slope, and at the horizon it reduces to a relation fixed by the slope of the weight. Those two degenerate rows are exactly the two admissibility conditions, which is why the discretisation is built on the full grid with no rows replaced.

How the second-order term is discretised is not a matter of taste. Expanding it by the product rule into the weight times the second derivative plus the derivative of the weight times the first derivative applies the operator directly to the interpolating polynomial, which is what the differential operator does. Leaving it in divergence form and building it as the differentiation matrix times the weight times the differentiation matrix does something else: it forms the flux p times the derivative at the nodes, silently replaces that by its interpolating polynomial of the grid degree, and differentiates the replacement. Because the weight is itself a polynomial of degree d - 1, the flux has degree beyond the grid and the replacement aliases, worst of all in the two endpoint rows where the admissibility conditions live. The two forms have the same continuum limit and give the same eigenvalue to every digit, but they do not give the same eigenvectors, and quantities built from the eigenvectors inherit the difference: with the expanded form the condition number of the mode settles under grid refinement, while with the divergence form it drifts without settling.

Returns
-------
dict holding the array operator_a, the second-order member of the pencil; the array operator_b, twice the differentiation matrix; the array nodes; the array derivative; and the float operator_b_smallest_singular_value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transmission_pencil(
    resolution: int,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Assemble the operator pencil of the left total transmission mode problem.

    Parameters
    ----------
    resolution : int
        Grid resolution N; the grid carries N + 1 nodes.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.

    Returns
    -------
    dict
        Under the keys operator_a, operator_b, nodes, derivative and
        operator_b_smallest_singular_value.

    Raises
    ------
    ValueError
        When the resolution, dimension, multipole number or spin label is outside the range the
        earlier steps accept.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transmission_pencil(
    resolution: int,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Reference implementation."""
    grid = _oracle_chebyshev_lobatto_operators(resolution)  # noqa: F821
    nodes = grid["nodes"]
    derivative = grid["derivative"]
    background = _oracle_compactified_coefficients(nodes, d, ell, s)  # noqa: F821
    weight = background["weight"]
    weight_derivative = background["weight_derivative"]
    reduced = background["reduced_potential"]
    second = derivative @ derivative
    operator_a = (weight[:, None] * second
                  + weight_derivative[:, None] * derivative
                  - np.diag(reduced))
    operator_b = 2.0 * derivative
    smallest = float(np.linalg.svd(operator_b, compute_uv=False)[-1])
    return {
        "operator_a": operator_a,
        "operator_b": operator_b,
        "nodes": nodes,
        "derivative": derivative,
        "operator_b_smallest_singular_value": smallest,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

    SETUP = """
import numpy as np
def digest(out):
    return (np.round(out["operator_a"], 7), np.round(out["operator_b"], 9))
"""
    return [
        {
            # the graded sector at a small resolution
            "setup": SETUP + FLAT,
            "call": "flat(digest(transmission_pencil(10, 14, 2, 2.0)))",
            "gold_call": "flat(digest(_oracle_transmission_pencil(10, 14, 2, 2.0)))",
        },
        {
            # the exact eigenpair of the four-dimensional vector sector: at ell = 2 the pencil is
            # annihilated by the polynomial 1 + 3 sigma / 4 at the eigenvalue -4, which is the
            # Schwarzschild algebraically special frequency; the second operator is singular
            "setup": SETUP + """
def exact_pair(fn):
    out = fn(18, 4, 2, 2.0)
    A, B, nodes = out["operator_a"], out["operator_b"], out["nodes"]
    psi = 1.0 + 0.75 * nodes
    residual = np.max(np.abs(A @ psi + 4.0 * (B @ psi)))
    return (int(residual < 1e-10),
            int(out["operator_b_smallest_singular_value"] < 1e-12),
            int(np.max(np.abs(B - 2.0 * out["derivative"])) == 0.0))
""" + FLAT,
            "call": "flat(exact_pair(transmission_pencil))",
            "gold_call": "flat(exact_pair(_oracle_transmission_pencil))",
        },
        {
            # boundary: the scalar sector in nine dimensions, and the degenerate endpoint rows, where
            # the second-order part contributes nothing at infinity because the weight vanishes there
            "setup": SETUP + """
def endpoints(fn):
    out = fn(16, 9, 2, 0.0)
    A, nodes = out["operator_a"], out["nodes"]
    return (np.round(A[0, :4], 9), np.round(A[-1, -4:], 9), round(float(nodes[0]), 14))
""" + FLAT,
            "call": "flat(endpoints(transmission_pencil))",
            "gold_call": "flat(endpoints(_oracle_transmission_pencil))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(resolution=10, d=14, ell=2, s=2.0)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(transmission_pencil, resolution=2), "
                    "verdict(transmission_pencil, d=3), "
                    "verdict(transmission_pencil, ell=-2), "
                    "verdict(transmission_pencil, s=-3.0)))",
            "gold_call": "flat((verdict(_oracle_transmission_pencil, resolution=2), "
                         "verdict(_oracle_transmission_pencil, d=3), "
                         "verdict(_oracle_transmission_pencil, ell=-2), "
                         "verdict(_oracle_transmission_pencil, s=-3.0)))",
        },
    ]
