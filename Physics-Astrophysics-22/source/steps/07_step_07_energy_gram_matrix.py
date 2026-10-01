"""
Given a resolution, a spacetime dimension, a multipole number, a spin label and a number of Gauss-Legendre quadrature points, return the Gram matrix of the energy inner product and the Gram matrix of the Lebesgue inner product on the space of nodal values of the Chebyshev-Lobatto grid of step 2, both obtained by exact integration of the Lagrange cardinal polynomials, together with the smallest eigenvalue of the energy Gram matrix. The Lebesgue inner product is one half of the integral of the product of the conjugated fields. The quadrature must be exact for the integrands, which requires at least resolution plus (d + 1) // 2 points.

A condition number has no meaning until a norm is chosen, and for a non-selfadjoint evolution problem the choice is not innocent: different norms give qualitatively different pictures of how sensitive the spectrum is. The norm with physical content here is the energy norm, obtained from the conserved energy of the master field associated with time translation along the adapted Eddington-Finkelstein time. Evaluating the stress tensor of the effective action of the master equation on a slice of constant adapted time and pushing the integral onto the compactified coordinate gives an inner product that is one half of the integral over the unit interval of the weight times the product of the conjugated derivatives, plus the reduced potential times the product of the fields. The weight and the reduced potential are exactly the two coefficient functions of step 1, so the norm and the operator are built from the same data.

Positive definiteness of this form is not automatic in the sector of interest. For gravitational vector perturbations in higher dimensions the reduced potential is negative over a neighbourhood of the horizon, so the second term of the integrand is negative there. What rescues the form is the degeneracy of the weight: near the horizon the weight vanishes linearly with slope d - 3, and a field concentrated in a layer of width w next to the horizon pays a gradient cost of order d - 3 while gaining only an amount of order w from the negative potential. This is a Hardy-type inequality, and it makes the form positive definite even when the potential is not positive. The statement can be checked directly by taking the smallest eigenvalue of the matrix representation.

The representation itself is the step where the construction is most easily got wrong. On the discretisation space the inner product is the Gram matrix of the Lagrange cardinal polynomials of the grid, and its entries are integrals of products of two such polynomials against the weight or against the reduced potential. Those integrands are polynomials of degree above twice the resolution, and the quadrature rule that belongs to the collocation nodes is exact only up to the resolution itself, so evaluating the Gram matrix by summing over the collocation nodes with their own quadrature weights aliases the integrand rather than computing it. The aliasing does not vanish as the grid is refined: it shifts the smallest eigenvalues of the Gram matrix, which are precisely the ones a dual norm is sensitive to, and the resulting condition numbers drift without settling. The integrals must be evaluated exactly, which a Gauss-Legendre rule with enough nodes does, the cardinal polynomials being evaluated at the quadrature points by barycentric interpolation. The same construction with the weight and the reduced potential replaced by one and zero gives the Lebesgue inner product, which carries no physical interpretation but is independent of the spacetime dimension and so serves as a uniform baseline.

Returns
-------
dict holding the array energy_gram, the (N + 1) by (N + 1) symmetric Gram matrix of the energy inner product; the array lebesgue_gram, the corresponding matrix of the Lebesgue inner product; and the float smallest_energy_eigenvalue, the smallest eigenvalue of the energy Gram matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_gram_matrix(
    resolution: int,
    d: int,
    ell: int,
    s: float,
    quadrature_points: int,
) -> dict:
    """Build the Gram matrices of the energy and Lebesgue inner products by exact integration.

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
    quadrature_points : int
        Number of Gauss-Legendre quadrature points.

    Returns
    -------
    dict
        Under the keys energy_gram, lebesgue_gram and smallest_energy_eigenvalue.

    Raises
    ------
    ValueError
        When the resolution, dimension, multipole number or spin label is outside the range the
        earlier steps accept, or when the quadrature point count is not an integer large enough to
        integrate the entries exactly.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cardinal_values(nodes, points):
    """Barycentric evaluation of the Chebyshev-Lobatto cardinal polynomials at arbitrary points."""
    count = nodes.size
    index = np.arange(count)
    bary = np.where((index == 0) | (index == count - 1), 0.5, 1.0) * (-1.0) ** index
    gap = points[:, None] - nodes[None, :]
    hit = np.isclose(gap, 0.0, rtol=0.0, atol=0.0)
    safe = np.where(hit, 1.0, gap)
    terms = bary[None, :] / safe
    values = terms / terms.sum(axis=1, keepdims=True)
    if np.any(hit):
        rows = np.any(hit, axis=1)
        values[rows] = hit[rows].astype(float)
    return values


def _oracle_energy_gram_matrix(
    resolution: int,
    d: int,
    ell: int,
    s: float,
    quadrature_points: int,
) -> dict:
    """Reference implementation."""
    grid = _oracle_chebyshev_lobatto_operators(resolution)  # noqa: F821
    nodes = grid["nodes"]
    derivative = grid["derivative"]
    _oracle_compactified_coefficients(nodes, d, ell, s)  # noqa: F821  validates d, ell and s
    if isinstance(quadrature_points, bool) or not isinstance(quadrature_points, (int, np.integer)):
        raise ValueError("quadrature_points must be an integer")
    required = int(resolution) + (int(d) + 1) // 2
    if int(quadrature_points) < required:
        raise ValueError("quadrature_points must be at least resolution plus (d + 1) // 2")

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    points = (1.0 - legendre_nodes) / 2.0
    weights = legendre_weights / 2.0
    cardinal = _cardinal_values(nodes, points)
    cardinal_derivative = cardinal @ derivative

    background = _oracle_compactified_coefficients(points, d, ell, s)  # noqa: F821
    weight = background["weight"]
    reduced = background["reduced_potential"]

    energy = 0.5 * (
        cardinal_derivative.T @ ((weights * weight)[:, None] * cardinal_derivative)
        + cardinal.T @ ((weights * reduced)[:, None] * cardinal)
    )
    energy = 0.5 * (energy + energy.T)
    lebesgue = 0.5 * (cardinal.T @ (weights[:, None] * cardinal))
    lebesgue = 0.5 * (lebesgue + lebesgue.T)
    smallest = float(np.linalg.eigvalsh(energy)[0])
    return {
        "energy_gram": energy,
        "lebesgue_gram": lebesgue,
        "smallest_energy_eigenvalue": smallest,
    }

# =============================================================================
# TEST CASES
# =============================================================================

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
    return (np.round(out["energy_gram"], 10), np.round(out["lebesgue_gram"], 12))
"""


def test_cases():
    return [
        {
            # the graded sector at a small resolution
            "setup": SETUP + FLAT,
            "call": "flat(digest(energy_gram_matrix(10, 14, 2, 2.0, 30)))",
            "gold_call": "flat(digest(_oracle_energy_gram_matrix(10, 14, 2, 2.0, 30)))",
        },
        {
            # the Gram matrices must reproduce the two integrals exactly on a test polynomial, the
            # energy form must stay positive definite although the reduced potential used for the
            # target, 56 - 108 sigma^11, is negative near the horizon, and adding quadrature points
            # must change nothing
            "setup": SETUP + """
def exactness(fn):
    out = fn(16, 14, 2, 2.0, 40)
    more = fn(16, 14, 2, 2.0, 90)
    nodes = (1.0 - np.cos(np.pi * np.arange(17) / 16)) / 2.0
    f = nodes ** 3 - 0.5 * nodes
    # 1/2 int p (f')^2 + q f^2 over [0, 1] with p = s^2 (1 - s^11), q = 56 - 108 s^11
    xs, ws = np.polynomial.legendre.leggauss(400)
    ts = (1.0 - xs) / 2.0
    wq = ws / 2.0
    p = ts ** 2 * (1.0 - ts ** 11)
    q = 56.0 - 108.0 * ts ** 11
    fp = 3.0 * ts ** 2 - 0.5
    ft = ts ** 3 - 0.5 * ts
    target = 0.5 * np.sum(wq * (p * fp ** 2 + q * ft ** 2))
    got = float(f @ (out["energy_gram"] @ f))
    lebesgue = float(f @ (out["lebesgue_gram"] @ f))
    lebesgue_target = 0.5 * np.sum(wq * ft ** 2)
    return (int(abs(got - target) < 1e-12),
            int(abs(lebesgue - lebesgue_target) < 1e-14),
            int(out["smallest_energy_eigenvalue"] > 0.0),
            int(np.max(np.abs(out["energy_gram"] - more["energy_gram"])) < 1e-12),
            int(q.min() < 0.0))
""" + FLAT,
            "call": "flat(exactness(energy_gram_matrix))",
            "gold_call": "flat(exactness(_oracle_energy_gram_matrix))",
        },
        {
            # boundary: the smallest admissible quadrature for the four-dimensional scalar sector
            "setup": SETUP + FLAT,
            "call": "flat(digest(energy_gram_matrix(8, 4, 2, 0.0, 10)))",
            "gold_call": "flat(digest(_oracle_energy_gram_matrix(8, 4, 2, 0.0, 10)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(resolution=10, d=14, ell=2, s=2.0, quadrature_points=30)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(energy_gram_matrix, quadrature_points=16), "
                    "verdict(energy_gram_matrix, quadrature_points=30.0), "
                    "verdict(energy_gram_matrix, resolution=3), "
                    "verdict(energy_gram_matrix, d=2), "
                    "verdict(energy_gram_matrix, s=-1.0)))",
            "gold_call": "flat((verdict(_oracle_energy_gram_matrix, quadrature_points=16), "
                         "verdict(_oracle_energy_gram_matrix, quadrature_points=30.0), "
                         "verdict(_oracle_energy_gram_matrix, resolution=3), "
                         "verdict(_oracle_energy_gram_matrix, d=2), "
                         "verdict(_oracle_energy_gram_matrix, s=-1.0)))",
        },
    ]
