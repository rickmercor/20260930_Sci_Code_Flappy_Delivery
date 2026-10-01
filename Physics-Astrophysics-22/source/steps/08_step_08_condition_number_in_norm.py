"""
Given a Gram matrix defining an inner product, the first-order member of the pencil and the right and left eigenvectors of step 4, return the eigenvector adjoint with respect to that inner product, the two norms, the modulus of the pairing and the resulting condition number.

The sensitivity of an eigenvalue to a perturbation of the operator is measured by its condition number, and for a generalised eigenvalue problem in which only the second-order member is perturbed the first-order shift of the eigenvalue is the perturbation sandwiched between the right eigenvector and a left eigenvector, divided by the same sandwich of the first-order member. Taking the supremum over perturbations of bounded size in a given norm turns that expression into the product of the norms of the two eigenvectors divided by the modulus of their pairing through the first-order operator. Because the norm is the energy norm rather than the Euclidean one, the left eigenvector that enters must be the one adjoint with respect to that inner product, not the ordinary left null vector.

The two are related by the Riesz map. The adjoint of an operator with respect to an inner product with Gram matrix E is E inverse times the conjugate transpose times E, so the vector annihilated by the energy adjoint of the shifted pencil is E inverse applied to the vector annihilated by the ordinary conjugate transpose. Writing the ordinary left null vector as z, the energy-adjoint eigenvector is E inverse z, its energy norm squared is z conjugated against E inverse z, and its energy pairing with the first-order operator applied to the right eigenvector collapses to the plain pairing of z with that vector, because the Gram matrices cancel. The condition number is therefore the square root of z against E inverse z, times the energy norm of the right eigenvector, divided by the modulus of the plain pairing. Skipping the Riesz map and using z directly in place of the adjoint vector is the natural slip and it changes the answer by tens of per cent.

Two invariances are worth stating because they decide what the number means. The expression is homogeneous of degree zero in each of the two eigenvectors separately, so the arbitrary scale and phase of either cancels and the value does not depend on the normalisation chosen in step 4. It is not invariant under a common rescaling of the pencil: multiplying both operators by a constant leaves the spectrum and the eigenvectors untouched while dividing every condition number by that constant. Only ratios of condition numbers computed with the same pencil are meaningful on their own, which is why the first-order member is held fixed at twice the derivative throughout. Written this way the value can fall below one, unlike the condition number of a standard eigenvalue problem.

Returns
-------
dict holding the complex array adjoint_vector, the Riesz image of the left vector; the float adjoint_norm, its norm in the given inner product; the float right_norm, the norm of the right eigenvector in the same inner product; the float pairing_modulus, the modulus of the pairing of the left vector with the first-order operator applied to the right eigenvector; and the float condition_number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def condition_number_in_norm(
    gram: np.ndarray,
    operator_b: np.ndarray,
    right_vector: np.ndarray,
    left_vector: np.ndarray,
) -> dict:
    """Form the eigenvalue condition number of the pencil in the inner product given by a Gram matrix.

    Parameters
    ----------
    gram : np.ndarray
        Gram matrix of the inner product.
    operator_b : np.ndarray
        First-order member of the pencil.
    right_vector : np.ndarray
        Right eigenvector.
    left_vector : np.ndarray
        Ordinary left null vector of the shifted pencil.

    Returns
    -------
    dict
        Under the keys adjoint_vector, adjoint_norm, right_norm, pairing_modulus and
        condition_number.

    Raises
    ------
    ValueError
        When the matrices are not square, finite and of the same shape, when the Gram matrix is not
        symmetric and positive definite, when either vector is not finite, non-zero and of matching
        length, or when the pairing vanishes.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as sla


def _validate_norm_inputs(gram, operator_b, right_vector, left_vector):
    G = np.asarray(gram, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] < 2:
        raise ValueError("gram must be a square array of side at least two")
    if not np.all(np.isfinite(G)):
        raise ValueError("gram must be finite")
    if not np.allclose(G, G.T, rtol=0.0, atol=1e-10 * max(1.0, float(np.max(np.abs(G))))):
        raise ValueError("gram must be symmetric")
    B = np.asarray(operator_b, dtype=complex)
    if B.shape != G.shape or not np.all(np.isfinite(B)):
        raise ValueError("operator_b must be a finite array of the same shape as gram")
    x = np.asarray(right_vector, dtype=complex)
    z = np.asarray(left_vector, dtype=complex)
    for name, v in (("right_vector", x), ("left_vector", z)):
        if v.ndim != 1 or v.size != G.shape[0]:
            raise ValueError(name + " must be a one-dimensional array of matching length")
        if not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0.0:
            raise ValueError(name + " must be finite and non-zero")
    return G, B, x, z


def _oracle_condition_number_in_norm(
    gram: np.ndarray,
    operator_b: np.ndarray,
    right_vector: np.ndarray,
    left_vector: np.ndarray,
) -> dict:
    """Reference implementation."""
    G, B, x, z = _validate_norm_inputs(gram, operator_b, right_vector, left_vector)
    try:
        factor = sla.cho_factor(G, lower=False)
    except sla.LinAlgError:
        raise ValueError("gram must be positive definite")
    adjoint = sla.cho_solve(factor, z)
    adjoint_norm_squared = float(np.real(np.conj(z) @ adjoint))
    right_norm_squared = float(np.real(np.conj(x) @ (G @ x)))
    if adjoint_norm_squared <= 0.0 or right_norm_squared <= 0.0:
        raise ValueError("gram must be positive definite")
    pairing = complex(np.conj(z) @ (B @ x))
    pairing_modulus = float(abs(pairing))
    if pairing_modulus == 0.0:
        raise ValueError("the pairing of the left vector with the first-order operator vanishes")
    adjoint_norm = float(np.sqrt(adjoint_norm_squared))
    right_norm = float(np.sqrt(right_norm_squared))
    return {
        "adjoint_vector": adjoint,
        "adjoint_norm": adjoint_norm,
        "right_norm": right_norm,
        "pairing_modulus": pairing_modulus,
        "condition_number": float(adjoint_norm * right_norm / pairing_modulus),
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
    if isinstance(x, complex):
        return (round(x.real, 9), round(x.imag, 9))
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

SETUP = """
import numpy as np
RNG = np.random.default_rng(20260924)
SIZE = 12
# a non-normal generalised eigenproblem A x = lambda B x built from a known eigendecomposition,
# A = B V diag(lam) V^-1: the right eigenvector is the first column of V and the ordinary left null
# vector is the conjugate of the first row of V^-1 B^-1, so no eigenvalue solver and no other step
# is needed, and the pairing z* B x equals one by construction
V = RNG.standard_normal((SIZE, SIZE)) + 1j * RNG.standard_normal((SIZE, SIZE))
B = RNG.standard_normal((SIZE, SIZE)) + 1j * RNG.standard_normal((SIZE, SIZE))
X = V[:, 0].copy()
Z = np.conj(np.linalg.solve(B.T, np.linalg.solve(V.T, np.eye(SIZE)[0])))
# a real symmetric positive definite Gram matrix, and a strongly anisotropic second one
R = RNG.standard_normal((SIZE, SIZE))
G = R @ R.T + SIZE * np.eye(SIZE)
G2 = np.diag(np.logspace(-2.0, 2.0, SIZE))
def digest(out):
    return (round(out["condition_number"], 9), round(out["adjoint_norm"], 8),
            round(out["right_norm"], 8), round(out["pairing_modulus"], 8))
"""


def test_cases():
    return [
        {
            # a non-normal eigenpair in a general inner product
            "setup": SETUP + FLAT,
            "call": "flat(digest(condition_number_in_norm(G, B, X, Z)))",
            "gold_call": "flat(digest(_oracle_condition_number_in_norm(G, B, X, Z)))",
        },
        {
            # the value is invariant under rescaling and rephasing either eigenvector and under a
            # common rescaling of the Gram matrix, scales as the reciprocal of a rescaling of the
            # first-order member, and changes when the inner product changes
            "setup": SETUP + """
def invariance(fn):
    base = fn(G, B, X, Z)["condition_number"]
    scaled = fn(G, B, 3.7 * np.exp(1.1j) * X, -0.2j * Z)["condition_number"]
    fifth = fn(G, 5.0 * B, X, Z)["condition_number"]
    rescaled_gram = fn(3.0 * G, B, X, Z)["condition_number"]
    other = fn(G2, B, X, Z)["condition_number"]
    return (int(abs(scaled / base - 1.0) < 1e-12),
            int(abs(fifth * 5.0 / base - 1.0) < 1e-12),
            int(abs(rescaled_gram / base - 1.0) < 1e-12),
            int(abs(other / base - 1.0) > 0.5),
            round(base, 9))
""" + FLAT,
            "call": "flat(invariance(condition_number_in_norm))",
            "gold_call": "flat(invariance(_oracle_condition_number_in_norm))",
        },
        {
            # boundary: the identity Gram matrix reduces the expression to the Euclidean condition
            # number with the left vector as its own Riesz image; and a two-by-two non-normal case in
            # closed form, x = (1, 0), z = (1, r) with r = c / (lambda1 - lambda2) = 1.5 and B = I,
            # for which kappa = sqrt(1 + r^2 g1 / g2) in the inner product diag(g1, g2)
            "setup": SETUP + """
def closed_forms(fn):
    out = fn(np.eye(SIZE), B, X, Z)
    direct = np.linalg.norm(Z) * np.linalg.norm(X) / abs(np.conj(Z) @ (B @ X))
    x2 = np.array([1.0, 0.0], dtype=complex)
    z2 = np.array([1.0, 1.5], dtype=complex)
    plain = fn(np.eye(2), np.eye(2), x2, z2)["condition_number"]
    weighted = fn(np.diag([4.0, 1.0]), np.eye(2), x2, z2)["condition_number"]
    return (int(abs(out["condition_number"] - direct) < 1e-12),
            int(np.max(np.abs(out["adjoint_vector"] - Z)) < 1e-12),
            int(abs(plain - np.sqrt(3.25)) < 1e-12),
            int(abs(weighted - np.sqrt(10.0)) < 1e-12))
""" + FLAT,
            "call": "flat(closed_forms(condition_number_in_norm))",
            "gold_call": "flat(closed_forms(_oracle_condition_number_in_norm))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(gram=G, operator_b=B, right_vector=X, left_vector=Z)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(condition_number_in_norm, gram=-np.eye(12)), "
                    "verdict(condition_number_in_norm, gram=np.ones((12, 12))), "
                    "verdict(condition_number_in_norm, right_vector=np.zeros(12, dtype=complex)), "
                    "verdict(condition_number_in_norm, left_vector=np.ones(5, dtype=complex)), "
                    "verdict(condition_number_in_norm, operator_b=np.zeros((12, 12)))))",
            "gold_call": "flat((verdict(_oracle_condition_number_in_norm, gram=-np.eye(12)), "
                         "verdict(_oracle_condition_number_in_norm, gram=np.ones((12, 12))), "
                         "verdict(_oracle_condition_number_in_norm, right_vector=np.zeros(12, dtype=complex)), "
                         "verdict(_oracle_condition_number_in_norm, left_vector=np.ones(5, dtype=complex)), "
                         "verdict(_oracle_condition_number_in_norm, operator_b=np.zeros((12, 12)))))",
        },
    ]
