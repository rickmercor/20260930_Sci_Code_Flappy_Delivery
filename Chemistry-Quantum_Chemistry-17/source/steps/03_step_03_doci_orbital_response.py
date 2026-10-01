"""
Evaluate the DOCI ground-state energy and its relaxed orbital gradient and Hessian.

A paired configuration space changes when its spatial orbitals rotate. Rediagonalizing its Hamiltonian relaxes the configuration coefficients, so the orbital curvature contains both the direct integral response and the response of the correlated ground state. A nondegenerate projected ground state defines these derivatives locally.

Returns
-------
return energy, coefficients, gradient, hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def doci_orbital_response(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray", orbitals: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Differentiate the lowest DOCI eigenvalue with respect to orbital rotations.

    All arrays are finite and real. Inputs must not be modified. NumPy and
    SciPy are available. The electronic Hamiltonian and paired determinant
    convention are those of doci_matrix. At a real orthogonal orbital frame
    U, define M(U) = doci_matrix(U.T h U, {U.T L_l U}, occupations).
    For every p < q in lexicographic order, K_(p,q) has entry +1 at (p,q),
    -1 at (q,p), and zero elsewhere. If A(theta) = sum_a theta_a K_a,
    E(theta) is the lowest eigenvalue of M(U exp(A(theta))). The requested
    gradient and Hessian are the first and second derivatives of E(theta)
    at theta=0. The configuration coefficients are rediagonalized for every
    theta; derivatives holding them fixed do not define this energy surface.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron matrix in Eh, 1 <= n <= 6.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    occupations : integer ndarray, shape (nd,k)
        Nonempty selection of distinct increasing orbital tuples, in any
        row order, with 1 <= k <= n. Orbital indices lie in [0,n).
    orbitals : real ndarray, shape (n,n)
        Orthogonal U; its columns are the spatial orbitals of the trial.
        Orthogonality is accepted within absolute tolerance 1e-10.

    Returns
    -------
    energy : float
        Lowest projected eigenvalue E(0), in Eh.
    coefficients : real ndarray, shape (nd,)
        Normalized ground eigenvector in the supplied occupation order.
        Its largest-magnitude component is positive, using the lowest index
        if several components have exactly equal maximum magnitude.
    gradient : real ndarray, shape (m,)
        Relaxed energy gradient in Eh per radian, m=n(n-1)/2.
    hessian : real ndarray, shape (m,m)
        Relaxed energy Hessian in Eh per radian squared, in the same order.
        For n=1, gradient and Hessian have shapes (0,) and (0,0).

    Raises
    ------
    ValueError
        For incompatible matrix/occupation shapes, nonsymmetric h or factors,
        invalid occupations, nonorthogonal orbitals, or a gap between the
        lowest two DOCI eigenvalues <= 1e-10 Eh. No gap is required when nd=1.
    """
    return energy, coefficients, gradient, hessian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oo_generators(n):
    pairs = [(p, q) for p in range(n) for q in range(p+1, n)]
    result = np.zeros((len(pairs), n, n))
    for a, (p, q) in enumerate(pairs):
        result[a, p, q] = 1.0
        result[a, q, p] = -1.0
    return result


def _oo_integral_jets(matrix, generators):
    m, n, _ = generators.shape
    first = np.zeros((m, n, n))
    second = np.zeros((m, m, n, n))
    for a, ka in enumerate(generators):
        first[a] = matrix @ ka - ka @ matrix
    for a, ka in enumerate(generators):
        for b, kb in enumerate(generators):
            second[a, b] = 0.5 * (
                first[a] @ kb - kb @ first[a]
                + first[b] @ ka - ka @ first[b]
            )
    return first, second


def _oo_projected_derivatives(h, factors, occupations, generators):
    h1, h2 = _oo_integral_jets(h, generators)
    factor_jets = [_oo_integral_jets(l, generators) for l in factors]
    m, nd = len(generators), len(occupations)
    first = np.zeros((m, nd, nd))
    second = np.zeros((m, m, nd, nd))
    for row, occ in enumerate(occupations):
        first[:, row, row] = 2*np.sum(h1[:, occ, occ], axis=-1)
        second[:, :, row, row] = 2*np.sum(h2[:, :, occ, occ], axis=-1)
        for l, (dl, ddl) in zip(factors, factor_jets):
            block = l[np.ix_(occ, occ)]
            dblock = dl[:, occ][:, :, occ]
            ddblock = ddl[:, :, occ][:, :, :, occ]
            trace = np.trace(block)
            dtrace = np.trace(dblock, axis1=-2, axis2=-1)
            ddtrace = np.trace(ddblock, axis1=-2, axis2=-1)
            first[:, row, row] += (
                4*trace*dtrace - 2*np.einsum('ij,aji->a', block, dblock)
            )
            second[:, :, row, row] += (
                4*np.outer(dtrace, dtrace) + 4*trace*ddtrace
                - 2*np.einsum('aij,bji->ab', dblock, dblock)
                - 2*np.einsum('ij,abji->ab', block, ddblock)
            )
        for col in range(row):
            added = set(occ) - set(occupations[col])
            removed = set(occupations[col]) - set(occ)
            if len(added) == len(removed) == 1:
                p, q = next(iter(added)), next(iter(removed))
                for l, (dl, ddl) in zip(factors, factor_jets):
                    value, dvalue, ddvalue = l[p,q], dl[:,p,q], ddl[:,:,p,q]
                    first[:, row, col] += 2*value*dvalue
                    second[:, :, row, col] += 2*(
                        np.outer(dvalue, dvalue) + value*ddvalue
                    )
                first[:, col, row] = first[:, row, col]
                second[:, :, col, row] = second[:, :, row, col]
    return first, second


def _oracle_doci_orbital_response(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray", orbitals: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    occupations = np.asarray(occupations, dtype=int)
    orbitals = np.asarray(orbitals, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 6:
        raise ValueError("h must be square with 1 <= n <= 6")
    n = len(h)
    if factors.ndim != 3 or factors.shape[1:] != (n,n) or not 1 <= len(factors) <= 8:
        raise ValueError("Incompatible two-electron factors")
    if (occupations.ndim != 2 or len(occupations) == 0
            or not 1 <= occupations.shape[1] <= n
            or len(np.unique(occupations, axis=0)) != len(occupations)):
        raise ValueError("Invalid occupations")
    if orbitals.shape != (n,n) or not np.allclose(
            orbitals.T @ orbitals, np.eye(n), rtol=0.0, atol=1e-10):
        raise ValueError("Orbitals must be orthogonal with shape (n,n)")
    transformed_h = orbitals.T @ h @ orbitals
    transformed_factors = np.asarray([orbitals.T @ l @ orbitals for l in factors])
    matrix = _oracle_doci_matrix(transformed_h, transformed_factors, occupations)
    energies, vectors = np.linalg.eigh(matrix)
    if len(energies) > 1 and energies[1] - energies[0] <= 1e-10:
        raise ValueError("The projected ground state is not isolated")
    coefficients = vectors[:, 0].copy()
    if coefficients[np.argmax(np.abs(coefficients))] < 0:
        coefficients *= -1
    first, second = _oo_projected_derivatives(
        transformed_h, transformed_factors, occupations, _oo_generators(n)
    )
    gradient = np.einsum('i,aij,j->a', coefficients, first, coefficients)
    hessian = np.einsum('i,abij,j->ab', coefficients, second, coefficients)
    couplings = np.einsum('im,aij,j->am', vectors[:,1:], first, coefficients)
    hessian += 2*(couplings / (energies[0] - energies[1:])) @ couplings.T
    return float(energies[0]), coefficients, gradient, 0.5*(hessian+hessian.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Check relaxed curvature, rotated frames, boundaries, and the gap guard."""
    setup = r"""import numpy as np
from scipy.linalg import expm

H = np.array([[-1.084145, 0.12, -0.08, 0.05], [0.12, -0.885553, 0.11, -0.06], [-0.08, 0.11, -0.763931, 0.09], [0.05, -0.06, 0.09, -0.742439]], dtype=float)
L = np.array([[[0.58, 0.196, -0.084, 0.112], [0.196, 0.43, 0.224, -0.056], [-0.084, 0.224, 0.36, 0.168], [0.112, -0.056, 0.168, 0.29]], [[0.16, -0.308, 0.112, 0.084], [-0.308, -0.22, 0.14, 0.196], [0.112, 0.14, 0.19, -0.224], [0.084, 0.196, -0.224, -0.14]], [[-0.12, 0.084, 0.252, -0.14], [0.084, 0.17, -0.168, 0.056], [0.252, -0.168, 0.08, 0.308], [-0.14, 0.056, 0.308, -0.1]]], dtype=float)
OCC = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)
U = np.eye(4)
"""
    checks = r"""
def _checked(fn, *args):
    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]
    try:
        result = fn(*copied)
    finally:
        for a, b in zip(before, copied):
            if isinstance(a, np.ndarray) and not np.array_equal(a, b):
                raise AssertionError("Input arrays must not be modified")
    n = len(args[0])
    m = n*(n-1)//2
    shapes = [(), (len(args[2]),), (m,), (m,m)]
    if not isinstance(result, tuple) or len(result) != len(shapes):
        raise AssertionError("Expected the documented four-part tuple")
    arrays = [np.asarray(x) for x in result]
    if any(x.shape != shape for x, shape in zip(arrays, shapes)):
        raise AssertionError("Incorrect response output shape")
    if any(np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in arrays):
        raise AssertionError("Expected finite real output")
    return np.concatenate([x.reshape(-1) for x in arrays])

def _raises(fn, *args):
    try:
        _checked(fn, *args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
"""
    rotated = """
A = np.array([[0., 0.17, -0.09, 0.03],
              [-0.17, 0., 0.11, -0.07],
              [0.09, -0.11, 0., 0.13],
              [-0.03, 0.07, -0.13, 0.]])
U = expm(A)
"""
    return [
        {"setup": setup + checks,
         "call": "_checked(doci_orbital_response, H, L, OCC, U)",
         "gold_call": "_checked(_oracle_doci_orbital_response, H, L, OCC, U)",
         "tol": 2e-7},
        {"setup": setup + rotated + checks,
         "call": "_checked(doci_orbital_response, H, L, OCC, U)",
         "gold_call": "_checked(_oracle_doci_orbital_response, H, L, OCC, U)",
         "tol": 2e-7},
        {"setup": setup + rotated + checks,
         "call": "_checked(doci_orbital_response, H, L, OCC[[4,0,2]], U)",
         "gold_call": "_checked(_oracle_doci_orbital_response, H, L, OCC[[4,0,2]], U)",
         "tol": 2e-7},
        {"setup": setup + rotated + checks,
         "call": "_checked(doci_orbital_response, H, L, np.array([[0,1,2,3]]), U)",
         "gold_call": "_checked(_oracle_doci_orbital_response, H, L, np.array([[0,1,2,3]]), U)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(doci_orbital_response, np.array([[-0.7]]), np.array([[[0.3]]]), np.array([[0]]), np.ones((1,1)))",
         "gold_call": "_checked(_oracle_doci_orbital_response, np.array([[-0.7]]), np.array([[[0.3]]]), np.array([[0]]), np.ones((1,1)))",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(doci_orbital_response, 1e-3*H, np.sqrt(1e-3)*L, OCC, U)",
         "gold_call": "_checked(_oracle_doci_orbital_response, 1e-3*H, np.sqrt(1e-3)*L, OCC, U)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_raises(doci_orbital_response, np.zeros((2,2)), np.zeros((1,2,2)), np.array([[0],[1]]), np.eye(2))",
         "gold_call": "_raises(_oracle_doci_orbital_response, np.zeros((2,2)), np.zeros((1,2,2)), np.array([[0],[1]]), np.eye(2))",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_raises(doci_orbital_response, H, L, OCC, np.ones((4,4)))",
         "gold_call": "_raises(_oracle_doci_orbital_response, H, L, OCC, np.ones((4,4)))",
         "tol": 2e-7},
    ]
