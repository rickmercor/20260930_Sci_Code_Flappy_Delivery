#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def reconstruct_gamma(J: "np.ndarray", X: "np.ndarray") -> "np.ndarray":
    import numpy as np
    J = np.asarray(J, dtype=float)
    X = np.asarray(X, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix")
    n4 = J.shape[0]
    if not np.allclose(J, J.T, atol=1e-9):
        raise ValueError("J must be symmetric")
    if X.ndim != 2 or X.shape[1] != n4:
        raise ValueError("X must have shape (m, n) matching J's dimension")
    n = X.shape[0]

    # Basis for the commutant of J within the symmetric matrices. When J's
    # eigenvalues are all simple, the commutant is exactly the monomial
    # span{I, J, J^2, ..., J^(n4-1)} (Cayley-Hamilton) -- use that directly,
    # so this case is byte-for-byte identical to the simple-eigenvalue-only
    # construction this task originally shipped with. Only when J has a
    # genuinely repeated eigenvalue does the commutant become strictly
    # larger (within a repeated eigenspace of dimension m, ANY symmetric
    # combination of its eigenvectors commutes with J, not just multiples
    # of the projector onto that eigenspace), so the monomial basis alone
    # is incomplete there and the eigenspace-block basis is required
    # instead. Detecting which case applies (rather than always taking the
    # eigenspace-block route) avoids letting np.linalg.eigh's floating-point
    # output become part of the answer for instances that don't need it --
    # eigh's exact eigenvectors are not bitwise-reproducible across
    # numpy/BLAS builds even for well-separated eigenvalues, and Stage 2's
    # extreme conditioning would otherwise amplify that platform-dependent
    # noise into a large, non-reproducible swing in the final answer.
    w = np.linalg.eigvalsh(J)
    w_sorted = np.sort(w)
    eig_scale = max(1.0, np.max(np.abs(w_sorted)))
    has_repeat = any(
        abs(w_sorted[i] - w_sorted[i - 1]) < 1e-7 * eig_scale
        for i in range(1, n4)
    )

    if not has_repeat:
        I4 = np.eye(n4)
        basis = [I4]
        for _ in range(1, n4):
            basis.append(basis[-1] @ J)
    else:
        w, Q = np.linalg.eigh(J)
        order = np.argsort(w)
        w = w[order]
        Q = Q[:, order]
        clusters = []
        cur = [0]
        for i in range(1, n4):
            if abs(w[i] - w[i - 1]) < 1e-7 * eig_scale:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        basis = []
        for cl in clusters:
            cols = Q[:, cl]
            m = len(cl)
            for a in range(m):
                for b in range(a, m):
                    if a == b:
                        B = np.outer(cols[:, a], cols[:, a])
                    else:
                        B = np.outer(cols[:, a], cols[:, b]) + np.outer(cols[:, b], cols[:, a])
                    basis.append(B)
    dim = len(basis)

    rows = []
    for B in basis:
        M = X @ B @ X.T
        row = []
        for i in range(n):
            for j in range(i, n):
                row.append(M[i, j])
        rows.append(row)
    Mat = np.array(rows).T
    n_eq = Mat.shape[0]
    _, sv, Vt = np.linalg.svd(Mat)
    # A 1-dimensional solution space for Gamma requires the null space of
    # Mat (as a map from the dim-dimensional commutant to the n_eq
    # orthogonality equations) to have dimension exactly 1. When the
    # commutant has more dimensions than there are equations (n_eq < dim,
    # possible once repeated eigenvalues enlarge the commutant), that
    # already guarantees a null space of dimension at least dim - n_eq,
    # and exactly 1 whenever Mat has full row rank; otherwise (n_eq >= dim)
    # the null space comes entirely from Mat's own rank deficiency, as
    # before. Both checks are relative to the appropriate end of the
    # singular-value spectrum, not both relative to sv[0], since the
    # spectrum can span many orders of magnitude before ever reaching the
    # null direction.
    structural_nullity = max(0, dim - n_eq)
    if structural_nullity > 1:
        raise ValueError("constraint system is underdetermined by more than one dimension")
    if structural_nullity == 1:
        if sv[-1] <= 1e-6 * sv[0]:
            raise ValueError("constraint matrix does not have full row rank, solution space is larger than 1-dimensional")
    else:
        if sv[-1] > 1e-6 * sv[0]:
            raise ValueError("constraint matrix has full rank, no null space (Gamma is not determined)")
        if len(sv) < 2 or sv[-2] < 1e6 * sv[-1]:
            raise ValueError("constraint matrix does not have rank one less than the commutant dimension, null space is not 1-dimensional")
    c = Vt[-1, :]
    Gamma = sum(c[k] * basis[k] for k in range(dim))
    Gamma = 0.5 * (Gamma + Gamma.T)
    # canonical convention, stated directly on Gamma: unit Frobenius norm,
    # trace(Gamma @ J) < 0.
    Gamma = Gamma / np.linalg.norm(Gamma, 'fro')
    if np.trace(Gamma @ J) > 0:
        Gamma = -Gamma
    return Gamma

import numpy as np


def leading_coefficient_A2(J: "np.ndarray", X: "np.ndarray", Gamma: "np.ndarray") -> "np.ndarray":
    import numpy as np
    J = np.asarray(J, dtype=float)
    X = np.asarray(X, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if X.ndim != 2 or J.shape[0] != J.shape[1] or X.shape[1] != J.shape[0]:
        raise ValueError("X and J have incompatible shapes")
    XJG = X @ J @ Gamma @ X.T
    if abs(np.linalg.det(XJG)) < 1e-12:
        raise ValueError("X J Gamma X^T is singular, A2 undefined")
    A2 = np.linalg.inv(XJG)
    A2 = 0.5 * (A2 + A2.T)
    return A2

import numpy as np


def linear_coefficient_A1(J: "np.ndarray", X: "np.ndarray", Gamma: "np.ndarray", A2: "np.ndarray") -> "np.ndarray":
    import numpy as np
    J = np.asarray(J, dtype=float)
    X = np.asarray(X, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    if A2.shape[0] != A2.shape[1]:
        raise ValueError("A2 must be square")
    if X.shape[0] != A2.shape[0]:
        raise ValueError("X and A2 have incompatible shapes")
    XJ2G = X @ (J @ J) @ Gamma @ X.T
    A1 = -A2 @ XJ2G @ A2
    A1 = 0.5 * (A1 + A1.T)
    return A1

import numpy as np


def constant_coefficient_A0(J: "np.ndarray", X: "np.ndarray", Gamma: "np.ndarray",
                                     A1: "np.ndarray", A2: "np.ndarray") -> "np.ndarray":
    import numpy as np
    J = np.asarray(J, dtype=float)
    X = np.asarray(X, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    if A2.shape[0] != A2.shape[1] or A1.shape != A2.shape:
        raise ValueError("A1 and A2 must be square matrices of the same shape")
    if abs(np.linalg.det(A2)) < 1e-12:
        raise ValueError("A2 is singular, A0 undefined")
    XJ3G = X @ (J @ J @ J) @ Gamma @ X.T
    A0 = -A2 @ XJ3G @ A2 + A1 @ np.linalg.inv(A2) @ A1
    A0 = 0.5 * (A0 + A0.T)
    return A0

import numpy as np


def build_conditioned_pencil(A2: "np.ndarray", condition_number: float) -> "np.ndarray":
    import numpy as np
    A2 = np.asarray(A2, dtype=float)
    if A2.shape[0] != A2.shape[1] or A2.shape[0] != 9:
        raise ValueError("A2 must be a 9x9 matrix")
    if not np.allclose(A2, A2.T, atol=1e-8):
        raise ValueError("A2 must be symmetric")
    if condition_number <= 1.0:
        raise ValueError("condition_number must be > 1")
    n = A2.shape[0]
    evals, evecs = np.linalg.eigh(A2)  # ascending order
    for k in range(evecs.shape[1]):
        col = evecs[:, k]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            evecs[:, k] = -col
    prescribed = 10.0 ** np.linspace(0.0, np.log10(condition_number), n)
    A = evecs @ np.diag(prescribed) @ evecs.T
    A = 0.5 * (A + A.T)
    return A

import numpy as np


def pencil_matrix_function(A: "np.ndarray", A0: "np.ndarray") -> "np.ndarray":
    import numpy as np
    A = np.asarray(A, dtype=float)
    A0 = np.asarray(A0, dtype=float)
    if A.shape[0] != A.shape[1] or A0.shape != A.shape:
        raise ValueError("A and A0 must be square matrices of the same shape")
    if not np.allclose(A, A.T, atol=1e-6):
        raise ValueError("A must be symmetric")
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("A must be positive definite")
    B = A0 @ A0
    B = 0.5 * (B + B.T)
    RA = L.T
    RAinv = np.linalg.inv(RA)
    S3 = RAinv.T @ B @ RAinv
    S3 = 0.5 * (S3 + S3.T)
    lam, Q = np.linalg.eigh(S3)
    if np.any(lam <= 0):
        raise ValueError("A^{-1}B has a non-positive eigenvalue, log undefined")
    S4 = Q @ np.diag(np.log(lam)) @ Q.T
    result = RA.T @ S4 @ RA
    return result

import numpy as np


def compute_pencil_entry(J: "np.ndarray", X: "np.ndarray", condition_number: float) -> float:
    Gamma = reconstruct_gamma(J, X)
    A2 = leading_coefficient_A2(J, X, Gamma)
    A1 = linear_coefficient_A1(J, X, Gamma, A2)
    A0 = constant_coefficient_A0(J, X, Gamma, A1, A2)
    A = build_conditioned_pencil(A2, condition_number)
    result = pencil_matrix_function(A, A0)
    return float(result[0, 0])
SCICODE_GOLD_EOF
