#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

# ORACLE SOLUTION
import numpy as np
def deformation_coefficients(r: float, s: float) -> tuple[complex, complex, float]:
    import numpy as np
    r = float(r); s = float(s)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    w = np.exp(2j*np.pi/3.0); wb = np.conj(w)
    f  = -(2.0/9.0)*(2.0*(r*s + w*r/s**2 + wb*s/r**2) - (1.0/(r*s) + wb*r**2/s + w*s**2/r))
    g1 = -(2.0/9.0)*(w*(r**2/s + s/r**2) + wb*(s**2/r + r/s**2) + r*s + 1.0/(r*s))
    g2 =  (1.0/9.0)*(3.0 + 1.0/(r*s) + s**2/r + r**2/s - 2.0*(r*s + s/r**2 + r/s**2))
    return complex(f), complex(g1), float(np.real(g2))

# ORACLE SOLUTION
import numpy as np

def build_local_interaction(
    r: float,
    s: float,
) -> tuple[np.ndarray, float]:
    import numpy as np

    f, g1, g2 = deformation_coefficients(r, s)

    w = np.exp(2j * np.pi / 3.0)
    sigma = np.diag([1.0, w, w**2]).astype(complex)

    tau = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        tau[(j + 1) % 3, j] = 1.0

    I3 = np.eye(3, dtype=complex)

    A = -(
        np.kron(sigma, sigma.conj().T)
        + 0.5 * f * (np.kron(tau, I3) + np.kron(I3, tau))
        + g1 * np.kron(tau, tau)
        + g2 * np.kron(tau, tau.conj().T)
    )

    B = A + A.conj().T
    epsilon = -float(np.linalg.eigvalsh(B)[0])
    h = B + epsilon * np.eye(9, dtype=complex)

    return 0.5 * (h + h.conj().T), epsilon

# ORACLE SOLUTION
import numpy as np

def excited_subspace_projector(
    H: np.ndarray,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    import numpy as np

    H = np.asarray(H, dtype=complex)
    tol = float(tol)

    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("H must be a nonempty square matrix")

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")

    if not np.allclose(H, H.conj().T, rtol=0.0, atol=1e-10):
        raise ValueError("H must be Hermitian")

    w, V = np.linalg.eigh(0.5 * (H + H.conj().T))

    if w[0] < -tol:
        raise ValueError("H must be positive semidefinite within tol")

    U = V[:, w > tol]
    P = U @ U.conj().T if U.shape[1] else np.zeros_like(H)

    return 0.5 * (P + P.conj().T), int((w <= tol).sum())

# ORACLE SOLUTION
import numpy as np

def level3_generator(
    x: np.ndarray,
    p2_perp: np.ndarray,
    delta: float,
    Y: np.ndarray,
) -> np.ndarray:
    import numpy as np

    x = np.asarray(x, dtype=complex)
    P = np.asarray(p2_perp, dtype=complex)
    Y = np.asarray(Y, dtype=complex)

    if x.shape != (9, 9) or not np.allclose(
        x, x.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("x must be Hermitian with shape (9,9)")

    if (
        P.shape != (9, 9)
        or not np.allclose(P, P.conj().T, atol=1e-10, rtol=0.0)
        or not np.allclose(P @ P, P, atol=1e-8, rtol=0.0)
    ):
        raise ValueError(
            "p2_perp must be a Hermitian projector with shape (9,9)"
        )

    if Y.shape != (9, 9) or not np.allclose(
        Y, Y.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("Y must be Hermitian with shape (9,9)")

    delta = float(delta)

    if not np.isfinite(delta):
        raise ValueError("delta must be finite")

    I3 = np.eye(3, dtype=complex)

    x1 = np.kron(x, I3)
    x2 = np.kron(I3, x)

    Z = P @ Y @ P

    q = (
        x1 @ x1
        + x1 @ x2
        + x2 @ x1
        - delta * x1
        + np.kron(I3, Z)
        - np.kron(Z, I3)
    )

    return 0.5 * (q + q.conj().T)

# HELPERS
import numpy as np
def _realify(C: "np.ndarray") -> "np.ndarray":
    import numpy as np
    return np.block([[C.real, -C.imag], [C.imag, C.real]])

def _hermitian_basis_on_range(P: "np.ndarray") -> list:
    import numpy as np
    w, V = np.linalg.eigh(P); W = V[:, w > 0.5]; m = W.shape[1]; basis = []
    for a in range(m):
        for b in range(a, m):
            Z = np.zeros((m, m), dtype=complex)
            if a == b:
                Z[a, a] = 1.0; basis.append(W @ Z @ W.conj().T)
            else:
                Z[a, b] = Z[b, a] = 1/np.sqrt(2); basis.append(W @ Z @ W.conj().T)
                Z = np.zeros((m, m), dtype=complex); Z[a, b] = 1j/np.sqrt(2); Z[b, a] = -1j/np.sqrt(2)
                basis.append(W @ Z @ W.conj().T)
    return basis

def _barrier_sdp_max(c: "np.ndarray", C: "np.ndarray", A: "np.ndarray", z0: "np.ndarray", tol: float) -> "np.ndarray":
    """Maximize c.z subject to C + sum_i z_i A_i >= 0 (real symmetric), starting from a strictly feasible z0.
    Primal log-det barrier with damped Newton steps; stops when n*mu < tol.  Deterministic."""
    import numpy as np
    A = np.asarray(A, dtype=float); n = C.shape[0]; k = len(c)
    z = np.array(z0, dtype=float)
    Q = C + np.tensordot(z, A, axes=1)
    if np.linalg.eigvalsh(Q).min() <= 0.0:
        raise ValueError("barrier start is not strictly feasible")
    mu = max(1.0, abs(float(c @ z)))/n
    while True:
        for _ in range(100):
            L = np.linalg.cholesky(Q); Linv = np.linalg.inv(L); Qinv = Linv.T @ Linv
            QA = np.einsum('ij,kjl->kil', Qinv, A)
            g = -c - mu*np.einsum('kii->k', QA)
            H = mu*np.einsum('kij,lji->kl', QA, QA)
            H = 0.5*(H + H.T) + 1e-14*np.eye(k)
            d = -np.linalg.solve(H, g)
            if float(-g @ d) < 1e-13:
                break
            phi0 = -float(c @ z) - mu*2.0*np.log(np.diag(L)).sum()
            t = 1.0
            while True:
                zn = z + t*d; Qn = C + np.tensordot(zn, A, axes=1)
                try:
                    Ln = np.linalg.cholesky(Qn)
                    if -float(c @ zn) - mu*2.0*np.log(np.diag(Ln)).sum() <= phi0 + 0.25*t*float(g @ d):
                        break
                except np.linalg.LinAlgError:
                    pass
                t *= 0.5
                if t < 1e-12:
                    break
            z, Q = zn, Qn
        if n*mu < tol:
            return z
        mu *= 0.15

# ORACLE SOLUTION
import numpy as np

def solve_certifiable_lti_bound(
    x: "np.ndarray",
    p2_perp: "np.ndarray",
    solver_tol: float = 1e-11,
) -> float:
    import numpy as np

    x = np.asarray(x, dtype=complex)
    P = np.asarray(p2_perp, dtype=complex)
    solver_tol = float(solver_tol)

    if x.shape != (9, 9) or not np.allclose(
        x, x.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("x must be Hermitian with shape (9,9)")

    if np.linalg.eigvalsh(0.5 * (x + x.conj().T))[0] < -1e-10:
        raise ValueError("x must be positive semidefinite")

    if (
        P.shape != (9, 9)
        or not np.allclose(P, P.conj().T, atol=1e-10, rtol=0.0)
        or not np.allclose(P @ P, P, atol=1e-8, rtol=0.0)
    ):
        raise ValueError(
            "p2_perp must be a Hermitian projector with shape (9,9)"
        )

    if not np.isfinite(solver_tol) or solver_tol <= 0.0:
        raise ValueError("solver_tol must be finite and > 0")

    I3 = np.eye(3, dtype=complex)
    x1 = np.kron(x, I3)
    x2 = np.kron(I3, x)

    G0 = x1 @ x1 + x1 @ x2 + x2 @ x1

    e3, V3 = np.linalg.eigh(x1 + x2)
    Wp = V3[:, e3 > 1e-10]
    n = Wp.shape[1]

    red = lambda Op: _realify(Wp.conj().T @ Op @ Wp)

    basis = _hermitian_basis_on_range(P)
    B = np.array(
        [
            red(np.kron(I3, Yb) - np.kron(Yb, I3))
            for Yb in basis
        ]
    )

    C = red(G0)
    X1 = red(x1)

    # Phase I: find a strictly feasible point at delta = 0.
    A1 = np.concatenate([[-np.eye(2 * n)], B])
    c1 = np.zeros(1 + len(B))
    c1[0] = 1.0

    z1 = np.zeros(1 + len(B))
    z1[0] = float(np.linalg.eigvalsh(C).min()) - 1.0

    z1 = _barrier_sdp_max(c1, C, A1, z1, 1e-3)

    if z1[0] <= 0.0:
        raise ValueError("SDP has no strictly feasible point at delta = 0")

    # Phase II: maximize delta from the feasible Phase-I gauge.
    A2 = np.concatenate([[-X1], B])
    c2 = np.zeros(1 + len(B))
    c2[0] = 1.0

    z2 = np.concatenate([[0.0], z1[1:]])
    z2 = _barrier_sdp_max(c2, C, A2, z2, solver_tol)

    delta = float(z2[0])
    y = z2[1:]

    Y = sum(cc * Yb for cc, Yb in zip(y, basis))
    Y = 0.5 * (Y + Y.conj().T)

    q3 = level3_generator(x, P, delta, Y)
    lmin = float(
        np.linalg.eigvalsh(Wp.conj().T @ q3 @ Wp)[0]
    )

    if lmin < -max(10.0 * solver_tol, 1e-10):
        raise ValueError("numerical feasibility check failed")

    return delta

# ORACLE SOLUTION
import numpy as np
def certified_bound_at_parameters(r: float, s: float, tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    import numpy as np
    r = float(r); s = float(s); tol = float(tol); solver_tol = float(solver_tol)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    if not np.isfinite(tol) or tol <= 0.0 or not np.isfinite(solver_tol) or solver_tol <= 0.0:
        raise ValueError("tol and solver_tol must be finite and > 0")
    h, _ = build_local_interaction(r, s)
    P2, _ = excited_subspace_projector(h, tol)      # H_2 = h, so P_2^perp = supp(h)
    return float(solve_certifiable_lti_bound(h, P2, solver_tol))

# ORACLE SOLUTION
import numpy as np
from collections.abc import Callable

def central_difference(
    fn: Callable[[float], float],
    r0: float,
    step: float,
) -> float:
    import numpy as np

    r0 = float(r0)
    step = float(step)

    if not callable(fn):
        raise ValueError("fn must be callable")

    if (
        not np.isfinite(r0)
        or not np.isfinite(step)
        or step <= 0.0
        or r0 - step <= 0.0
    ):
        raise ValueError(
            "r0 and step must be finite, step > 0 and r0 - step > 0"
        )

    fp = float(fn(r0 + step))
    fm = float(fn(r0 - step))

    if not np.isfinite(fp) or not np.isfinite(fm):
        raise ValueError("fn returned a non-finite value")

    return float((fp - fm) / (2.0 * step))

# ORACLE SOLUTION
import numpy as np
def gap_bound_sensitivity(r: float = 0.347, s: float = 0.783, step: float = 1e-4,
                                  tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    import numpy as np
    r = float(r); s = float(s); step = float(step)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    if not np.isfinite(step) or step <= 0.0 or r - step <= 0.0:
        raise ValueError("step must be finite, > 0 and smaller than r")
    fn = lambda rr: certified_bound_at_parameters(rr, s, tol, solver_tol)
    return float(central_difference(fn, r, step))
SCICODE_GOLD_EOF
