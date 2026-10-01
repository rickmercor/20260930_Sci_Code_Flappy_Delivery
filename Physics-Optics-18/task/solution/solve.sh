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


def _qssf_array(value, ndim, *, real=False, nonempty=True):
    """
    Validate numerical arrays without changing caller-owned storage.
    """
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "biufc":
            raise ValueError("Expected a numerical dtype")
        if real and np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("Expected real data")
        array = np.asarray(
            raw.real if real else raw, dtype=float if real else complex
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected numerical array") from error
    if array.ndim != ndim or (nonempty and array.size == 0):
        raise ValueError("Invalid array rank or empty input")
    if not np.isfinite(array).all():
        raise ValueError("Array entries must be finite")
    return array


def _qssf_scalar(value, *, positive=False, nonnegative=False):
    """
    Validate a finite real scalar.
    """
    array = _qssf_array(value, 0, real=True)
    result = float(array)
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError("Scalar is outside the supported domain")
    return result


def _qssf_pair(U, V):
    """
    Validate a pair of equally sized, nonempty square matrices.
    """
    U = _qssf_array(U, 2)
    V = _qssf_array(V, 2)
    if U.shape[0] != U.shape[1] or V.shape != U.shape:
        raise ValueError("U and V must be matching square matrices")
    return U, V


def qssf_dispersion_propagator(
    w: "np.ndarray",
    dz: float,
    d2: float = 1.0,
    beta3: float = 0.08,
    A0: float = 1.5,
    gamma: float = 1.0,
) -> "np.ndarray":
    w = _qssf_array(w, 1, real=True)
    dz = _qssf_scalar(dz, nonnegative=True)
    d2 = _qssf_scalar(d2, positive=True)
    beta3 = _qssf_scalar(beta3)
    A0 = _qssf_scalar(A0, positive=True)
    gamma = _qssf_scalar(gamma, positive=True)
    if beta3 == 0:
        raise ValueError("This window rule requires a nonzero cubic term")
    D = 0.5 * d2 * w**2 + beta3 * w**3
    P = np.exp(0.5j * D * dz)
    coeffs = [beta3, 0.5 * d2, 0.0, 0.5 * gamma * A0**2]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.abs(r.imag) < 1e-10]
    if len(real_roots) != 1 or not np.isfinite(D).all():
        raise ValueError(
            "The real band center must be numerically representable"
        )
    w_RR = float(real_roots[0])
    out = np.empty((3, len(w)), dtype=complex)
    out[0] = P
    out[1] = D
    out[2] = w_RR
    return out

import numpy as np


def qssf_coupled_step(
    A: "np.ndarray",
    U: "np.ndarray",
    V: "np.ndarray",
    P: "np.ndarray",
    dz: float,
    gamma: float = 1.0,
) -> "np.ndarray":
    A = _qssf_array(A, 1)
    P = _qssf_array(P, 1)
    U, V = _qssf_pair(U, V)
    dz = _qssf_scalar(dz, nonnegative=True)
    gamma = _qssf_scalar(gamma)
    if A.shape != P.shape or U.shape != (A.size, A.size):
        raise ValueError("Envelope, propagator and maps must match")

    A_half = np.fft.ifft(P * np.fft.fft(A))
    intensity = np.abs(A_half) ** 2
    A_new = np.fft.ifft(
        P * np.fft.fft(A_half * np.exp(1j * gamma * intensity * dz))
    )
    alpha = 2.0 * gamma * intensity
    mu = gamma * A_half**2
    kappa = np.sqrt(3.0) * abs(gamma) * intensity
    scale = dz * np.sinc(kappa * dz / np.pi)
    u = np.cos(kappa * dz) + 1j * alpha * scale
    v = 1j * mu * scale

    U_half = P[:, None] * U
    V_half = P[:, None] * V
    Ut = np.fft.ifft(
        np.fft.fft(U_half, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    Vt = np.fft.ifft(
        np.fft.ifft(V_half, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    Ut_new = u[:, None] * Ut + v[:, None] * np.conj(Vt)
    Vt_new = u[:, None] * Vt + v[:, None] * np.conj(Ut)
    U_freq = np.fft.fft(
        np.fft.ifft(Ut_new, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    V_freq = np.fft.fft(
        np.fft.fft(Vt_new, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    return np.vstack((A_new, P[:, None] * U_freq, P[:, None] * V_freq))

import numpy as np


def qssf_symplecticity_check(
    U: "np.ndarray", V: "np.ndarray"
) -> "np.ndarray":
    U, V = _qssf_pair(U, V)
    Nt = U.shape[0]
    identity = np.eye(Nt, dtype=complex)
    comm1 = U @ U.conj().T - V @ V.conj().T - identity
    comm2 = U @ V.T - V @ U.T
    err1 = float(np.linalg.norm(comm1) / Nt)
    err2 = float(np.linalg.norm(comm2) / Nt)
    return np.array([err1, err2], dtype=float)

import numpy as np


def qssf_second_moments(
    U: "np.ndarray", V: "np.ndarray", window: "np.ndarray"
) -> "np.ndarray":
    U, V = _qssf_pair(U, V)
    window = np.asarray(window)
    if window.ndim != 1 or window.dtype.kind not in "iu":
        raise ValueError("window must be a one-dimensional integer array")
    if (
        np.any(window < 0)
        or np.any(window >= U.shape[0])
        or len(np.unique(window)) != len(window)
    ):
        raise ValueError("window indices must be unique and in range")
    errors = qssf_symplecticity_check(U, V)
    if np.max(errors) > 1e-9:
        raise ValueError("U and V must define a canonical map")
    Uw = U[window, :]
    Vw = V[window, :]
    Nw = np.conj(Vw) @ Vw.T
    Mw = Uw @ Vw.T
    n = len(window)
    B = np.real(Nw) + 0.5 * np.eye(n)
    Vq = np.block(
        [
            [B + np.real(Mw), np.imag(Mw) + np.imag(Nw)],
            [np.imag(Mw) - np.imag(Nw), B - np.real(Mw)],
        ]
    )
    return (Vq + Vq.T) / 2.0

import numpy as np


def _qssf_covariance(Vq):
    """Validate and symmetrize a covariance without mutating it."""
    Vq = _qssf_array(Vq, 2, real=True, nonempty=False)
    if Vq.shape[0] != Vq.shape[1] or Vq.shape[0] % 2:
        raise ValueError("Vq must be square and even-dimensional")
    if not np.allclose(Vq, Vq.T, rtol=0.0, atol=1e-10):
        raise ValueError("Vq must be symmetric within tolerance 1e-10")
    return (Vq + Vq.T) / 2.0


def _qssf_omega(n):
    """Return the grouped-quadrature symplectic form."""
    return np.block(
        [[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]
    )


def qssf_williamson_decomposition(Vq: "np.ndarray") -> "np.ndarray":
    Vq = _qssf_covariance(Vq)
    n = Vq.shape[0] // 2
    if n == 0:
        return np.empty((1, 0), dtype=float)
    try:
        factor = np.linalg.cholesky(Vq)
    except np.linalg.LinAlgError as error:
        raise ValueError("Vq must be positive definite") from error
    omega = _qssf_omega(n)
    hermitian = 1j * factor.T @ omega @ factor
    hermitian = (hermitian + hermitian.conj().T) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    nu = values[n:][::-1]
    if not np.isfinite(nu).all() or nu[-1] < 0.5 - 1e-10:
        raise ValueError("Covariance violates the uncertainty relation")

    # Positive eigenvectors encode real canonical pairs. The minus sign
    # fixes Omega's orientation. Orthogonality holds also at degeneracies.
    positive = vectors[:, n:][:, ::-1]
    orthogonal = np.sqrt(2.0) * np.concatenate(
        (positive.real, -positive.imag), axis=1
    )
    diagonal = np.concatenate((nu, nu))
    transform = (
        np.sqrt(diagonal)[:, None] * np.linalg.solve(factor.T, orthogonal).T
    )
    return np.vstack((diagonal, transform))

import numpy as np


def qssf_williamson_entropy(
    Vq: "np.ndarray", decomposition: "np.ndarray"
) -> "np.ndarray":
    Vq = _qssf_covariance(Vq)
    decomposition = _qssf_array(decomposition, 2, real=True, nonempty=False)
    n = Vq.shape[0] // 2
    if decomposition.shape != (2 * n + 1, 2 * n):
        raise ValueError("Malformed Williamson decomposition")
    if n == 0:
        return np.array([0.0, 0.0, 1.0, 1.0])
    # The certificate tolerance cannot certify Vq's tighter physicality
    # domain. Validate the actual covariance independently of the packet.
    qssf_williamson_decomposition(Vq)
    diagonal, transform = decomposition[0], decomposition[1:]
    nu = diagonal[:n].copy()
    if (
        not np.allclose(diagonal[n:], nu, rtol=0.0, atol=1e-10)
        or np.any(np.diff(nu) > 1e-10)
        or np.min(nu) < 0.5 - 1e-10
    ):
        raise ValueError("Invalid thermal variance spectrum")
    omega = _qssf_omega(n)
    error_cov = np.linalg.norm(
        transform @ Vq @ transform.T - np.diag(diagonal)
    ) / max(1.0, np.linalg.norm(diagonal))
    error_symp = np.linalg.norm(transform @ omega @ transform.T - omega) / (
        2 * n
    )
    if (
        not np.isfinite([error_cov, error_symp]).all()
        or max(error_cov, error_symp) > 1e-8
    ):
        raise ValueError("Inconsistent canonical coordinate map")
    nu = np.maximum(nu, 0.5)
    nu[np.abs(nu - 0.5) <= 1e-12] = 0.5
    nk = nu - 0.5
    occupied = nk[nk > 0]
    entropy = float(
        np.sum(
            (occupied + 1) * np.log1p(occupied) - occupied * np.log(occupied)
        )
    )
    renyi = float(np.sum(np.log(2 * nu)))
    purity = float(np.exp(-renyi))
    if occupied.size:
        scaled = occupied / np.max(occupied)
        effective = float(np.sum(scaled) ** 2 / np.sum(scaled**2))
    else:
        effective = 1.0
    return np.concatenate(([entropy, renyi, purity, effective], nu))

import numpy as np


def qssf_full_simulation(
    n_steps: int = 40,
    dz: float = 0.01,
    Nt: int = 128,
    T: float = 40.0,
    A0: float = 1.5,
    beta3: float = 0.08,
    d2: float = 1.0,
    gamma: float = 1.0,
    delta_w: float = 3.0,
    quantity: str = "entropy",
) -> float:
    if (
        not isinstance(n_steps, (int, np.integer))
        or isinstance(n_steps, (bool, np.bool_))
        or n_steps < 0
    ):
        raise ValueError("n_steps must be a nonnegative integer")
    if (
        not isinstance(Nt, (int, np.integer))
        or isinstance(Nt, (bool, np.bool_))
        or Nt < 2
        or Nt % 2
    ):
        raise ValueError("Nt must be an even integer >= 2")
    if quantity not in ("entropy", "purity", "K_eff", "photons"):
        raise ValueError("Unknown diagnostic quantity")
    T = _qssf_scalar(T, positive=True)
    delta_w = _qssf_scalar(delta_w, positive=True)
    dz = _qssf_scalar(dz, nonnegative=True)
    dt = T / Nt
    t = (np.arange(Nt) - Nt / 2) * dt
    w = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)

    res1 = qssf_dispersion_propagator(
        w, dz, d2=d2, beta3=beta3, A0=A0, gamma=gamma
    )
    P = res1[0]
    w_RR = float(res1[2, 0].real)

    window = np.where(np.abs(w - w_RR) <= delta_w / 2.0)[0]

    A = A0 / np.cosh(A0 * t)
    U = np.eye(Nt, dtype=complex)
    V = np.zeros((Nt, Nt), dtype=complex)

    errors = qssf_symplecticity_check(U, V)
    if not np.isfinite(errors).all() or np.max(errors) > 1e-9:
        raise RuntimeError("Initial commutator check failed")

    for _ in range(n_steps):
        state = qssf_coupled_step(A, U, V, P, dz, gamma)
        A = state[0]
        boundary = Nt + 1
        U = state[1:boundary]
        V = state[boundary:]
        errors = qssf_symplecticity_check(U, V)
        if not np.isfinite(errors).all() or np.max(errors) > 1e-9:
            raise RuntimeError(
                "Quantum propagation lost canonical commutators"
            )

    Vq = qssf_second_moments(U, V, window)
    decomposition = qssf_williamson_decomposition(Vq)
    res7 = qssf_williamson_entropy(Vq, decomposition)

    SvN = float(res7[0])
    purity = float(res7[2])
    K_eff = float(res7[3])

    if quantity == "purity":
        return purity
    elif quantity == "K_eff":
        return K_eff
    elif quantity == "photons":
        return float(max((np.trace(Vq) - len(window)) / 2.0, 0.0))
    return SvN
SCICODE_GOLD_EOF
