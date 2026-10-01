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


def build_convdiff_operator(
    state: dict,
    n: int,
    D: float,
) -> float:
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    if not (isinstance(n, (int, np.integer)) and n >= 1):
        raise ValueError("n must be a positive integer")
    if not (isinstance(D, (int, float, np.integer, np.floating)) and float(D) > 0.0):
        raise ValueError("D must be positive")

    h = 1.0 / (n + 1)

    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = 2.0
        if i > 0:
            L[i, i - 1] = -1.0
        if i + 1 < n:
            L[i, i + 1] = -1.0

    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        C[i, i] = 1.0
        if i > 0:
            C[i, i - 1] = -1.0

    I = np.eye(n)

    A = (
        float(D) / h**2 * (np.kron(L, I) + np.kron(I, L))
        + 1.0 / h * (np.kron(C, I) + np.kron(I, C.T))
    )

    b = np.ones(n * n, dtype=float)
    b /= np.linalg.norm(b)

    state["A"] = A
    state["b"] = b
    state["n"] = n
    state["N"] = n * n
    state["h"] = h
    state["D"] = float(D)

    return float(np.linalg.norm(A, ord="fro"))

import numpy as np


def _draw_sparse_sign(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    zeta = min(rows, 8)
    S = np.zeros((rows, cols), dtype=float)

    for j in range(cols):
        idx = rng.choice(rows, size=zeta, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=zeta)
        S[idx, j] = signs / np.sqrt(zeta)

    return S


def sparse_sign_sketch(
    state: dict,
    seed: int,
    rows: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    if "N" not in state:
        raise ValueError("operator must be constructed first")
    if not (isinstance(rows, (int, np.integer)) and rows >= 1):
        raise ValueError("rows must be positive")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    S = _draw_sparse_sign(rng, rows, int(state["N"]))

    state["rng"] = rng
    state["S"] = S
    state["sketch_rows"] = rows
    state["sketch_seed"] = int(seed)

    return float(np.linalg.norm(S, ord="fro"))

import numpy as np


def _draw_sparse_sign_cycle(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    zeta = min(rows, 8)
    S = np.zeros((rows, cols), dtype=float)

    for j in range(cols):
        idx = rng.choice(rows, size=zeta, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=zeta)
        S[idx, j] = signs / np.sqrt(zeta)

    return S


def truncated_arnoldi_cycle(
    state: dict,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    required = ("A", "b", "S", "rng")
    if any(key not in state for key in required):
        raise ValueError("state is missing required data")
    if not (isinstance(t, (int, np.integer)) and t >= 1):
        raise ValueError("t must be >= 1")
    if not (isinstance(tau, (int, float)) and float(tau) > 1.0):
        raise ValueError("tau must be > 1")
    if not (isinstance(s0, (int, np.integer)) and s0 >= 1):
        raise ValueError("s0 must be positive")
    if not (isinstance(eta, (int, float)) and float(eta) > 0.0):
        raise ValueError("eta must be positive")
    if not (isinstance(mmax, (int, np.integer)) and mmax >= 1):
        raise ValueError("mmax must be positive")

    A = np.asarray(state["A"], dtype=float)
    bstart = np.asarray(state["b"], dtype=float)
    S = np.asarray(state["S"], dtype=float)
    rng = state["rng"]

    beta = float(np.linalg.norm(bstart))
    if beta == 0.0:
        raise ValueError("starting vector must be nonzero")

    b1 = bstart / beta
    Bcols = [b1]

    s = S.shape[0]
    P = (S @ b1).reshape(-1, 1)

    H = np.zeros((mmax + 1, mmax), dtype=float)
    hsub = []

    m = mmax

    for j in range(1, mmax + 1):
        ej = A @ Bcols[j - 1]

        first_i = max(1, j - t + 1)

        for i in range(first_i, j + 1):
            hij = float(np.dot(Bcols[i - 1], ej))
            ej = ej - Bcols[i - 1] * hij
            H[i - 1, j - 1] = hij

        hj1j = float(np.linalg.norm(ej))
        if hj1j <= np.finfo(float).eps:
            raise ValueError("Arnoldi breakdown")

        H[j, j - 1] = hj1j
        hsub.append(hj1j)

        bj1 = ej / hj1j
        Bcols.append(bj1)

        P = np.column_stack((P, S @ bj1))

        if np.linalg.cond(P) > float(tau):
            m = j
            break

        if s < float(eta) * (j + 1):
            Sincr = _draw_sparse_sign_cycle(rng, s0, A.shape[0])
            S = np.vstack((S, Sincr))

            Bpartial = np.column_stack(Bcols)
            P = np.vstack((P, Sincr @ Bpartial))

            s += s0

        m = j

    Bm = np.column_stack(Bcols[:m])
    Hm = H[:m, :m]
    bmp1 = Bcols[m]

    state["cycle_Bm"] = Bm
    state["cycle_Hm"] = Hm
    state["cycle_bmp1"] = bmp1
    state["cycle_hmp1_m"] = float(hsub[m - 1])
    state["cycle_subdiag"] = np.asarray(hsub[:m - 1], dtype=float)
    state["S"] = S
    state["sketch_rows"] = S.shape[0]
    state["beta"] = beta

    return float(m)

import numpy as np


def rank1_harmonic_update(state: dict) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    required = (
        "A",
        "S",
        "cycle_Bm",
        "cycle_Hm",
        "cycle_bmp1",
        "cycle_hmp1_m",
    )
    if any(key not in state for key in required):
        raise ValueError("state is missing rank-1 update data")

    A = np.asarray(state["A"], dtype=float)
    S = np.asarray(state["S"], dtype=float)
    Bm = np.asarray(state["cycle_Bm"], dtype=float)
    Hm = np.asarray(state["cycle_Hm"], dtype=float)
    bmp1 = np.asarray(state["cycle_bmp1"], dtype=float)
    hmp1_m = float(state["cycle_hmp1_m"])

    if not np.isclose(np.linalg.norm(bmp1), 1.0, atol=1e-10):
        raise ValueError("bmp1 must be unit norm")

    ABm = A @ Bm
    SABm = S @ ABm
    SBm = S @ Bm
    Sbmp1 = S @ bmp1

    G = SABm.T @ SBm
    rhs = SABm.T @ Sbmp1

    cm = np.linalg.solve(G, rhs)

    residual = bmp1 - Bm @ cm
    alpha = float(np.linalg.norm(residual))

    if alpha <= np.finfo(float).eps:
        raise ValueError("rank-1 correction is singular")

    ebmp1 = residual / alpha

    Hf = Hm.copy()
    Hf[:, -1] += cm * hmp1_m

    htilde = alpha * hmp1_m

    state["cm"] = cm
    state["alpha"] = alpha
    state["corrected_bmp1"] = ebmp1
    state["corrected_Hm"] = Hf
    state["corrected_hmp1_m"] = htilde

    return float(np.linalg.norm(Hf - Hm, ord="fro"))

import numpy as np


def arnoldi_like_approx(state: dict) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    required = ("cycle_Bm", "corrected_Hm", "beta")
    if any(key not in state for key in required):
        raise ValueError("state is missing reduced approximation data")

    Bm = np.asarray(state["cycle_Bm"], dtype=float)
    Hm = np.asarray(state["corrected_Hm"])

    beta = float(state["beta"])
    m = Bm.shape[1]

    if Hm.shape != (m, m):
        raise ValueError("invalid reduced matrix")

    eigvals, V = np.linalg.eig(Hm)
    if np.any(np.real(eigvals) <= 0.0):
        raise ValueError("principal inverse square root requires positive-real spectrum")

    Vinv = np.linalg.inv(V)
    fHm = V @ np.diag(eigvals ** (-0.5)) @ Vinv

    e1 = np.zeros(m, dtype=complex)
    e1[0] = 1.0

    f0 = Bm @ (fHm @ (beta * e1))

    state["f0"] = np.real_if_close(f0)

    return float(np.linalg.norm(f0))

import numpy as np


def _phi(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    out = np.ones(values.shape, dtype=complex)

    for th in theta:
        out *= values - th

    return out


def _quadrature_value(
    z: complex,
    theta: np.ndarray,
    gamma: float,
    beta: float,
    nodes: int,
) -> complex:
    x, w = np.polynomial.legendre.leggauss(nodes)

    q = 0.5 * (x + 1.0)
    weights = 0.5 * w

    v = q / (1.0 - q)
    jac = 1.0 / (1.0 - q) ** 2

    denominator = _phi(-v**2, theta) * (v**2 + z)

    integral = np.sum(weights * jac / denominator)

    return (2.0 * gamma * beta / np.pi) * integral


def restart_error_scalar_kernel(
    theta: np.ndarray,
    gamma: float,
    beta: float,
    z: complex,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    import numpy as np

    theta = np.asarray(theta, dtype=complex)

    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array")
    if r1 < 2 or maxnodes < r1:
        raise ValueError("invalid quadrature sizes")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if not np.isfinite(gamma) or not np.isfinite(beta):
        raise ValueError("gamma and beta must be finite")

    nodes = int(r1)

    previous = _quadrature_value(
        complex(z), theta, float(gamma), float(beta), nodes
    )

    nodes *= 2

    while nodes <= maxnodes:
        current = _quadrature_value(
            complex(z), theta, float(gamma), float(beta), nodes
        )

        if abs(current - previous) <= tol:
            return float(np.real(current))

        previous = current
        nodes *= 2

    return float(np.real(previous))

import numpy as np


def _scalar_kernel_internal(
    z: complex,
    theta: np.ndarray,
    gamma: float,
    beta: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> complex:
    theta = np.asarray(theta, dtype=complex)

    def phi(v):
        out = np.ones(v.shape, dtype=complex)
        for th in theta:
            out *= v - th
        return out

    def estimate(nodes):
        x, w = np.polynomial.legendre.leggauss(nodes)

        q = 0.5 * (x + 1.0)
        weights = 0.5 * w

        v = q / (1.0 - q)
        jac = 1.0 / (1.0 - q) ** 2

        integrand = 1.0 / (phi(-v**2) * (v**2 + z))
        return (2.0 * gamma * beta / np.pi) * np.sum(
            weights * jac * integrand
        )

    nodes = int(r1)
    previous = estimate(nodes)
    nodes *= 2

    while nodes <= maxnodes:
        current = estimate(nodes)

        if abs(current - previous) <= tol:
            return current

        previous = current
        nodes *= 2

    return previous


def error_kernel_matrix_apply(
    state: dict,
    theta_prev: np.ndarray,
    gamma_prev: float,
    beta_prev: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    if "second_corrected_Hm" not in state:
        raise ValueError("second corrected reduced matrix is missing")

    Hm = np.asarray(state["second_corrected_Hm"])

    if Hm.ndim != 2 or Hm.shape[0] != Hm.shape[1]:
        raise ValueError("current reduced matrix must be square")

    eigvals, V = np.linalg.eig(Hm)

    if np.linalg.cond(V) > 1e14:
        raise ValueError("current reduced matrix is numerically diagonalization-singular")

    Vinv = np.linalg.inv(V)

    values = np.array(
        [
            _scalar_kernel_internal(
                z,
                np.asarray(theta_prev, dtype=complex),
                gamma_prev,
                beta_prev,
                tol,
                maxnodes,
                r1,
            )
            for z in eigvals
        ],
        dtype=complex,
    )

    errHm = V @ np.diag(values) @ Vinv

    state["errHm"] = np.real_if_close(errHm)

    return float(np.linalg.norm(errHm, ord="fro"))

import numpy as np


def sketch_and_restart_pipeline(
    n: int,
    D: float,
    sketch_seed: int,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax0: int,
    entry_index: int,
) -> float:
    import numpy as np

    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be positive")
    if float(D) <= 0.0:
        raise ValueError("D must be positive")
    if not isinstance(sketch_seed, (int, np.integer)):
        raise ValueError("sketch_seed must be an integer")
    if t < 1:
        raise ValueError("t must be >= 1")
    if tau <= 1.0:
        raise ValueError("tau must be > 1")
    if s0 < 1:
        raise ValueError("s0 must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if mmax0 < 1:
        raise ValueError("mmax0 must be positive")

    N = n * n

    if not (isinstance(entry_index, (int, np.integer))
            and 0 <= entry_index < N):
        raise ValueError("entry_index out of range")

    state = {}

    # ---------------------------------------------------------------
    # STEP 01
    # ---------------------------------------------------------------
    build_convdiff_operator(
        state,
        n,
        D,
    )

    # ---------------------------------------------------------------
    # STEP 02
    # ---------------------------------------------------------------
    sparse_sign_sketch(
        state,
        sketch_seed,
        s0,
    )

    # ---------------------------------------------------------------
    # STEP 03: first cycle
    # ---------------------------------------------------------------
    truncated_arnoldi_cycle(
        state,
        t,
        tau,
        s0,
        eta,
        mmax0,
    )

    first_m = int(state["cycle_Bm"].shape[1])

    # Preserve first-cycle state under explicit names.
    state["first_Bm"] = state["cycle_Bm"]
    state["first_Hm"] = state["cycle_Hm"]
    state["first_bmp1"] = state["cycle_bmp1"]
    state["first_hmp1_m"] = state["cycle_hmp1_m"]
    state["first_subdiag"] = state["cycle_subdiag"]
    state["first_beta"] = state["beta"]
    state["first_S"] = state["S"]

    # ---------------------------------------------------------------
    # STEP 04: first correction
    # ---------------------------------------------------------------
    rank1_harmonic_update(state)

    state["first_corrected_Hm"] = state["corrected_Hm"]
    state["first_corrected_bmp1"] = state["corrected_bmp1"]
    state["first_corrected_hmp1_m"] = state["corrected_hmp1_m"]

    # ---------------------------------------------------------------
    # STEP 05: zero-restart approximation
    # ---------------------------------------------------------------
    arnoldi_like_approx(state)

    state["first_f0"] = state["f0"]

    # ---------------------------------------------------------------
    # First-cycle restart data
    # ---------------------------------------------------------------
    theta0 = np.linalg.eigvals(state["first_corrected_Hm"])

    subdiag0 = state["first_subdiag"]

    if subdiag0.size:
        gamma0 = (
            float(np.prod(subdiag0))
            * float(state["first_corrected_hmp1_m"])
        )
    else:
        gamma0 = float(state["first_corrected_hmp1_m"])

    beta0 = float(state["first_beta"])

    state["theta0"] = theta0
    state["gamma0"] = gamma0
    state["beta0"] = beta0

    # ---------------------------------------------------------------
    # STEP 06
    #
    # Call the public/scientific subproblem for one deterministic
    # kernel evaluation. The full matrix application follows in Step 07.
    # ---------------------------------------------------------------
    z_probe = complex(theta0[0])

    restart_error_scalar_kernel(
        theta0,
        gamma0,
        beta0,
        z_probe,
        1e-6,
        64,
        4,
    )

    # ---------------------------------------------------------------
    # STEP 03 again: second cycle.
    #
    # The corrected residual direction becomes the second starting
    # vector, and mmax is fixed to the first-cycle dimension.
    # ---------------------------------------------------------------
    second_state = {
        "A": state["A"],
        "b": state["first_corrected_bmp1"],
        "S": state["first_S"].copy(),
        "rng": state["rng"],
    }

    truncated_arnoldi_cycle(
        second_state,
        t,
        tau,
        s0,
        eta,
        first_m,
    )

    state["second_Bm"] = second_state["cycle_Bm"]
    state["second_Hm"] = second_state["cycle_Hm"]
    state["second_bmp1"] = second_state["cycle_bmp1"]
    state["second_hmp1_m"] = second_state["cycle_hmp1_m"]
    state["second_subdiag"] = second_state["cycle_subdiag"]
    state["second_beta"] = second_state["beta"]
    state["second_S"] = second_state["S"]
    state["rng"] = second_state["rng"]

    # ---------------------------------------------------------------
    # STEP 04 again: second correction
    # ---------------------------------------------------------------
    update_state = {
        "A": state["A"],
        "S": state["second_S"],
        "cycle_Bm": state["second_Bm"],
        "cycle_Hm": state["second_Hm"],
        "cycle_bmp1": state["second_bmp1"],
        "cycle_hmp1_m": state["second_hmp1_m"],
    }

    rank1_harmonic_update(update_state)

    state["second_corrected_Hm"] = update_state["corrected_Hm"]
    state["second_corrected_bmp1"] = update_state["corrected_bmp1"]
    state["second_corrected_hmp1_m"] = update_state["corrected_hmp1_m"]

    # ---------------------------------------------------------------
    # STEP 07
    # ---------------------------------------------------------------
    error_kernel_matrix_apply(
        state,
        state["theta0"],
        state["gamma0"],
        state["beta0"],
        1e-6,
        64,
        4,
    )

    # ---------------------------------------------------------------
    # Final restart correction
    # ---------------------------------------------------------------
    H1 = state["second_corrected_Hm"]
    B1 = state["second_Bm"]
    beta1 = float(state["second_beta"])
    errH1 = state["errHm"]

    e1 = np.zeros(H1.shape[0], dtype=complex)
    e1[0] = 1.0

    g1 = B1 @ (errH1 @ (beta1 * e1))

    f0 = np.asarray(state["first_f0"], dtype=complex)
    f1 = f0 + g1

    value = np.real_if_close(f1[entry_index])

    if np.iscomplexobj(value):
        value = np.real(value)

    return float(value)
SCICODE_GOLD_EOF
