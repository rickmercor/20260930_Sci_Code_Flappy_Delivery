#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def sep_orth_R_lead_entry(A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int) -> float:
    import numpy as np

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    k = int(k)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    bn = np.linalg.norm(b)
    if bn < 1e-15:
        raise ValueError("b must be nonzero")
    Y[:, 0] = b / bn

    for it in range(k):
        wX = AX[:, :it] @ R[:it, it] + E @ Y[:, it]
        wY = A @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1.0 / beta
        g = X[:, :it].T @ wX
        wX = wX - X[:, :it] @ g
        for _ in range(2):
            corr = X[:, :it].T @ wX
            wX = wX - X[:, :it] @ corr
        alpha = float(np.linalg.norm(wX))
        if alpha < 1e-15:
            raise ValueError("lucky breakdown in top orthonormalization")
        X[:, it] = wX / alpha
        AX[:, it] = A @ X[:, it]
        if it == 0:
            preR = np.array([[alpha]], dtype=float)
        else:
            preR = np.block(
                [
                    [np.eye(it), g.reshape(-1, 1)],
                    [np.zeros((1, it)), np.array([[alpha]])],
                ]
            )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    return float(R[0, -1])

def compressed_offdiag_energy(
    A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int
) -> float:
    import numpy as np

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    k = int(k)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    EY = np.zeros((n, k + 1), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    bn = np.linalg.norm(b)
    if bn < 1e-15:
        raise ValueError("b must be nonzero")
    Y[:, 0] = b / bn

    for it in range(k):
        wE = E @ Y[:, it]
        EY[:, it] = wE
        wX = AX[:, :it] @ R[:it, it] + wE
        wY = A @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1.0 / beta
        g = X[:, :it].T @ wX
        wX = wX - X[:, :it] @ g
        for _ in range(2):
            corr = X[:, :it].T @ wX
            wX = wX - X[:, :it] @ corr
        alpha = float(np.linalg.norm(wX))
        if alpha < 1e-15:
            raise ValueError("lucky breakdown in top orthonormalization")
        X[:, it] = wX / alpha
        AX[:, it] = A @ X[:, it]
        if it == 0:
            preR = np.array([[alpha]], dtype=float)
        else:
            preR = np.block(
                [
                    [np.eye(it), g.reshape(-1, 1)],
                    [np.zeros((1, it)), np.array([[alpha]])],
                ]
            )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    XEY = X.T @ EY[:, :-1]
    return float(np.linalg.norm(XEY, ord="fro") ** 2)

def frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    k = int(k)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    EY = np.zeros((n, k + 1), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    Hbot = np.zeros((k + 1, k), dtype=float)
    bn = np.linalg.norm(b)
    if bn < 1e-15:
        raise ValueError("b must be nonzero")
    Y[:, 0] = b / bn

    for it in range(k):
        wE = E @ Y[:, it]
        EY[:, it] = wE
        wX = AX[:, :it] @ R[:it, it] + wE
        wY = A @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        Hbot[: it + 1, it] = h
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
        Hbot[it + 1, it] = beta
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1.0 / beta
        g = X[:, :it].T @ wX
        wX = wX - X[:, :it] @ g
        for _ in range(2):
            corr = X[:, :it].T @ wX
            wX = wX - X[:, :it] @ corr
        alpha = float(np.linalg.norm(wX))
        if alpha < 1e-15:
            raise ValueError("lucky breakdown in top orthonormalization")
        X[:, it] = wX / alpha
        AX[:, it] = A @ X[:, it]
        if it == 0:
            preR = np.array([[alpha]], dtype=float)
        else:
            preR = np.block(
                [
                    [np.eye(it), g.reshape(-1, 1)],
                    [np.zeros((1, it)), np.array([[alpha]])],
                ]
            )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    Yk = Y[:, :-1]
    H = np.block(
        [[X.T @ AX, X.T @ EY[:, :-1]], [np.zeros((k, k)), Hbot[:-1].copy()]]
    )
    U = np.block([[X, np.zeros((n, k))], [np.zeros((n, k)), Yk]])
    rhs = np.zeros(2 * k, dtype=float)
    rhs[k:] = Yk.T @ b
    v = (U @ (expm(H) @ rhs))[:n]
    return float(v @ b)

def exact_frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,):
        raise ValueError("incompatible shapes")
    big = np.block([[A, E], [np.zeros((n, n)), A]])
    v = (expm(big) @ np.concatenate([np.zeros(n), b]))[:n]
    return float(v @ b)

def approximate_total_network_sensitivity(
    A: np.ndarray, i: int, j: int, k: int
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    k = int(k)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    M = A.T
    E = np.ones((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    b[j] = 1.0

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    EY = np.zeros((n, k + 1), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    Hbot = np.zeros((k + 1, k), dtype=float)
    Y[:, 0] = b / np.linalg.norm(b)

    for it in range(k):
        wE = E @ Y[:, it]
        EY[:, it] = wE
        wX = AX[:, :it] @ R[:it, it] + wE
        wY = M @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        Hbot[: it + 1, it] = h
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
        Hbot[it + 1, it] = beta
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1.0 / beta
        g = X[:, :it].T @ wX
        wX = wX - X[:, :it] @ g
        for _ in range(2):
            corr = X[:, :it].T @ wX
            wX = wX - X[:, :it] @ corr
        alpha = float(np.linalg.norm(wX))
        if alpha < 1e-15:
            raise ValueError("lucky breakdown in top orthonormalization")
        X[:, it] = wX / alpha
        AX[:, it] = M @ X[:, it]
        if it == 0:
            preR = np.array([[alpha]], dtype=float)
        else:
            preR = np.block(
                [
                    [np.eye(it), g.reshape(-1, 1)],
                    [np.zeros((1, it)), np.array([[alpha]])],
                ]
            )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    Yk = Y[:, :-1]
    H = np.block(
        [[X.T @ AX, X.T @ EY[:, :-1]], [np.zeros((k, k)), Hbot[:-1].copy()]]
    )
    U = np.block([[X, np.zeros((n, k))], [np.zeros((n, k)), Yk]])
    rhs = np.zeros(2 * k, dtype=float)
    rhs[k:] = Yk.T @ b
    v = (U @ (expm(H) @ rhs))[:n]
    return float(v[i])

def exact_total_network_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n):
        raise ValueError("invalid inputs")
    M = A.T
    E = np.ones((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    b[j] = 1.0
    big = np.block([[M, E], [np.zeros((n, n)), M]])
    out = expm(big) @ np.concatenate([np.zeros(n), b])
    return float(out[i])

def estrada_edge_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n):
        raise ValueError("invalid inputs")
    return float(expm(A.T)[i, j])

def sensitivity_ratio_pipeline(
    A: np.ndarray, i: int, j: int, k: int
) -> float:
    import numpy as np

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    k = int(k)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    AT = A.T
    E = np.ones((n, n), dtype=float)
    ej = np.zeros(n, dtype=float)
    ej[j] = 1.0

    # Chain prior * only (shared Studio namespace). Never call public
    # names: those can be rebound to the candidate on delivery.
    r_lead = sep_orth_R_lead_entry(AT, E, ej, k)
    offdiag_energy = compressed_offdiag_energy(AT, E, ej, k)
    v_approx = frechet_action_start_overlap(AT, E, ej, k)
    v_exact = exact_frechet_action_start_overlap(AT, E, ej)
    s_tn = approximate_total_network_sensitivity(A, i, j, k)
    s_tn_exact = exact_total_network_sensitivity(A, i, j)
    s_ee = estrada_edge_sensitivity(A, i, j)

    for name, val in (
        ("sep_orth_R_lead_entry", r_lead),
        ("compressed_offdiag_energy", offdiag_energy),
        ("frechet_action_start_overlap", v_approx),
        ("exact_frechet_action_start_overlap", v_exact),
        ("approximate_total_network_sensitivity", s_tn),
        ("exact_total_network_sensitivity", s_tn_exact),
        ("estrada_edge_sensitivity", s_ee),
    ):
        if not np.isfinite(val):
            raise ValueError(f"{name} returned a non-finite value: {val!r}")

    if offdiag_energy < 0:
        raise ValueError(
            "compressed_offdiag_energy must be non-negative (it is a squared Frobenius norm)"
        )

    def _same_ballpark(approx, exact, factor=5.0, atol=1e-8):
        if abs(exact) < atol:
            return abs(approx) < atol * factor
        ratio = approx / exact
        return (1.0 / factor) <= ratio <= factor

    if not _same_ballpark(v_approx, v_exact):
        raise ValueError(
            "frechet_action_start_overlap is not within a plausible range of "
            "exact_frechet_action_start_overlap"
        )
    if not _same_ballpark(s_tn, s_tn_exact):
        raise ValueError(
            "approximate_total_network_sensitivity is not within a plausible "
            "range of exact_total_network_sensitivity"
        )

    if abs(s_ee) < 1e-15:
        raise ValueError("Estrada edge sensitivity is zero")
    return float(s_tn / s_ee)
SCICODE_GOLD_EOF
