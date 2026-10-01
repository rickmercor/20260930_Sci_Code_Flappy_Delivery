#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def two_qubit_pauli(code: int) -> np.ndarray:
    import numpy as np

    if type(code) is not int:
        raise ValueError("code must be an integer")
    if code < 0 or code > 15:
        raise ValueError("code must lie in {0, ..., 15}")
    I = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    local = (I, X, Y, Z)
    return np.kron(local[code // 4], local[code % 4])

def fsim_unitary(theta: float, phi: float) -> np.ndarray:
    import numpy as np

    theta = float(theta)
    phi = float(phi)
    if theta != theta or phi != phi or abs(theta) == float("inf") or abs(phi) == float("inf"):
        raise ValueError("theta and phi must be finite")
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    eye = np.eye(4, dtype=complex)
    rxx = np.cos(theta / 2.0) * eye - 1.0j * np.sin(theta / 2.0) * XX
    ryy = np.cos(theta / 2.0) * eye - 1.0j * np.sin(theta / 2.0) * YY
    cp = np.diag([1.0, 1.0, 1.0, np.exp(-1.0j * phi)]).astype(complex)
    return cp @ rxx @ ryy

def pauli_transfer_matrix(unitary: np.ndarray) -> np.ndarray:
    import numpy as np

    U = np.asarray(unitary, dtype=complex)
    if U.shape != (4, 4):
        raise ValueError("unitary must have shape (4, 4)")
    if not np.all(np.isfinite(U)):
        raise ValueError("unitary must be finite")
    if np.linalg.norm(U.conj().T @ U - np.eye(4)) > 1e-8:
        raise ValueError("unitary must be unitary")
    chi = np.zeros((16, 16), dtype=float)
    Ud = U.conj().T
    for alpha in range(16):
        Pa = two_qubit_pauli(alpha)
        for beta in range(16):
            Pb = two_qubit_pauli(beta)
            chi[alpha, beta] = float(np.real(np.trace(Pa @ U @ Pb @ Ud) / 4.0))
    return chi

def retained_pair_table(chi: np.ndarray, tau: float) -> np.ndarray:
    import numpy as np
    raw = np.asarray(chi)
    if raw.shape != (16, 16):
        raise ValueError("chi must have shape (16, 16)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("chi must be finite")
    as_c = np.asarray(raw, dtype=complex)
    if np.any(np.abs(as_c.imag) > 0.0):
        raise ValueError("chi must be real")
    C = np.asarray(as_c.real, dtype=float)
    tau = float(tau)
    if tau != tau or tau == float("inf") or tau < 0.0:
        raise ValueError("tau must be a nonnegative finite number")
    rows = []
    for alpha in range(16):
        for beta in range(16):
            if alpha == 0 and beta == 0:
                continue
            value = float(C[alpha, beta])
            if value == 0.0:
                continue
            if abs(value) >= tau:
                rows.append((float(alpha), float(beta), value))
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)

def pair_jointly_accessible(pair_a: np.ndarray, pair_b: np.ndarray) -> float:
    import numpy as np
    a = np.asarray(pair_a, dtype=float).reshape(-1)
    b = np.asarray(pair_b, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3:
        raise ValueError("each pair must be a length-3 row")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("pair rows must be finite")
    def _code(x):
        xr = float(x)
        if abs(xr - round(xr)) > 1e-12:
            raise ValueError("Pauli labels must be integers")
        c = int(round(xr))
        if c < 0 or c > 15:
            raise ValueError("Pauli labels must lie in {0, ..., 15}")
        return c
    def _local_ok(p, q):
        return p == 0 or q == 0 or p == q
    a0, a1 = _code(a[0]), _code(a[1])
    b0, b1 = _code(b[0]), _code(b[1])
    out_ok = _local_ok(a0 // 4, b0 // 4) and _local_ok(a0 % 4, b0 % 4)
    inn_ok = _local_ok(a1 // 4, b1 // 4) and _local_ok(a1 % 4, b1 % 4)
    if out_ok and inn_ok:
        return 1.0
    return 0.0

def partition_shot_overhead(pair_table: np.ndarray) -> float:
    import numpy as np
    table = np.asarray(pair_table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("pair_table must have shape (K, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("pair_table must be finite")
    k = int(table.shape[0])
    if k == 0:
        return 0.0
    for i in range(k):
        for j in (0, 1):
            raw = float(table[i, j])
            if abs(raw - round(raw)) > 1e-12:
                raise ValueError("Pauli labels must be integers")
            code = int(round(raw))
            if code < 0 or code > 15:
                raise ValueError("Pauli labels must lie in {0, ..., 15}")
        if float(table[i, 2]) == 0.0:
            raise ValueError("zero coefficients are excluded from support")
    order = sorted(
        range(k),
        key=lambda i: (
            -round(float(table[i, 2]) ** 2, 12),
            float(table[i, 0]),
            float(table[i, 1]),
        ),
    )
    remaining = list(order)
    total = 0.0
    def _same_identity_class(i, j):
        ai, bi = int(round(float(table[i, 0]))), int(round(float(table[i, 1])))
        aj, bj = int(round(float(table[j, 0]))), int(round(float(table[j, 1])))
        return (ai == 0) == (aj == 0) and (bi == 0) == (bj == 0)
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        keep = []
        for cand in remaining:
            ok = True
            for mem in members:
                if pair_jointly_accessible(table[cand], table[mem]) < 0.5:
                    ok = False
                    break
                if not _same_identity_class(cand, mem):
                    ok = False
                    break
            if ok:
                members.append(cand)
            else:
                keep.append(cand)
        u = table[members, 2]
        l1 = float(np.sum(np.abs(u)))
        l2sq = float(np.sum(u * u))
        if l2sq <= 0.0:
            raise ValueError("a block has vanishing coefficient mass")
        total += (l1 * l1) / l2sq
        remaining = keep
    return float(total)

def total_effective_support(theta: float, phi: float, tau: float) -> float:
    import numpy as np
    tau = float(tau)
    if tau != tau or tau == float("inf") or tau < 0.0:
        raise ValueError("tau must be a nonnegative finite number")
    unitary = fsim_unitary(theta, phi)
    chi = pauli_transfer_matrix(unitary)
    table = retained_pair_table(chi, tau)
    return float(partition_shot_overhead(table))
SCICODE_GOLD_EOF
