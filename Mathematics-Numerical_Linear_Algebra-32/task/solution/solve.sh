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

def construct_flat_spectrum_matrix(m, n, s, seed):
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    m, n = int(m), int(n)
    if m < 1 or n < 1 or m < n:
        raise ValueError("require m >= n >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    return U @ np.diag(s) @ V.T

import numpy as np

def _pack_three(A, B, C):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    header = np.array(
        [A.shape[0], A.shape[1], B.shape[0], B.shape[1], C.shape[0], C.shape[1]],
        dtype=float,
    )
    return np.concatenate([header, A.ravel(), B.ravel(), C.ravel()])


def _unpack_three(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 6:
        raise ValueError("pack is too short")
    mA, nA, mB, nB, mC, nC = [int(round(float(x))) for x in pack[:6]]
    if min(mA, nA, mB, nB, mC, nC) < 1:
        raise ValueError("packed shapes must be positive")
    need = 6 + mA * nA + mB * nB + mC * nC
    if pack.size != need:
        raise ValueError("pack length does not match header")
    i = 6
    A = pack[i : i + mA * nA].reshape(mA, nA)
    i += mA * nA
    B = pack[i : i + mB * nB].reshape(mB, nB)
    i += mB * nB
    C = pack[i:].reshape(mC, nC)
    return A, B, C


def draw_gaussian_test_matrices(m, n, s_width, d, l, seed):
    ints = (m, n, s_width, d, l)
    if not all(isinstance(v, (int, np.integer)) for v in ints):
        raise ValueError("m, n, s_width, d, and l must be integers")
    m, n, s_width, d, l = (int(v) for v in ints)
    if m < 1 or n < 1:
        raise ValueError("require m >= 1 and n >= 1")
    if s_width < 1:
        raise ValueError("require s_width >= 1")
    if not (l > s_width and d > s_width):
        raise ValueError("require min(l, d) > s_width")
    if s_width > n or s_width > m:
        raise ValueError("require s_width <= min(m, n)")
    rng = np.random.default_rng(int(seed))
    Omega = rng.standard_normal((n, s_width))
    Psi = rng.standard_normal((d, m))
    Phi = rng.standard_normal((n, l))
    return _pack_three(Omega, Psi, Phi)

import numpy as np

def _pack_three(A, B, C):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    header = np.array(
        [A.shape[0], A.shape[1], B.shape[0], B.shape[1], C.shape[0], C.shape[1]],
        dtype=float,
    )
    return np.concatenate([header, A.ravel(), B.ravel(), C.ravel()])


def _unpack_three(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 6:
        raise ValueError("pack is too short")
    mA, nA, mB, nB, mC, nC = [int(round(float(x))) for x in pack[:6]]
    if min(mA, nA, mB, nB, mC, nC) < 1:
        raise ValueError("packed shapes must be positive")
    need = 6 + mA * nA + mB * nB + mC * nC
    if pack.size != need:
        raise ValueError("pack length does not match header")
    i = 6
    A = pack[i : i + mA * nA].reshape(mA, nA)
    i += mA * nA
    B = pack[i : i + mB * nB].reshape(mB, nB)
    i += mB * nB
    C = pack[i:].reshape(mC, nC)
    return A, B, C


def form_one_pass_sketches(A, test_pack):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    if m < 1 or n < 1:
        raise ValueError("A must be nonempty")
    Omega, Psi, Phi = _unpack_three(test_pack)
    if Omega.shape[0] != n:
        raise ValueError("Omega must have n rows")
    if Psi.shape[1] != m:
        raise ValueError("Psi must have m columns")
    if Phi.shape[0] != n:
        raise ValueError("Phi must have n rows")
    Y = A @ Omega
    W = Psi @ A
    Z = A @ Phi
    return _pack_three(Y, W, Z)

import numpy as np

def amplify_rangefinder(Y, Z, q):
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Y.ndim != 2 or Z.ndim != 2:
        raise ValueError("Y and Z must be 2D")
    if Y.shape[0] != Z.shape[0]:
        raise ValueError("Y and Z must have the same number of rows")
    if Y.shape[1] < 1 or Z.shape[1] < Y.shape[1]:
        raise ValueError("require l >= s_width >= 1")
    if not isinstance(q, (int, np.integer)):
        raise ValueError("q must be an integer")
    q = int(q)
    if q < 1:
        raise ValueError("require q >= 1")
    Yhat = np.array(Y, dtype=float, copy=True)
    for _ in range(q):
        core = Z.T @ Yhat
        X, R = np.linalg.qr(core, mode="reduced")
        if np.any(np.abs(np.diag(R)) < 1e-14):
            raise ValueError("amplifier core is rank deficient")
        Yhat = Z @ X
    return Yhat

import numpy as np

def rangefinder_thin_factors(Yhat):
    Yhat = np.asarray(Yhat, dtype=float)
    if Yhat.ndim != 2:
        raise ValueError("Yhat must be 2D")
    m, s_width = Yhat.shape
    if m < s_width or s_width < 1:
        raise ValueError("require m >= s_width >= 1")
    Q, R = np.linalg.qr(Yhat, mode="reduced")
    if np.any(np.abs(np.diag(R)) < 1e-14):
        raise ValueError("rangefinder is rank deficient")
    return np.vstack([Q, R])

import numpy as np

def sketched_coefficient_matrix(Q, Psi, W):
    Q = np.asarray(Q, dtype=float)
    Psi = np.asarray(Psi, dtype=float)
    W = np.asarray(W, dtype=float)
    if Q.ndim != 2 or Psi.ndim != 2 or W.ndim != 2:
        raise ValueError("Q, Psi, and W must be 2D")
    m, s_width = Q.shape
    d, m_psi = Psi.shape
    d_w, n = W.shape
    if m < 1 or s_width < 1:
        raise ValueError("Q must be nonempty")
    if m_psi != m:
        raise ValueError("Psi must have as many columns as Q has rows")
    if d_w != d:
        raise ValueError("W must have as many rows as Psi")
    if d < s_width:
        raise ValueError("require d >= s_width")
    if n < 1:
        raise ValueError("W must have at least one column")
    B, *_ = np.linalg.lstsq(Psi @ Q, W, rcond=None)
    return np.asarray(B, dtype=float)

import numpy as np

def reconstruction_entry(Q, B, rank, row, col):
    Q = np.asarray(Q, dtype=float)
    B = np.asarray(B, dtype=float)
    if Q.ndim != 2 or B.ndim != 2:
        raise ValueError("Q and B must be 2D")
    if Q.shape[1] != B.shape[0]:
        raise ValueError("Q and B have incompatible inner dimensions")
    if not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")
    rank = int(rank)
    max_rank = min(B.shape)
    if rank < 1 or rank > max_rank:
        raise ValueError("rank must lie between 1 and min(B.shape)")
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        raise ValueError("row and col must be integers")
    row, col = int(row), int(col)
    if row < 0 or row >= Q.shape[0] or col < 0 or col >= B.shape[1]:
        raise ValueError("row and col are out of bounds")
    Ub, sb, Vbh = np.linalg.svd(B, full_matrices=False)
    Ahat = (Q @ Ub[:, :rank]) * sb[:rank] @ Vbh[:rank]
    return float(Ahat[row, col])

import importlib.util
import sys
from pathlib import Path

import numpy as np

def _step_roots():
    roots = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    roots.append(Path.cwd())
    for entry in list(sys.path):
        if entry:
            roots.append(Path(entry))
    out = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            continue
        if key in seen or not root.is_dir():
            continue
        seen.add(key)
        out.append(root)
    return out


def _load_step(key):
    """Load a sibling step module whose filename ends with ``{key}.py``."""
    for root in _step_roots():
        matches = sorted(root.glob(f"*{key}.py"))
        if not matches:
            continue
        path = matches[0]
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(f"could not load sub-problem {key}")


def _bind_oracle(public_name):
    g = globals()
    oracle_name = "" + public_name
    if oracle_name in g and callable(g[oracle_name]):
        return g[oracle_name]
    if public_name in g and callable(g[public_name]) and public_name != "run_sketch_power_entry":
        fn = g[public_name]
        try:
            mod = _load_step(public_name)
            return getattr(mod, oracle_name, fn)
        except ImportError:
            return fn
    mod = _load_step(public_name)
    return getattr(mod, oracle_name)


construct_flat_spectrum_matrix = _bind_oracle("construct_flat_spectrum_matrix")
draw_gaussian_test_matrices = _bind_oracle("draw_gaussian_test_matrices")
form_one_pass_sketches = _bind_oracle("form_one_pass_sketches")
amplify_rangefinder = _bind_oracle("amplify_rangefinder")
rangefinder_thin_factors = _bind_oracle("rangefinder_thin_factors")
sketched_coefficient_matrix = _bind_oracle("sketched_coefficient_matrix")
reconstruction_entry = _bind_oracle("reconstruction_entry")


def _unpack_three(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 6:
        raise ValueError("pack is too short")
    mA, nA, mB, nB, mC, nC = [int(round(float(x))) for x in pack[:6]]
    if min(mA, nA, mB, nB, mC, nC) < 1:
        raise ValueError("packed shapes must be positive")
    need = 6 + mA * nA + mB * nB + mC * nC
    if pack.size != need:
        raise ValueError("pack length does not match header")
    i = 6
    A = pack[i : i + mA * nA].reshape(mA, nA)
    i += mA * nA
    B = pack[i : i + mB * nB].reshape(mB, nB)
    i += mB * nB
    C = pack[i:].reshape(mC, nC)
    return A, B, C


def run_sketch_power_entry(
    m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col
):
    if not isinstance(q, (int, np.integer)):
        raise ValueError("q must be an integer")
    q = int(q)
    if q < 1:
        raise ValueError("require q >= 1")
    A = construct_flat_spectrum_matrix(m, n, s, data_seed)
    test_pack = draw_gaussian_test_matrices(m, n, s_width, d, l, sketch_seed)
    _, Psi, _ = _unpack_three(test_pack)
    sketch_pack = form_one_pass_sketches(A, test_pack)
    Y, W, Z = _unpack_three(sketch_pack)
    Yhat = amplify_rangefinder(Y, Z, q)
    stacked = rangefinder_thin_factors(Yhat)
    Q = stacked[: A.shape[0]]
    B = sketched_coefficient_matrix(Q, Psi, W)
    return reconstruction_entry(Q, B, rank, row, col)
SCICODE_GOLD_EOF
