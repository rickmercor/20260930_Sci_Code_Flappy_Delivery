"""
the deterministic subsampled trigonometric sketch operator.

Build S = sqrt(m/s) D F E at the given (p, q) using this task's fixed

number-theoretic surrogates for E and D and an orthogonal type-II DCT

for F. Wrong index conventions for the signs or selected rows, sorted

row order, or a non-orthogonal transform are not accepted.

Returns
-------
ndarray of shape (s, m), float64: the explicit sketch operator S.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_sketch_operator(m, s, p, q):
    """m: ambient row count; s: sketch dimension with 1 <= s <= m; p: a prime
    strictly greater than m; q: a positive integer. Returns (s, m) float64:
    the deterministic subsampled trigonometric sketch at these parameters.
    Raises ValueError for bad sizes, a non-prime or too-small p, or a q that
    does not yield s distinct selected rows."""
    return np.zeros((s, m))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.fft import dct


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    f = 3
    while f * f <= v:
        if v % f == 0:
            return False
        f += 2
    return True


def _oracle_build_sketch_operator(m, s, p, q):
    for name, v in (("m", m), ("s", s), ("p", p), ("q", q)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    m, s, p, q = int(m), int(s), int(p), int(q)
    if m < 1 or s < 1:
        raise ValueError("m and s must be positive")
    if s > m:
        raise ValueError("require s <= m")
    if p <= m or not _is_prime(p):
        raise ValueError("p must be a prime strictly greater than m")
    if q < 1:
        raise ValueError("q must be positive")

    residues = {(k * k) % p for k in range(1, p)}
    signs = np.array(
        [1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)],
        dtype=np.float64,
    )
    rows = np.array([(q * k) % m for k in range(s)], dtype=int)
    if len(np.unique(rows)) != s:
        raise ValueError("stride q does not produce s distinct rows modulo m")

    F = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm="ortho")
    S = np.sqrt(m / s) * (F[rows, :] * signs[None, :])
    return np.asarray(S, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    # Hard cases: 0-based Legendre, sorted row order, and non-ortho DCT each
    # disagree with the oracle on every added instance below.
    return [
        {"setup": 'm, s, p, q = 128, 64, 131, 79', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 32, 32, 37, 5', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 16, 4, 17, 3', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 64, 16, 67, 11', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 48, 12, 53, 7', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 96, 24, 97, 13', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 80, 20, 83, 17', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 40, 10, 41, 9', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 72, 18, 73, 11', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 60, 15, 61, 7', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        # Fingerprints / wrong-route traps (0-based, sorted, non-ortho)
        {"setup": 'S = build_sketch_operator(96, 24, 97, 13)', "call": 'float(S[0, 0])', "gold_call": '0.204124145232', "tol": 1e-12},
        {"setup": 'S = build_sketch_operator(96, 24, 97, 13)', "call": 'float(np.linalg.norm(S))', "gold_call": '9.797958971133', "tol": 1e-12},
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 96, 24, 97, 13\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns0 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(m)], dtype=np.float64)\nrows = np.array([(q * k) % m for k in range(s)], dtype=int)\nF = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm=\"ortho\")\nS0 = np.sqrt(m / s) * (F[rows, :] * signs0[None, :])', "call": 'float(np.linalg.norm(S - S0) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 96, 24, 97, 13\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns1 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)], dtype=np.float64)\nrows = np.sort(np.array([(q * k) % m for k in range(s)], dtype=int))\nF = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm=\"ortho\")\nSs = np.sqrt(m / s) * (F[rows, :] * signs1[None, :])', "call": 'float(np.linalg.norm(S - Ss) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'm, s, p, q = 104, 26, 107, 19', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        {"setup": 'm, s, p, q = 120, 30, 127, 23', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
        # Non-ortho DCT (norm=None) must disagree with the instance sketch
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 96, 24, 97, 13\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns1 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)], dtype=np.float64)\nrows = np.array([(q * k) % m for k in range(s)], dtype=int)\nF = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm=None)\nSn = np.sqrt(m / s) * (F[rows, :] * signs1[None, :])', "call": 'float(np.linalg.norm(S - Sn) > 10.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'S = build_sketch_operator(128, 64, 131, 79)', "call": 'float(S[0, 0])', "gold_call": '0.125', "tol": 1e-12},
        # Extra traps on the gold instance: 0-based signs / sorted rows / DCT-III
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 128, 64, 131, 79\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns0 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(m)], dtype=np.float64)\nrows = np.array([(q * k) % m for k in range(s)], dtype=int)\nF = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm=\"ortho\")\nS0 = np.sqrt(m / s) * (F[rows, :] * signs0[None, :])', "call": 'float(np.linalg.norm(S - S0) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 128, 64, 131, 79\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns1 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)], dtype=np.float64)\nrows = np.sort(np.array([(q * k) % m for k in range(s)], dtype=int))\nF = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm=\"ortho\")\nSs = np.sqrt(m / s) * (F[rows, :] * signs1[None, :])', "call": 'float(np.linalg.norm(S - Ss) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'from scipy.fft import dct\nimport numpy as np\nm, s, p, q = 128, 64, 131, 79\nS = build_sketch_operator(m, s, p, q)\nresidues = {(k * k) % p for k in range(1, p)}\nsigns1 = np.array([1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)], dtype=np.float64)\nrows = np.array([(q * k) % m for k in range(s)], dtype=int)\nF = dct(np.eye(m, dtype=np.float64), type=3, axis=0, norm=\"ortho\")\nS3 = np.sqrt(m / s) * (F[rows, :] * signs1[None, :])', "call": 'float(np.linalg.norm(S - S3) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'm, s, p, q = 112, 28, 113, 15', "call": 'build_sketch_operator(m, s, p, q)', "gold_call": '_oracle_build_sketch_operator(m, s, p, q)', "tol": 1e-12},
    ]
