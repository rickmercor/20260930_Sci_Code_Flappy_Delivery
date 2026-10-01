#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def sp8_family_coefficients(index: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from math import comb

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer")
    index = int(index)
    if index < 0 or index > 7:
        raise ValueError("index must satisfy 0 <= index <= 7")

    # Tabulated members for L = 4, 5, 6, 7, stored as b0..b8.
    table = {
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        6: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, -7.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0, -48.0, 21.0],
        4: [0.0, 0.0, 0.0, 0.0, 0.0, 56.0, -140.0, 120.0, -35.0],
    }
    if index >= 4:
        return np.array(table[index], dtype=float)

    # Reflection: p_L(x) = 1 - p_(7-L)(1 - x). Expanding (1 - x)**k with the
    # binomial theorem turns the mirrored polynomial back into monomials.
    mirror = np.array(table[7 - index], dtype=float)
    coeffs = np.zeros(9, dtype=float)
    for k in range(9):
        if mirror[k] == 0.0:
            continue
        for j in range(k + 1):
            coeffs[j] -= mirror[k] * comb(k, j) * (-1.0) ** j
    coeffs[0] += 1.0
    return coeffs

def select_family_member(lam_lumo: float, lam_homo: float,
                                 iteration: int) -> np.ndarray:
    import numpy as np
    from math import comb

    for name, val in (("lam_lumo", lam_lumo), ("lam_homo", lam_homo)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    lam_lumo = float(lam_lumo)
    lam_homo = float(lam_homo)
    if not (0.0 <= lam_lumo < lam_homo <= 1.0):
        raise ValueError("require 0 <= lam_lumo < lam_homo <= 1")
    if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
        raise ValueError("iteration must be an integer")
    iteration = int(iteration)
    if iteration < 1:
        raise ValueError("iteration must be >= 1")

    def family(L):
        # Same construction as sub-problem 01, repeated locally so that this
        # oracle can be executed on its own.
        c = 8.0 * comb(7, L)
        b = np.zeros(9, dtype=float)
        for j in range(7 - L + 1):
            b[L + j + 1] += c * comb(7 - L, j) * (-1.0) ** j / (L + j + 1)
        return b

    def value(b, x):
        acc = 0.0
        for i in range(8, -1, -1):
            acc = acc * x + b[i]
        return float(acc)

    kappa = 0.01
    if lam_lumo < kappa and 1.0 - kappa < lam_homo:
        # Acceleration is exhausted at both ends: alternate the two central
        # members, which together give asymptotic order 20 over two iterations.
        index = 3 if iteration % 2 == 1 else 4
    else:
        gaps = [value(family(L), lam_homo) - value(family(L), lam_lumo)
                for L in range(8)]
        index = int(np.argmax(np.asarray(gaps, dtype=float)))

    b = family(index)
    return np.array([float(index), value(b, lam_lumo), value(b, lam_homo)],
                    dtype=float)

def degree_eight_evaluation_coefficients(coeffs: np.ndarray) -> np.ndarray:
    import numpy as np

    b = np.asarray(coeffs, dtype=float)
    if b.ndim != 1 or b.shape[0] != 9:
        raise ValueError("coeffs must be a one-dimensional array of length 9")
    if not np.all(np.isfinite(b)):
        raise ValueError("coeffs must be finite")
    if b[8] == 0.0:
        raise ValueError("the polynomial must have degree exactly eight (b8 != 0)")

    # The leading three matched monomials fix f4, c1 and the two sums
    # t2 = d2 + e2 and t1 = d1 + e1 outright.
    f4 = b[8]
    c1 = b[7] / (2.0 * f4)
    t2 = b[6] / f4 - c1 ** 2
    t1 = b[5] / f4 - c1 * t2

    # Tenth condition: the discriminant of the quadratic for e2 is set to one,
    # which both removes the square root and pins d0.
    d0 = 0.25 * (1.0 - t2 ** 2 + 4.0 * b[4] / f4 - 4.0 * c1 * t1)
    e2 = 0.5 * (t2 + 1.0)
    d2 = 0.5 * (t2 - 1.0)

    # With e2 - d2 = 1 the remaining split of t1 is explicit.
    e1 = c1 * d0 + t1 * e2 - b[3] / f4
    d1 = t1 - e1

    f2 = b[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = b[1] - f4 * d0 * e1
    f0 = b[0]

    return np.array([c1, d0, d1, d2, e1, e2, f0, f1, f2, f4], dtype=float)

def workspace_rearrangement_scalars(evaluation: np.ndarray) -> np.ndarray:
    import numpy as np

    k = np.asarray(evaluation, dtype=float)
    if k.ndim != 1 or k.shape[0] != 10:
        raise ValueError("evaluation must be a one-dimensional array of length 10")
    if not np.all(np.isfinite(k)):
        raise ValueError("evaluation must be finite")

    c1, d1, d2 = k[0], k[2], k[3]
    e1, f1, f2 = k[4], k[7], k[8]

    quarter = 0.25 * c1 ** 2
    r2 = d2 - quarter
    r1 = d1 - 0.5 * c1 * r2
    r3 = e1 - d1 - 0.5 * c1
    r4 = f1 - f2 * (e1 - d1)
    return np.array([r1, r2, r3, r4], dtype=float)

def apply_degree_eight_polynomial(matrix: np.ndarray, evaluation: np.ndarray,
                                          scalars: np.ndarray,
                                          drop_tolerance: float = 0.0) -> np.ndarray:
    import numpy as np

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square two-dimensional array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")

    k = np.asarray(evaluation, dtype=float)
    if k.ndim != 1 or k.shape[0] != 10 or not np.all(np.isfinite(k)):
        raise ValueError("evaluation must be a finite array of length 10")
    r = np.asarray(scalars, dtype=float)
    if r.ndim != 1 or r.shape[0] != 4 or not np.all(np.isfinite(r)):
        raise ValueError("scalars must be a finite array of length 4")
    if isinstance(drop_tolerance, bool) or not isinstance(
        drop_tolerance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("drop_tolerance must be a real number")
    tolerance = float(drop_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("drop_tolerance must be finite and nonnegative")
    if tolerance > 0.0:
        if not np.allclose(a, a.T, rtol=0.0, atol=1e-10):
            raise ValueError("matrix must be symmetric when filtering is requested")
        a = 0.5 * (a + a.T)

    c1, d0 = k[0], k[1]
    f0, f2, f4 = k[6], k[8], k[9]
    r1, r2, r3, r4 = r[0], r[1], r[2], r[3]

    eye = np.eye(a.shape[0], dtype=float)
    m1 = a.copy()

    def multiply(left, right):
        product = left @ right
        if tolerance > 0.0:
            product = 0.5 * (product + product.T)
            product[np.abs(product) < tolerance] = 0.0
        return product

    m2 = multiply(m1, m1)              # first non-scalar multiplication
    m2 = m2 + 0.5 * c1 * m1
    m3 = multiply(m2, m2)              # second non-scalar multiplication

    # Rebuild the two factors of the final product inside the same three slots.
    m3 = m3 + r1 * m1
    m3 = m3 + r2 * m2
    m2 = m2 + r3 * m1
    m1 = r4 * m1 + f2 * m2 + f0 * eye
    m2 = m2 + m3
    m3 = m3 + d0 * eye

    m1 = f4 * multiply(m2, m3) + m1    # third non-scalar multiplication
    return m1

def equidistant_step_spectrum(size: int, mu: float, gap: float) -> np.ndarray:
    import numpy as np

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError("size must be an integer")
    size = int(size)
    if size < 2:
        raise ValueError("size must be at least 2")
    for name, val in (("mu", mu), ("gap", gap)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    mu = float(mu)
    gap = float(gap)
    if not 0.0 < mu < 1.0:
        raise ValueError("mu must satisfy 0 < mu < 1")
    if gap <= 0.0:
        raise ValueError("gap must be positive")

    lam_lumo = mu - 0.5 * gap
    lam_homo = mu + 0.5 * gap
    if not 0.0 < lam_lumo < lam_homo < 1.0:
        raise ValueError("the gap must lie strictly inside the unit interval")

    n_occ = int(round(size * (1.0 - mu)))
    if n_occ < 2 or size - n_occ < 2:
        raise ValueError("both blocks must hold at least two eigenvalues")

    lower = np.linspace(0.0, lam_lumo, size - n_occ)
    upper = np.linspace(lam_homo, 1.0, n_occ)
    return np.concatenate([lower, upper])

def symmetric_matrix_from_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    import numpy as np

    ev = np.asarray(eigenvalues, dtype=float)
    if ev.ndim != 1 or ev.shape[0] < 1:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(ev)):
        raise ValueError("eigenvalues must be finite")

    n = ev.shape[0]
    idx = np.arange(1, n + 1, dtype=float)
    q = np.sqrt(2.0 / (n + 1.0)) * np.sin(np.pi * np.outer(idx, idx) / (n + 1.0))
    matrix = q @ (ev[:, None] * q)
    return 0.5 * (matrix + matrix.T)

def idempotency_trace(matrix: np.ndarray) -> float:
    import numpy as np

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square two-dimensional array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")

    # For a symmetric argument the trace of the square is the squared Frobenius
    # norm, so no matrix-matrix multiplication is needed here.
    return float(np.trace(a) - np.sum(a * a))

def critical_initial_gap(size: int = 120, mu: float = 0.35,
                                 multiplications: int = 30,
                                 target_trace: float = 0.1,
                                 gap_lower: float = 2.5e-4,
                                 gap_upper: float = 3.5e-4,
                                 bisection_steps: int = 40,
                                 drop_tolerance: float = 1.0e-6) -> float:
    import numpy as np

    for name, value, minimum in (
        ("size", size, 2),
        ("multiplications", multiplications, 3),
        ("bisection_steps", bisection_steps, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    size = int(size)
    multiplications = int(multiplications)
    bisection_steps = int(bisection_steps)

    for name, value in (
        ("mu", mu),
        ("target_trace", target_trace),
        ("gap_lower", gap_lower),
        ("gap_upper", gap_upper),
        ("drop_tolerance", drop_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    mu = float(mu)
    target_trace = float(target_trace)
    left = float(gap_lower)
    right = float(gap_upper)
    drop_tolerance = float(drop_tolerance)
    if not (0.0 < left < right):
        raise ValueError("require 0 < gap_lower < gap_upper")
    if drop_tolerance < 0.0:
        raise ValueError("drop_tolerance must be nonnegative")

    def residual(gap):
        eigenvalues = equidistant_step_spectrum(size, mu, gap)
        iterate = symmetric_matrix_from_spectrum(eigenvalues)
        lam_lumo = mu - 0.5 * gap
        lam_homo = mu + 0.5 * gap

        for iteration in range(1, multiplications // 3 + 1):
            state = np.asarray(
                select_family_member(lam_lumo, lam_homo, iteration),
                dtype=float,
            )
            index = int(round(float(state[0])))
            lam_lumo = float(state[1])
            lam_homo = float(state[2])

            coeffs = sp8_family_coefficients(index)
            evaluation = degree_eight_evaluation_coefficients(coeffs)
            scalars = workspace_rearrangement_scalars(evaluation)
            iterate = apply_degree_eight_polynomial(
                iterate, evaluation, scalars, drop_tolerance
            )

        return float(idempotency_trace(iterate) - target_trace)

    f_left = residual(left)
    f_right = residual(right)
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if np.signbit(f_left) == np.signbit(f_right):
        raise ValueError("gap endpoints must bracket the target trace")

    for _ in range(bisection_steps):
        middle = 0.5 * (left + right)
        f_middle = residual(middle)
        if f_middle == 0.0:
            return float(middle)
        if np.signbit(f_middle) == np.signbit(f_left):
            left = middle
            f_left = f_middle
        else:
            right = middle
            f_right = f_middle

    return float(0.5 * (left + right))
SCICODE_GOLD_EOF
