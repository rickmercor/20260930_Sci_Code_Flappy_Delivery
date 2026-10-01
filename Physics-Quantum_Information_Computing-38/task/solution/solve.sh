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
def build_bloch_generator(hamiltonian: 'np.ndarray', jump_operators: 'np.ndarray') -> 'np.ndarray':
    """Reference implementation (Pauli projection of the Lindblad superoperator)."""
    import numpy as np

    def _as_complex(value):
        try:
            return np.asarray(value, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("operators must be numeric arrays") from None

    ham = _as_complex(hamiltonian)
    if ham.shape != (2, 2) or not np.all(np.isfinite(ham)):
        raise ValueError("hamiltonian must be a finite (2, 2) matrix")
    if np.any(np.abs(ham) > 10.0):
        raise ValueError("hamiltonian entries must have modulus <= 10")
    if np.max(np.abs(ham - ham.conj().T)) > 1e-12:
        raise ValueError("hamiltonian must be Hermitian")
    jumps = _as_complex(jump_operators)
    if jumps.size == 0:
        jumps = np.zeros((0, 2, 2), dtype=complex)
    if jumps.ndim != 3 or jumps.shape[1:] != (2, 2):
        raise ValueError("jump_operators must have shape (m, 2, 2)")
    if not np.all(np.isfinite(jumps)):
        raise ValueError("jump_operators must be finite")
    if jumps.shape[0] > 16 or np.any(np.abs(jumps) > 10.0):
        raise ValueError("at most 16 jump operators with entry modulus <= 10 are supported")

    basis = [
        np.eye(2, dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    ]

    def _lindblad(rho):
        out = -1.0j * (ham @ rho - rho @ ham)
        for jump in jumps:
            dag = jump.conj().T
            out = out + jump @ rho @ dag - 0.5 * (dag @ jump @ rho + rho @ dag @ jump)
        return out

    # rho = (u_0 I + x sigma_x + y sigma_y + z sigma_z) / 2 with u_0 = 1, and
    # d u_a / d t = Tr(sigma_a L(rho)), so column b holds Tr(sigma_a L(P_b)) / 2.
    generator = np.zeros((4, 4))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for col, element in enumerate(basis):
                image = _lindblad(element)
                if not np.all(np.isfinite(image)):
                    raise ValueError("nonfinite Lindblad image")
                for row in range(1, 4):
                    generator[row, col] = 0.5 * np.trace(basis[row] @ image).real
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("nonfinite generator evaluation") from exc
    if not np.all(np.isfinite(generator)):
        raise ValueError("nonfinite Bloch generator")
    return generator

import numpy as np
from scipy.linalg import expm
def propagate_averaged_drift(
    alpha: float,
    beta: float,
    bloch0: 'np.ndarray',
    delta: float,
    n_intervals: int,
    drive_generator: 'np.ndarray',
    dephasing_generator: 'np.ndarray',
) -> 'np.ndarray':
    """Reference implementation (block matrix exponential for the Frechet derivatives)."""
    import numpy as np
    from scipy.linalg import expm

    def _is_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _as_real(value, shape, name):
        try:
            mat = np.asarray(value, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite real array") from None
        if mat.shape != shape or not np.all(np.isfinite(mat)) or np.any(mat.imag != 0.0):
            raise ValueError(f"{name} must be a finite real array of shape {shape}")
        return mat.real.copy()

    if not (_is_number(alpha) and _is_number(beta) and _is_number(delta)):
        raise ValueError("alpha, beta and delta must be finite real numbers")
    if not (-5.0 <= alpha <= 5.0 and 0.0 <= beta <= 3.0 and 1e-4 <= delta <= 1.0):
        raise ValueError("alpha, beta or delta is outside the supported numerical range")
    if not (_is_integer(n_intervals) and 1 <= n_intervals <= 128):
        raise ValueError("n_intervals must be an integer in [1, 128]")
    if (int(n_intervals) - 1) * float(delta) > 4.0:
        raise ValueError("the last returned sample time must be <= 4")
    start = _as_real(bloch0, (3,), "bloch0")
    if np.linalg.norm(start) > 1.0 + 1e-12:
        raise ValueError("bloch0 must lie in the Bloch ball")
    drive = _as_real(drive_generator, (4, 4), "drive_generator")
    damp = _as_real(dephasing_generator, (4, 4), "dephasing_generator")
    if any(np.any(np.abs(mat) > 2.0) or np.any(mat[0] != 0.0) for mat in (drive, damp)):
        raise ValueError("generators must have zero first rows and entries within [-2, 2]")

    a, b = float(alpha), float(beta)
    gen = a * drive + b * b * damp
    if not np.all(np.isfinite(gen)):
        raise ValueError("nonfinite averaged generator")
    symmetric = 0.5 * (gen[1:, 1:] + gen[1:, 1:].T)
    if np.linalg.eigvalsh(symmetric)[-1] > 1e-12:
        raise ValueError("the homogeneous Bloch generator must be contractive")
    # exp(t M) of the block upper-triangular M carries exp(t G) on the diagonal
    # and the Frechet derivatives of exp(t G) along dG/dalpha, dG/dbeta above it;
    # the same expression is regular when G is defective (critical damping).
    block = np.zeros((12, 12))
    block[0:4, 0:4] = gen
    block[0:4, 4:8] = drive
    block[0:4, 8:12] = 2.0 * b * damp
    block[4:8, 4:8] = gen
    block[8:12, 8:12] = gen
    if not np.all(np.isfinite(block)):
        raise ValueError("nonfinite sensitivity generator")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step = expm(float(delta) * block)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("failed finite propagation") from exc
    if not np.all(np.isfinite(step)):
        raise ValueError("nonfinite propagator")
    state = np.zeros((12, 2))
    u0 = np.concatenate([[1.0], start])
    state[4:8, 0] = u0
    state[8:12, 1] = u0
    drift = np.empty((int(n_intervals), 3))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for j in range(int(n_intervals)):
                z_value = state[7, 0]
                dz_alpha = state[3, 0]
                dz_beta = state[3, 1]
                # h = 2 beta z; the beta-derivative also acts on L itself.
                drift[j] = (2.0 * b * z_value, 2.0 * b * dz_alpha, 2.0 * z_value + 2.0 * b * dz_beta)
                if j + 1 < int(n_intervals):
                    state = step @ state
                    if not np.all(np.isfinite(state)):
                        raise ValueError("nonfinite propagated state or sensitivity")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("nonfinite drift evaluation") from exc
    if not np.all(np.isfinite(drift)):
        raise ValueError("nonfinite averaged drift or derivative")
    return drift

import numpy as np
def evaluate_contrast_model(
    alpha: float,
    beta: float,
    mean_increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Reference implementation (left-endpoint discretisation of the contrast)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(alpha) and _is_number(beta) and _is_number(delta)):
        raise ValueError("alpha, beta and delta must be finite real numbers")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not _is_function(drift_fn):
        raise ValueError("drift_fn must be callable")
    try:
        increments = np.asarray(mean_increments, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("mean_increments must be a numeric array") from None
    if increments.ndim != 1 or increments.size == 0 or not np.all(np.isfinite(increments)):
        raise ValueError("mean_increments must be a non-empty finite 1-D array")
    rows = np.asarray(drift_fn(float(alpha), float(beta)), dtype=float)
    if rows.shape != (increments.size, 3) or not np.all(np.isfinite(rows)):
        raise ValueError("drift_fn must return a finite (n, 3) array")

    step = float(delta)
    drift = rows[:, 0]
    grads = rows[:, 1:]
    # The drift is frozen at the left end of each interval in both sums.
    value = float(drift @ increments - 0.5 * step * drift @ drift)
    residual = increments - drift * step
    gradient = grads.T @ residual
    information = step * grads.T @ grads
    return np.array([value, gradient[0], gradient[1],
                     information[0, 0], information[0, 1], information[1, 1]])

import numpy as np
def locate_contrast_candidates(
    contrast_fn: "Callable[[float, float], float]",
    alpha_bounds: tuple,
    beta_bounds: tuple,
    spacing: float = 0.05,
    max_candidates: int = 8,
    min_separation: float = 0.15,
) -> 'np.ndarray':
    """Reference implementation (eight-neighbour peak test, greedy separation)."""
    import numpy as np

    def _is_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _is_function(value):
        return callable(value)

    def _axis(bounds, name):
        try:
            low, high = bounds
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold two numbers") from None
        if not (_is_number(low) and _is_number(high) and low < high):
            raise ValueError(f"{name} must be finite with low < high")
        if low < -100.0 or high > 100.0:
            raise ValueError(f"{name} must lie within [-100, 100]")
        width = float(high) - float(low)
        count = round(width / spacing)
        if count < 1 or count > 2000 or abs(count * spacing - width) > 1e-9 * max(1.0, width):
            raise ValueError(f"spacing must divide the {name} range")
        axis = float(low) + spacing * np.arange(count + 1)
        axis[-1] = float(high)
        if not np.all(np.isfinite(axis)):
            raise ValueError("nonfinite grid axis")
        return axis

    if not _is_function(contrast_fn):
        raise ValueError("contrast_fn must be callable")
    if not (_is_number(spacing) and 1e-4 <= spacing <= 200.0):
        raise ValueError("spacing must lie in [1e-4, 200]")
    if not (_is_integer(max_candidates) and max_candidates >= 1):
        raise ValueError("max_candidates must be an integer >= 1")
    if not (_is_number(min_separation) and min_separation >= 0.0):
        raise ValueError("min_separation must be finite and >= 0")
    spacing = float(spacing)
    alphas = _axis(alpha_bounds, "alpha_bounds")
    betas = _axis(beta_bounds, "beta_bounds")
    if alphas.size * betas.size > 250000:
        raise ValueError("the grid must contain at most 250000 points")

    values = np.empty((alphas.size, betas.size))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            try:
                v = np.asarray(contrast_fn(float(a), float(b)), dtype=complex)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("contrast_fn must return a finite real scalar") from exc
            if v.shape != () or not np.isfinite(v):
                raise ValueError("contrast_fn must return a finite scalar")
            if v.imag != 0.0:
                raise ValueError("contrast_fn must return a real scalar")
            values[i, j] = float(v.real)

    padded = np.full((alphas.size + 2, betas.size + 2), -np.inf)
    padded[1:-1, 1:-1] = values
    peak = np.ones(values.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = padded[1 + di:1 + di + alphas.size, 1 + dj:1 + dj + betas.size]
            peak &= values >= shifted
    index = np.flatnonzero(peak.ravel())
    # Stable sort on the negated value keeps row-major order among ties.
    order = index[np.argsort(-values.ravel()[index], kind="stable")]
    chosen = []
    for flat in order:
        i, j = divmod(int(flat), betas.size)
        point = np.array([alphas[i], betas[j]])
        if all(np.hypot(*(point - q)) >= min_separation - 1e-9 for q in chosen):
            chosen.append(point)
            if len(chosen) == max_candidates:
                break
    result = np.array(chosen, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite selected candidates")
    return result

import numpy as np
def maximize_contrast(
    model_fn: "Callable[[float, float], np.ndarray]",
    starts: 'np.ndarray',
    alpha_bounds: tuple,
    beta_bounds: tuple,
    tolerance: float = 1e-10,
) -> 'np.ndarray':
    """Reference implementation (bounded Gauss-Newton ascent, then Newton polish)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _pair(bounds, name):
        try:
            low, high = bounds
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold two numbers") from None
        if not (_is_number(low) and _is_number(high) and low < high):
            raise ValueError(f"{name} must be finite with low < high")
        return float(low), float(high)

    if not _is_function(model_fn):
        raise ValueError("model_fn must be callable")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    (a_lo, a_hi), (b_lo, b_hi) = _pair(alpha_bounds, "alpha_bounds"), _pair(beta_bounds, "beta_bounds")
    lo, hi = np.array([a_lo, b_lo]), np.array([a_hi, b_hi])
    points = np.asarray(starts, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("starts must be a finite (m, 2) array with m >= 1")

    def _eval(p):
        out = np.asarray(model_fn(float(p[0]), float(p[1])), dtype=float)
        if out.shape != (6,) or not np.all(np.isfinite(out)):
            raise ValueError("model_fn must return a finite length-6 array")
        return out[0], out[1:3].copy(), np.array([[out[3], out[4]], [out[4], out[5]]])

    def _free(p, grad):
        return np.flatnonzero(~(((p <= lo) & (grad < 0.0)) | ((p >= hi) & (grad > 0.0))))

    def _gauss_newton(p, grad, info):
        idx, move = _free(p, grad), np.zeros(2)
        if idx.size:
            block = info[np.ix_(idx, idx)]
            if not np.all(np.linalg.eigvalsh(block) > 0.0):
                raise ValueError("information matrix must be positive definite")
            move[idx] = np.linalg.solve(block, grad[idx])
        return move

    def _hessian(p, grad, idx):
        cols = []
        for k in idx:
            h = 1e-6 * max(1.0, abs(p[k]))
            e = np.zeros(2)
            e[k] = h
            if p[k] - h >= lo[k] and p[k] + h <= hi[k]:
                cols.append((_eval(p + e)[1] - _eval(p - e)[1]) / (2.0 * h))
            elif p[k] + h <= hi[k]:
                cols.append((_eval(p + e)[1] - grad) / h)
            else:
                cols.append((grad - _eval(p - e)[1]) / h)
        mat = np.array(cols).T[np.ix_(idx, range(len(idx)))]
        return 0.5 * (mat + mat.T)

    def _ascend(start):
        p = np.clip(start, lo, hi)
        for _ in range(5000):
            value, grad, info = _eval(p)
            move = _gauss_newton(p, grad, info)
            if np.max(np.abs(np.clip(p + move, lo, hi) - p)) <= 1e-7:
                break
            scale = 1.0
            while scale >= 1e-12:
                trial = np.clip(p + scale * move, lo, hi)
                if _eval(trial)[0] >= value:
                    break
                scale *= 0.5
            else:
                break
            p = trial
        # Gauss-Newton converges only linearly here; finish with Newton steps.
        for _ in range(100):
            value, grad, info = _eval(p)
            idx = _free(p, grad)
            if idx.size == 0:
                break
            hess = _hessian(p, grad, idx)
            move = np.zeros(2)
            if np.all(np.linalg.eigvalsh(hess) < 0.0):
                move[idx] = -np.linalg.solve(hess, grad[idx])
            else:
                move = _gauss_newton(p, grad, info)
            new = np.clip(p + move, lo, hi)
            shift, p = np.max(np.abs(new - p)), new
            if shift <= 0.01 * tolerance:
                break
        return p, _eval(p)[0]

    best_point, best_value = None, -np.inf
    for start in points:
        point, value = _ascend(start)
        if value > best_value:
            best_point, best_value = point, value
    return np.asarray(best_point, dtype=float)

import numpy as np
def estimate_sandwich_covariance(
    alpha: float,
    beta: float,
    increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Reference implementation (centred per-record scores, plug-in information)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(alpha) and _is_number(beta) and _is_number(delta)):
        raise ValueError("alpha, beta and delta must be finite real numbers")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not _is_function(drift_fn):
        raise ValueError("drift_fn must be callable")
    try:
        data = np.asarray(increments, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("increments must be a numeric array") from None
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("increments must be a finite (N, n) array with N >= 2")
    rows = np.asarray(drift_fn(float(alpha), float(beta)), dtype=float)
    if rows.shape != (data.shape[1], 3) or not np.all(np.isfinite(rows)):
        raise ValueError("drift_fn must return a finite (n, 3) array")

    step = float(delta)
    drift, grads = rows[:, 0], rows[:, 1:]
    information = step * grads.T @ grads
    eigen = np.linalg.eigvalsh(information)
    if not eigen[-1] > 0.0 or eigen[0] <= 1e-12 * eigen[-1]:
        raise ValueError("the plug-in information is not positive definite")
    scores = (data - drift * step) @ grads                 # (N, 2) score contributions
    centred = scores - scores.mean(axis=0)
    meat = centred.T @ centred / data.shape[0]
    bread = np.linalg.inv(information)
    covariance = bread @ meat @ bread
    return 0.5 * (covariance + covariance.T)

import numpy as np
def compute_studentized_statistics(
    theta_hat: 'np.ndarray',
    theta_ref: 'np.ndarray',
    covariance: 'np.ndarray',
    n_records: int,
) -> 'np.ndarray':
    """Reference implementation (eigen-decomposition for the symmetric root)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _vector(value, name):
        try:
            vec = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric") from None
        if vec.shape != (2,) or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must be a finite length-2 vector")
        return vec

    estimate = _vector(theta_hat, "theta_hat")
    reference = _vector(theta_ref, "theta_ref")
    if not (_is_integer(n_records) and n_records >= 1):
        raise ValueError("n_records must be an integer >= 1")
    try:
        cov = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("covariance must be numeric") from None
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be a finite (2, 2) matrix")
    scale = np.max(np.abs(cov))
    if np.max(np.abs(cov - cov.T)) > 1e-12 * scale:
        raise ValueError("covariance must be symmetric")
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    if not values[0] > 0.0:
        raise ValueError("covariance must be positive definite")

    scaled = np.sqrt(float(n_records)) * (estimate - reference)
    marginal = scaled / np.sqrt(np.diag(cov))
    inv_root = vectors @ np.diag(values ** -0.5) @ vectors.T
    joint = inv_root @ scaled
    distance = float(joint @ joint)
    return np.array([marginal[0], marginal[1], joint[0], joint[1], distance])

import numpy as np
def assess_nominal_calibration(
    records: 'np.ndarray',
    delta: float = 0.4,
    bloch0: tuple = (0.0, -0.8, 0.6),
    theta_nominal: tuple = (-1.5, 0.55),
    alpha_bounds: tuple = (-5.0, 5.0),
    beta_bounds: tuple = (0.05, 3.0),
    grid_spacing: float = 0.05,
    tolerance: float = 1e-10,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    try:
        data = np.asarray(records, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("records must be a numeric array") from None
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("records must be a finite (N, n) array with N >= 2")
    def _bounded_pair(bounds, lower, upper):
        try:
            pair = np.asarray(bounds, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("search bounds must be finite real pairs") from None
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.any(pair.imag != 0.0):
            raise ValueError("search bounds must be finite real pairs")
        low, high = pair.real
        if not lower <= low < high <= upper:
            raise ValueError("search box must lie within [-5, 5] x [0, 3]")
        return (float(low), float(high))

    alpha_bounds = _bounded_pair(alpha_bounds, -5.0, 5.0)
    beta_bounds = _bounded_pair(beta_bounds, 0.0, 3.0)
    n_records, n_intervals = data.shape
    increments = np.diff(np.concatenate([np.zeros((n_records, 1)), data], axis=1), axis=1)
    mean_increments = increments.mean(axis=0)
    start = bloch0

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    drive = build_bloch_generator(0.5 * sigma_x, np.zeros((0, 2, 2), dtype=complex))
    dephasing = build_bloch_generator(np.zeros((2, 2), dtype=complex), np.array([sigma_z]))

    def _drift_fn(alpha, beta):
        return propagate_averaged_drift(
            alpha, beta, start, delta, n_intervals, drive, dephasing
        )

    def _model_fn(alpha, beta):
        return evaluate_contrast_model(alpha, beta, mean_increments, delta, _drift_fn)

    def _contrast_fn(alpha, beta):
        return float(_model_fn(alpha, beta)[0])

    starts = locate_contrast_candidates(
        _contrast_fn, alpha_bounds, beta_bounds, grid_spacing, 8, 0.15
    )
    theta_hat = maximize_contrast(_model_fn, starts, alpha_bounds, beta_bounds, tolerance)
    covariance = estimate_sandwich_covariance(
        theta_hat[0], theta_hat[1], increments, delta, _drift_fn
    )
    statistics = compute_studentized_statistics(
        theta_hat, np.asarray(theta_nominal, dtype=float), covariance, n_records
    )
    return float(statistics[4])
SCICODE_GOLD_EOF
