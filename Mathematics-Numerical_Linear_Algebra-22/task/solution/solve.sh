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


def condenser_parameters(c: float, d: float) -> np.ndarray:
    """Reference implementation."""
    c = float(c)
    d = float(d)
    if not (np.isfinite(c) and np.isfinite(d)):
        raise ValueError("c and d must be finite")
    if not (c < d < 0.0):
        raise ValueError("endpoints must satisfy c < d < 0")

    varsigma = c + np.sqrt(c * (c - d))
    varrho = 2.0 * c - varsigma

    eta1 = (varsigma - d) / (d - varrho)
    eta2 = (varsigma - c) / (c - varrho)
    mu = 1.0 - (eta1 / eta2) ** 2

    return np.array([varsigma, varrho, eta1, eta2, mu], dtype=float)

import numpy as np
from scipy.special import ellipk, ellipj


def zolotarev_poles_nodes(params: np.ndarray, n: int) -> np.ndarray:
    """Reference implementation."""
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or params.size != 5:
        raise ValueError("params must be a one-dimensional array of length 5")
    if not np.all(np.isfinite(params)):
        raise ValueError("params must be finite")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)

    varsigma, varrho, eta1, eta2, mu = (float(params[0]), float(params[1]),
                                        float(params[2]), float(params[3]),
                                        float(params[4]))
    if not (0.0 <= mu < 1.0):
        raise ValueError("modulus must lie in [0, 1)")
    if eta2 == 0.0:
        raise ValueError("second symmetric-condenser endpoint must be nonzero")

    bigj = ellipk(mu)
    poles = np.empty(n, dtype=float)
    nodes = np.empty(n, dtype=float)
    for i in range(1, n + 1):
        v = (2 * n - 2 * i + 1) * bigj / (2 * n)
        w = eta2 * ellipj(v, mu)[2]
        poles[i - 1] = (varsigma + varrho * w) / (1.0 + w)
        nodes[i - 1] = (varsigma - varrho * w) / (1.0 - w)

    return np.vstack([poles, nodes])

import numpy as np
from decimal import Decimal, localcontext


def interpolation_residues(poles: np.ndarray, nodes: np.ndarray,
                                   t: float) -> np.ndarray:
    """Reference implementation."""
    poles = np.asarray(poles, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    if poles.ndim != 1 or nodes.ndim != 1:
        raise ValueError("poles and nodes must be one-dimensional")
    if poles.size != nodes.size or poles.size < 1:
        raise ValueError("poles and nodes must have equal positive length")
    if not (np.all(np.isfinite(poles)) and np.all(np.isfinite(nodes))):
        raise ValueError("poles and nodes must be finite")
    if not np.all(poles < 0.0):
        raise ValueError("all poles must be strictly negative")
    if not np.all(nodes >= 0.0):
        raise ValueError("all nodes must be nonnegative")
    t = float(t)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and strictly positive")

    n = poles.size
    with localcontext() as ctx:
        ctx.prec = 60
        dt = Decimal(repr(t))
        dnodes = [Decimal(repr(float(x))) for x in nodes]
        dpoles = [Decimal(repr(float(x))) for x in poles]

        # Augmented system built and eliminated entirely in extended precision.
        aug = [[Decimal(1) / (dnodes[i] - dpoles[j]) for j in range(n)]
               + [(-dt * dnodes[i]).exp()] for i in range(n)]

        for k in range(n):
            piv = max(range(k, n), key=lambda r: abs(aug[r][k]))
            if aug[piv][k] == 0:
                raise ValueError("interpolation system is singular")
            aug[k], aug[piv] = aug[piv], aug[k]
            for r in range(k + 1, n):
                fac = aug[r][k] / aug[k][k]
                for cc in range(k, n + 1):
                    aug[r][cc] -= fac * aug[k][cc]

        sol = [Decimal(0)] * n
        for i in range(n - 1, -1, -1):
            acc = aug[i][n] - sum(aug[i][j] * sol[j] for j in range(i + 1, n))
            sol[i] = acc / aug[i][i]

        return np.asarray([float(v) for v in sol], dtype=float)

import numpy as np
from scipy.special import ellipk, ellipj


def _h_config(c, d, n):
    """Extremal poles and interpolation nodes of degree n for [c,d] u [0,inf)."""
    varsigma = c + np.sqrt(c * (c - d))
    varrho = 2.0 * c - varsigma
    eta1 = (varsigma - d) / (d - varrho)
    eta2 = (varsigma - c) / (c - varrho)
    mu = 1.0 - (eta1 / eta2) ** 2
    if not (mu < 1.0):
        raise ValueError("reduced modulus is not strictly below one in working precision")
    bigj = ellipk(mu)
    poles = np.empty(n, dtype=float)
    nodes = np.empty(n, dtype=float)
    for i in range(1, n + 1):
        w = eta2 * ellipj((2 * n - 2 * i + 1) * bigj / (2 * n), mu)[2]
        poles[i - 1] = (varsigma + varrho * w) / (1.0 + w)
        nodes[i - 1] = (varsigma - varrho * w) / (1.0 - w)
    return poles, nodes


def discrete_uniform_error(c: float, d: float, n: int, times: np.ndarray,
                                   z_samples: np.ndarray) -> float:
    """Reference implementation."""
    c = float(c)
    d = float(d)
    if not (np.isfinite(c) and np.isfinite(d)):
        raise ValueError("c and d must be finite")
    if not (c < d < 0.0):
        raise ValueError("endpoints must satisfy c < d < 0")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)

    times = np.asarray(times, dtype=float)
    z_samples = np.asarray(z_samples, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times must be a one-dimensional non-empty array")
    if z_samples.ndim != 1 or z_samples.size < 1:
        raise ValueError("z_samples must be a one-dimensional non-empty array")
    if not np.all(np.isfinite(times)) or not np.all(times > 0.0):
        raise ValueError("times must be finite and strictly positive")
    if not np.all(np.isfinite(z_samples)) or not np.all(z_samples >= 0.0):
        raise ValueError("z_samples must be finite and nonnegative")

    poles, nodes = _h_config(c, d, n)
    cauchy = 1.0 / (nodes[:, None] - poles[None, :])

    worst = 0.0
    for t in times:
        alpha = np.linalg.solve(cauchy, np.exp(-t * nodes))
        approx = (alpha[None, :] / (z_samples[:, None] - poles[None, :])).sum(axis=1)
        dev = float(np.abs(approx - np.exp(-t * z_samples)).max())
        if dev > worst:
            worst = dev
    return float(worst)

import numpy as np


def truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, trunc: int,
                                    tau: float) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be one-dimensional of length matching A")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must be finite")
    if isinstance(trunc, bool) or not isinstance(trunc, (int, np.integer)) or int(trunc) < 1:
        raise ValueError("trunc must be a positive integer")
    trunc = int(trunc)
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 1.0:
        raise ValueError("tau must be finite and greater than one")

    ndim = A.shape[0]
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        raise ValueError("b must have nonzero norm")

    basis = [b / beta]
    hess = np.zeros((ndim + 1, ndim), dtype=float)

    for j in range(ndim):
        w = A @ basis[j]
        for i in range(max(0, j - trunc + 1), j + 1):
            hess[i, j] = float(basis[i] @ w)
            w = w - hess[i, j] * basis[i]
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            raise ValueError("recurrence broke down with a zero remainder")
        hess[j + 1, j] = nrm
        basis.append(w / nrm)

        bmat = np.array(basis).T
        if np.linalg.cond(bmat) > tau:
            m = j + 1
            packed = np.zeros((ndim + m + 1, m + 1), dtype=float)
            packed[:ndim, :] = bmat
            packed[ndim:, :m] = hess[:m + 1, :m]
            return packed

    raise ValueError("condition threshold not exceeded before basis filled the space")

import numpy as np


def harmonic_rank_one_update(A: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    packed = np.asarray(packed, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if packed.ndim != 2 or packed.shape[1] < 2:
        raise ValueError("packed must be two-dimensional with at least two columns")
    if not np.all(np.isfinite(packed)):
        raise ValueError("packed must be finite")

    m = packed.shape[1] - 1
    ndim = packed.shape[0] - m - 1
    if ndim < 1:
        raise ValueError("packed row count is inconsistent with its column count")
    if ndim != A.shape[0]:
        raise ValueError("packed leading block width does not match the order of A")

    basis = packed[:ndim, :m]
    trailing = packed[:ndim, m]
    hess = packed[ndim:, :m]

    image = A @ basis
    gram = image.T @ basis
    rhs = image.T @ trailing
    if not np.all(np.isfinite(gram)) or np.linalg.matrix_rank(gram) < m:
        raise ValueError("the square system determining the coefficient vector is singular")
    try:
        c_m = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the square system determining the coefficient vector is singular") from exc

    h_tilde = hess[:m, :m].copy()
    h_tilde[:, m - 1] = h_tilde[:, m - 1] + c_m * hess[m, m - 1]
    return h_tilde

import numpy as np
import scipy.linalg as sla


def evaluate_propagator(A: np.ndarray, b: np.ndarray, n: int, c: float,
                                d: float, times: np.ndarray, z_samples: np.ndarray,
                                t_eval: float, trunc: int, tau: float,
                                index: int, err_bound: float) -> float:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be one-dimensional of length matching A")
    err_bound = float(err_bound)
    if not np.isfinite(err_bound) or err_bound <= 0.0:
        raise ValueError("err_bound must be finite and positive")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer")
    index = int(index)
    if not (0 <= index < A.shape[0]):
        raise ValueError("index out of range for the dimension of A")

    achieved = discrete_uniform_error(c, d, n, times, z_samples)  # noqa: F821
    if not (achieved <= err_bound):
        raise ValueError("interval does not meet the stated error bound")

    params = condenser_parameters(c, d)  # noqa: F821
    config = zolotarev_poles_nodes(params, n)  # noqa: F821
    poles = config[0]
    nodes = config[1]

    alpha = interpolation_residues(poles, nodes, t_eval)  # noqa: F821

    packed = truncated_arnoldi_basis(A, b, trunc, tau)  # noqa: F821
    m = packed.shape[1] - 1
    ndim = packed.shape[0] - m - 1
    basis = packed[:ndim, :m]

    h_tilde = harmonic_rank_one_update(A, packed)  # noqa: F821

    sqrt_h = np.real(sla.sqrtm(h_tilde))
    e1 = np.zeros(m, dtype=float)
    e1[0] = 1.0
    ident = np.eye(m, dtype=float)

    acc = np.zeros(m, dtype=float)
    for k in range(poles.size):
        acc = acc + alpha[k] * np.linalg.solve(sqrt_h - poles[k] * ident, e1)

    beta = float(np.linalg.norm(b))
    result = beta * (basis @ acc)
    return float(result[index])
SCICODE_GOLD_EOF
