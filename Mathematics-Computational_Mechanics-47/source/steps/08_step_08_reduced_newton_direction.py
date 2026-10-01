"""
Solve one reduced Newton system for the implicit level of a projected polynomial model.

The step direction is obtained entirely from the offline Gram matrix and the

reduced state, so no full-order vector is formed and the cost of an iteration

depends only on the reduced dimension and the number of inputs.

Returns
-------
tuple[np.ndarray, float] holding the step direction of shape (n,) and the 2-norm of the test-basis-projected residual before the step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def reduced_newton_direction(
    gram: np.ndarray,
    coefficients: np.ndarray,
    reduced_state: np.ndarray,
    current_input: np.ndarray,
    time_step: float,
    alpha_zero: float,
    beta_zero: float,
    scheme: str,
) -> tuple[np.ndarray, float]:
    """Return one reduced Newton direction and the current convergence measure.

    Let K be the six-block matrix

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    whose Gram matrix K^T K is supplied as gram, and let r = K coefficients be
    the approximate residual at the implicit level. Let J be the derivative of
    r with respect to the reduced state at reduced_state, with all preceding
    levels held fixed, so that J = K Z for a coefficient matrix Z determined by
    alpha_zero, beta_zero, time_step, reduced_state and current_input.

    The test basis is Phi, the first block of K, when scheme is "galerkin", and
    is J when scheme is "lspg". The step direction p solves

        (Psi^T J) p = -(Psi^T r),

    and the returned convergence measure is the 2-norm of Psi^T r. The routine
    receives no full-order quantity other than gram and must not form one.

    Parameters
    ----------
    gram : np.ndarray
        Finite real symmetric array of shape (d, d) with
        d = 2 * n + n**2 + n_u * n + 1 + n_u.
    coefficients : np.ndarray
        Finite real array of shape (d,) representing the residual in the
        column basis.
    reduced_state : np.ndarray
        Finite real array of shape (n,) holding the current implicit level.
    current_input : np.ndarray
        Finite real array of shape (n_u,) holding the input at the implicit
        level.
    time_step : float
        Finite strictly positive step size.
    alpha_zero : float
        Finite state coefficient of the implicit level.
    beta_zero : float
        Finite rate coefficient of the implicit level.
    scheme : str
        Either "galerkin" or "lspg".

    Returns
    -------
    step_direction : np.ndarray
        Float array of shape (n,).
    criterion_norm : float
        Native Python float equal to the 2-norm of the test-basis-projected
        residual before the step.

    Raises
    ------
    ValueError
        If any array has the wrong rank or an inconsistent shape, if any entry
        is not finite, if time_step is not finite and positive, or if scheme is
        not one of the two recognised names.
    """
    return (np.empty(0, dtype=float), 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_newton_direction(
    gram: np.ndarray,
    coefficients: np.ndarray,
    reduced_state: np.ndarray,
    current_input: np.ndarray,
    time_step: float,
    alpha_zero: float,
    beta_zero: float,
    scheme: str,
    kronecker_function=None,
) -> tuple[np.ndarray, float]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    matrix = _finite("gram", gram, 2)
    coeff = _finite("coefficients", coefficients, 1)
    state = _finite("reduced_state", reduced_state, 1)
    control = _finite("current_input", current_input, 1)
    if not isinstance(scheme, str) or scheme not in ("galerkin", "lspg"):
        raise ValueError("scheme must be 'galerkin' or 'lspg'")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    for name, value in (("alpha_zero", alpha_zero), ("beta_zero", beta_zero)):
        if not _is_number(value) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real scalar")

    n_reduced = state.shape[0]
    n_inputs = control.shape[0]
    if n_reduced < 1 or n_inputs < 1:
        raise ValueError("reduced_state and current_input must be non-empty")
    total = 2 * n_reduced + n_reduced * n_reduced + n_inputs * n_reduced + 1 + n_inputs
    if matrix.shape != (total, total):
        raise ValueError("gram must be square with the column-basis dimension")
    if coeff.shape[0] != total:
        raise ValueError("coefficients must match the column-basis dimension")

    step = float(time_step)
    lead_state = float(alpha_zero)
    lead_rate = float(beta_zero)
    identity = np.eye(n_reduced, dtype=float)
    square_jacobian = (_oracle_kronecker_square_jacobian
                       if kronecker_function is None else kronecker_function)

    first = n_reduced
    second = 2 * n_reduced
    third = second + n_reduced * n_reduced
    fourth = third + n_inputs * n_reduced

    factor = np.zeros((total, n_reduced), dtype=float)
    factor[:first, :] = lead_state * identity
    factor[first:second, :] = -step * lead_rate * identity
    factor[second:third, :] = -step * lead_rate * square_jacobian(state)
    factor[third:fourth, :] = -step * lead_rate * np.kron(control.reshape(n_inputs, 1), identity)

    if scheme == "lspg":
        # The test basis is the full-order Jacobian itself, so both sides are
        # contracted through the Gram matrix rather than through the trial block.
        weighted = matrix @ factor
        system = factor.T @ weighted
        gradient = factor.T @ (matrix @ coeff)
    else:
        trial_rows = matrix[:first, :]
        system = trial_rows @ factor
        gradient = trial_rows @ coeff

    direction = np.linalg.solve(system, -gradient)
    return np.asarray(direction, dtype=float), float(np.linalg.norm(gradient))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
        "def _sigt(parts, scale):\n"
        "    return float(sum((i + 1.0) * _sig(p, 1.0) for i, p in enumerate(parts)) / scale)\n"
        "def _setup(full, red, n_u, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    basis, _ = np.linalg.qr(rng.standard_normal((full, red)))\n"
        "    c = rng.standard_normal(full)\n"
        "    a = rng.standard_normal((full, full))\n"
        "    f = np.zeros((full, full * full))\n"
        "    for r in range(full):\n"
        "        for p in range(full):\n"
        "            for q in range(p, full):\n"
        "                if (r + p + q) % 3 == 0:\n"
        "                    f[r, p * full + q] = 0.4 * rng.standard_normal()\n"
        "    b = rng.standard_normal((full, n_u))\n"
        "    nn = 0.3 * rng.standard_normal((full, n_u * full))\n"
        "    cols = np.concatenate([basis, a @ basis, f @ np.kron(basis, basis),\n"
        "                           nn @ np.kron(np.eye(n_u), basis),\n"
        "                           c.reshape(-1, 1), b], axis=1)\n"
        "    return basis, c, a, f, b, nn, cols, cols.T @ cols\n"
        "def _coeff(hist, uhist, dt, alphas, betas):\n"
        "    n = hist.shape[1]\n"
        "    n_u = uhist.shape[1]\n"
        "    d = 2 * n + n * n + n_u * n + 1 + n_u\n"
        "    out = np.zeros(d)\n"
        "    o1, o2, o3, o4 = n, 2 * n, 2 * n + n * n, 2 * n + n * n + n_u * n\n"
        "    for j in range(hist.shape[0]):\n"
        "        out[:o1] += alphas[j] * hist[j]\n"
        "        out[o1:o2] += -dt * betas[j] * hist[j]\n"
        "        out[o2:o3] += -dt * betas[j] * np.kron(hist[j], hist[j])\n"
        "        out[o3:o4] += -dt * betas[j] * np.kron(uhist[j], hist[j])\n"
        "        out[o4] += -dt * betas[j]\n"
        "        out[o4 + 1:] += -dt * betas[j] * uhist[j]\n"
        "    return out\n"
        "dt = 0.05\n"
        "alphas = np.array([1.0, -1.0])\n"
        "betas = np.array([1.0, 0.0])\n"
        "hist = np.array([[0.5, -0.9, 0.3], [0.2, -0.7, 0.45]])\n"
        "uhist = np.array([[0.8, -0.3], [0.8, -0.3]])\n"
    )
    return [
        # Case 1: the Galerkin direction on a generic instance.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(8, 3, 2, 31)\n"
                "coeff = _coeff(hist, uhist, dt, alphas, betas)\n"
            ),
            "call": (
                "_sigt(reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, "
                "'galerkin'), 1e0)"
            ),
            "gold_call": (
                "_sigt(_oracle_reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, "
                "'galerkin'), 1e0)"
            ),
        },
        # Case 2: the least-squares direction on the same instance.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(8, 3, 2, 31)\n"
                "coeff = _coeff(hist, uhist, dt, alphas, betas)\n"
            ),
            "call": (
                "_sigt(reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, "
                "'lspg'), 1e0)"
            ),
            "gold_call": (
                "_sigt(_oracle_reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, "
                "'lspg'), 1e0)"
            ),
        },
        # Case 3: a single reduced mode and a single input, with Crank-Nicolson
        #     weights so the implicit rate coefficient is not one.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(5, 1, 1, 12)\n"
                "h1 = np.array([[0.6], [0.4]])\n"
                "u1 = np.array([[0.9], [0.7]])\n"
                "coeff = _coeff(h1, u1, dt, np.array([1.0, -1.0]), np.array([0.5, 0.5]))\n"
            ),
            "call": (
                "_sigt(reduced_newton_direction(gram, coeff, h1[0], u1[0], dt, 1.0, 0.5, 'lspg'), 1e0)"
            ),
            "gold_call": (
                "_sigt(_oracle_reduced_newton_direction(gram, coeff, h1[0], u1[0], dt, 1.0, 0.5, "
                "'lspg'), 1e0)"
            ),
        },
        # --- Decisive: both directions must satisfy the stated Newton system
        #     when the residual and its derivative are rebuilt at full order by
        #     central differences, and the least-squares direction must be the
        #     one that annihilates the full-order gradient rather than the
        #     gradient of the trial-basis-projected residual.
        {
            "setup": reducer + (
                "def against_full_order():\n"
                "    basis, c, a, f, b, nn, cols, gram = _setup(8, 3, 2, 31)\n"
                "    u = uhist[0]\n"
                "    prev = hist[1]\n"
                "    def res(xh):\n"
                "        st = basis @ xh\n"
                "        rate = c + a @ st + f @ np.kron(st, st) + b @ u + nn @ np.kron(u, st)\n"
                "        return st - basis @ prev - dt * rate\n"
                "    x0 = hist[0]\n"
                "    n = x0.size\n"
                "    jac = np.zeros((basis.shape[0], n))\n"
                "    h = 1e-6\n"
                "    for r in range(n):\n"
                "        e = np.zeros(n)\n"
                "        e[r] = h\n"
                "        jac[:, r] = (res(x0 + e) - res(x0 - e)) / (2.0 * h)\n"
                "    r0 = res(x0)\n"
                "    coeff = _coeff(hist, uhist, dt, alphas, betas)\n"
                "    pg, ng = reduced_newton_direction(gram, coeff, x0, u, dt, 1.0, 1.0, 'galerkin')\n"
                "    pl, nl = reduced_newton_direction(gram, coeff, x0, u, dt, 1.0, 1.0, 'lspg')\n"
                "    okg = np.max(np.abs(basis.T @ jac @ pg + basis.T @ r0)) < 1e-6\n"
                "    okl = np.max(np.abs(jac.T @ jac @ pl + jac.T @ r0)) < 1e-6\n"
                "    okng = abs(ng - float(np.linalg.norm(basis.T @ r0))) < 1e-8\n"
                "    oknl = abs(nl - float(np.linalg.norm(jac.T @ r0))) < 1e-6\n"
                "    return int(okg) + 2 * int(okl) + 4 * int(okng) + 8 * int(oknl)\n"
            ),
            "call": "against_full_order()",
            "gold_call": "15",
        },
        # --- Decisive: assembling the least-squares system from the residual
        #     already projected onto the trial basis reproduces the Galerkin
        #     direction, so the two schemes must not agree on this instance.
        {
            "setup": reducer + (
                "def schemes_differ():\n"
                "    basis, c, a, f, b, nn, cols, gram = _setup(8, 3, 2, 31)\n"
                "    coeff = _coeff(hist, uhist, dt, alphas, betas)\n"
                "    pg, _ = reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, 'galerkin')\n"
                "    pl, _ = reduced_newton_direction(gram, coeff, hist[0], uhist[0], dt, 1.0, 1.0, 'lspg')\n"
                "    return int(float(np.max(np.abs(pg - pl))) > 1e-4)\n"
            ),
            "call": "schemes_differ()",
            "gold_call": "1",
        },
        # Case 6: a vanishing residual gives a vanishing direction and measure.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(6, 2, 2, 8)\n"
                "d = gram.shape[0]\n"
                "zero = np.zeros(d)\n"
            ),
            "call": (
                "_sigt(reduced_newton_direction(gram, zero, np.array([0.3, -0.2]), "
                "np.array([0.5, 0.1]), dt, 1.0, 1.0, 'lspg'), 1e0)"
            ),
            "gold_call": (
                "_sigt(_oracle_reduced_newton_direction(gram, zero, np.array([0.3, -0.2]), "
                "np.array([0.5, 0.1]), dt, 1.0, 1.0, 'lspg'), 1e0)"
            ),
        },
        # Case 7: invalid scheme name.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(6, 2, 2, 8)\n"
                "coeff = np.zeros(gram.shape[0])\n"
                "def run_model():\n"
                "    try:\n"
                "        reduced_newton_direction(gram, coeff, np.array([0.3, -0.2]), np.array([0.5, 0.1]), dt, 1.0, 1.0, 'petrov')\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_reduced_newton_direction(gram, coeff, np.array([0.3, -0.2]), np.array([0.5, 0.1]), dt, 1.0, 1.0, 'petrov')\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 8: invalid Gram dimension.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn, cols, gram = _setup(6, 2, 2, 8)\n"
                "small = gram[:-1, :-1]\n"
                "coeff = np.zeros(small.shape[0])\n"
                "def run_model():\n"
                "    try:\n"
                "        reduced_newton_direction(small, coeff, np.array([0.3, -0.2]), np.array([0.5, 0.1]), dt, 1.0, 1.0, 'lspg')\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_reduced_newton_direction(small, coeff, np.array([0.3, -0.2]), np.array([0.5, 0.1]), dt, 1.0, 1.0, 'lspg')\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
