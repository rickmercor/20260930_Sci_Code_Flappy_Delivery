"""
Reconstruct the unknown Hamiltonian and jump-operator scale factors of a continuously monitored qutrit from averaged observation data, by maximizing a discrete contrast function built from the deterministic averaged dynamics evaluated at a trial parameter.



The maximization follows the source paper's own prescribed numerical procedure: the contrast is first evaluated on a uniform grid over the whole bounded parameter space, grid points that are local-maximum candidates (no neighbor scores higher) are selected as well-separated starting points, and each selected candidate is then refined by a bounded Gauss-Newton-type local optimization (equivalently, a bounded nonlinear least-squares fit, since maximizing this contrast is exactly the same as minimizing a weighted sum of squared residuals between the trial drift and the data), with the best-scoring refined candidate reported as the estimate. This combination is required because the contrast surface is not globally well-behaved: a single local run from one starting point can converge to a poor local optimum.

Returns
-------
return theta_hat
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discrete_contrast_estimator(seed_gen: int, Y_avg: "np.ndarray", T: float, n: int) -> "np.ndarray":
    """Maximize the discrete contrast function to estimate (alpha, beta).

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The trial Hamiltonian and jump operator at a
        candidate ``(alpha, beta)`` are ``alpha * H0`` and ``beta * L0``, and the
        deterministic dynamics used to build the contrast start from the fixed
        initial state ``rho0 = |0><0|`` (the first computational basis state),
        independent of ``seed_gen``.
    Y_avg : np.ndarray
        (n,) real array, the trajectory-averaged observation increment at each step.
    T : float
        Total observation time, T > 0.
    n : int
        Number of equal discrete time steps, n >= 1, matching Y_avg's length.

    Returns
    -------
    theta_hat : np.ndarray
        (2,) real array, the fitted (alpha_hat, beta_hat) maximizing the contrast,
        obtained by a uniform grid search over [0, 2] x [0, 2] with spacing 0.1,
        selecting up to 8 well-separated local-maximum candidates (minimum Euclidean
        separation 0.15), refining each by a bounded nonlinear least-squares fit
        constrained to [0, 2] x [0, 2], and returning the refined candidate with the
        largest contrast value.

    Raises
    ------
    ValueError
        If T <= 0, n < 1, or Y_avg does not have length n.
    """
    return theta_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import least_squares


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _b_func(A: "np.ndarray", rho: "np.ndarray") -> float:
    return 2 * np.real(np.trace(A.conj().T @ rho))


def _contrast_and_h(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> "tuple":
    alpha, beta = theta
    H = alpha * H0
    L = beta * L0
    traj = _oracle_averaged_lindblad_evolution(H, np.array([L]), rho0, T, n)
    h = np.array([_b_func(L, traj[j]) for j in range(n)])
    return h


def _contrast_value(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> float:
    dt = T / n
    h = _contrast_and_h(theta, H0, L0, Y_avg, T, n, rho0)
    return float(np.sum(h * Y_avg - 0.5 * h ** 2 * dt))


def _residuals(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> "np.ndarray":
    dt = T / n
    h = _contrast_and_h(theta, H0, L0, Y_avg, T, n, rho0)
    return np.sqrt(dt) * (h - Y_avg / dt)


def _oracle_discrete_contrast_estimator(seed_gen: int, Y_avg: "np.ndarray", T: float, n: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    if Y_avg.shape[0] != n:
        raise ValueError("Y_avg must have length n")
    H0, L0 = _build_qutrit_generators(seed_gen)
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0

    lo, hi = 0.0, 2.0
    grid_step = 0.1
    vals = np.round(np.arange(lo, hi + 1e-9, grid_step), 10)
    C = np.zeros((len(vals), len(vals)))
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            C[i, j] = _contrast_value((a, b), H0, L0, Y_avg, T, n, rho0)

    candidates = []
    ni, nj = C.shape
    for i in range(ni):
        for j in range(nj):
            v = C[i, j]
            is_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < ni and 0 <= jj < nj and C[ii, jj] > v:
                        is_max = False
            if is_max:
                candidates.append((v, vals[i], vals[j]))
    candidates.sort(key=lambda x: -x[0])

    selected = []
    for v, a, b in candidates:
        if all(np.hypot(a - sa, b - sb) >= 0.15 for _, sa, sb in selected):
            selected.append((v, a, b))
        if len(selected) >= 8:
            break
    if not selected:
        i, j = np.unravel_index(np.argmax(C), C.shape)
        selected = [(C[i, j], vals[i], vals[j])]

    best_theta = None
    best_val = -np.inf
    for _, a0, b0 in selected:
        sol = least_squares(_residuals, x0=[a0, b0], bounds=([lo, lo], [hi, hi]), method="trf",
                             args=(H0, L0, Y_avg, T, n, rho0))
        theta = sol.x
        val = _contrast_value(theta, H0, L0, Y_avg, T, n, rho0)
        if val > best_val:
            best_val = val
            best_theta = theta
    return np.array(best_theta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_helpers = """
import numpy as np

def _build_generators(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    H0 = (A + A.conj().T)/2
    L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    return H0, L0

def _lindblad_drift(rho, H, L):
    comm = -1j*(H@rho - rho@H)
    diss = L@rho@L.conj().T - 0.5*(L.conj().T@L@rho + rho@L.conj().T@L)
    return comm + diss

def _proj_physical(rho):
    rho = (rho + rho.conj().T)/2
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        d = rho.shape[0]
        return np.eye(d, dtype=complex)/d
    w = w/s
    return (v*w) @ v.conj().T

def _b_func(A, rho):
    return 2*np.real(np.trace(A.conj().T @ rho))

def _simulate_true_trajectory(H, L, rho0, T, n, n_sub, rng):
    delta = T/n/n_sub
    rho = np.array(rho0, dtype=complex).copy()
    Y = np.zeros(n)
    for j in range(n):
        y_accum = 0.0
        for _ in range(n_sub):
            b_val = _b_func(L, rho)
            H_back = L@rho + rho@L.conj().T - b_val*rho
            dW = rng.standard_normal(1)[0]*np.sqrt(delta)
            y_accum += b_val*delta + dW
            rho_new = rho + delta*_lindblad_drift(rho, H, L) + H_back*dW
            rho = _proj_physical(rho_new)
        Y[j] = y_accum
    return Y

def _simulate_Y_avg(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub):
    H0, L0 = _build_generators(seed_gen)
    H = alpha0*H0
    L = beta0*L0
    rho0 = np.zeros((3,3), dtype=complex); rho0[0,0]=1.0
    rng_noise = np.random.default_rng(seed_noise)
    Y = np.zeros((N,n))
    for i in range(N):
        Y[i] = _simulate_true_trajectory(H, L, rho0, T, n, n_sub, rng_noise)
    return Y.mean(axis=0)
"""
    return [
        # --- Normal: small instance, data simulated near a known parameter pair; the
        #     contrast has a single interior maximizer well ahead of the only other
        #     grid candidate, so the result does not depend on how tied candidates
        #     are ordered ---
        {
            "setup": setup_helpers + """
seed_gen = 31
Y_avg = _simulate_Y_avg(31, 32, 1.0, 0.8, 1.0, 8, 30, 5)
T = 1.0
n = 8
""",
            "call": "discrete_contrast_estimator(seed_gen, Y_avg.copy(), T, n)",
            "gold_call": "_oracle_discrete_contrast_estimator(seed_gen, Y_avg.copy(), T, n)",
            "tol": 1e-3,
        },
        # --- Boundary: zero observation data, a genuine continuum of maximizers (alpha, 0)
        #     for any alpha; check the equivalence-class properties (beta~0, contrast~0)
        #     rather than the specific alpha chosen, since any point on this ridge is valid ---
        {
            "setup": setup_helpers + """
seed_gen = 7
Y_avg = np.zeros(3)
T = 0.3
n = 3

def _build_generators_local(seed):
    return _build_generators(seed)

def summarize(fn):
    theta_hat = np.asarray(fn(seed_gen, Y_avg.copy(), T, n))
    H0, L0 = _build_generators_local(seed_gen)
    rho0 = np.zeros((3,3), dtype=complex); rho0[0,0]=1.0
    alpha, beta = theta_hat
    L = beta*L0
    H = alpha*H0
    dt = T/n
    from scipy.linalg import expm
    d = 3
    I = np.eye(d)
    Lsup = -1j*(np.kron(I,H) - np.kron(H.T,I)) + np.kron(L.conj(),L) - 0.5*(np.kron(I,L.conj().T@L) + np.kron((L.conj().T@L).T,I))
    Esup = expm(Lsup*dt)
    rho = rho0.copy()
    total = 0.0
    for _ in range(n):
        vec = Esup @ rho.reshape(-1,1,order='F')
        rho = vec.reshape(3,3,order='F')
        rho = (rho+rho.conj().T)/2
        h_val = _b_func(L, rho)
        total += h_val*0.0 - 0.5*h_val**2*dt
    in_box = bool(0.0 <= alpha <= 2.0 and 0.0 <= beta <= 2.0)
    beta_small = bool(abs(beta) < 1e-3)
    contrast_near_zero = bool(abs(total) < 1e-3)
    return (in_box, beta_small, contrast_near_zero)
""",
            "call": "summarize(discrete_contrast_estimator)",
            "gold_call": "summarize(_oracle_discrete_contrast_estimator)",
        },
        # --- Edge: the actual task instance's data and settings ---
        {
            "setup": setup_helpers + """
seed_gen = 101
Y_avg = _simulate_Y_avg(101, 202, 1.3, 0.8, 1.0, 40, 30, 25)
T = 1.0
n = 40
""",
            "call": "discrete_contrast_estimator(seed_gen, Y_avg.copy(), T, n)",
            "gold_call": "_oracle_discrete_contrast_estimator(seed_gen, Y_avg.copy(), T, n)",
            "tol": 1e-3,
        },
    ]
