"""
Simulate the observation record produced by continuously monitoring a qutrit whose true Hamiltonian and jump operator are fixed multiples of two seeded generators, and average the resulting noisy increments across many independent trajectories.

What is physically recorded is the continuous measurement current, whose increments are set by the conditional state: that conditional state follows a nonlinear stochastic master equation driven by measurement back-action, not a deterministic drift with additive noise on top. Each trajectory is simulated by Euler-Maruyama stepping of this nonlinear equation on a fine sub-grid, with the resulting state projected back onto the physical set after every fine step (Hermitian part taken, negative eigenvalues clipped to zero, rescaled to unit trace) to prevent the finite-step update from leaving that set, each fine observation increment formed from the state at the start of its sub-step, and the fine increments aggregated onto the coarser observation grid. This averaged data is what a maximum-contrast estimator for continuously monitored open quantum systems uses to reconstruct the unknown Hamiltonian and jump-operator scales.

Returns
-------
return Y_avg
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_averaged_observation_data(seed_gen: int, seed_noise: int, alpha0: float, beta0: float, T: float, n: int, N: int, n_sub: int) -> "np.ndarray":
    """Simulate N averaged noisy observation increments for a monitored qutrit.

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The true generators are ``H = alpha0 * H0``
        and ``L = beta0 * L0``. The monitored qutrit starts in the fixed initial
        state ``rho0 = |0><0|`` (the first computational basis state), independent
        of ``seed_gen``.
    seed_noise : int
        RNG seed used to draw the observation noise: with
        ``rng_noise = np.random.default_rng(seed_noise)``, one
        ``rng_noise.standard_normal(1)`` value is drawn per fine Euler-Maruyama
        sub-step, in trajectory-major, then coarse-step-major, then sub-step-minor
        order (all draws for trajectory 0 before any for trajectory 1, and so on).
    alpha0 : float
        True Hamiltonian scale, so that the true Hamiltonian is alpha0 * H0.
    beta0 : float
        True jump-operator scale, so that the true jump operator is beta0 * L0.
    T : float
        Total observation time, T > 0.
    n : int
        Number of equal coarse observation steps, n >= 1.
    N : int
        Number of independent observation trajectories to average, N >= 1.
    n_sub : int
        Number of Euler-Maruyama sub-steps simulated within each coarse observation
        step, n_sub >= 1. The true conditional-state trajectory and its observation
        increments are generated on this finer grid and then aggregated onto the n
        coarse steps.

    Returns
    -------
    Y_avg : np.ndarray
        (n,) real array, the trajectory-averaged observation increment at each coarse
        step.

    Raises
    ------
    ValueError
        If T <= 0, n < 1, N < 1, or n_sub < 1.
    """
    return Y_avg

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _b_func(A: "np.ndarray", rho: "np.ndarray") -> float:
    return 2 * np.real(np.trace(A.conj().T @ rho))


def _lindblad_drift(rho: "np.ndarray", H: "np.ndarray", L: "np.ndarray") -> "np.ndarray":
    comm = -1j * (H @ rho - rho @ H)
    diss = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return comm + diss


def _proj_physical(rho: "np.ndarray") -> "np.ndarray":
    rho = (rho + rho.conj().T) / 2
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    w = w / s
    return (v * w) @ v.conj().T


def _simulate_true_trajectory(H: "np.ndarray", L: "np.ndarray", rho0: "np.ndarray", T: float, n: int, n_sub: int, rng: "np.random.Generator") -> "np.ndarray":
    delta = T / n / n_sub
    rho = np.array(rho0, dtype=complex).copy()
    Y = np.zeros(n)
    for j in range(n):
        y_accum = 0.0
        for _ in range(n_sub):
            b_val = _b_func(L, rho)
            H_back = L @ rho + rho @ L.conj().T - b_val * rho
            dW = rng.standard_normal(1)[0] * np.sqrt(delta)
            y_accum += b_val * delta + dW
            rho_new = rho + delta * _lindblad_drift(rho, H, L) + H_back * dW
            rho = _proj_physical(rho_new)
        Y[j] = y_accum
    return Y


def _oracle_simulate_averaged_observation_data(seed_gen: int, seed_noise: int, alpha0: float, beta0: float, T: float, n: int, N: int, n_sub: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    if N < 1:
        raise ValueError("N must be at least 1")
    if n_sub < 1:
        raise ValueError("n_sub must be at least 1")
    H0, L0 = _build_qutrit_generators(seed_gen)
    H = alpha0 * H0
    L = beta0 * L0
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0
    rng_noise = np.random.default_rng(seed_noise)
    Y = np.zeros((N, n))
    for i in range(N):
        Y[i] = _simulate_true_trajectory(H, L, rho0, T, n, n_sub, rng_noise)
    return Y.mean(axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: small n, N, generic seeds ---
        {
            "setup": """
import numpy as np
seed_gen = 5
seed_noise = 6
alpha0 = 1.0
beta0 = 0.5
T = 0.5
n = 4
N = 3
n_sub = 5
""",
            "call": "simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
            "gold_call": "_oracle_simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
        },
        # --- Boundary: N=1 (no averaging across trajectories), n_sub=1 (no fine sub-stepping) ---
        {
            "setup": """
import numpy as np
seed_gen = 7
seed_noise = 8
alpha0 = 0.7
beta0 = 1.2
T = 0.3
n = 3
N = 1
n_sub = 1
""",
            "call": "simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
            "gold_call": "_oracle_simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
        },
        # --- Edge: the actual task instance's generators and true parameters ---
        {
            "setup": """
import numpy as np
seed_gen = 101
seed_noise = 202
alpha0 = 1.3
beta0 = 0.8
T = 1.0
n = 40
N = 30
n_sub = 25
""",
            "call": "simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
            "gold_call": "_oracle_simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)",
        },
    ]
