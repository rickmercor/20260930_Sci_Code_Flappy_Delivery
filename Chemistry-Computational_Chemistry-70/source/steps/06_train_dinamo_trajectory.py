"""
Implement a function that trains DINaMo's trajectory-unsupervised,

physics-informed neural solver on one initial-value problem and returns its

predicted position trajectory at the same collocation frames used by Steps 3

and 5. The predicted trajectory must be built from a hard initial-condition

ansatz (so the initial position, velocity, and acceleration are exact by

construction) plus a small neural correction, trained by minimizing only

residuals of Newton's equation, momentum conservation, and energy

conservation evaluated on the network's own predicted path -- never against

any simulator-generated position, velocity, force, or energy.

DINaMo represents an entire short-time trajectory as a single differentiable

function of time, r_theta(t) = r0 + t*v0 + 0.5*a0*t^2 + T^2*s^3*g_theta(s),

where s = t/T, a0 = F_LJ(r0)/m is the exact analytic Lennard-Jones

acceleration at the initial state, and g_theta is a small ISRU-activated MLP

correction. Because the leading polynomial already satisfies the initial

position, velocity, and acceleration exactly, the network only has to learn

the higher-order nonlinear departure. Velocity and acceleration are obtained

by differentiating this single function of time (not from separate network

outputs), so training minimizes a weighted sum of the Newton residual

||m*a_theta - F_LJ(r_theta)||^2, a momentum-conservation residual, and an

energy-conservation residual, with no simulator trajectory ever entering the

objective. This is the second of the two independent solvers this problem

compares against the exact reference trajectory of Step 3.

Returns
-------
np.ndarray of shape (n_steps + 1, N, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def train_dinamo_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                             box_length: float, n_steps: int, dt: float = 5e-4,
                             cutoff: float = 2.937, epsilon: float = 1.0,
                             sigma: float = 1.0, hidden_width: int = 64,
                             n_epochs: int = 3000, lr: float = 1e-3, seed: int = 0,
                             w_newton: float = 1.0, w_momentum: float = 0.02,
                             w_energy: float = 0.02, energy_scale: float = 0.1,
                             final_layer_scale: float = 1e-3) -> np.ndarray:
    """
    Parameters
    ----------
    positions0 : np.ndarray
        Array of shape (N, 3) with N >= 2, the initial particle positions.
    velocities0 : np.ndarray
        Array of shape (N, 3), matching `positions0`, the initial particle
        velocities.
    box_length : float
        Side length of the cubic periodic box.
    n_steps : int
        Number of collocation intervals; the trajectory is returned at the
        n_steps + 1 frames t_b = b*dt for b = 0, ..., n_steps. Must be >= 0.
    dt : float, optional
        Spacing between collocation frames. Default 5e-4.
    cutoff : float, optional
        Lennard-Jones interaction cutoff. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.
    hidden_width : int, optional
        Width of each hidden layer of the correction network. Default 64.
    n_epochs : int, optional
        Number of full-batch Adam training epochs, >= 0. A value of 0 returns
        the untrained ansatz. Default 3000.
    lr : float, optional
        Adam learning rate. Default 1e-3.
    seed : int, optional
        Seed for `np.random.default_rng`, controlling the network's initial
        weights. Default 0.
    w_newton, w_momentum, w_energy : float, optional
        Non-negative weights on the three residual terms. Defaults 1.0, 0.02,
        0.02.
    energy_scale : float, optional
        Per-atom energy normalization scale for the energy residual. Default
        0.1.
    final_layer_scale : float, optional
        Factor multiplying the correction network's final layer weights and
        bias at initialization, >= 0. Default 1e-3.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_steps + 1, N, 3): the predicted positions at t_b =
        b*dt for b = 0, ..., n_steps, wrapped into the primary periodic box
        `[0, box_length)`.

    Raises
    ------
    ValueError
        If `positions0` does not have shape (N, 3) with N >= 2, or
        contains non-finite values; if `velocities0` does not match
        `positions0`'s shape or contains non-finite values; if
        `box_length`, `dt`, `lr`, or `energy_scale` is not finite and > 0;
        if `n_steps` or `n_epochs` is not a non-negative integer; if
        `cutoff` is not finite, > 0, and < `box_length` / 2; if `epsilon`
        or `sigma` is not finite and > 0; if `hidden_width` is not a
        positive integer; if `seed` is not a non-negative integer; if
        `w_newton`, `w_momentum`, or `w_energy` is not finite and >= 0; or
        if `final_layer_scale` is not finite and >= 0.
"""
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_train_dinamo_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                     box_length: float, n_steps: int, dt: float = 5e-4,
                                     cutoff: float = 2.937, epsilon: float = 1.0,
                                     sigma: float = 1.0, hidden_width: int = 64,
                                     n_epochs: int = 3000, lr: float = 1e-3, seed: int = 0,
                                     w_newton: float = 1.0, w_momentum: float = 0.02,
                                     w_energy: float = 0.02, energy_scale: float = 0.1,
                                     final_layer_scale: float = 1e-3) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    def _is_nonneg_int(x):
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 0

    positions0 = np.asarray(positions0, dtype=float)
    velocities0 = np.asarray(velocities0, dtype=float)

    if positions0.ndim != 2 or positions0.shape[1] != 3 or positions0.shape[0] < 2:
        raise ValueError("positions0 must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions0)):
        raise ValueError("positions0 must contain only finite values.")
    if velocities0.shape != positions0.shape:
        raise ValueError("velocities0 must have the same shape as positions0.")
    if not np.all(np.isfinite(velocities0)):
        raise ValueError("velocities0 must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not _is_nonneg_int(n_steps):
        raise ValueError("n_steps must be a non-negative integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")
    if not (isinstance(hidden_width, (int, np.integer)) and not isinstance(hidden_width, bool) and hidden_width > 0):
        raise ValueError("hidden_width must be a positive integer.")
    if not _is_nonneg_int(n_epochs):
        raise ValueError("n_epochs must be a non-negative integer.")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and > 0.")
    if not _is_nonneg_int(seed):
        raise ValueError("seed must be a non-negative integer.")
    for name, w in (("w_newton", w_newton), ("w_momentum", w_momentum), ("w_energy", w_energy)):
        if not np.isfinite(w) or w < 0:
            raise ValueError(f"{name} must be finite and >= 0.")
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("energy_scale must be finite and > 0.")
    if not np.isfinite(final_layer_scale) or final_layer_scale < 0:
        raise ValueError("final_layer_scale must be finite and >= 0.")

    n = positions0.shape[0]
    out_dim = 3 * n
    t_window = n_steps * dt if n_steps > 0 else dt
    h_fd = 1e-4  # finite-difference step for velocity/acceleration; the ballistic
                 # (quadratic) part of the ansatz is differenced exactly, so only
                 # the network correction carries a controlled O(h^2) error.

    # ---- shared shifted-force Lennard-Jones force/energy, and its
    # vector-Jacobian product (needed because training backpropagates the
    # Newton residual's dependence on position through the force law) ----
    def _min_image(diff):
        return diff - box_length * np.round(diff / box_length)

    def _lj_force_energy(pos):
        diff = pos[:, None, :] - pos[None, :, :]
        diff = _min_image(diff)
        dist = np.linalg.norm(diff, axis=-1)
        iu, ju = np.triu_indices(n, k=1)
        d = dist[iu, ju]
        mask = d <= cutoff
        force = np.zeros((n, 3))
        energy = 0.0
        if np.any(mask):
            ii, jj = iu[mask], ju[mask]
            dd = d[mask]
            dvec = diff[ii, jj]
            sr6 = (sigma / dd) ** 6
            sr12 = sr6 * sr6
            Fr = 24.0 * epsilon / dd * (2.0 * sr12 - sr6)
            Ur = 4.0 * epsilon * (sr12 - sr6)
            src6 = (sigma / cutoff) ** 6
            src12 = src6 * src6
            Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)
            Uc = 4.0 * epsilon * (src12 - src6)
            Fsf = Fr - Fc
            Usf = Ur - Uc + (dd - cutoff) * Fc
            rhat = dvec / dd[:, None]
            fij = Fsf[:, None] * rhat
            np.add.at(force, ii, fij)
            np.add.at(force, jj, -fij)
            energy = float(np.sum(Usf))
        return force, energy

    def _lj_force_vjp(pos, v):
        """Given an incoming gradient v (N,3) w.r.t. the force output,
        return the resulting gradient (N,3) w.r.t. pos: v^T @ (dF/dpos),
        using the standard pairwise "M-matrix" identity
        d[F(r)*rhat]/dr_i = F'(r)*rhat(x)rhat + (F(r)/r)*(I - rhat(x)rhat)."""
        diff = pos[:, None, :] - pos[None, :, :]
        diff = _min_image(diff)
        dist = np.linalg.norm(diff, axis=-1)
        iu, ju = np.triu_indices(n, k=1)
        d = dist[iu, ju]
        mask = d <= cutoff
        grad = np.zeros((n, 3))
        if np.any(mask):
            ii, jj = iu[mask], ju[mask]
            dd = d[mask]
            dvec = diff[ii, jj]
            rhat = dvec / dd[:, None]
            sr6 = (sigma / dd) ** 6
            sr12 = sr6 * sr6
            Fr = 24.0 * epsilon / dd * (2.0 * sr12 - sr6)
            src6 = (sigma / cutoff) ** 6
            src12 = src6 * src6
            Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)
            Fsf = Fr - Fc
            dFr = (24.0 * epsilon / dd ** 2) * (7.0 * sr6 - 26.0 * sr12)
            dv = v[ii] - v[jj]
            rhat_dot_dv = np.sum(rhat * dv, axis=1)
            term = (dFr - Fsf / dd)[:, None] * rhat_dot_dv[:, None] * rhat + (Fsf / dd)[:, None] * dv
            np.add.at(grad, ii, term)
            np.add.at(grad, jj, -term)
        return grad

    # ---- ISRU activation phi(x) = x/sqrt(1+x^2) and its derivative ----
    def _phi(x):
        return x / np.sqrt(1.0 + x * x)

    def _phi_prime(x):
        return (1.0 + x * x) ** (-1.5)

    # ---- correction network: Linear(1,H) -> ISRU -> Linear(H,H) -> ISRU -> Linear(H,3N) ----
    rng = np.random.default_rng(seed)

    def _kaiming(fan_in, fan_out):
        bound = 1.0 / np.sqrt(fan_in)
        W = rng.uniform(-bound, bound, size=(fan_out, fan_in))
        b = rng.uniform(-bound, bound, size=(fan_out,))
        return W, b

    W1, b1 = _kaiming(1, hidden_width)
    W2, b2 = _kaiming(hidden_width, hidden_width)
    W3, b3 = _kaiming(hidden_width, out_dim)
    W3 = W3 * final_layer_scale
    b3 = b3 * final_layer_scale
    params = [W1, b1, W2, b2, W3, b3]

    def _net_forward(S):
        Z1 = S @ params[0].T + params[1]
        A1 = _phi(Z1)
        Z2 = A1 @ params[2].T + params[3]
        A2 = _phi(Z2)
        G = A2 @ params[4].T + params[5]
        return G, (S, Z1, A1, Z2, A2)

    def _net_backward(dG, cache):
        S, Z1, A1, Z2, A2 = cache
        dW3 = dG.T @ A2
        db3 = dG.sum(axis=0)
        dA2 = dG @ params[4]
        dZ2 = dA2 * _phi_prime(Z2)
        dW2 = dZ2.T @ A1
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ params[2]
        dZ1 = dA1 * _phi_prime(Z1)
        dW1 = dZ1.T @ S
        db1 = dZ1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    # ---- hard initial-condition ansatz: analytic Taylor part + correction ----
    a0, _ = _lj_force_energy(positions0)  # mass = 1

    def _r_ballistic(t_arr):
        t = t_arr.reshape(-1, 1, 1)
        return positions0[None, :, :] + t * velocities0[None, :, :] + 0.5 * a0[None, :, :] * t * t

    def _r_full(t_arr):
        s_arr = t_arr / t_window
        S = s_arr.reshape(-1, 1)
        G, cache = _net_forward(S)
        G = G.reshape(-1, n, 3)
        coeff = (t_window ** 2) * (s_arr ** 3)
        R = _r_ballistic(t_arr) + coeff.reshape(-1, 1, 1) * G
        return R, (cache, coeff)

    def _r_full_backward(dR, aux):
        cache, coeff = aux
        dG = (dR * coeff.reshape(-1, 1, 1)).reshape(dR.shape[0], -1)
        return _net_backward(dG, cache)

    t_grid = np.array([b * dt for b in range(n_steps + 1)], dtype=float)
    B = len(t_grid)

    # ---- Adam optimizer state ----
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for epoch in range(1, n_epochs + 1):
        R0, aux0 = _r_full(t_grid)
        Rp, auxp = _r_full(t_grid + h_fd)
        Rm, auxm = _r_full(t_grid - h_fd)

        V = (Rp - Rm) / (2 * h_fd)
        A = (Rp - 2 * R0 + Rm) / (h_fd ** 2)

        F0 = np.zeros_like(R0)
        U0 = np.zeros(B)
        for b in range(B):
            F0[b], U0[b] = _lj_force_energy(R0[b])

        e = A - F0
        newton_per_frame = np.sum(e ** 2, axis=(1, 2)) / n

        P = V.sum(axis=1)
        P0 = P[0]
        p_res = np.sum((P - P0) ** 2, axis=1)

        KE = 0.5 * np.sum(V ** 2, axis=(1, 2))
        E = KE + U0
        E0 = E[0]
        e_res = ((E - E0) / (n * energy_scale)) ** 2

        # ---- backward pass ----
        dNewton = np.full(B, w_newton / B)
        dPres = np.full(B, w_momentum / B)
        dEres = np.full(B, w_energy / B)

        dE = dEres * 2 * (E - E0) / (n * energy_scale) ** 2
        dE_total = dE.copy()
        dE_total[0] -= np.sum(dE)

        dV_from_KE = dE_total.reshape(-1, 1, 1) * V
        dU0_total = dE_total.copy()

        dP = dPres.reshape(-1, 1) * 2 * (P - P0)
        dP_total = dP.copy()
        dP_total[0] -= np.sum(dP, axis=0)
        dV_from_P = np.repeat(dP_total[:, None, :], n, axis=1)

        de = dNewton.reshape(-1, 1, 1) * 2 * e / n
        dA_total = de
        dF0_total = -de

        dV_total = dV_from_KE + dV_from_P

        dRp = dV_total / (2 * h_fd) + dA_total / (h_fd ** 2)
        dRm = -dV_total / (2 * h_fd) + dA_total / (h_fd ** 2)
        dR0 = -2 * dA_total / (h_fd ** 2)

        for b in range(B):
            dR0[b] += _lj_force_vjp(R0[b], dF0_total[b])
            dR0[b] += (-F0[b]) * dU0_total[b]

        grads = [np.zeros_like(p) for p in params]
        for dR, aux in ((dR0, aux0), (dRp, auxp), (dRm, auxm)):
            g = _r_full_backward(dR, aux)
            for i in range(len(grads)):
                grads[i] += g[i]

        for i in range(len(params)):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grads[i]
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_state[i] / (1 - beta1 ** epoch)
            v_hat = v_state[i] / (1 - beta2 ** epoch)
            params[i] = params[i] - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    R_final, _ = _r_full(t_grid)
    trajectory = R_final % box_length
    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 3 atoms, a handful of steps, a small network
            # and short training run (kept small for test speed).
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.6, 1.1, 1.0],\n"
                "    [1.0, 1.6, 1.2],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.1, -0.1, 0.0],\n"
                "    [-0.1, 0.1, 0.1],\n"
                "    [0.0, 0.0, -0.1],\n"
                "])\n"
                "box_length = 8.0\n"
                "cutoff = 2.0\n"
                "n_steps = 5\n"
                "dt = 1e-3\n"
            ),
            "call": (
                "train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, "
                "cutoff=cutoff, hidden_width=8, n_epochs=30, lr=1e-2, seed=0)"
            ),
            "gold_call": (
                "_oracle_train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, "
                "cutoff=cutoff, hidden_width=8, n_epochs=30, lr=1e-2, seed=0)"
            ),
        },
        {
            # Boundary case: n_epochs = 0 and final_layer_scale = 0.0, so the
            # correction network contributes exactly zero and the returned
            # trajectory must equal the pure analytic Taylor ansatz
            # r0 + t*v0 + 0.5*a0*t^2, independent of the (untrained) network.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.7, 1.0, 1.0],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.2, 0.0, 0.0],\n"
                "    [-0.2, 0.1, 0.0],\n"
                "])\n"
                "box_length = 8.0\n"
                "cutoff = 2.5\n"
                "n_steps = 3\n"
                "dt = 1e-3\n"
            ),
            "call": (
                "train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, "
                "cutoff=cutoff, n_epochs=0, final_layer_scale=0.0)"
            ),
            "gold_call": (
                "_oracle_train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, "
                "cutoff=cutoff, n_epochs=0, final_layer_scale=0.0)"
            ),
        },
        {
            # Edge case: n_steps = 0, a single collocation frame at t=0,
            # which must equal positions0 wrapped into the box regardless of
            # training.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [0.5, 0.5, 0.5],\n"
                "    [9.3, -1.1, 2.2],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.1, 0.0, 0.0],\n"
                "    [-0.1, 0.0, 0.0],\n"
                "])\n"
                "box_length = 8.0\n"
                "cutoff = 2.0\n"
                "n_steps = 0\n"
            ),
            "call": (
                "train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, "
                "cutoff=cutoff, hidden_width=8, n_epochs=10)"
            ),
            "gold_call": (
                "_oracle_train_dinamo_trajectory(positions0, velocities0, box_length, n_steps, "
                "cutoff=cutoff, hidden_width=8, n_epochs=10)"
            ),
        },
    ]
