#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def lj_force_energy(positions: np.ndarray, box_length: float, cutoff: float,
                             epsilon: float = 1.0, sigma: float = 1.0) -> tuple:
    """Reference implementation."""

    import numpy as np

    def _min_image_pair_data(pos, box):
        """Local helper: minimum-image pairwise displacement vectors and distances."""
        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - box * np.round(diff / box)
        dist = np.linalg.norm(diff, axis=-1)
        return diff, dist

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions.shape[0]
    diff, dist = _min_image_pair_data(positions, box_length)

    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    mask = d <= cutoff

    forces = np.zeros_like(positions)
    potential_energy = 0.0

    if np.any(mask):
        d_in = d[mask]
        dvec_in = diff[iu, ju][mask]

        sr6 = (sigma / d_in) ** 6
        sr12 = sr6 * sr6
        U = 4.0 * epsilon * (sr12 - sr6)
        F = 24.0 * epsilon / d_in * (2.0 * sr12 - sr6)

        src6 = (sigma / cutoff) ** 6
        src12 = src6 * src6
        Uc = 4.0 * epsilon * (src12 - src6)
        Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)

        Usf = U - Uc + (d_in - cutoff) * Fc
        Fsf = F - Fc

        rhat = dvec_in / d_in[:, None]
        fij = Fsf[:, None] * rhat

        ii = iu[mask]
        jj = ju[mask]
        np.add.at(forces, ii, fij)
        np.add.at(forces, jj, -fij)
        potential_energy = float(np.sum(Usf))

    return forces, potential_energy

def generate_initial_condition(n_atoms: int, density: float, temperature: float,
                                        pos_seed: int, vel_seed: int,
                                        min_separation: float = 0.8, cutoff: float = 2.937,
                                        epsilon: float = 1.0, sigma: float = 1.0,
                                        minimize_steps: int = 100, burn_in_steps: int = 1000,
                                        dt: float = 5e-4, rescale_interval: int = 20,
                                        max_attempts: int = 200000) -> tuple:
    """Reference implementation."""
    import numpy as np

    def _is_nonneg_int(x):
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 0

    if not (isinstance(n_atoms, (int, np.integer)) and not isinstance(n_atoms, bool) and n_atoms >= 2):
        raise ValueError("n_atoms must be an integer >= 2.")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and > 0.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0.")
    if not _is_nonneg_int(pos_seed):
        raise ValueError("pos_seed must be a non-negative integer.")
    if not _is_nonneg_int(vel_seed):
        raise ValueError("vel_seed must be a non-negative integer.")
    if not _is_nonneg_int(minimize_steps):
        raise ValueError("minimize_steps must be a non-negative integer.")
    if not _is_nonneg_int(burn_in_steps):
        raise ValueError("burn_in_steps must be a non-negative integer.")
    if not (isinstance(rescale_interval, (int, np.integer)) and rescale_interval > 0):
        raise ValueError("rescale_interval must be a positive integer.")
    if not (isinstance(max_attempts, (int, np.integer)) and max_attempts > 0):
        raise ValueError("max_attempts must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")

    box_length = (n_atoms / density) ** (1.0 / 3.0)

    if not np.isfinite(min_separation) or min_separation <= 0 or min_separation >= box_length / 2:
        raise ValueError("min_separation must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    def _min_image(vec, box):
        return vec - box * np.round(vec / box)

    def _minimize(pos, n_iter):
        if n_iter == 0:
            return pos
        pos = pos.copy()
        max_disp = 0.005
        for _ in range(n_iter):
            F, _ = lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
            fmax = np.max(np.linalg.norm(F, axis=1))
            if fmax < 1e-6:
                break
            s = max_disp / max(fmax, 1e-12)
            pos = (pos + s * F) % box_length
        return pos

    def _rescale(v, dof):
        ke = 0.5 * np.sum(v ** 2)
        t_inst = 2.0 * ke / dof
        return v * np.sqrt(temperature / t_inst)

    def _burn_in(pos, vel, n_steps, dof):
        if n_steps == 0:
            return pos, vel
        pos = pos.copy()
        vel = vel.copy()
        F, _ = lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
        for step in range(1, n_steps + 1):
            pos = (pos + vel * dt + 0.5 * F * dt * dt) % box_length
            F_new, _ = lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
            vel = vel + 0.5 * (F + F_new) * dt
            F = F_new
            if step % rescale_interval == 0:
                vel = _rescale(vel, dof)
        vel = vel - vel.mean(axis=0, keepdims=True)
        vel = _rescale(vel, dof)
        return pos, vel

    rng_pos = np.random.default_rng(pos_seed)
    positions = np.zeros((n_atoms, 3))
    placed = 0
    attempts = 0
    while placed < n_atoms:
        candidate = rng_pos.random(3) * box_length
        ok = True
        for k in range(placed):
            d = np.linalg.norm(_min_image(candidate - positions[k], box_length))
            if d < min_separation:
                ok = False
                break
        if ok:
            positions[placed] = candidate
            placed += 1
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Could not place {n_atoms} particles with min_separation="
                f"{min_separation} in box_length={box_length} within "
                f"{max_attempts} attempts; density may be too high."
            )

    positions = _minimize(positions, minimize_steps)

    rng_vel = np.random.default_rng(vel_seed)
    velocities = rng_vel.normal(loc=0.0, scale=np.sqrt(temperature), size=(n_atoms, 3))
    velocities = velocities - velocities.mean(axis=0, keepdims=True)
    dof = 3 * n_atoms - 3
    velocities = _rescale(velocities, dof)

    positions, velocities = _burn_in(positions, velocities, burn_in_steps, dof)

    return positions, velocities, float(box_length)

def generate_reference_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                           box_length: float, n_steps: int, dt: float = 5e-4,
                                           cutoff: float = 2.937, epsilon: float = 1.0,
                                           sigma: float = 1.0) -> np.ndarray:
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

    n = positions0.shape[0]
    pos = positions0.copy() % box_length
    vel = velocities0.copy()

    trajectory = np.zeros((n_steps + 1, n, 3))
    trajectory[0] = pos

    force, _ = lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
    for step in range(1, n_steps + 1):
        pos = (pos + vel * dt + 0.5 * force * dt * dt) % box_length
        force_new, _ = lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
        vel = vel + 0.5 * (force + force_new) * dt
        force = force_new
        trajectory[step] = pos

    return trajectory

def rbl_force(positions: np.ndarray, box_length: float, core_cutoff: float,
                       cutoff: float, batch_size: int, rng: np.random.Generator,
                       epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Reference implementation."""

    import numpy as np

    def _pair_force_magnitude(r):
        sr6 = (sigma / r) ** 6
        sr12 = sr6 * sr6
        F = 24.0 * epsilon / r * (2.0 * sr12 - sr6)
        src6 = (sigma / cutoff) ** 6
        src12 = src6 * src6
        Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)
        return F - Fc

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not np.isfinite(core_cutoff) or core_cutoff <= 0:
        raise ValueError("core_cutoff must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= core_cutoff or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > core_cutoff, and < box_length / 2.")
    if not (isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an instance of np.random.Generator.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    diff = diff - box_length * np.round(diff / box_length)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    force_star = np.zeros((n, 3))
    for i in range(n):
        d_row = dist[i]
        core_idx = np.where(d_row <= core_cutoff)[0]
        shell_idx = np.where((d_row > core_cutoff) & (d_row <= cutoff))[0]

        contribution = np.zeros(3)
        if core_idx.size > 0:
            d_core = d_row[core_idx]
            f_core = _pair_force_magnitude(d_core)
            rhat = diff[i, core_idx] / d_core[:, None]
            contribution += (f_core[:, None] * rhat).sum(axis=0)

        n_shell = shell_idx.size
        if n_shell > 0:
            if n_shell <= batch_size:
                sample = shell_idx
                weight = 1.0
            else:
                sample = rng.choice(shell_idx, size=batch_size, replace=False)
                weight = n_shell / batch_size
            d_shell = d_row[sample]
            f_shell = _pair_force_magnitude(d_shell)
            rhat = diff[i, sample] / d_shell[:, None]
            contribution += weight * (f_shell[:, None] * rhat).sum(axis=0)

        force_star[i] = contribution

    forces = force_star - force_star.sum(axis=0, keepdims=True) / n
    return forces

def generate_rbl_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                     box_length: float, n_steps: int, core_cutoff: float,
                                     cutoff: float, batch_size: int, seed: int,
                                     dt: float = 5e-4, epsilon: float = 1.0,
                                     sigma: float = 1.0) -> np.ndarray:
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
    if not np.isfinite(core_cutoff) or core_cutoff <= 0:
        raise ValueError("core_cutoff must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= core_cutoff or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > core_cutoff, and < box_length / 2.")
    if not (isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not _is_nonneg_int(seed):
        raise ValueError("seed must be a non-negative integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions0.shape[0]
    rng = np.random.default_rng(seed)
    pos = positions0.copy() % box_length
    vel = velocities0.copy()

    trajectory = np.zeros((n_steps + 1, n, 3))
    trajectory[0] = pos

    force = rbl_force(pos, box_length, core_cutoff, cutoff, batch_size, rng, epsilon, sigma)
    for step in range(1, n_steps + 1):
        pos = (pos + vel * dt + 0.5 * force * dt * dt) % box_length
        force_new = rbl_force(pos, box_length, core_cutoff, cutoff, batch_size, rng, epsilon, sigma)
        vel = vel + 0.5 * (force + force_new) * dt
        force = force_new
        trajectory[step] = pos

    return trajectory

def train_dinamo_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
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

def trajectory_rmse(predicted_trajectory: np.ndarray, reference_trajectory: np.ndarray,
                             box_length: float) -> float:
    """Reference implementation."""
    import numpy as np
    predicted_trajectory = np.asarray(predicted_trajectory, dtype=float)
    reference_trajectory = np.asarray(reference_trajectory, dtype=float)

    if (predicted_trajectory.ndim != 3 or predicted_trajectory.shape[2] != 3
            or predicted_trajectory.shape[0] < 1 or predicted_trajectory.shape[1] < 2):
        raise ValueError("predicted_trajectory must have shape (B, N, 3) with B >= 1 and N >= 2.")
    if not np.all(np.isfinite(predicted_trajectory)):
        raise ValueError("predicted_trajectory must contain only finite values.")
    if reference_trajectory.shape != predicted_trajectory.shape:
        raise ValueError("reference_trajectory must have the same shape as predicted_trajectory.")
    if not np.all(np.isfinite(reference_trajectory)):
        raise ValueError("reference_trajectory must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")

    b, n, _ = predicted_trajectory.shape
    diff = predicted_trajectory - reference_trajectory
    diff = diff - box_length * np.round(diff / box_length)
    sq_sum = np.sum(diff ** 2)
    rmse = np.sqrt(sq_sum / (3 * n * b))
    return float(rmse)

def orchestrator(n_atoms: int = 50, density: float = 0.15,
                          temperature: float = 2.50487911765, pos_seed: int = 12345,
                          vel_seed: int = 987654, n_steps: int = 240, dt: float = 5e-4,
                          cutoff: float = 2.937, core_cutoff: float = 1.5,
                          batch_size: int = 10, rbl_seed: int = 20260911,
                          hidden_width: int = 64, n_epochs: int = 3000, lr: float = 1e-3,
                          dinamo_seed: int = 0, epsilon: float = 1.0, sigma: float = 1.0,
                          min_separation: float = 0.8, minimize_steps: int = 100,
                          burn_in_steps: int = 1000, rescale_interval: int = 20) -> float:
    """Reference implementation."""
    import numpy as np

    positions0, velocities0, box_length = generate_initial_condition(
        n_atoms, density, temperature, pos_seed, vel_seed,
        min_separation=min_separation, cutoff=cutoff, epsilon=epsilon, sigma=sigma,
        minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, dt=dt,
        rescale_interval=rescale_interval,
    )

    reference_trajectory = generate_reference_trajectory(
        positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff,
        epsilon=epsilon, sigma=sigma,
    )

    rbl_trajectory = generate_rbl_trajectory(
        positions0, velocities0, box_length, n_steps, core_cutoff, cutoff,
        batch_size, rbl_seed, dt=dt, epsilon=epsilon, sigma=sigma,
    )

    dinamo_trajectory = train_dinamo_trajectory(
        positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff,
        epsilon=epsilon, sigma=sigma, hidden_width=hidden_width, n_epochs=n_epochs,
        lr=lr, seed=dinamo_seed,
    )

    rmse_rbl = trajectory_rmse(rbl_trajectory, reference_trajectory, box_length)
    rmse_dinamo = trajectory_rmse(dinamo_trajectory, reference_trajectory, box_length)

    if rmse_dinamo <= 0.0:
        raise ValueError(
            "RMSE_DINaMo is 0 (e.g. because n_steps = 0), so log10(RMSE_RBL / "
            "RMSE_DINaMo) is undefined."
        )

    with np.errstate(divide="ignore"):
        return float(np.log10(rmse_rbl / rmse_dinamo))
SCICODE_GOLD_EOF
