"""
Advance the peridynamic body to the final time with the asynchronous variational velocity-Verlet scheme, updating bond states as the integration proceeds.

Each material point's interaction potential carries its own local update grid, whose spacing is given by local_steps. Local grids all begin at time zero. The global grid is the union of the local grids, and every global node advances positions and velocities for the whole body.

Force contributions are exchanged as impulses rather than applied instantaneously. When a potential reaches one of its local nodes, the internal forces of the bonds it owns are evaluated at the configuration current at that node and deposited into a backward and a forward impulse register, weighted respectively by the local interval preceding and the local interval following that node. The backward weight is omitted at a potential's first local node and the forward weight at its last. Each deposit places the contribution on both points joined by the bond, with the same signs and volume weights as the internal force accumulation.

Each advance from one global node to the next consists of three operations in this order. Velocities receive half of the forward impulses accumulated at the node being left, together with half of the external body-force impulse over the interval. Positions then advance using the resulting velocity. At the node just reached, both impulse registers are cleared, the active potentials update their bond states and deposit at the new configuration, and velocities then receive half of the backward impulses accumulated there together with the remaining half of the body-force impulse. Consequently the two half updates of an advance use forces evaluated at the two different configurations that bound it, and the first and last global nodes each contribute one half update rather than two.

Bond states are updated immediately before each deposit, on the bonds owned by the potential being updated. Bond states begin from initial_bond_states, and the body begins at rest with zero displacement.

Only the displacement field at the final time is returned, with one row per material point in the lattice ordering.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2) with B >= 1; any bond index lies outside the range of lattice; micromodulus does not have shape (N,) or is not finite and positive; critical_stretch_squared does not have shape (B,) or is not finite and positive; no_fail_points does not have shape (N,) or contains a value other than 0 or 1; body_force does not have shape (N, 2) or contains non-finite values; local_steps does not have shape (N,) or is not finite and positive; final_time is not finite and strictly positive; density is not finite and strictly positive; influence_exponent is not finite or is greater than or equal to 3; initial_bond_states does not have shape (B,) matching bonds, or contains a value other than 0 or 1; any local step is not a positive integer multiple of the smallest local step within a relative tolerance of 1e-12; final_time is not a positive integer multiple of the smallest local step, or of every local step, within a relative tolerance of 1e-12; any bond has non-positive reference length; any bond has non-positive deformed length.

A variational integrator is obtained by discretising the action rather than the equations of motion. Approximating the potential contribution over each interval with the trapezoidal rule and taking variations of the resulting discrete action produces the velocity-Verlet scheme, with the characteristic staggered velocity update. Because the scheme descends from a discrete variational principle it is symplectic and satisfies a discrete Noether theorem, so momenta associated with symmetries are conserved exactly and the energy shows no secular drift.

The asynchronous extension assigns every interaction potential its own sequence of update times and merges them into one global ordered sequence. Kinetic contributions live on the global grid, since all points must share a common notion of position and velocity, while potential contributions live on the local grids. Taking the variation with this structure shows that a potential's force enters the dynamics weighted by the length of its own local intervals, split between the interval before and the interval after the node at which it is evaluated. This is why the scheme is naturally expressed with impulse registers rather than with accelerations: the quantity exchanged at an event is a force multiplied by a time interval, and the endpoints of each local grid carry only one of the two half-contributions.

The order of the resulting method depends on where the forces of each half update are evaluated. Second order requires that the half update opening an interval use the configuration at its start and the half update closing it use the configuration at its end. Taking both from the same configuration, or applying the external loading at full weight at the first advance, produces a scheme that is still stable and still conservative in appearance but is only first-order accurate, and the degradation is visible on a smooth problem where no other error source is active.

The efficiency gain is that a coarse region is evaluated only at its own, larger intervals, while remaining correctly coupled to a fine region evaluated more often; the reported saving in internal-force evaluations relative to a globally synchronous scheme is substantial. Where fracture is present the scheme is no longer strictly conservative, because breaking a bond removes stored energy irreversibly. Evaluating the failure criterion at each potential's own update events, rather than once per global step, keeps the bond history consistent with the multirate structure.

Returns
-------
np.ndarray, an (N, 2) float array of displacements at final_time, in the lattice ordering
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrate_avv(lattice: np.ndarray, bonds: np.ndarray, micromodulus: np.ndarray,
                  critical_stretch_squared: np.ndarray, no_fail_points: np.ndarray,
                  body_force: np.ndarray, local_steps: np.ndarray, final_time: float,
                  density: float, influence_exponent: float,
                  initial_bond_states: np.ndarray) -> np.ndarray:
    '''Integrate the body to final_time with the asynchronous velocity-Verlet scheme.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    critical_stretch_squared : np.ndarray
        (B,) float array of squared critical stretch, one per directed bond.
    no_fail_points : np.ndarray
        (N,) array marking points exempt from failure, 1 for exempt and 0 otherwise.
    body_force : np.ndarray
        (N, 2) float array of external body-force density, constant in time.
    local_steps : np.ndarray
        (N,) float array giving each potential's local update interval.
    final_time : float
        Time at which the integration stops.
    density : float
        Mass density of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.
    initial_bond_states : np.ndarray
        (B,) array of starting bond states, 1 for intact and 0 for broken.

    Returns
    -------
    displacement : np.ndarray
        (N, 2) float array of displacements at final_time, in the lattice ordering.
    '''
    displacement = np.zeros((len(np.asarray(lattice)), 2), dtype=float)
    return displacement  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_integrate_avv(lattice: np.ndarray, bonds: np.ndarray, micromodulus: np.ndarray,
                          critical_stretch_squared: np.ndarray, no_fail_points: np.ndarray,
                          body_force: np.ndarray, local_steps: np.ndarray, final_time: float,
                          density: float, influence_exponent: float,
                          initial_bond_states: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    sc2 = np.asarray(critical_stretch_squared, dtype=float)
    nf = np.asarray(no_fail_points, dtype=float)
    b = np.asarray(body_force, dtype=float)
    hs = np.asarray(local_steps, dtype=float)
    T = float(final_time)
    rho = float(density)
    a = float(influence_exponent)
    mu0 = np.asarray(initial_bond_states, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    N = lat.shape[0]
    if bd.ndim != 2 or bd.shape[1] != 2 or bd.shape[0] < 1:
        raise ValueError("bonds must have shape (B, 2) with B >= 1")
    if bd.min() < 0 or bd.max() >= N:
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != N or not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must have shape (N,) and be finite and > 0")
    if (sc2.ndim != 1 or sc2.shape[0] != bd.shape[0] or not np.all(np.isfinite(sc2))
            or np.any(sc2 <= 0.0)):
        raise ValueError("critical_stretch_squared must have shape (B,) and be finite and > 0")
    if nf.ndim != 1 or nf.shape[0] != N or not np.all((nf == 0.0) | (nf == 1.0)):
        raise ValueError("no_fail_points must have shape (N,) containing only 0 and 1")
    if b.shape != (N, 2) or not np.all(np.isfinite(b)):
        raise ValueError("body_force must have shape (N, 2) and be finite")
    if hs.ndim != 1 or hs.shape[0] != N or not np.all(np.isfinite(hs)) or np.any(hs <= 0.0):
        raise ValueError("local_steps must have shape (N,) and be finite and > 0")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError("final_time must be finite and > 0")
    if not (np.isfinite(rho) and rho > 0.0):
        raise ValueError("density must be finite and > 0")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")
    if mu0.ndim != 1 or mu0.shape[0] != bd.shape[0] or not np.all((mu0 == 0.0) | (mu0 == 1.0)):
        raise ValueError("initial_bond_states must have shape (B,) containing only 0 and 1")

    h_base = hs.min()
    ratio = hs / h_base
    n_i = np.rint(ratio).astype(np.int64)
    if np.any(np.abs(ratio - n_i) > 1e-12 * ratio) or np.any(n_i < 1):
        raise ValueError("every local step must be a positive integer multiple of the smallest")
    M = int(round(T / h_base))
    if M < 1 or abs(M * h_base - T) > 1e-12 * T:
        raise ValueError("final_time must be a positive integer multiple of the smallest local step")
    if np.any(np.abs(np.rint(T / hs) * hs - T) > 1e-12 * T):
        raise ValueError("final_time must be a positive integer multiple of every local step")
    M_i = np.rint(T / hs).astype(np.int64)

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    coef = cm[nbr] * R ** (-a)
    vol_n = lat[nbr, 4]
    vol_o = lat[own, 4]
    allowed = (nf[own] == 0.0) & (nf[nbr] == 0.0)
    step_b = hs[own]
    last_b = M_i[own]

    u = np.zeros((N, 2), dtype=float)
    v = np.zeros((N, 2), dtype=float)
    mu = mu0 == 1.0
    f_minus = np.zeros((N, 2), dtype=float)
    f_plus = np.zeros((N, 2), dtype=float)

    def _deposit(sel, w_minus, w_plus):
        position = lat[:, :2] + u
        d = position[nbr[sel]] - position[own[sel]]
        r = np.hypot(d[:, 0], d[:, 1])
        if np.any(r <= 0.0):
            raise ValueError("bonds must have strictly positive deformed length")
        s = (r - R[sel]) / R[sel]
        brk = mu[sel] & allowed[sel] & (s * s >= sc2[sel])
        if brk.any():
            m = mu[sel]
            m[brk] = False
            mu[sel] = m
        f = (coef[sel] * mu[sel] * s / r)[:, None] * d
        g_own = f * vol_n[sel][:, None]
        g_rea = f * vol_o[sel][:, None]
        if w_minus is not None:
            np.add.at(f_minus, own[sel], w_minus[:, None] * g_own)
            np.add.at(f_minus, nbr[sel], -w_minus[:, None] * g_rea)
        if w_plus is not None:
            np.add.at(f_plus, own[sel], w_plus[:, None] * g_own)
            np.add.at(f_plus, nbr[sel], -w_plus[:, None] * g_rea)

    all_bonds = np.arange(bd.shape[0])
    _deposit(all_bonds, None, step_b)
    for k in range(M):
        v += f_plus / (2.0 * rho) + (h_base / (2.0 * rho)) * b
        u += h_base * v
        f_minus[:] = 0.0
        f_plus[:] = 0.0
        sel = np.nonzero((k + 1) % n_i[own] == 0)[0]
        if sel.size:
            tau = (k + 1) // n_i[own[sel]]
            _deposit(sel, step_b[sel], np.where(tau < last_b[sel], step_b[sel], 0.0))
        v += f_minus / (2.0 * rho) + (h_base / (2.0 * rho)) * b
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the full production configuration ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
def _load(lat, sigma, height):
    b = np.zeros((len(lat), 2)); dx = lat[:, 2]
    top = lat[:, 1] > height - dx; bot = lat[:, 1] < dx
    b[top, 1] = sigma / dx[top]; b[bot, 1] = -sigma / dx[bot]
    return b
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0179)).astype(float)
body_force = _load(lattice, 14e6, 0.02)
local_steps = np.where(lattice[:, 0] < 0.02, 4e-8, 2e-8)
final_time = 2e-5
density = 2440.0
influence_exponent = 2.0
initial_bond_states = np.ones(len(bonds), dtype=float)
""",
            "call": "integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
            "gold_call": "_oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
        },
        # --- Boundary: all potentials on a single common grid ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
def _load(lat, sigma, height):
    b = np.zeros((len(lat), 2)); dx = lat[:, 2]
    top = lat[:, 1] > height - dx; bot = lat[:, 1] < dx
    b[top, 1] = sigma / dx[top]; b[bot, 1] = -sigma / dx[bot]
    return b
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, 0.0041, 0.006)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0059)).astype(float)
body_force = _load(lattice, 30e6, 0.008)
density = 2440.0
influence_exponent = 2.0
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.full(len(lattice), 2e-8)
final_time = 4e-6
""",
            "call": "integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
            "gold_call": "_oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
        },
        # --- Edge: three-to-one ratio between the two local grids ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
def _load(lat, sigma, height):
    b = np.zeros((len(lat), 2)); dx = lat[:, 2]
    top = lat[:, 1] > height - dx; bot = lat[:, 1] < dx
    b[top, 1] = sigma / dx[top]; b[bot, 1] = -sigma / dx[bot]
    return b
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, 0.0041, 0.006)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0059)).astype(float)
body_force = _load(lattice, 30e6, 0.008)
density = 2440.0
influence_exponent = 2.0
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.where(lattice[:, 0] < 0.008, 6e-8, 2e-8)
final_time = 4.2e-6
""",
            "call": "integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
            "gold_call": "_oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
        },
        # --- Edge: integration started from a partially broken bond state ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
def _load(lat, sigma, height):
    b = np.zeros((len(lat), 2)); dx = lat[:, 2]
    top = lat[:, 1] > height - dx; bot = lat[:, 1] < dx
    b[top, 1] = sigma / dx[top]; b[bot, 1] = -sigma / dx[bot]
    return b
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, 0.0041, 0.006)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0059)).astype(float)
body_force = _load(lattice, 30e6, 0.008)
density = 2440.0
influence_exponent = 2.0
initial_bond_states = np.ones(len(bonds), dtype=float)
initial_bond_states[::5] = 0.0
local_steps = np.where(lattice[:, 0] < 0.008, 4e-8, 2e-8)
final_time = 4e-6
""",
            "call": "integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
            "gold_call": "_oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
        },
        # --- Edge: thresholds raised so far that no bond can fail ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
def _load(lat, sigma, height):
    b = np.zeros((len(lat), 2)); dx = lat[:, 2]
    top = lat[:, 1] > height - dx; bot = lat[:, 1] < dx
    b[top, 1] = sigma / dx[top]; b[bot, 1] = -sigma / dx[bot]
    return b
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, 0.0041, 0.006)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0) * 1e12
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0059)).astype(float)
body_force = _load(lattice, 30e6, 0.008)
density = 2440.0
influence_exponent = 2.0
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.where(lattice[:, 0] < 0.008, 4e-8, 2e-8)
final_time = 4e-6
""",
            "call": "integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
            "gold_call": "_oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, density, influence_exponent, initial_bond_states)",
        },
        # --- Invalid: local step not an integer multiple of the smallest ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = np.zeros(len(lattice), dtype=float)
body_force = np.zeros((len(lattice), 2), dtype=float)
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.where(lattice[:, 0] < 0.008, 6.8e-8, 2e-8)
final_time = 4e-6
def run_model():
    try:
        integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 2440.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 2440.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: final time not an integer multiple of every local step ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = np.zeros(len(lattice), dtype=float)
body_force = np.zeros((len(lattice), 2), dtype=float)
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.where(lattice[:, 0] < 0.008, 4e-8, 2e-8)
final_time = 2.5e-8
def run_model():
    try:
        integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 2440.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 2440.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive density ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
def _sc2(lat, bd, cm, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    return 3.0 * Gc / (2.0 * cm[n] * lat[n, 3] ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.008], [0.008, 0.016]], [0.002, 0.001], 0.008, 3.015)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
critical_stretch_squared = _sc2(lattice, bonds, micromodulus, 135.0, 2.0)
no_fail_points = np.zeros(len(lattice), dtype=float)
body_force = np.zeros((len(lattice), 2), dtype=float)
initial_bond_states = np.ones(len(bonds), dtype=float)
local_steps = np.where(lattice[:, 0] < 0.008, 4e-8, 2e-8)
final_time = 4e-6
def run_model():
    try:
        integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 0.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_avv(lattice, bonds, micromodulus, critical_stretch_squared, no_fail_points, body_force, local_steps, final_time, 0.0, 2.0, initial_bond_states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
