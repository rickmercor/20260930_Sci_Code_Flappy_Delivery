"""
Implement free_transport_step to stream particles along their characteristic
lines and compute their microscopic numerical flux at periodic cell interfaces.

Particles leaving the domain [0, 1) wrap around periodically, so transport
through a domain boundary does not remove a particle. Each particle carries
its own mass. The time step permits at most one interface crossing per
particle. Update particle positions and return the signed mass current at
each periodic interface.

Returns
-------
tuple[np.ndarray, np.ndarray], (x_new, H_mi_free), wrapped positions and the signed microscopic interface current, shape (N_x,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def free_transport_step(
    x_p: np.ndarray,
    xi_p: np.ndarray,
    w_p: np.ndarray,
    dx: float,
    N_x: int,
    dt: float,
    t_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_p : numpy.ndarray
        Particle positions, shape (n_particles,).
    xi_p : numpy.ndarray
        Particle velocities xi = v * mu, shape (n_particles,). Range [-1, 1]
        for v = 1.
    w_p : numpy.ndarray
        Per-particle mass, shape (n_particles,).
    dx : float
        Cell width.
    N_x : int
        Number of cells.
    dt : float
        Time step size.
    t_f : numpy.ndarray
        Free transport time for each particle, shape (n_particles,).

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (x_new, H_mi_free) where x_new is the updated particle positions
        (wrapped to [0, 1)), and H_mi_free is the microscopic flux at each
        periodic interface, shape (N_x,).

    Raises
    ------
    ValueError
        If N_x < 1, or dx or dt is not positive.
    """
    return x_new, H_mi_free

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_free_transport_step(
    x_p: np.ndarray,
    xi_p: np.ndarray,
    w_p: np.ndarray,
    dx: float,
    N_x: int,
    dt: float,
    t_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream particles with periodic wrapping and O(1) interface check."""
    if N_x < 1 or dx <= 0 or dt <= 0:
        raise ValueError("need N_x >= 1 and positive dx and dt")
    n = len(x_p)
    H_mi_free = np.zeros(N_x)
    x_new = np.empty(n)

    for p in range(n):
        x0 = x_p[p]
        xi = xi_p[p]
        w = w_p[p]
        x1 = x0 + xi * t_f[p]
        crossed_periodic_face = x1 < 0.0 or x1 >= 1.0
        if x1 < 0.0:
            x1 += 1.0
        elif x1 >= 1.0:
            x1 -= 1.0
        cell0 = int(x0 / dx) % N_x
        cell1 = int(x1 / dx) % N_x
        if cell1 != cell0 or crossed_periodic_face:
            if xi > 0.0:
                H_mi_free[(cell0 + 1) % N_x] += w / dt
            elif xi < 0.0:
                H_mi_free[cell0] -= w / dt
        x_new[p] = x1

    return x_new, H_mi_free

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # One-cell periodic mesh: wrapping still contributes an interface flux
        {
            "setup": """import numpy as np
dx = 1.0
N_x = 1
x_p = np.array([0.9])
xi_p = np.array([1.0])
w_p = np.array([0.01])
t_f = np.array([0.2])
dt = 0.2
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
        # Normal: two particles moving in opposite directions within domain
        {
            "setup": """import numpy as np
dx = 0.5
N_x = 2
x_p = np.array([0.25, 0.75])
xi_p = np.array([1.0, -1.0])
w_p = np.array([0.01, 0.01])
t_f = np.array([0.1, 0.1])
dt = 0.1
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
        # Boundary: particle wraps around right edge periodically
        {
            "setup": """import numpy as np
dx = 0.25
N_x = 4
x_p = np.array([0.9])
xi_p = np.array([1.0])
w_p = np.array([0.005])
t_f = np.array([0.2])
dt = 0.2
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
        # Edge: particle wraps around left edge periodically
        {
            "setup": """import numpy as np
dx = 0.25
N_x = 4
x_p = np.array([0.05])
xi_p = np.array([-1.0])
w_p = np.array([0.005])
t_f = np.array([0.1])
dt = 0.1
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
        # Normal: interior interface crossed in opposite directions by two particles, two non-crossing particles
        {
            "setup": """import numpy as np
dx = 0.25
N_x = 4
x_p = np.array([0.24, 0.26, 0.60, 0.51])
xi_p = np.array([1.0, -1.0, 0.5, -0.1])
w_p = np.array([0.01, 0.02, 0.03, 0.04])
t_f = np.array([0.05, 0.05, 0.05, 0.05])
dt = 0.05
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
        # Edge: partial flight, the collisional particle stops before the interface it would otherwise cross
        {
            "setup": """import numpy as np
dx = 0.25
N_x = 4
x_p = np.array([0.24, 0.24])
xi_p = np.array([1.0, 1.0])
w_p = np.array([0.01, 0.01])
t_f = np.array([0.005, 0.05])
dt = 0.05
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 6) for a in _oracle_free_transport_step(x_p.copy(), xi_p.copy(), w_p.copy(), dx, N_x, dt, t_f.copy()))",
        },
    ]
