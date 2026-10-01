"""
End-to-end meshfree Poisson experiment on the frozen point cloud.

Run the whole pipeline on the frozen point cloud of the problem statement and report the volume-weighted relative discrete L2 error of the meshfree solution against the exact one. Build the cloud from a ``grid_size`` by ``grid_size`` tensor grid of equally spaced coordinates on the unit square, listed with the first coordinate varying slowest. A node is a boundary node when either of its grid coordinates is 0 or 1. With ``h`` the grid spacing, every interior node is displaced by ``amplitude * h * sin(6 x + 2 y)`` in the first coordinate and ``amplitude * h * cos(2 x + 5 y)`` in the second, both evaluated at the undisplaced coordinates; boundary nodes are not displaced. The graph radius is ``epsilon_factor * h``. Solve the discrete conservation law with the manufactured exact solution ``sin(pi x) sin(pi y)``, so that the problem being solved is ``-Laplacian u = f`` with forcing ``f = 2 pi^2 sin(pi x) sin(pi y)``, and the exact solution supplying the Dirichlet data. Return a native Python float.

Returns
-------
error : float Volume-weighted relative discrete L2 error of the meshfree solution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pointcloud_poisson_error(
    grid_size: int, amplitude: float, epsilon_factor: float, domain_measure: float
) -> float:
    '''End-to-end meshfree Poisson experiment on the frozen point cloud.

    Parameters
    ----------
    grid_size : int
        Number of grid points per coordinate direction.
    amplitude : float
        Interior displacement amplitude in units of the grid spacing.
    epsilon_factor : float
        Graph radius in units of the grid spacing.
    domain_measure : float
        Measure of the unit square.

    Returns
    -------
    error : float
        Volume-weighted relative discrete L2 error of the meshfree solution.

    Raises
    ------
    ValueError
        If ``grid_size`` is not an integer of at least 3, if any scalar
        parameter is nonfinite, or if ``epsilon_factor`` or
        ``domain_measure`` is not positive.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pointcloud_poisson_error(grid_size, amplitude, epsilon_factor, domain_measure):
    import numpy as np

    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)):
        raise ValueError("grid_size must be an integer")
    k = int(grid_size)
    if k < 3:
        raise ValueError("grid_size must be at least 3")
    try:
        amp = float(amplitude)
        eps_factor = float(epsilon_factor)
        measure = float(domain_measure)
    except (TypeError, ValueError) as exc:
        raise ValueError("scalar parameters must be real-valued") from exc
    if (np.ndim(amplitude) != 0 or np.ndim(epsilon_factor) != 0
            or np.ndim(domain_measure) != 0):
        raise ValueError("amplitude, epsilon_factor and domain_measure must be scalars")
    if not np.isfinite(amp):
        raise ValueError("amplitude must be finite")
    if not np.isfinite(eps_factor) or eps_factor <= 0.0:
        raise ValueError("epsilon_factor must be finite and positive")
    if not np.isfinite(measure) or measure <= 0.0:
        raise ValueError("domain_measure must be finite and positive")
    xs = np.linspace(0.0, 1.0, k)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P0 = np.stack([X.ravel(), Y.ravel()], axis=1)
    h = 1.0 / (k - 1)
    flags = (np.isclose(P0[:, 0], 0.0) | np.isclose(P0[:, 0], 1.0) |
             np.isclose(P0[:, 1], 0.0) | np.isclose(P0[:, 1], 1.0))
    P = P0.copy()
    P[~flags, 0] += amp * h * np.sin(6.0 * P0[~flags, 0] + 2.0 * P0[~flags, 1])
    P[~flags, 1] += amp * h * np.cos(2.0 * P0[~flags, 0] + 5.0 * P0[~flags, 1])
    eps = eps_factor * h

    volumes = _oracle_assemble_virtual_node_volumes(P, flags, eps, measure)
    system = _oracle_assemble_moment_constraint_system(P, flags, eps, volumes)
    areas = _oracle_solve_edge_areas(P, eps, system)
    d0 = _oracle_build_coboundary(P, eps)
    K = _oracle_assemble_hodge_laplacian(d0, areas)
    exact = np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
    forcing = 2.0 * np.pi ** 2 * exact
    u = _oracle_solve_dirichlet_problem(K, volumes, flags, forcing, exact)
    return float(_oracle_volume_weighted_relative_error(u, exact, volumes, flags))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": '',
            "call": 'pointcloud_poisson_error(9, 0.18, 2.3, 1.0)',
            "gold_call": '_oracle_pointcloud_poisson_error(9, 0.18, 2.3, 1.0)',
        },
        {
            "setup": '',
            "call": 'pointcloud_poisson_error(7, 0.18, 2.3, 1.0)',
            "gold_call": '_oracle_pointcloud_poisson_error(7, 0.18, 2.3, 1.0)',
        },
        {
            "setup": '',
            "call": 'pointcloud_poisson_error(5, 0.18, 2.6, 1.0)',
            "gold_call": '_oracle_pointcloud_poisson_error(5, 0.18, 2.6, 1.0)',
        },
        {
            "setup": '',
            "call": 'pointcloud_poisson_error(9, 0.0, 2.3, 1.0)',
            "gold_call": '_oracle_pointcloud_poisson_error(9, 0.0, 2.3, 1.0)',
        },
        {
            "setup": (
                'def _value_error(thunk):\n'
                '    try:\n'
                '        thunk()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                '    return 0\n'
            ),
            "call": '_value_error(lambda: pointcloud_poisson_error(2, 0.18, 2.3, 1.0))',
            "gold_call": '_value_error(lambda: _oracle_pointcloud_poisson_error(2, 0.18, 2.3, 1.0))',
        },
    ]
