"""
Assemble the benchmark convection-diffusion matrix from its velocity field, build its adaptive residual-based sparse approximate inverse column by column and return the Frobenius norm of $AM - I$.

The Frobenius norm $\|AM - I\|_F$ is the quantity the preconditioner minimizes column by column, since $\min \|AM - I\|_F^2 = \sum_{k=1}^{n} \min \|A m_k - e_k\|_2^2$, so it measures how far the adaptive patterns bring $M$ toward the inverse.

Returns
-------
float: the Frobenius norm $\|A M - I\|_F$ of the benchmark preconditioner.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_preconditioner_benchmark(
    n_side: int = 18,
    flow_x: float = 30.0,
    flow_y: float = 60.0,
    flow_center: float = 0.37,
    tolerance: float = 0.1,
    threshold: float = 0.25,
    cap: int = 3,
    max_passes: int = 3,
    *, seed_mode: str = "power",
) -> float:
    r"""Return $\lVert A M - I \rVert_F$ for the adaptive preconditioner of the benchmark matrix.

    The cells have edges $x_i = (i/n)^{1.7}$ and $y_j = (j/n)^{1.3}$ for
    $i, j = 0, \dots, n$ with $n$ = ``n_side``, and midpoint centres
    $(X_i, Y_j)$. Cell speeds are
    $v_x = \text{flow\_x} \, (1 + X_i) \cos(2 \pi Y_j)$ and
    $v_y = \text{flow\_y} \, \sin(2 \pi (X_i - \text{flow\_center}))$. The
    assembly data are

    $$D_x[j,i] = 0.08 \left[ 1 + 0.6 \sin^2(\pi x_i) \cos^2(2 \pi Y_j) \right]$$

    $$D_y[j,i] = 0.12 \left[ 1 + 0.5 \cos^2(\pi X_i) \sin^2(2 \pi y_{j+1}) \right]$$

    $$\text{reaction}[j,i] = 0.1 + 0.05 \cos^2(\pi X_i) \sin^2(2 \pi Y_j)$$

    $$\rho_{\mathrm{left}}[j] = 0.2 + 2 \sin^2(2 \pi Y_j), \quad
    \rho_{\mathrm{right}}[j] = 0.5 + 3 \cos^2(2 \pi Y_j)$$

    $A$ is the finite-volume, $y$-periodic, $x$-Robin operator returned by
    ``assemble_upwind_convection_diffusion``, including its volume scaling.
    $M$ is the right preconditioner whose column $k$ is the column returned by
    ``compute_adaptive_column(A, k, tolerance, threshold, cap, max_passes,
    seed_mode=seed_mode)``, every column being computed independently. Return
    $\lVert A M - I \rVert_F$.

    Parameters
    ----------
    n_side : int
        Number of cells per axis, at least 1.
    flow_x : float
        Finite amplitude of the $x$ velocity component.
    flow_y : float
        Finite amplitude of the $y$ velocity component.
    flow_center : float
        Finite $x$ phase shift in the sine defining the $y$ velocity.
    tolerance : float
        Finite positive column residual tolerance $\epsilon$.
    threshold : float
        Finite nonnegative relative admission threshold $\delta$.
    cap : int
        Maximum number of rows admitted per pass, at least 1.
    max_passes : int
        Maximum number of passes per column, at least 0.
    seed_mode : str
        ``"power"`` for the $e_k$, $A e_k$, $A^2 e_k$ seed supports, or
        ``"diagonal"`` to start every column from its diagonal position
        alone.

    Returns
    -------
    float
        The Frobenius norm $\lVert A M - I \rVert_F$.

    Raises
    ------
    ValueError
        If ``n_side`` is not an integer of at least 1, a flow parameter is not
        a finite number, ``seed_mode`` is neither ``"power"`` nor
        ``"diagonal"``, or any argument passed on to the earlier steps is
        outside their stated domains (booleans are rejected).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_preconditioner_benchmark(
    n_side: int = 18,
    flow_x: float = 30.0,
    flow_y: float = 60.0,
    flow_center: float = 0.37,
    tolerance: float = 0.1,
    threshold: float = 0.25,
    cap: int = 3,
    max_passes: int = 3,
    *, seed_mode: str = "power",
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not (_is_integer(n_side) and n_side >= 1):
        raise ValueError("n_side must be an integer of at least 1")
    for value in (flow_x, flow_y, flow_center):
        if not _is_number(value):
            raise ValueError("flow parameters must be finite numbers")
    if seed_mode not in ("power", "diagonal"):
        raise ValueError("seed_mode must be 'power' or 'diagonal'")
    size = int(n_side)
    edges = np.arange(size + 1, dtype=float) / size
    xe, ye = edges**1.7, edges**1.3
    xc, yc = (xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2
    x_grid, y_grid = np.meshgrid(xc, yc, indexing="xy")
    velocity_x = float(flow_x)*(1+x_grid)*np.cos(2*np.pi*y_grid)
    velocity_y = float(flow_y)*np.sin(2*np.pi*(x_grid-float(flow_center)))
    xf, yf = np.meshgrid(xe, yc, indexing="xy")
    dx_face = 0.08*(1+0.6*np.sin(np.pi*xf)**2*np.cos(2*np.pi*yf)**2)
    xf, yf = np.meshgrid(xc, ye[1:], indexing="xy")
    dy_face = 0.12*(1+0.5*np.cos(np.pi*xf)**2*np.sin(2*np.pi*yf)**2)
    geometry = dict(x_edges=xe, y_edges=ye, diffusion_x=dx_face, diffusion_y=dy_face,
                    reaction=0.1+0.05*np.cos(np.pi*x_grid)**2*np.sin(2*np.pi*y_grid)**2,
                    robin_left=0.2+2*np.sin(2*np.pi*yc)**2,
                    robin_right=0.5+3*np.cos(2*np.pi*yc)**2)
    matrix = _oracle_assemble_upwind_convection_diffusion(velocity_x, velocity_y, fitted=geometry)
    unknowns = matrix.shape[0]
    preconditioner = np.zeros((unknowns, unknowns))
    # Frobenius-norm minimization decouples M into independent columns.
    for k in range(unknowns):
        vector, _, _ = _oracle_compute_adaptive_column(
            matrix, k, tolerance, threshold, cap, max_passes, seed_mode=seed_mode
        )
        preconditioner[:, k] = vector
    defect = matrix @ preconditioner - np.eye(unknowns)
    value = float(np.linalg.norm(defect, "fro"))
    if not np.isfinite(value):
        raise ValueError("the Frobenius norm must be finite")
    return value

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Fitted-grid coverage, both seed modes, and the declared invalid inputs."""
    status = ('import numpy as np\n'
              'def _status(fn):\n'
              '    try:\n'
              '        fn()\n'
              '        return 0\n'
              '    except ValueError:\n'
              '        return 1\n'
              '    except Exception:\n'
              '        return 2\n')
    return [
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(6)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(6)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(8, -25.0, 40.0, 0.55, 0.15, 0.2, 2, 2)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(8, -25.0, 40.0, 0.55, 0.15, 0.2, 2, 2)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(11, 12.0, -50.0, 0.3, 0.05, 0.3, 4, 4)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(11, 12.0, -50.0, 0.3, 0.05, 0.3, 4, 4)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(1)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(1)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(2)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(2)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(9, 0.0, 0.0, 0.5, 0.2, 0.25, 3, 0)',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(9, 0.0, 0.0, 0.5, 0.2, 0.25, 3, 0)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'evaluate_preconditioner_benchmark(6, seed_mode="diagonal")',
            'gold_call': '_oracle_evaluate_preconditioner_benchmark(6, seed_mode="diagonal")',
        },
        {
            'setup': status,
            'call': '_status(lambda: evaluate_preconditioner_benchmark(0))',
            'gold_call': '_status(lambda: _oracle_evaluate_preconditioner_benchmark(0))',
        },
        {
            'setup': status,
            'call': '_status(lambda: evaluate_preconditioner_benchmark(6, 30.0, 60.0, 0.37, -0.1))',
            'gold_call': '_status(lambda: _oracle_evaluate_preconditioner_benchmark(6, 30.0, 60.0, 0.37, -0.1))',
        },
        {
            'setup': status,
            'call': '_status(lambda: evaluate_preconditioner_benchmark(4, seed_mode="both"))',
            'gold_call': '_status(lambda: _oracle_evaluate_preconditioner_benchmark(4, seed_mode="both"))',
        },
    ]
