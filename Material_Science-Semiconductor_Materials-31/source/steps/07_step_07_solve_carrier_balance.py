"""
Assemble and solve the conservative quadratic carrier-balance equations.

Use the tensor-product quadratic nodal field and the joined control intervals.

Within each element, the two internal cuts in either coordinate create six face segments in each normal direction.

For an oriented segment shared by the negative-side node $K$ and positive-side node $L$, the linear functional $J^h+J^t$ enters the matrix with signs $(-,+)$ and $J^f$ enters the load with signs $(+,-)$.

Thus each interior row satisfies

$$

-*\sum_*{e\subset\partial V}(J_e^h+J_e^t)

=\int_Vf\,dA+*\sum_*{e\subset\partial V}J_e^f.

$$

Dirichlet values replace the boundary equations.

Rows and columns use increasing horizontal index inside increasing vertical index.

The returned array includes boundary nodes and has vertical index first.

Returns
-------
A two-dimensional float array containing the solved nodal carrier field with vertical index first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_carrier_balance(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> np.ndarray:
    r"""Assemble and solve the conservative quadratic carrier-balance equations.

    Parameters
    ----------
    x_knots, y_knots : np.ndarray
        Strictly increasing finite primary coordinate vectors, each of length at least two.
    diffusion : float
        Positive dimensionless diffusion coefficient.
    drift : np.ndarray
        Shape $(2,)$; finite constant horizontal and vertical drift.
    source : np.ndarray
        Shape $(4,)$; coefficients $[f_0,f_x,f_y,f_{xy}]$.
    boundary : np.ndarray
        Shape $(3,)$; Dirichlet coefficients $[g_0,g_x,g_y]$.
    partition : float, optional
        Dual cut fraction $p$ in $(0,1/2)$.
    trace_fraction : float, optional
        Positive half-width fraction $r<p$; all face Peclet magnitudes must be at most 1000.

    Returns
    -------
    np.ndarray
        Shape $(2m_y-1,2m_x-1)$, the nodal carrier field including contacts.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements fail, knots are not increasing,
        diffusion is non-positive, $0<r<p<1/2$ fails, a face Peclet magnitude
        exceeds 1000, or the discrete system is singular or has no finite solution.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _transport_data(
    x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
):
    x = _oracle_build_dual_partition(x_knots, partition)
    y = _oracle_build_dual_partition(y_knots, partition)
    alpha = _positive_scalar(diffusion, "diffusion")
    drift = _finite_array(drift, "drift", (2,))
    source = _finite_array(source, "source", (4,))
    boundary = _finite_array(boundary, "boundary", (3,))
    fraction = _positive_scalar(trace_fraction, "trace_fraction")
    if fraction >= partition:
        raise ValueError("trace_fraction must be smaller than partition")
    return x, y, alpha, drift, source, boundary, fraction


def _assemble_carrier_balance(
    x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
):
    x, y, alpha, drift, source, boundary, fraction = _transport_data(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    nx, ny = len(x), len(y)
    matrix = np.zeros((nx * ny, nx * ny))
    load = np.empty(nx * ny)
    for j in range(ny):
        for i in range(nx):
            xm, ym = (x[i, 1] + x[i, 2]) / 2, (y[j, 1] + y[j, 2]) / 2
            area = (x[i, 2] - x[i, 1]) * (y[j, 2] - y[j, 1])
            load[j * nx + i] = area * (
                source[0] + source[1] * xm + source[2] * ym + source[3] * xm * ym
            )
    fractions = np.array([0.0, partition, 1 - partition, 1.0])
    for ey in range((ny - 1) // 2):
        for ex in range((nx - 1) // 2):
            bounds = np.array(
                [x[2 * ex, 0], x[2 * ex + 2, 0], y[2 * ey, 0], y[2 * ey + 2, 0]]
            )
            widths = bounds[[1, 3]] - bounds[[0, 2]]
            lower = bounds[[0, 2]]
            indices = np.array(
                [(2 * ey + j) * nx + 2 * ex + i for j in range(3) for i in range(3)]
            )
            for axis in (0, 1):
                normal, tangent = _oriented_frame(axis)
                ell = fraction * widths[axis]
                kernel = _oracle_compute_normal_kernel(ell, alpha, drift[axis])
                cuts = lower[axis] + widths[axis] * fractions
                other = 1 - axis
                ends = lower[other] + widths[other] * fractions
                for cut in (1, 2):
                    for segment in range(3):
                        center = np.empty(2)
                        center[axis] = cuts[cut]
                        center[other] = (ends[segment] + ends[segment + 1]) / 2
                        length = ends[segment + 1] - ends[segment]
                        coefficients = _oracle_transform_quadratic_traces(
                            bounds, center, axis
                        )
                        flux = _oracle_compute_homogeneous_face(
                            coefficients, ell, length, alpha, kernel
                        )
                        flux += _oracle_compute_transverse_face(
                            coefficients,
                            length,
                            alpha,
                            float(drift @ tangent),
                            kernel[2:],
                        )
                        correction = _oracle_compute_source_face(
                            source, center, axis, length, kernel[2:]
                        )
                        if axis == 0:
                            left = (2 * ey + segment) * nx + 2 * ex + cut - 1
                            right = left + 1
                        else:
                            left = (2 * ey + cut - 1) * nx + 2 * ex + segment
                            right = left + nx
                        matrix[left, indices] -= flux
                        matrix[right, indices] += flux
                        load[left] += correction
                        load[right] -= correction
    for j in range(ny):
        for i in range(nx):
            if i in (0, nx - 1) or j in (0, ny - 1):
                index = j * nx + i
                matrix[index] = 0
                matrix[index, index] = 1
                load[index] = (
                    boundary[0] + boundary[1] * x[i, 0] + boundary[2] * y[j, 0]
                )
    return matrix, load


def _oracle_solve_carrier_balance(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    matrix, load = _assemble_carrier_balance(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    try:
        result = np.linalg.solve(matrix, load)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the discrete system is singular") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("the discrete solution must be finite")
    return result.reshape(2 * len(y_knots) - 1, 2 * len(x_knots) - 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .22, .57, 1.])\ny = np.array([0., .31, .64, 1.])\nalpha = .03\nbeta = np.array([1.7, -.9])\nf = np.array([1.,1.,2.,3.])\ng = np.array([1.,.2,.3])\np, r = .27, .18\n",
            "call": "solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .4, 1.]); y = np.array([0., .6, 1.])\nalpha = .1; beta = np.zeros(2)\nf = np.array([2., 0., 0., 0.]); g = np.zeros(3)\np, r = 1/3, .2\n",
            "call": "solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .15, .7, 1.]); y = np.array([0., .2, .8, 1.])\nalpha = .025; beta = np.array([-.8,1.4])\nf = np.array([.5,2.,-.5,1.]); g = np.array([.8,.1,.2])\np, r = .31, .22\n",
            "call": "solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_solve_carrier_balance(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .22, .57, 1.])\ny = np.array([0., .31, .64, 1.])\nalpha = .03\nbeta = np.array([1.7, -.9])\nf = np.array([1.,1.,2.,3.])\ng = np.array([1.,.2,.3])\np, r = .27, .18\nr = p\ndef _exception_code(function):\n    try:\n        function(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(solve_carrier_balance)",
            "gold_call": "_exception_code(_oracle_solve_carrier_balance)",
        },
    ]
