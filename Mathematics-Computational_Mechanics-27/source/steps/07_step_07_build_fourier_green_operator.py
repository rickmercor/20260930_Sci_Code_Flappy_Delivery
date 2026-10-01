"""
This step builds the multiplier once, for a given voxel count and cell edge, so that the solver can apply the inverse of the constant-coefficient operator by a forward transform, a pointwise product and an inverse transform.

Only one voxel is assembled. Every voxel of the periodic mesh is a translate of that reference voxel, which is what makes the operator a convolution, so the whole preconditioner is determined by a matrix of twenty-four rows and twenty-four columns together with the corner offsets. The multiplier carries no material data: it is a preconditioner, so its quality but not the solution it converges to depends on how well it resembles the real operator.

The preconditioner of the enriched system is block diagonal: the identity on the enriched degrees of freedom and, on the standard degrees of freedom, the constant-coefficient operator $(A^0)_{ij} = \int_Y \operatorname{sym} \operatorname{grad}(N_i) : \operatorname{sym} \operatorname{grad}(N_j) dV$, which uses the symmetrised gradient so that it is consistent with the internal scaling and carries no material data at all. On a regular periodic grid this operator is a convolution, so its inverse is a matrix-valued multiplier in Fourier space and can be applied in $\mathcal{O}(n^3 \log n)$ operations.

Assembling the operator for one reference voxel gives a matrix $A^0_v$ of twenty-four rows and twenty-four columns over the eight voxel corners. For a frequency $k$ the eight corner phases $c_j(k) = \exp(2 \pi i Y_j \cdot k / n)$, with $Y$ the corner offsets in voxel units, build the matrix $Z(k)$ of twenty-four rows and three columns whose $j$-th block is $c_j(k) I_3$, and the multiplier is $\widehat{G}(k) = [Z(k)^{H} A^0_v Z(k)]^{-1}$ with $\widehat{G}(0) = 0$. The zero frequency is annihilated rather than inverted, which enforces the mean-free condition on the standard displacement fluctuation.

Returns
-------
np.ndarray, the Fourier Green operator at every discrete frequency.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_fourier_green_operator(n_voxels: int, cell_size: float):
    """Build the block Fourier multiplier of the inverse constant-coefficient operator.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge, at least two.
    cell_size : float
        Edge length of the cubic periodic cell, strictly positive.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (n_voxels, n_voxels, n_voxels, 3, 3).

    Raises
    ------
    ValueError
        If n_voxels is below two, or if cell_size is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

R2 = np.sqrt(2.0)
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
TETRAHEDRA = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))

def _strain_column(gradient, direction):
    """Mandel strain of the vector field N e_direction whose scalar gradient is given."""
    gx, gy, gz = gradient
    if direction == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / R2, gy / R2])
    if direction == 1:
        return np.array([0.0, gy, 0.0, gz / R2, 0.0, gx / R2])
    return np.array([0.0, 0.0, gz, gy / R2, gx / R2, 0.0])

def _shape_gradients(vertices):
    """Gradients of the four linear basis functions of a tetrahedron."""
    return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T


def _voxel_reference_stiffness(spacing):
    """Constant-coefficient voxel stiffness of the six linear tetrahedra, Mandel identity."""
    matrix = np.zeros((24, 24))
    corners = spacing * CORNERS
    for tetrahedron in TETRAHEDRA:
        vertices = corners[list(tetrahedron)]
        gradients = _shape_gradients(vertices)
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        operator = np.zeros((6, 12))
        for node in range(4):
            for direction in range(3):
                operator[:, 3 * node + direction] = _strain_column(gradients[node], direction)
        local = volume * (operator.T @ operator)
        index = np.array([3 * tetrahedron[node] + direction
                          for node in range(4) for direction in range(3)])
        matrix[np.ix_(index, index)] += local
    return matrix


def _green(n_voxels, cell_size):
    """Block Fourier representation of the inverse constant-coefficient operator."""
    spacing = float(cell_size) / int(n_voxels)
    voxel = _voxel_reference_stiffness(spacing)
    green = np.zeros((n_voxels, n_voxels, n_voxels, 3, 3), dtype=complex)
    frequencies = np.arange(n_voxels)
    for k1 in frequencies:
        for k2 in frequencies:
            for k3 in frequencies:
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue
                phase = np.exp(2j * np.pi * (CORNERS @ np.array([k1, k2, k3])) / n_voxels)
                projector = np.zeros((24, 3), dtype=complex)
                for corner in range(8):
                    projector[3 * corner:3 * corner + 3, :] = phase[corner] * np.eye(3)
                green[k1, k2, k3] = np.linalg.inv(
                    projector.conj().T @ voxel @ projector)
    return green


def _oracle_build_fourier_green_operator(n_voxels: int, cell_size: float):
    """Reference implementation."""
    if int(n_voxels) < 2:
        raise ValueError("n_voxels must be at least two")
    if float(cell_size) <= 0.0:
        raise ValueError("cell_size must be strictly positive")
    return _green(int(n_voxels), float(cell_size))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
def summarize(g):
    return (g.shape, round(float(np.abs(g[0, 0, 0]).max()), 14),
            tuple(round(float(x), 12) for x in np.real(g[1, 1, 1]).ravel()),
            round(float(np.abs(np.imag(g[1, 1, 1])).max()), 12))
""",
            "call": "summarize(build_fourier_green_operator(4, 1.0))",
            "gold_call": "summarize(_oracle_build_fourier_green_operator(4, 1.0))",
        },
        {
            "setup": """import numpy as np
def scaling(fn):
    # the operator has the dimensions of an inverse length, so halving the cell edge at
    # fixed voxel count doubles every multiplier
    a = fn(4, 1.0)
    b = fn(4, 0.5)
    hermitian = np.abs(a - np.conj(np.transpose(a, (0, 1, 2, 4, 3)))).max()
    return (round(float(np.abs(b - 2.0 * a).max()), 12), round(float(hermitian), 12))
""",
            "call": "scaling(build_fourier_green_operator)",
            "gold_call": "scaling(_oracle_build_fourier_green_operator)",
        },
        {
            "setup": """def run(fn):
    codes = []
    for args in [(1, 1.0), (4, 0.0)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(build_fourier_green_operator)",
            "gold_call": "run(_oracle_build_fourier_green_operator)",
        },
    ]
