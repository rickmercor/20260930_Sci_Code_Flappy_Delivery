"""
Fourier-space inverse of the homogeneous periodic P1 operator.




The X-FFT block preconditioner uses the constant-coefficient standard finite-element operator and an identity block for scaled enrichment degrees of freedom. Six P1 tetrahedra form the reference voxel. For every nonzero discrete frequency, voxel corner phase factors reduce its 24 by 24 stiffness to a 3 by 3 Hermitian symbol whose pseudoinverse is Green's operator. The zero-frequency symbol is set to zero to enforce the mean-free displacement fluctuation.




Inputs

------

n_voxels : int

    Number of periodic voxels on each cell edge.




Returns

-------

green : np.ndarray of shape (n, n, n, 3, 3)

    Complex Fourier multipliers for the standard displacement block.

Returns
-------
complex128 array with shape (n, n, n, 3, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_fourier_green_operator(n_voxels: int) -> np.ndarray:
    """Build the cached Fourier Green operator for the standard FE block.

    Parameters
    ----------
    n_voxels : int
        Number of equal periodic voxels along each axis, at least two.

    Returns
    -------
    green : np.ndarray
        Complex array of 3 by 3 inverse symbols at all discrete frequencies.
    Raises
    ------
    ValueError
        If n_voxels is not an integer of at least two.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=int,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=int,
)

def _shape_gradients(vertices):
    interpolation = np.column_stack([np.ones(4), vertices])
    return np.linalg.inv(interpolation)[1:, :].T


def _mandel_column(gradient, component):
    gx, gy, gz = gradient
    root_two = np.sqrt(2.0)
    if component == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / root_two, gy / root_two])
    if component == 1:
        return np.array([0.0, gy, 0.0, gz / root_two, 0.0, gx / root_two])
    return np.array([0.0, 0.0, gz, gy / root_two, gx / root_two, 0.0])


def _reference_voxel_stiffness(voxel_edge):
    coordinates = voxel_edge * _CUBE_VERTICES
    stiffness = np.zeros((24, 24), dtype=float)
    for tetrahedron in _CUBE_TETS:
        vertices = coordinates[tetrahedron]
        gradients = _shape_gradients(vertices)
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        strain_matrix = np.zeros((6, 12), dtype=float)
        for local_node in range(4):
            for component in range(3):
                strain_matrix[:, 3 * local_node + component] = _mandel_column(
                    gradients[local_node], component
                )
        local_stiffness = volume * (strain_matrix.T @ strain_matrix)
        dofs = np.array(
            [
                3 * int(node) + component
                for node in tetrahedron
                for component in range(3)
            ]
        )
        stiffness[np.ix_(dofs, dofs)] += local_stiffness
    return stiffness


def _oracle_build_fourier_green_operator(n_voxels: int) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(n_voxels, (int, np.integer)) or int(n_voxels) < 2:
        raise ValueError("n_voxels must be an integer of at least two")
    n_voxels = int(n_voxels)
    voxel_stiffness = _reference_voxel_stiffness(1.0 / n_voxels)
    green = np.zeros(
        (n_voxels, n_voxels, n_voxels, 3, 3),
        dtype=complex,
    )
    for frequency in np.ndindex(n_voxels, n_voxels, n_voxels):
        if frequency == (0, 0, 0):
            continue
        wavevector = 2.0 * np.pi * np.asarray(frequency) / n_voxels
        phases = np.exp(1j * (_CUBE_VERTICES @ wavevector))
        phase_matrix = np.zeros((24, 3), dtype=complex)
        for node in range(8):
            phase_matrix[3 * node : 3 * node + 3] = phases[node] * np.eye(3)
        symbol = phase_matrix.conj().T @ voxel_stiffness @ phase_matrix
        green[frequency] = np.linalg.pinv(symbol, rcond=1e-12)
    return green

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
def summarize(green):
    return (
        green.shape,
        np.round(green[1, 1, 1].real, 12).tolist(),
        round(float(np.max(np.abs(green - np.swapaxes(green.conj(), -1, -2)))), 14),
        round(float(np.linalg.norm(green[0, 0, 0])), 14),
    )
""",
            "call": "summarize(build_fourier_green_operator(3))",
            "gold_call": "summarize(_oracle_build_fourier_green_operator(3))",
        },
        {
            "setup": """import numpy as np
def summarize(green):
    return (green.shape, round(float(np.trace(green[1, 0, 0]).real), 12))
""",
            "call": "summarize(build_fourier_green_operator(2))",
            "gold_call": "summarize(_oracle_build_fourier_green_operator(2))",
        },
        {
            "setup": """def run_model():
    try:
        build_fourier_green_operator(1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_fourier_green_operator(1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
