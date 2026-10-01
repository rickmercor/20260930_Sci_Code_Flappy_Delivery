"""
Build the real 4x4 matrix of one symmetry-respecting two-qubit MERA block from its rotation
angles. Returns the matrix as a float array.

Angles are in units of pi. With Pauli matrices X, Y and the two-qubit basis ordered by the
index 2*q_left + q_right (|00>, |01>, |10>, |11>), define XY(x) = exp(-i pi x X(x)Y / 2) and
YX(x) = exp(-i pi x Y(x)X / 2); both products X(x)Y and Y(x)X square to the identity and are
purely imaginary, so each gate is a real rotation. A two-angle block [x1, x2] is G = YX(x1)
XY(x2) (the two factors commute). A four-angle block [y1, y2, x1, x2] applies Ry(y1) on the
left qubit and Ry(y2) on the right qubit AFTER that product: (Ry(y1) (x) Ry(y2)) G, with
Ry(y) = exp(-i pi y Y / 2) = [[cos(pi y/2), -sin(pi y/2)], [sin(pi y/2), cos(pi y/2)]].
Return the real matrix, not a complex one.

Returns
-------
np.ndarray: real 4x4 block matrix, rows = output basis index, columns = input.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def block_gate(angles: list) -> "np.ndarray":
    """Build the real 4x4 matrix of one symmetry-respecting two-qubit MERA block from its
    rotation angles. Returns the matrix as a float array.

    Args:
        angles: list of 2 floats [x1, x2] or 4 floats [y1, y2, x1, x2], units of pi.

    Returns:
        np.ndarray: real 4x4 block matrix, rows = output basis index, columns = input.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _block_gate(angles):
    a = [float(v) for v in angles]
    x1, x2 = a[-2], a[-1]
    cp, sp = np.cos(np.pi * (x1 + x2) / 2.0), np.sin(np.pi * (x1 + x2) / 2.0)
    cm, sm = np.cos(np.pi * (x2 - x1) / 2.0), np.sin(np.pi * (x2 - x1) / 2.0)
    g = np.array([[cp, 0.0, 0.0, -sp],
                  [0.0, cm, sm, 0.0],
                  [0.0, -sm, cm, 0.0],
                  [sp, 0.0, 0.0, cp]])
    if len(a) == 4:
        c1, s1 = np.cos(np.pi * a[0] / 2.0), np.sin(np.pi * a[0] / 2.0)
        c2, s2 = np.cos(np.pi * a[1] / 2.0), np.sin(np.pi * a[1] / 2.0)
        g = np.kron(np.array([[c1, -s1], [s1, c1]]), np.array([[c2, -s2], [s2, c2]])) @ g
    return g


def _oracle_block_gate(angles: list) -> "np.ndarray":
    return _block_gate(angles)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; angles = [0.16148386263647466, -0.056145691508003415]',
            'call': 'block_gate(*copy.deepcopy((angles,)))',
            'gold_call': '_oracle_block_gate(*copy.deepcopy((angles,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; angles = [-0.061861621487427465, 0.08198065488845328]',
            'call': 'block_gate(*copy.deepcopy((angles,)))',
            'gold_call': '_oracle_block_gate(*copy.deepcopy((angles,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; angles = [-0.08645507352591686, -0.08645536601575388, 0.06035718626700043, 0.06035714232872677]',
            'call': 'block_gate(*copy.deepcopy((angles,)))',
            'gold_call': '_oracle_block_gate(*copy.deepcopy((angles,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; angles = [0.5, -0.35]',
            'call': 'block_gate(*copy.deepcopy((angles,)))',
            'gold_call': '_oracle_block_gate(*copy.deepcopy((angles,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; angles = [0.0, 0.0, 0.25, 0.1]',
            'call': 'block_gate(*copy.deepcopy((angles,)))',
            'gold_call': '_oracle_block_gate(*copy.deepcopy((angles,)))',
        },
    ]
