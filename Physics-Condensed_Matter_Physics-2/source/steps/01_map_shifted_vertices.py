"""
Map an XYZ spin interaction and its quadratic counterterm.

The three-Majorana spin representation permits a shifted expansion

without projecting away the copy Hilbert space.

Returns
-------
complex ndarray (V,6): ordered residual vertices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def map_shifted_vertices(
    couplings: np.ndarray, hopping: np.ndarray
) -> np.ndarray:
    """Encode V=H_Maj-H0 as ordered even Majorana monomials.

    Parameters
    ----------
    couplings : ndarray, shape (3,L,L)
        Finite real symmetric zero-diagonal XYZ couplings, L even and >=2.
        H_spin=sum_{a=0}^2 sum_{j<k} couplings[a,j,k] S_j^a S_k^a.
        Color order is x,y,z; hbar=1 and S=Pauli/2.
    hopping : ndarray, shape (3,L,L)
        Finite real skew-symmetric tensor, the same shape as couplings.
        H0=(i/2) sum_{a,j,k} hopping[a,j,k] rho_{aL+j} rho_{aL+k}.
        There is no scalar offset. The supplied reference is held fixed;
        it is not assumed to solve a self-consistency equation.

    Returns
    -------
    vertices : complex ndarray, shape (V,6)
        Each row is [coefficient, degree, i1, i2, i3, i4] and means
        coefficient*rho_i1*...*rho_i_degree in precisely that order.
        Majoranas satisfy {rho_p,rho_q}=2 delta_pq and are color-major.
        Use S_j^a=-(i/2) rho_{bL+j} rho_{cL+j}, with cyclic (a,b,c).
        First emit the nonzero spin bonds in increasing (a,j,k), j<k,
        retaining the spin-product field order (b,j),(c,j),(b,k),(c,k).
        Then emit the nonzero counterterm bonds in the same index order.
        They have degree two; unused indices are -1. Retain every nonzero
        coupling separately; an empty table has shape (0,6).

    Raises
    ------
    ValueError
        For nonfinite, nonreal, incorrectly shaped or incorrectly symmetric
        inputs, nonzero diagonals, odd L, L<2, or unequal tensor shapes.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def _numeric(value, name, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if real and np.any(a.imag != 0):
        raise ValueError(f"{name} must be real")
    return a.real if real else a


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} is outside the supported range")
    return int(value)


def _temperature(beta):
    b = _numeric(beta, "beta", True)
    if b.ndim != 0 or b <= 0:
        raise ValueError("beta must be positive")
    return float(b)


def _tensor(value, name, antisymmetric):
    a = _numeric(value, name, True)
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"{name} must have shape (3,L,L)")
    n = a.shape[1]
    if n < 2 or n % 2:
        raise ValueError("L must be positive and even, at least two")
    target = -a.swapaxes(-1, -2) if antisymmetric else a.swapaxes(-1, -2)
    if not np.allclose(a, target, atol=1e-12, rtol=0):
        raise ValueError(f"{name} has the wrong exchange symmetry")
    if np.max(np.abs(np.diagonal(a, axis1=1, axis2=2))) > 1e-12:
        raise ValueError(f"{name} must have a zero diagonal")
    return a


def _observable(sites, component, tau, beta, n):
    pair = _numeric(sites, "sites", True)
    if pair.shape != (2,) or np.any(pair != np.floor(pair)):
        raise ValueError("sites must contain two integer indices")
    if np.any((pair < 0) | (pair >= n)):
        raise ValueError("site index is out of range")
    color = _integer(component, "component", 0, 2)
    time = _numeric(tau, "tau", True)
    if time.ndim != 0 or not 0 <= time <= beta:
        raise ValueError("tau must lie in [0,beta]")
    return pair.astype(int), color, float(time)


def _vertices(value, dimension):
    v = _numeric(value, "vertices")
    if v.ndim != 2 or v.shape[1] != 6:
        raise ValueError("vertices must have shape (V,6)")
    for row in v:
        if np.any(row[1:].imag != 0) or np.any(
            row[1:].real != np.floor(row[1:].real)
        ):
            raise ValueError("vertex degrees and indices must be integers")
        degree = int(row[1].real)
        if degree not in (2, 4):
            raise ValueError("vertex degree must be two or four")
        indices = row[2 : 2 + degree].real
        if np.any((indices < 0) | (indices >= dimension)):
            raise ValueError("Majorana index is out of range")
        if np.any(row[2 + degree :] != -1):
            raise ValueError("unused vertex indices must be -1")
    return v


def _oracle_map_shifted_vertices(
    couplings: np.ndarray, hopping: np.ndarray
) -> np.ndarray:
    j = _tensor(couplings, "couplings", False)
    a = _tensor(hopping, "hopping", True)
    if j.shape != a.shape:
        raise ValueError("couplings and hopping must have identical shapes")
    n = j.shape[1]
    rows = []
    for color in range(3):
        b, c = (color + 1) % 3, (color + 2) % 3
        for left in range(n):
            for right in range(left + 1, n):
                if j[color, left, right] != 0:
                    rows.append(
                        [
                            -j[color, left, right] / 4,
                            4,
                            b * n + left,
                            c * n + left,
                            b * n + right,
                            c * n + right,
                        ]
                    )
    for color in range(3):
        for left in range(n):
            for right in range(left + 1, n):
                if a[color, left, right] != 0:
                    rows.append(
                        [
                            -1j * a[color, left, right],
                            2,
                            color * n + left,
                            color * n + right,
                            -1,
                            -1,
                        ]
                    )
    return np.asarray(rows, complex).reshape(-1, 6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    ("Return normal, boundary, edge and " "invalid controls.")
    return [
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.array(
    [
        [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.1, 0.0],
            [0.0, 1.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.7, 0.0],
        ],
    ]
)
h = np.array(
    [
        [
            [0.0, 0.23, 0.0, 0.0],
            [-0.23, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, -0.31, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.31, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.19],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-0.19, 0.0, 0.0, 0.0],
        ],
    ]
)
beta = 2.4
tau = 0.83
sites = np.array([0, 0])
component = 2
""",
            "call": "map_shifted_vertices(j.copy(), h.copy())",
            "gold_call": "_oracle_map_shifted_vertices(j.copy(), h.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.array(
    [
        [
            [0.0, 0.8, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.6, 0.0],
            [0.0, -0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.55],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.75],
            [0.55, 0.0, 0.75, 0.0],
        ],
    ]
)
h = np.array(
    [
        [
            [0.0, 0.21, 0.0, 0.0],
            [-0.21, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, -0.13, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.13, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, -0.12],
            [0.0, 0.0, 0.0, 0.18],
            [0.0, 0.0, 0.0, 0.0],
            [0.12, -0.18, 0.0, 0.0],
        ],
    ]
)
beta = 1.8
tau = 0.47
sites = np.array([0, 3])
component = 2
""",
            "call": "map_shifted_vertices(j.copy(), h.copy())",
            "gold_call": "_oracle_map_shifted_vertices(j.copy(), h.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.array(
    [
        [[0.0, 0.7], [0.7, 0.0]],
        [[0.0, 1.2], [1.2, 0.0]],
        [[0.0, -0.3], [-0.3, 0.0]],
    ]
)
h = np.array(
    [
        [[0.0, 0.19], [-0.19, 0.0]],
        [[0.0, -0.27], [0.27, 0.0]],
        [[0.0, 0.13], [-0.13, 0.0]],
    ]
)
beta = 2.1
tau = 0.68
sites = np.array([0, 1])
component = 0
""",
            "call": "map_shifted_vertices(j.copy(), h.copy())",
            "gold_call": "_oracle_map_shifted_vertices(j.copy(), h.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.array(
    [
        [[0.0, 0.7], [0.7, 0.0]],
        [[0.0, 1.2], [1.2, 0.0]],
        [[0.0, -0.3], [-0.3, 0.0]],
    ]
)
h = np.array(
    [
        [[0.0, 0.19], [-0.19, 0.0]],
        [[0.0, -0.27], [0.27, 0.0]],
        [[0.0, 0.13], [-0.13, 0.0]],
    ]
)
beta = 2.1
tau = 0.68
sites = np.array([0, 1])
component = 0
j[:] = 0
h[:] = 0
""",
            "call": "map_shifted_vertices(j.copy(), h.copy())",
            "gold_call": "_oracle_map_shifted_vertices(j.copy(), h.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.array(
    [
        [[0.0, 0.7], [0.7, 0.0]],
        [[0.0, 1.2], [1.2, 0.0]],
        [[0.0, -0.3], [-0.3, 0.0]],
    ]
)
h = np.array(
    [
        [[0.0, 0.19], [-0.19, 0.0]],
        [[0.0, -0.27], [0.27, 0.0]],
        [[0.0, 0.13], [-0.13, 0.0]],
    ]
)
beta = 2.1
tau = 0.68
sites = np.array([0, 1])
component = 0
j[0, 0, 1] += 1


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "_expect_value_error(lambda: " "map_shifted_vertices(j, h))"
            ),
            "gold_call": (
                "_expect_value_error(lambda: "
                "_oracle_map_shifted_vertices(j, h))"
            ),
            "tol": 1e-09,
        },
    ]
