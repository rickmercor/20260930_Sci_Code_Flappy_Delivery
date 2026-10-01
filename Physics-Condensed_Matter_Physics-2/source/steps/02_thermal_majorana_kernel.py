"""
Compute equilibrium Majorana contractions at arbitrary imaginary times.

The thermal propagator must retain zero modes, the fermionic time-order

sign, and the one-sided equal-time contraction.

Returns
-------
complex ndarray (T,T,3L,3L): ordinary thermal contractions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def thermal_majorana_kernel(
    hopping: np.ndarray, beta: float, times: np.ndarray
) -> np.ndarray:
    """Compute the color-preserving free thermal two-point kernel.

    Parameters
    ----------
    hopping : ndarray, shape (3,L,L)
        Finite real skew-symmetric zero-diagonal matrices, L even and >=2.
        H0=(i/2) sum_{a,j,k} hopping[a,j,k] rho_{aL+j} rho_{aL+k},
        with {rho_p,rho_q}=2 delta_pq and color-major indexing.
    beta : float
        Finite positive inverse temperature.
    times : ndarray, shape (T,)
        Nonempty finite real vector in [0,beta]. Unsorted or repeated
        times are allowed. Imaginary-time evolution is exp(t H0).

    Returns
    -------
    kernel : complex ndarray, shape (T,T,3L,3L)
        For positive times[u]-times[v], the entry [u,v,p,q] is
        <rho_p(times[u]) rho_q(times[v])>_0. For a negative difference
        it is -<rho_q(times[v]) rho_p(times[u])>_0. At equal times it
        is the ordered contraction <rho_p rho_q>_0, including diagonal
        value one. This kernel has no extra factor -i. Cross-color
        entries vanish. Include the full Gaussian thermal ensemble.

    Raises
    ------
    ValueError
        For invalid tensor shape/symmetry, L, nonfinite or nonreal inputs,
        beta<=0, or empty, nonvector or out-of-interval times.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_thermal_majorana_kernel(
    hopping: np.ndarray, beta: float, times: np.ndarray
) -> np.ndarray:
    a = _tensor(hopping, "hopping", True)
    beta = _temperature(beta)
    t = _numeric(times, "times", True)
    if t.ndim != 1 or t.size == 0 or np.any((t < 0) | (t > beta)):
        raise ValueError("times must be a nonempty vector in [0,beta]")
    n = a.shape[1]
    delta = t[:, None] - t[None, :]
    wrapped = np.where(delta < 0, delta + beta, delta)
    sign = np.where(delta < 0, -1.0, 1.0)
    result = np.zeros((t.size, t.size, 3 * n, 3 * n), complex)
    for color in range(3):
        energy, basis = np.linalg.eigh(1j * a[color])
        log_weight = (
            np.log(2.0)
            - 2 * wrapped[..., None] * energy
            - np.logaddexp(0.0, -2 * beta * energy)
        )
        block = np.einsum(
            "ik,abk,jk->abij", basis, np.exp(log_weight), basis.conj()
        )
        result[
            :, :, color * n : (color + 1) * n, color * n : (color + 1) * n
        ] = (sign[..., None, None] * block)
    return result

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
            "call": (
                "thermal_majorana_kernel(h.copy(), beta, "
                "np.array([1.9, 0.0, .83, .83, beta]))"
            ),
            "gold_call": (
                "_oracle_thermal_majorana_kernel(h.copy(),"
                " beta, np.array([1.9, 0.0, .83, .83, "
                "beta]))"
            ),
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
            "call": (
                "thermal_majorana_kernel(h.copy(), 400.0,"
                " np.array([0.0, .001, 399.99, 400.0]))"
            ),
            "gold_call": (
                "_oracle_thermal_majorana_kernel(h.copy(),"
                " 400.0, np.array([0.0, .001, 399.99, "
                "400.0]))"
            ),
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
h[:] = 0
""",
            "call": (
                "thermal_majorana_kernel(h.copy(), beta, "
                "np.array([0.0, .3, beta]))"
            ),
            "gold_call": (
                "_oracle_thermal_majorana_kernel(h.copy(),"
                " beta, np.array([0.0, .3, beta]))"
            ),
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
h *= -1
""",
            "call": (
                "thermal_majorana_kernel(h.copy(), beta, "
                "np.array([tau, .0]))"
            ),
            "gold_call": (
                "_oracle_thermal_majorana_kernel(h.copy(),"
                " beta, np.array([tau, .0]))"
            ),
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


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "_expect_value_error(lambda: "
                "thermal_majorana_kernel(h, beta, "
                "np.array([-0.1])))"
            ),
            "gold_call": (
                "_expect_value_error(lambda: "
                "_oracle_thermal_majorana_kernel(h, beta,"
                " np.array([-0.1])))"
            ),
            "tol": 1e-09,
        },
    ]
