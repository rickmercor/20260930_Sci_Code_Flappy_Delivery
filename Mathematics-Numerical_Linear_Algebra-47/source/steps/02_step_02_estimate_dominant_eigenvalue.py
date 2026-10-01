"""
Estimate the dominant eigenvalue of the diffusion operator by a Jacobian-free power iteration, and return both the converged estimate and the safeguarded value used for stabilized-explicit stage selection.

The diffusion operator is the same conservative-flux operator used elsewhere in this problem: face coefficients D_{j+1/2} = (D_j + D_{j+1}) / 2, interior-node value (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2, and zero at both boundary nodes of every species. It is evaluated at the initial state y0 only.

Weights for the norm are formed once from the initial state as w_i = rtol |y0_i| + atol, and the weighted root-mean-square norm of a vector v is the square root of the mean of (v_i / w_i)^2.

The iteration starts from v drawn as numpy.random.default_rng(seed).standard_normal(y0.size). Each pass performs, in order: set sigma to the reciprocal of the weighted root-mean-square norm of the current v; form the matrix-vector product as the difference quotient (f_D(y0 + sigma v) - f_D(y0)) / sigma; update the eigenvalue estimate by the Rayleigh quotient (v dot Jv) / (v dot v); replace v by Jv divided by its Euclidean norm. From the second pass onward, the iteration stops as soon as the absolute difference between the current and previous eigenvalue estimates is less than tau times the absolute value of the current estimate.

The returned safeguarded value is q_lambda times the absolute value of the converged estimate.

The function raises ValueError when y0 is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y0.size // 3; when y0 or D_nodes contains a non-finite value; when any of dx, rtol, atol, tau, q_lambda is not a finite positive scalar; when max_iter is less than 2; when an iteration produces an image vector of zero Euclidean norm; or when the stopping criterion is not met within max_iter passes.

Stabilized explicit methods for parabolic problems extend their stability interval along the negative real axis by adding internal stages, with the extent of the interval growing quadratically in the stage count. This turns the choice of stage count into a stability condition rather than an accuracy one: the method must be given enough stages that the whole diffusion spectrum lies inside the stability interval at the step size in use. That requires an estimate of the dominant eigenvalue of the diffusion Jacobian, and the failure mode is asymmetric. An overestimate merely costs extra stages and therefore extra work, whereas an underestimate places part of the spectrum outside the stability region and the integration diverges.

Two routes to the estimate are in common use. An analytical bound built from the grid spacing and a representative coefficient value is cheap and requires no evaluations, but it can be substantially wrong when the coefficient varies in space, since it must implicitly commit to a single coefficient value for the whole domain. The alternative, favoured in large kinetic and gyrokinetic applications where forming or storing a Jacobian is impractical, is a matrix-free power iteration in which each matrix-vector product is replaced by a difference quotient of the right-hand-side function. The perturbation size for that difference quotient is not arbitrary: scaling it by the reciprocal of the same weighted root-mean-square norm used for solution error control ties the perturbation to the magnitudes the solution components actually carry, so that components near zero are not swamped by components of order one.

Because a power iteration is terminated on a relative-change criterion rather than run to convergence, its converged value approaches the dominant eigenvalue from below in magnitude and is systematically an underestimate at any practical tolerance. This is why the estimate is not used directly. Multiplying by a safety factor slightly larger than one, whose minimum admissible size can be related to the stopping tolerance itself, restores a guaranteed bound. The interaction of estimator tolerance, safety factor, and the rounding of the stage count to an integer determines both whether the resulting scheme is stable and, when several nearby estimates round to different integers, which member of a discrete family of stabilized schemes actually advances the solution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_dominant_eigenvalue(y0: np.ndarray, D_nodes: np.ndarray, dx: float,
                                 seed: int, rtol: float, atol: float, tau: float,
                                 q_lambda: float, max_iter: int = 200) -> np.ndarray:
    '''Estimate the dominant eigenvalue of the diffusion operator, matrix-free.

    Parameters
    ----------
    y0 : np.ndarray
        Initial state vector of shape (3n,), stacking the three species as
        [u; v; w]. The operator is linearized and the weights are formed here.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    seed : int
        Seed for the initial iterate, drawn with numpy.random.default_rng.
    rtol : float
        Relative tolerance entering the weighted root-mean-square weights, positive.
    atol : float
        Absolute floor entering the weighted root-mean-square weights, positive.
    tau : float
        Relative-change stopping tolerance for successive eigenvalue estimates,
        positive.
    q_lambda : float
        Multiplicative safety factor applied to the converged estimate, positive.
    max_iter : int
        Maximum number of iteration passes permitted, at least 2.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,): the converged eigenvalue estimate (negative for a
        diffusion operator), the safeguarded magnitude q_lambda times its
        absolute value, and the number of passes performed, as a float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_estimate_dominant_eigenvalue(y0: np.ndarray, D_nodes: np.ndarray,
                                         dx: float, seed: int, rtol: float,
                                         atol: float, tau: float, q_lambda: float,
                                         max_iter: int = 200) -> np.ndarray:
    """Reference implementation."""
    y0 = np.asarray(y0, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y0.ndim != 1 or y0.size % 3 != 0:
        raise ValueError("y0 must be a 1D array whose length is a multiple of 3")
    n = y0.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y0.size // 3")
    if not np.isfinite(y0).all() or not np.isfinite(D_nodes).all():
        raise ValueError("y0 and D_nodes must contain only finite values")
    for _name, _val in (("dx", dx), ("rtol", rtol), ("atol", atol),
                        ("tau", tau), ("q_lambda", q_lambda)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
        if float(_val) <= 0.0:
            raise ValueError(_name + " must be positive")
    if int(max_iter) < 2:
        raise ValueError("max_iter must be at least 2")
    dx = float(dx)
    rtol = float(rtol)
    atol = float(atol)
    tau = float(tau)
    q_lambda = float(q_lambda)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _diffusion(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            s = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (s[2:] - s[1:-1])
                       - D_face[:-1] * (s[1:-1] - s[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out

    weights = rtol * np.abs(y0) + atol
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(y0.size)
    base = _diffusion(y0)
    lam_prev = 0.0
    lam = 0.0
    iters = 0
    converged = False
    for k in range(int(max_iter)):
        iters = k + 1
        sigma = 1.0 / np.sqrt(np.mean((v / weights) ** 2))
        jv = (_diffusion(y0 + sigma * v) - base) / sigma
        lam = float((v @ jv) / (v @ v))
        nrm = np.linalg.norm(jv)
        if nrm == 0.0:
            raise ValueError("power iteration produced a zero image vector")
        v = jv / nrm
        if k > 0 and abs(lam - lam_prev) < tau * abs(lam):
            converged = True
            break
        lam_prev = lam
    if not converged:
        raise ValueError("power iteration did not converge within max_iter")

    return np.array([lam, q_lambda * abs(lam), float(iters)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration and protocol parameters ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
s = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + s, 2.9 / 1.05 + s, 2.9 + s])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 12, 1e-4, 1e-11, 0.003, 1.1)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 12, 1e-4, 1e-11, 0.003, 1.1)",
        },
        # --- Normal: same grid, different seed and looser stopping tolerance ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
s = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + s, 2.9 / 1.05 + s, 2.9 + s])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 3, 1e-4, 1e-11, 0.05, 1.2)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 3, 1e-4, 1e-11, 0.05, 1.2)",
        },
        # --- Normal: coarse grid with constant diffusion coefficient ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 3, 1e-4, 1e-11, 0.01, 1.2)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 3, 1e-4, 1e-11, 0.01, 1.2)",
        },
        # --- Boundary: minimal admissible grid, one interior node per species ---
        {
            "setup": """import numpy as np
D_nodes = np.array([0.2, 0.5, 0.9])
dx = 0.25
y0 = np.array([1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.0, 2.0, 1.0])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 5, 1e-4, 1e-11, 0.01, 1.1)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 5, 1e-4, 1e-11, 0.01, 1.1)",
        },
        # --- Edge: strongly varying coefficient and a tight stopping tolerance ---
        {
            "setup": """import numpy as np
n = 33
x = np.arange(n) / (n - 1.0)
D_nodes = 0.1 * (1.0 + 0.99 * np.sin(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([np.ones(n), 2.0 * np.ones(n), 3.0 * np.ones(n)])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 1, 1e-4, 1e-11, 1e-4, 1.05)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 1, 1e-4, 1e-11, 1e-4, 1.05)",
        },
        # --- Edge: unit safety factor leaves the estimate magnitude unchanged ---
        {
            "setup": """import numpy as np
n = 17
x = np.arange(n) / (n - 1.0)
D_nodes = 0.25 * (1.0 + 0.5 * np.cos(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([1.0 + 0.2 * x, 2.0 - 0.3 * x, 3.0 + 0.1 * x])
""",
            "call": "estimate_dominant_eigenvalue(y0, D_nodes, dx, 8, 1e-3, 1e-10, 0.02, 1.0)",
            "gold_call": "_oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 8, 1e-3, 1e-10, 0.02, 1.0)",
        },
        # --- Invalid: state length not a multiple of 3 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        estimate_dominant_eigenvalue(np.zeros(10), np.ones(3), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(np.zeros(10), np.ones(3), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: diffusion coefficient array length mismatch ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        estimate_dominant_eigenvalue(np.zeros(9), np.ones(4), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(np.zeros(9), np.ones(4), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive stopping tolerance ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
def run_model():
    try:
        estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 0.0, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 0.0, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive safety factor ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
def run_model():
    try:
        estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 0.01, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 0.01, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: stopping tolerance unreachable within the iteration budget ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y0 = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
def run_model():
    try:
        estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 1e-16, 1.1, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(y0, D_nodes, dx, 0, 1e-4, 1e-11, 1e-16, 1.1, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: identically zero diffusion coefficient ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        estimate_dominant_eigenvalue(np.ones(9), np.zeros(3), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_dominant_eigenvalue(np.ones(9), np.zeros(3), 1.0, 0, 1e-4, 1e-11, 0.01, 1.1)
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
