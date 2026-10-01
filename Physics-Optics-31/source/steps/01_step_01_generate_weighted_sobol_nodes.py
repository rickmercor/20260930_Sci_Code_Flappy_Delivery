"""
Generate deterministic importance-weighted Sobol phase-space nodes.

The mesh-free quadrature uses the Gaussian phase-space covariance Sigma = block_diag(gamma0^-1 + gamma^-1, gamma0 + gamma) to map a scrambled Sobol net around the target center.

Returns
-------
np.ndarray: finite float rows whose first and second d columns are position and momentum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
from scipy.stats import norm, qmc
import numpy as np

def generate_weighted_sobol_nodes(
    q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    n_nodes: int, hbar: float = 1.0, seed: int = 23,
) -> np.ndarray:
    """Return normally distributed Sobol nodes ordered as ``[q, p]``.

    Parameters are length-d centers, d-by-d positive-definite width matrices,
    a positive power-of-two node count (including one), positive ``hbar``,
    and integer ``seed``.
    The deterministic Python realization is SciPy's scrambled ``qmc.Sobol`` in
    dimension ``2*d`` followed by ``random_base2(log2(n_nodes))``. Map its
    inverse-normal values ``xi`` as
    ``[q0,p0] + sqrt(hbar) * xi @ chol(Sigma).T``, where
    ``Sigma = block_diag(inv(gamma0)+inv(gamma), gamma0+gamma)`` and the
    Cholesky factor is lower triangular.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_nodes, 2*d)``.

    Raises
    ------
    ValueError
        If centers or widths are misaligned or non-finite, either width is not
        symmetric positive-definite, ``n_nodes`` is not a power-of-two integer,
        ``hbar`` is not positive finite, or ``seed`` is not an integer.
    """
    return nodes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral, Real
from scipy.stats import norm, qmc
import numpy as np

def _oracle_generate_weighted_sobol_nodes(
    q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    n_nodes: int, hbar: float = 1.0, seed: int = 23,
) -> np.ndarray:
    """Reference implementation."""
    q0, p0 = np.asarray(q0, float), np.asarray(p0, float)
    g0, g = np.asarray(gamma0, float), np.asarray(gamma, float)
    if q0.ndim != 1 or p0.shape != q0.shape or q0.size < 1:
        raise ValueError("q0 and p0 must be aligned vectors")
    d = q0.size
    if g0.shape != (d, d) or g.shape != (d, d):
        raise ValueError("width matrices must be d by d")
    if not np.all(np.isfinite(np.r_[q0, p0, g0.ravel(), g.ravel()])):
        raise ValueError("inputs must be finite")
    if not np.allclose(g0, g0.T) or not np.allclose(g, g.T):
        raise ValueError("width matrices must be symmetric")
    try:
        np.linalg.cholesky(g0); np.linalg.cholesky(g)
    except np.linalg.LinAlgError as exc:
        raise ValueError("width matrices must be positive definite") from exc
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, Integral):
        raise ValueError("n_nodes must be a power-of-two integer")
    n = int(n_nodes)
    if n < 1 or n & (n - 1):
        raise ValueError("n_nodes must be a power of two")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    sigma = np.block([[np.linalg.inv(g0) + np.linalg.inv(g), np.zeros((d, d))],
                      [np.zeros((d, d)), g0 + g]])
    u = qmc.Sobol(2 * d, scramble=True, seed=int(seed)).random_base2(int(np.log2(n)))
    standard = norm.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
    return np.r_[q0, p0] + np.sqrt(float(hbar)) * standard @ np.linalg.cholesky(sigma).T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc\n'
               'q=np.array([-2.4,.65]);p=np.array([4.,0.]);g0=np.diag([.8,4.]);g=4*g0;n=32',
      'call': 'float(np.dot(generate_weighted_sobol_nodes(q,p,g0,g,n,1.,23).ravel(),np.arange(1,128+1,dtype=float)))',
      'gold_call': 'float(np.dot(_oracle_generate_weighted_sobol_nodes(q,p,g0,g,n,1.,23).ravel(),np.arange(1,128+1,dtype=float)))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc\n'
               'q=np.array([0.]);p=np.array([0.]);g0=np.array([[2.]]);g=np.array([[.5]]);n=2',
      'call': 'float(np.dot(generate_weighted_sobol_nodes(q,p,g0,g,n,.5,0).ravel(),np.arange(1,4+1,dtype=float)))',
      'gold_call': 'float(np.dot(_oracle_generate_weighted_sobol_nodes(q,p,g0,g,n,.5,0).ravel(),np.arange(1,4+1,dtype=float)))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc\n'
               'q=np.array([0.]);p=np.array([0.]);g0=np.array([[1.]]);g=np.array([[2.]]);n=6\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: generate_weighted_sobol_nodes(q,p,g0,g,n))',
      'gold_call': 'status(lambda: _oracle_generate_weighted_sobol_nodes(q,p,g0,g,n))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc\n'
               'q=np.array([0.]); p=np.array([0.]); g0=np.array([[2.]]); g=np.array([[.5]])',
      'call': 'float(np.dot(generate_weighted_sobol_nodes(q,p,g0,g,1,.5,23).ravel(),np.array([1.,2.])))',
      'gold_call': 'float(np.dot(_oracle_generate_weighted_sobol_nodes(q,p,g0,g,1,.5,23).ravel(),np.array([1.,2.])))'}]
