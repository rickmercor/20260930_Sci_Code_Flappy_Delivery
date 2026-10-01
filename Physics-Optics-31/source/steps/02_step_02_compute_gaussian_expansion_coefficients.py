"""
Evaluate the overlap-to-weight ratio at every phase-space node.

The mesh-free ansatz assigns each normalized representing Gaussian its target-overlap divided by the unnormalized Gaussian sampling weight and coherent-state measure.

Returns
-------
np.ndarray: finite complex coefficients aligned one-to-one with the Sobol nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np

def compute_gaussian_expansion_coefficients(
    nodes: np.ndarray, q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return coefficients for normalized Gaussian basis states.

    ``nodes`` has shape ``(N,2*d)`` and widths are real symmetric positive
    definite. For ``z=(q,p)``, use
    ``W(z)=exp(-(z-[q0,p0]) @ inv(Sigma) @ (z-[q0,p0])/(2*hbar))`` with
    ``Sigma=block_diag(inv(gamma0)+inv(gamma),gamma0+gamma)``. The normalized
    Gaussian is proportional to ``det(gamma)^0.25`` times
    ``exp((-(x-q)@gamma@(x-q)/2 + 1j*p@(x-q))/hbar)``. Return its conjugate
    overlap with the normalized target Gaussian divided by
    ``N*(2*pi*hbar)^d*W(z)``.

    Returns
    -------
    np.ndarray
        Complex coefficient vector of length N.

    Raises
    ------
    ValueError
        If the nodes, centers, or widths are misaligned or non-finite, either
        width is not symmetric positive-definite, or ``hbar`` is not positive finite.
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np

def _oracle_compute_gaussian_expansion_coefficients(
    nodes: np.ndarray, q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference implementation using the closed Gaussian overlap."""
    z = np.asarray(nodes, float)
    q0, p0 = np.asarray(q0, float), np.asarray(p0, float)
    g0, g = np.asarray(gamma0, float), np.asarray(gamma, float)
    if q0.ndim != 1 or q0.size < 1 or p0.shape != q0.shape:
        raise ValueError("q0 and p0 must be aligned nonempty vectors")
    d = q0.size
    if z.ndim != 2 or z.shape[1] != 2 * d or p0.shape != q0.shape or z.shape[0] < 1:
        raise ValueError("nodes and centers have incompatible shapes")
    if g0.shape != (d, d) or g.shape != (d, d):
        raise ValueError("width matrices must be d by d")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[z.ravel(), q0, p0, g0.ravel(), g.ravel()])):
        raise ValueError("inputs must be finite")
    if not np.allclose(g0, g0.T) or not np.allclose(g, g.T):
        raise ValueError("width matrices must be symmetric")
    try:
        np.linalg.cholesky(g0); np.linalg.cholesky(g)
    except np.linalg.LinAlgError as exc:
        raise ValueError("width matrices must be positive definite") from exc
    h = float(hbar); n = z.shape[0]
    sigma = np.block([[np.linalg.inv(g0) + np.linalg.inv(g), np.zeros((d, d))],
                      [np.zeros((d, d)), g0 + g]])
    dz = z - np.r_[q0, p0]
    weight = np.exp(-np.einsum("ni,ij,nj->n", dz, np.linalg.inv(sigma), dz) / (2 * h))
    a = g0 + g
    inva = np.linalg.inv(a)
    pref = (np.linalg.det(g0) * np.linalg.det(g)) ** .25
    pref *= (2 * np.pi * h) ** (d / 2) / ((np.pi * h) ** (d / 2) * np.sqrt(np.linalg.det(a)))
    q, p = z[:, :d], z[:, d:]
    b = q @ g.T + q0 @ g0.T + 1j * (p0 - p)
    constant = (-.5 * np.einsum("ni,ij,nj->n", q, g, q)).astype(complex)
    constant += -.5 * q0 @ g0 @ q0 + 1j * np.einsum("ni,ni->n", p, q) - 1j * p0 @ q0
    exponent = (constant + .5 * np.einsum("ni,ij,nj->n", b, inva, b)) / h
    overlap = pref * np.exp(exponent)
    return overlap / (n * (2 * np.pi * h) ** d * weight)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'q=np.array([-2.4,.65]);p=np.array([4.,0.]);g0=np.diag([.8,4.]);g=4*g0;z=np.array([[-2.4,.65,4.,0.],[-2.,.3,3.7,.2],[-2.8,.9,4.4,-.15],[-1.9,.8,4.1,.35]])',
      'call': '(lambda '
              'a:float(np.dot(a.real,np.arange(1,4+1))+3*np.dot(a.imag,np.arange(4,0,-1))))(compute_gaussian_expansion_coefficients(z,q,p,g0,g))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real,np.arange(1,4+1))+3*np.dot(a.imag,np.arange(4,0,-1))))(_oracle_compute_gaussian_expansion_coefficients(z,q,p,g0,g))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'q=np.array([0.]);p=np.array([0.]);g0=np.array([[2.]]);g=np.array([[.5]]);z=np.array([[0.,0.]])',
      'call': '(lambda '
              'a:float(np.dot(a.real,np.arange(1,1+1))+3*np.dot(a.imag,np.arange(1,0,-1))))(compute_gaussian_expansion_coefficients(z,q,p,g0,g,.5))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real,np.arange(1,1+1))+3*np.dot(a.imag,np.arange(1,0,-1))))(_oracle_compute_gaussian_expansion_coefficients(z,q,p,g0,g,.5))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'q=np.array([0.]);p=np.array([0.]);g0=np.array([[1.]]);g=np.array([[2.]]);z=np.array([[0.,0.]])\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: compute_gaussian_expansion_coefficients(z,q,p,g0,g,0.0))',
      'gold_call': 'status(lambda: _oracle_compute_gaussian_expansion_coefficients(z,q,p,g0,g,0.0))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'z=np.zeros((1,4)); q=np.zeros(2); p=np.zeros(2); g0=np.eye(2); g=np.eye(2)\n'
               'g=np.array([[1.,1.],[0.,1.]])\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: compute_gaussian_expansion_coefficients(z,q,p,g0,g))',
      'gold_call': 'status(lambda: _oracle_compute_gaussian_expansion_coefficients(z,q,p,g0,g))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'z=np.zeros((1,4)); q=np.zeros(2); p=np.zeros(2); g0=np.eye(2); g=np.eye(2)\n'
               'g0=np.array([[1.,1.],[0.,1.]])\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: compute_gaussian_expansion_coefficients(z,q,p,g0,g))',
      'gold_call': 'status(lambda: _oracle_compute_gaussian_expansion_coefficients(z,q,p,g0,g))'}]
