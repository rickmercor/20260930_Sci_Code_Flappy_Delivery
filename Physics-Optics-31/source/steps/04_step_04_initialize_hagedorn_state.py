"""
Convert phase-space nodes and a basis width into the propagated parameter state.

The factorization C=P Q^{-1} encodes the initial complex width while satisfying the Gaussian compatibility law.

Returns
-------
tuple[np.ndarray, ...]: aligned q, p, Q, P, and zero-action arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np
def initialize_hagedorn_state(
    nodes: np.ndarray, gamma: np.ndarray, hbar: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize ``q, p, Q, P, S`` for all phase-space nodes.

    ``nodes`` has shape ``(N,2*d)`` and ``gamma`` is real symmetric positive
    definite. Set ``q`` and ``p`` to copies of the two node blocks,
    ``Q_j = sqrt(hbar) * gamma**(-1/2)``,
    ``P_j = 1j * sqrt(hbar) * gamma**(1/2)``, and ``S_j = 0`` for every node,
    using the symmetric eigendecomposition roots.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Shapes ``(N,d)``, ``(N,d)``, ``(N,d,d)``, ``(N,d,d)``, and ``(N,)``.

    Raises
    ------
    ValueError
        If nodes are not a finite ``(N,2*d)`` array, ``gamma`` is not a finite
        symmetric positive-definite ``(d,d)`` matrix, or ``hbar`` is not
        positive finite.
    """
    return q, p, Q, P, S

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np
def _oracle_initialize_hagedorn_state(
    nodes: np.ndarray, gamma: np.ndarray, hbar: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of the paper's initial conditions."""
    z = np.asarray(nodes, float)
    g = np.asarray(gamma, float)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("nodes must have shape (N,2*d)")
    d = z.shape[1] // 2
    if g.shape != (d, d) or not np.all(np.isfinite(np.r_[z.ravel(), g.ravel()])):
        raise ValueError("gamma must be finite and d by d")
    if not np.allclose(g, g.T):
        raise ValueError("gamma must be symmetric")
    values, vectors = np.linalg.eigh(g)
    if np.any(values <= 0):
        raise ValueError("gamma must be positive definite")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    root = (vectors * np.sqrt(values)) @ vectors.T
    invroot = (vectors * (1 / np.sqrt(values))) @ vectors.T
    q, p = z[:, :d].copy(), z[:, d:].copy()
    qmat = np.broadcast_to(np.sqrt(float(hbar)) * invroot, (len(z), d, d)).copy()
    pmat = np.broadcast_to(1j * np.sqrt(float(hbar)) * root, (len(z), d, d)).copy()
    action = np.zeros(len(z), float)
    return q, p, qmat, pmat, action

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    projection = "float((lambda a:sum(np.dot(np.asarray(x).real.ravel(),np.arange(1,np.asarray(x).size+1)) + 2*np.sum(np.asarray(x).imag) for x in a))({call}))"
    return [
        {"setup":"import numpy as np\nfrom numbers import Real\nz=np.arange(24,dtype=float).reshape(6,4)/10;g=np.diag([3.2,16.])", "call":projection.format(call="initialize_hagedorn_state(z,g)"), "gold_call":projection.format(call="_oracle_initialize_hagedorn_state(z,g)")},
        {"setup":"import numpy as np\nfrom numbers import Real\nz=np.array([[0.,.2],[.3,-.1]]);g=np.array([[.5]])", "call":projection.format(call="initialize_hagedorn_state(z,g,.25)"), "gold_call":projection.format(call="_oracle_initialize_hagedorn_state(z,g,.25)")},
        {"setup":"import numpy as np\nfrom numbers import Real\nz=np.array([[0.,0.]]);g=np.array([[-1.]])\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2", "call":"status(lambda: initialize_hagedorn_state(z,g))", "gold_call":"status(lambda: _oracle_initialize_hagedorn_state(z,g))"},
    ]
