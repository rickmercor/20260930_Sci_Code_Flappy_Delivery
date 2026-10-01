"""
Validate the actual covariance's physicality and the supplied canonical coordinates at their respective tolerances, then evaluate entropy, purity and effective occupied mode number.

Validate Williamson coordinates and evaluate the Gaussian diagnostics.



Inputs are a physical covariance Vq in grouped quadrature order and a

decomposition of shape (2*n+1,2*n). Row 0 contains duplicated descending

symplectic eigenvalues; the remaining rows contain the real coordinate map

W. Require W Vq W.T = D and W Omega W.T = Omega. Different valid canonical

bases must yield the same diagnostics.



Use normalized Frobenius tolerances 1e-8 for both identities: divide the

covariance residual by max(1,||D||_F) and the symplectic residual by 2*n.

The two halves of row 0 must agree to absolute tolerance 1e-10, and its

first half must be positive and nonincreasing to that tolerance.



Vq has the preceding step's real, finite, symmetric, positive-definite,

even-dimensional domain. Its actual symplectic eigenvalues must all be

at least 0.5-1e-10. Validate physicality independently of the approximate

supplied certificate: its looser identity tolerance does not override

Vq's domain. Invalid covariance or inconsistent decomposition raises

ValueError. Do not modify inputs. Empty inputs return [0,0,1,1].



Use the thermal variances in D once per physical mode. Clamp nu below 0.5

within physicality tolerance 1e-10 to 0.5, and nu within 1e-12 above 0.5

to vacuum. For occupations n_k=nu_k-0.5, return von Neumann entropy in nats,

Renyi-2 entropy, purity, effective occupied Williamson mode number, then

descending nu. Use 0*ln(0)=0 and effective mode number 1 at vacuum. The

effective mode number is the inverse participation ratio of normalized

occupations. This is state entropy; it measures entanglement only when

the covariance is reduced from a globally pure state.

Returns
-------
float ndarray (4+n,): SvN, S2, purity, K_eff, descending nu.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_williamson_entropy(
    Vq: "np.ndarray", decomposition: "np.ndarray"
) -> "np.ndarray":
    """Validate a canonical decomposition and evaluate its diagnostics.

    Parameters
    ----------
    Vq : "np.ndarray"
        Real physical covariance (2*n, 2*n), grouped quadrature order.
    decomposition : "np.ndarray"
        Real array (2*n+1, 2*n): diag(D), then W from the preceding step.

    Returns
    -------
    result : "np.ndarray"
        Float vector (4+n,), [SvN, S2, purity, K_eff, nu...].

    Raises
    ------
    ValueError
        For invalid covariance, malformed or nonfinite decomposition,
        unsorted or nonphysical spectrum, or failed canonical identities.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qssf_williamson_entropy(
    Vq: "np.ndarray", decomposition: "np.ndarray"
) -> "np.ndarray":
    Vq = _qssf_covariance(Vq)
    decomposition = _qssf_array(decomposition, 2, real=True, nonempty=False)
    n = Vq.shape[0] // 2
    if decomposition.shape != (2 * n + 1, 2 * n):
        raise ValueError("Malformed Williamson decomposition")
    if n == 0:
        return np.array([0.0, 0.0, 1.0, 1.0])
    # The certificate tolerance cannot certify Vq's tighter physicality
    # domain. Validate the actual covariance independently of the packet.
    _oracle_qssf_williamson_decomposition(Vq)
    diagonal, transform = decomposition[0], decomposition[1:]
    nu = diagonal[:n].copy()
    if (
        not np.allclose(diagonal[n:], nu, rtol=0.0, atol=1e-10)
        or np.any(np.diff(nu) > 1e-10)
        or np.min(nu) < 0.5 - 1e-10
    ):
        raise ValueError("Invalid thermal variance spectrum")
    omega = _qssf_omega(n)
    error_cov = np.linalg.norm(
        transform @ Vq @ transform.T - np.diag(diagonal)
    ) / max(1.0, np.linalg.norm(diagonal))
    error_symp = np.linalg.norm(transform @ omega @ transform.T - omega) / (
        2 * n
    )
    if (
        not np.isfinite([error_cov, error_symp]).all()
        or max(error_cov, error_symp) > 1e-8
    ):
        raise ValueError("Inconsistent canonical coordinate map")
    nu = np.maximum(nu, 0.5)
    nu[np.abs(nu - 0.5) <= 1e-12] = 0.5
    nk = nu - 0.5
    occupied = nk[nk > 0]
    entropy = float(
        np.sum(
            (occupied + 1) * np.log1p(occupied) - occupied * np.log(occupied)
        )
    )
    renyi = float(np.sum(np.log(2 * nu)))
    purity = float(np.exp(-renyi))
    if occupied.size:
        scaled = occupied / np.max(occupied)
        effective = float(np.sum(scaled) ** 2 / np.sum(scaled**2))
    else:
        effective = 1.0
    return np.concatenate(([entropy, renyi, purity, effective], nu))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return normal, boundary and edge-case specifications."""
    return [
        {
            "setup": """
import numpy as np

Vq = np.empty((0, 0))
decomposition = np.empty((1, 0))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np

Vq = 0.5 * np.eye(8)
decomposition = np.vstack((np.full(8, 0.5), np.eye(8)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

nu = np.array([1.2, 0.8, 0.6])
n = len(nu)
rng = np.random.default_rng(51)
H = rng.normal(size=(2 * n, 2 * n))
H = 0.2 * (H + H.T)
omega = np.block(
    [
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))],
    ]
)
S = expm(omega @ H)
diagonal = np.tile(nu, 2)
Vq = S @ np.diag(diagonal) @ S.T
decomposition = np.vstack((diagonal, np.linalg.inv(S)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

nu = np.array([0.9, 0.9, 0.5, 0.5])
n = len(nu)
rng = np.random.default_rng(52)
H = rng.normal(size=(2 * n, 2 * n))
H = 0.2 * (H + H.T)
omega = np.block(
    [
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))],
    ]
)
S = expm(omega @ H)
diagonal = np.tile(nu, 2)
Vq = S @ np.diag(diagonal) @ S.T
decomposition = np.vstack((diagonal, np.linalg.inv(S)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

nu = np.array([0.5, 0.5])
n = len(nu)
rng = np.random.default_rng(53)
H = rng.normal(size=(2 * n, 2 * n))
H = 0.2 * (H + H.T)
omega = np.block(
    [
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))],
    ]
)
S = expm(omega @ H)
diagonal = np.tile(nu, 2)
Vq = S @ np.diag(diagonal) @ S.T
decomposition = np.vstack((diagonal, np.linalg.inv(S)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

nu = np.array([0.500000003, 0.500000001])
n = len(nu)
rng = np.random.default_rng(54)
H = rng.normal(size=(2 * n, 2 * n))
H = 0.2 * (H + H.T)
omega = np.block(
    [
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))],
    ]
)
S = expm(omega @ H)
diagonal = np.tile(nu, 2)
Vq = S @ np.diag(diagonal) @ S.T
decomposition = np.vstack((diagonal, np.linalg.inv(S)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-09,
        },
        {
            "setup": """
import numpy as np

Vq = 0.6 * np.eye(2)
decomposition = np.vstack(([0.6, 0.6], 2 * np.eye(2)))


def invalid(fn):
    try:
        fn(Vq.copy(), decomposition.copy())
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_williamson_entropy)
""",
            "gold_call": """
invalid(_oracle_qssf_williamson_entropy)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

Vq = 0.8 * np.eye(2)
decomposition = np.vstack(([0.6, 0.6], np.eye(2)))


def invalid(fn):
    try:
        fn(Vq.copy(), decomposition.copy())
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_williamson_entropy)
""",
            "gold_call": """
invalid(_oracle_qssf_williamson_entropy)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

Vq = 0.4 * np.eye(2)
decomposition = np.vstack(([0.4, 0.4], np.eye(2)))


def invalid(fn):
    try:
        fn(Vq.copy(), decomposition.copy())
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_williamson_entropy)
""",
            "gold_call": """
invalid(_oracle_qssf_williamson_entropy)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

Vq = (0.5 - 1e-9) * np.eye(2)
decomposition = np.vstack(([0.5, 0.5], np.eye(2)))


def invalid(fn):
    try:
        fn(Vq.copy(), decomposition.copy())
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_williamson_entropy)
""",
            "gold_call": """
invalid(_oracle_qssf_williamson_entropy)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

Vq = 0.49999999995 * np.eye(2)
decomposition = np.vstack(([0.5, 0.5], 1.0 * np.eye(2)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

Vq = 0.5 * np.eye(2)
decomposition = np.vstack(([0.5, 0.5], 1.000000003 * np.eye(2)))
""",
            "call": """
qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "gold_call": """
_oracle_qssf_williamson_entropy(Vq.copy(), decomposition.copy())
""",
            "tol": 1e-10,
        },
    ]
