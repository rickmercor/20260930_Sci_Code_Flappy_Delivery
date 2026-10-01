"""
Construct a real canonical coordinate map that diagonalizes the reduced covariance. Retain every symplectic mode, including degenerate modes; accept equivalent choices of canonical basis.

Construct canonical coordinates for a reduced Gaussian covariance.







Williamson's theorem supplies a real symplectic change of coordinates W



with W Vq W.T = D and W Omega W.T = Omega, where quadratures are grouped



as (q_1,...,q_n,p_1,...,p_n), Omega = [[0,I],[-I,0]], and



D = diag(nu_1,...,nu_n,nu_1,...,nu_n). The positive nu are ordered from



largest to smallest, retaining multiplicity. These coordinates describe



the independent thermal modes underlying the QSSF entropy diagnostics.







Return an array of shape (2*n+1, 2*n): row 0 contains diag(D), and the



remaining rows contain W. No particular eigenvector phase, sign, or basis



within a degenerate symplectic eigenspace is prescribed. Any W satisfying



both identities is accepted. Tests compare the spectrum and the identities,



not individual entries of W. Reconstruction and symplectic Frobenius errors



must be <= 1e-8 after division by max(1, ||D||_F) and 2*n, respectively.







Vq must be finite and real, square and even-dimensional, symmetric within



absolute tolerance 1e-10, positive definite and physical: every nu must



be >= 0.5-1e-10. Symmetrize within that tolerance. Retain unrounded nu in



the factorization; vacuum-roundoff conventions are applied by the entropy



step. The empty (0,0) input returns shape (1,0). Invalid data raises



ValueError. Caller inputs must not be modified.

Returns
-------
float ndarray (2*n+1, 2*n): duplicated descending nu in row 0, canonical map W in the remaining rows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_williamson_decomposition(Vq: "np.ndarray") -> "np.ndarray":
    """Return a Williamson spectrum and a canonical diagonalizing map.

    Parameters
    ----------
    Vq : "np.ndarray"
        Physical real covariance of shape (2*n, 2*n), in grouped order.

    Returns
    -------
    result : "np.ndarray"
        Real array (2*n+1, 2*n). First row: duplicated descending nu.
        Remaining rows: W with W Vq W.T = D and W Omega W.T = Omega.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite or nonreal entries, nonsymmetry,
        lack of positive definiteness or violation of physicality.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _qssf_covariance(Vq):
    """Validate and symmetrize a covariance without mutating it."""
    Vq = _qssf_array(Vq, 2, real=True, nonempty=False)
    if Vq.shape[0] != Vq.shape[1] or Vq.shape[0] % 2:
        raise ValueError("Vq must be square and even-dimensional")
    if not np.allclose(Vq, Vq.T, rtol=0.0, atol=1e-10):
        raise ValueError("Vq must be symmetric within tolerance 1e-10")
    return (Vq + Vq.T) / 2.0


def _qssf_omega(n):
    """Return the grouped-quadrature symplectic form."""
    return np.block(
        [[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]
    )


def _oracle_qssf_williamson_decomposition(Vq: "np.ndarray") -> "np.ndarray":
    Vq = _qssf_covariance(Vq)
    n = Vq.shape[0] // 2
    if n == 0:
        return np.empty((1, 0), dtype=float)
    try:
        factor = np.linalg.cholesky(Vq)
    except np.linalg.LinAlgError as error:
        raise ValueError("Vq must be positive definite") from error
    omega = _qssf_omega(n)
    hermitian = 1j * factor.T @ omega @ factor
    hermitian = (hermitian + hermitian.conj().T) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    nu = values[n:][::-1]
    if not np.isfinite(nu).all() or nu[-1] < 0.5 - 1e-10:
        raise ValueError("Covariance violates the uncertainty relation")

    # Positive eigenvectors encode real canonical pairs. The minus sign
    # fixes Omega's orientation. Orthogonality holds also at degeneracies.
    positive = vectors[:, n:][:, ::-1]
    orthogonal = np.sqrt(2.0) * np.concatenate(
        (positive.real, -positive.imag), axis=1
    )
    diagonal = np.concatenate((nu, nu))
    transform = (
        np.sqrt(diagonal)[:, None] * np.linalg.solve(factor.T, orthogonal).T
    )
    return np.vstack((diagonal, transform))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return normal, boundary and edge-case specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "Vq = np.diag([0.6, 0.8, 1.2, 0.6, 0.8, 1.2])\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "Vq = 0.5 * np.eye(8)\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "Vq = np.empty((0, 0))\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "nu = np.array([0.51, 0.8, 1.4])\n"
                "n = len(nu)\n"
                "rng = np.random.default_rng(11)\n"
                "H = rng.normal(size=(2 * n, 2 * n))\n"
                "H = 0.6 * (H + H.T) / np.sqrt(2 * n)\n"
                "omega = np.block(\n"
                "    [\n"
                "        [np.zeros((n, n)), np.eye(n)],\n"
                "        [-np.eye(n), np.zeros((n, n))],\n"
                "    ]\n"
                ")\n"
                "S = expm(omega @ H)\n"
                "Vq = S @ np.diag(np.tile(nu, 2)) @ S.T\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "nu = np.array([0.75, 0.75, 0.75, 0.75])\n"
                "n = len(nu)\n"
                "rng = np.random.default_rng(12)\n"
                "H = rng.normal(size=(2 * n, 2 * n))\n"
                "H = 0.5 * (H + H.T) / np.sqrt(2 * n)\n"
                "omega = np.block(\n"
                "    [\n"
                "        [np.zeros((n, n)), np.eye(n)],\n"
                "        [-np.eye(n), np.zeros((n, n))],\n"
                "    ]\n"
                ")\n"
                "S = expm(omega @ H)\n"
                "Vq = S @ np.diag(np.tile(nu, 2)) @ S.T\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "nu = np.array([0.5, 0.5, 0.9, 0.9])\n"
                "n = len(nu)\n"
                "rng = np.random.default_rng(13)\n"
                "H = rng.normal(size=(2 * n, 2 * n))\n"
                "H = 0.4 * (H + H.T) / np.sqrt(2 * n)\n"
                "omega = np.block(\n"
                "    [\n"
                "        [np.zeros((n, n)), np.eye(n)],\n"
                "        [-np.eye(n), np.zeros((n, n))],\n"
                "    ]\n"
                ")\n"
                "S = expm(omega @ H)\n"
                "Vq = S @ np.diag(np.tile(nu, 2)) @ S.T\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "nu = np.array([0.5, 0.5])\n"
                "n = len(nu)\n"
                "rng = np.random.default_rng(14)\n"
                "H = rng.normal(size=(2 * n, 2 * n))\n"
                "H = 1.2 * (H + H.T) / np.sqrt(2 * n)\n"
                "omega = np.block(\n"
                "    [\n"
                "        [np.zeros((n, n)), np.eye(n)],\n"
                "        [-np.eye(n), np.zeros((n, n))],\n"
                "    ]\n"
                ")\n"
                "S = expm(omega @ H)\n"
                "Vq = S @ np.diag(np.tile(nu, 2)) @ S.T\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "from scipy.linalg import expm\n"
                "\n"
                "\n"
                "def check(fn, covariance):\n"
                "    result = np.asarray(fn(covariance.copy()))\n"
                "    n = len(covariance) // 2\n"
                "    assert result.shape == (2 * n + 1, 2 * n)\n"
                "    assert np.isrealobj(result) and np.isfinite("
                "result).all()\n"
                "    if n == 0:\n"
                "        return np.array([0.0])\n"
                "    diagonal, W = result[0], result[1:]\n"
                "    nu = diagonal[:n]\n"
                "    omega = np.block(\n"
                "        [\n"
                "            [np.zeros((n, n)), np.eye(n)],\n"
                "            [-np.eye(n), np.zeros((n, n))],\n"
                "        ]\n"
                "    )\n"
                "    error_cov = np.linalg.norm(W @ covariance @ "
                "W.T - np.diag(diagonal)) / max(\n"
                "        1.0, np.linalg.norm(diagonal)\n"
                "    )\n"
                "    error_symp = np.linalg.norm(W @ omega @ W.T "
                "- omega) / (2 * n)\n"
                "    valid = (\n"
                "        np.allclose(diagonal[n:], nu, rtol=0, at"
                "ol=1e-10)\n"
                "        and np.all(np.diff(nu) <= 1e-10)\n"
                "        and np.min(nu) >= 0.5 - 1e-10\n"
                "        and error_cov <= 1e-8\n"
                "        and error_symp <= 1e-8\n"
                "    )\n"
                "    return np.concatenate((nu, [0.0 if valid els"
                "e 1.0]))\n"
                "\n"
                "\n"
                "nu = np.array([0.7, 0.70000001, 1.3])\n"
                "n = len(nu)\n"
                "rng = np.random.default_rng(15)\n"
                "H = rng.normal(size=(2 * n, 2 * n))\n"
                "H = 0.7 * (H + H.T) / np.sqrt(2 * n)\n"
                "omega = np.block(\n"
                "    [\n"
                "        [np.zeros((n, n)), np.eye(n)],\n"
                "        [-np.eye(n), np.zeros((n, n))],\n"
                "    ]\n"
                ")\n"
                "S = expm(omega @ H)\n"
                "Vq = S @ np.diag(np.tile(nu, 2)) @ S.T\n"
            ),
            "call": ("check(qssf_williamson_decomposition, Vq)"),
            "gold_call": ("check(_oracle_qssf_williamson_decomposition, Vq)"),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def invalid(fn):\n"
                "    try:\n"
                "        fn(-np.eye(2))\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": ("invalid(qssf_williamson_decomposition)"),
            "gold_call": ("invalid(_oracle_qssf_williamson_decomposition)"),
            "tol": 0.0,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def invalid(fn):\n"
                "    try:\n"
                "        fn(0.4 * np.eye(2))\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": ("invalid(qssf_williamson_decomposition)"),
            "gold_call": ("invalid(_oracle_qssf_williamson_decomposition)"),
            "tol": 0.0,
        },
    ]
