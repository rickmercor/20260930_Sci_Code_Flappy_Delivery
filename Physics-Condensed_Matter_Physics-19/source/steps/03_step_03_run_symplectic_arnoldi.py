"""
Generate a skew-orthonormal polynomial basis on a set of nodes, together with the upper Hessenberg matrix of its multiplication-by-x recurrence.

Skew-orthogonal polynomials obey no three-term recurrence, so x times a basis polynomial expands over all lower members; building the basis from repeated multiplication by x keeps that full recurrence explicit and avoids ill-conditioned monomial moments.

Returns
-------
tuple[np.ndarray, np.ndarray]: the basis S and the Hessenberg matrix H.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_symplectic_arnoldi(
    gram: np.ndarray,
    nodes: np.ndarray,
    n_vectors: int,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a skew-orthonormal Krylov basis and its Hessenberg recurrence matrix.

    Functions are represented by their values at the ``L`` nodes and the skew
    product is ``<f, g> = f @ gram @ g``. Column 0 of ``S`` is the constant
    function 1. Use the normalization and reorthogonalization conventions of
    ``skew_orthonormalize``. ``H`` is upper Hessenberg and records the
    multiplication-by-node recurrence for the returned basis. Hence
    ``nodes[:, None] * S[:, :-1] = S @ H``, column ``j`` is a polynomial of
    degree ``j`` in the node values, and consecutive columns ``(2k, 2k + 1)``
    form skew-orthonormal pairs.

    Parameters
    ----------
    gram : np.ndarray
        Real skew-symmetric array of shape ``(L, L)``.
    nodes : np.ndarray
        Node values, shape ``(L,)``.
    n_vectors : int
        Number of basis columns, ``1 <= n_vectors <= L``.
    eta : float
        Reorthogonalization threshold passed to each reduction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(S, H)`` with shapes ``(L, n_vectors)`` and
        ``(n_vectors, n_vectors - 1)``.

    Raises
    ------
    ValueError
        If ``gram`` is not a finite ``(L, L)`` array, ``nodes`` is not a
        finite array of shape ``(L,)``, ``n_vectors`` is not an integer with
        ``1 <= n_vectors <= L`` (booleans are rejected), or any reduction
        raises ``ValueError``.
    """
    return basis, hessenberg

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_symplectic_arnoldi(
    gram: np.ndarray,
    nodes: np.ndarray,
    n_vectors: int,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation (symplectic Arnoldi iteration)."""
    import numpy as np

    form = np.asarray(gram, dtype=float)
    points = np.asarray(nodes, dtype=float)
    if form.ndim != 2 or form.shape[0] != form.shape[1] or not np.all(np.isfinite(form)):
        raise ValueError("gram must be a finite square array")
    size = form.shape[0]
    if points.shape != (size,) or not np.all(np.isfinite(points)):
        raise ValueError("nodes must be a finite array of shape (L,)")
    if (isinstance(n_vectors, bool) or not isinstance(n_vectors, (int, np.integer))
            or not 1 <= n_vectors <= size):
        raise ValueError("n_vectors must be an integer with 1 <= n_vectors <= L")
    count = int(n_vectors)
    basis = np.zeros((size, count))
    hessenberg = np.zeros((count, count - 1))
    # ESR3m start: r11 = 1, so the first vector is the unscaled constant.
    basis[:, 0] = 1.0
    for j in range(1, count):
        candidate = points * basis[:, j - 1]
        column, coefficients = _oracle_skew_orthonormalize(
            basis[:, :j], candidate, form, eta
        )
        basis[:, j] = column
        hessenberg[: j + 1, j - 1] = coefficients
    return basis, hessenberg

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [{'setup': 'import numpy as np\n'
               'def _form(q, n):\n'
               '    x = np.arange(n, dtype=float)\n'
               '    w = q ** (0.5 * x)\n'
               '    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               'def _red(out, scale):\n'
               '    S, H = (np.asarray(a, dtype=float) for a in out)\n'
               '    return (S / scale, H / scale, S.shape, H.shape)\n'
               'G = _form(0.3, 10)\n'
               'x = np.arange(10.0)\n',
      'call': '_red(run_symplectic_arnoldi(G.copy(), x.copy(), 6), 1.0e3)',
      'gold_call': '_red(_oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 6), 1.0e3)'},
     {'setup': 'import numpy as np\n'
               'def _form(q, n):\n'
               '    x = np.arange(n, dtype=float)\n'
               '    w = q ** (0.5 * x)\n'
               '    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               'def _red(out, scale):\n'
               '    S, H = (np.asarray(a, dtype=float) for a in out)\n'
               '    return (S / scale, H / scale, S.shape, H.shape)\n'
               'G = _form(0.5, 14)\n'
               'x = np.arange(14.0)\n',
      'call': '_red(run_symplectic_arnoldi(G.copy(), x.copy(), 5), 1.0e3)',
      'gold_call': '_red(_oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 5), 1.0e3)'},
     {'setup': 'import numpy as np\n'
               'def _form(q, n):\n'
               '    x = np.arange(n, dtype=float)\n'
               '    w = q ** (0.5 * x)\n'
               '    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               'def _red(out, scale):\n'
               '    S, H = (np.asarray(a, dtype=float) for a in out)\n'
               '    return (S / scale, H / scale, S.shape, H.shape)\n'
               'G = _form(0.3, 6)\n'
               'x = np.arange(6.0)\n',
      'call': '_red(run_symplectic_arnoldi(G.copy(), x.copy(), 1), 1.0)',
      'gold_call': '_red(_oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 1), 1.0)'},
     {'setup': 'import numpy as np\n'
               'def _form(q, n):\n'
               '    x = np.arange(n, dtype=float)\n'
               '    w = q ** (0.5 * x)\n'
               '    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               'def _red(out, scale):\n'
               '    S, H = (np.asarray(a, dtype=float) for a in out)\n'
               '    return (S / scale, H / scale, S.shape, H.shape)\n'
               'G = _form(0.2, 30)\n'
               'x = np.arange(30.0)\n',
      'call': 'float(run_symplectic_arnoldi(G.copy(), x.copy(), 8)[1][7, 6])',
      'gold_call': 'float(_oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 8)[1][7, 6])'},
     {'setup': 'import numpy as np\n'
               'def _form(q, n):\n'
               '    x = np.arange(n, dtype=float)\n'
               '    w = q ** (0.5 * x)\n'
               '    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               'def _red(out, scale):\n'
               '    S, H = (np.asarray(a, dtype=float) for a in out)\n'
               '    return (S / scale, H / scale, S.shape, H.shape)\n'
               'a = np.random.default_rng(2).normal(size=(9, 9))\n'
               'G = a - a.T\n'
               'x = np.linspace(-1.0, 1.0, 9)\n',
      'call': '_red(run_symplectic_arnoldi(G.copy(), x.copy(), 4, 0.9), 1.0)',
      'gold_call': '_red(_oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 4, 0.9), 1.0)'},
     {'setup': 'import numpy as np\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'x = np.arange(5.0)\n'
               'G = 0.5 * np.sign(x[None, :] - x[:, None])\n',
      'call': '_status(lambda: run_symplectic_arnoldi(G.copy(), x.copy(), 6))',
      'gold_call': '_status(lambda: _oracle_run_symplectic_arnoldi(G.copy(), x.copy(), 6))'}]
