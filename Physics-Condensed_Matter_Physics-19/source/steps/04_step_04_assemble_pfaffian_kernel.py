"""
Assemble the 2 x 2 matrix-valued Pfaffian kernel of a finite beta = 1 discrete particle ensemble from its skew-orthonormal polynomials.

An N-particle ensemble with density proportional to prod |h_i - h_j| prod w(h_i) is a Pfaffian point process, so all of its correlation functions are Pfaffians of one skew-symmetric kernel with a 2 x 2 block per site; such a kernel is not unique, but all valid choices share their Pfaffian minors.

Returns
-------
np.ndarray: the (2L, 2L) skew-symmetric Pfaffian kernel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def assemble_pfaffian_kernel(polys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a Pfaffian correlation kernel of a finite beta = 1 ensemble on sites.

    The sites are ``x = 0, 1, ..., L - 1`` with positive weights ``w(x)`` given
    by ``weights``. Column ``k`` of ``polys`` holds the values at the sites of a
    polynomial ``R_k`` of degree ``k``, ``k = 0, ..., N - 1`` with ``N`` even,
    and the family is skew-orthonormal under the skew product
    ``<f, g> = sum_{x, y} f(x) g(y) * 0.5 * sign(y - x) * w(x) * w(y)``:
    ``<R_{2k}, R_{2k+1}> = 1`` and every other skew product between members of
    different pairs vanishes. The ensemble places ``N`` particles on distinct
    sites with probability proportional to
    ``prod_{i<j} |h_i - h_j| * prod_i w(h_i)``.

    Return a real skew-symmetric ``(2L, 2L)`` array ``K`` whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x`` and such that, for every
    set ``X`` of distinct sites, the Pfaffian of the principal submatrix of
    ``K`` on the rows and columns of the sites in ``X`` (sites in increasing
    order) equals the probability that every site of ``X`` is occupied. The
    Pfaffian convention is ``Pf([[0, a], [-a, 0]]) = a``.

    Parameters
    ----------
    polys : np.ndarray
        Array of shape ``(L, N)``, ``N`` even and positive.
    weights : np.ndarray
        Positive finite site weights, shape ``(L,)``.

    Returns
    -------
    np.ndarray
        The kernel ``K`` of shape ``(2L, 2L)``.

    Raises
    ------
    ValueError
        If ``polys`` is not a finite two-dimensional array with an even,
        positive number of columns, or ``weights`` is not a finite positive
        array of shape ``(L,)`` matching the rows of ``polys``.
    """
    return kernel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_pfaffian_kernel(polys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference implementation (the paper's convention: Pfaffian, not quaternion determinant)."""
    import numpy as np

    family = np.asarray(polys, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if family.ndim != 2 or family.shape[1] == 0 or family.shape[1] % 2 == 1:
        raise ValueError("polys must have shape (L, N) with N even and positive")
    if not np.all(np.isfinite(family)):
        raise ValueError("polys must be finite")
    size = family.shape[0]
    if weight.shape != (size,) or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("weights must be finite, positive and of shape (L,)")
    sites = np.arange(size, dtype=float)
    sign = np.sign(sites[:, None] - sites[None, :])
    # psi_k(x) = 1/2 sum_y R_k(y) sign(x - y) w(y)
    transforms = 0.5 * sign @ (family * weight[:, None])
    even, odd = family[:, 0::2], family[:, 1::2]
    psi_even, psi_odd = transforms[:, 0::2], transforms[:, 1::2]
    # S(x, y) = w(x) sum_k [R_{2k+1}(x) psi_{2k}(y) - R_{2k}(x) psi_{2k+1}(y)]
    mixed = weight[:, None] * (odd @ psi_even.T - even @ psi_odd.T)
    # D(x, y) = w(x) w(y) sum_k [R_{2k}(x) R_{2k+1}(y) - R_{2k+1}(x) R_{2k}(y)]
    derivative = np.outer(weight, weight) * (even @ odd.T - odd @ even.T)
    # J(x, y) = sum_k [psi_{2k+1}(x) psi_{2k}(y) - psi_{2k}(x) psi_{2k+1}(y)] - sign(x - y)/2
    integral = psi_odd @ psi_even.T - psi_even @ psi_odd.T - 0.5 * sign
    kernel = np.zeros((2 * size, 2 * size))
    kernel[0::2, 0::2] = integral
    kernel[0::2, 1::2] = mixed.T
    kernel[1::2, 0::2] = -mixed
    kernel[1::2, 1::2] = -derivative
    return kernel

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [{'setup': 'import numpy as np\n'
               'def _pf(A):\n'
               '    A = np.array(A, dtype=float)\n'
               '    n = A.shape[0]\n'
               '    out = 1.0\n'
               '    for k in range(0, n - 1, 2):\n'
               '        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n'
               '        if p != k + 1:\n'
               '            A[[k + 1, p], :] = A[[p, k + 1], :]\n'
               '            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n'
               '            out = -out\n'
               '        if A[k, k + 1] == 0.0:\n'
               '            return 0.0\n'
               '        out *= A[k, k + 1]\n'
               '        t = A[k, k + 2:] / A[k, k + 1]\n'
               '        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + '
               '1], t)\n'
               '    return float(out)\n'
               'def _blocks(K, sites):\n'
               '    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n'
               '    return K[np.ix_(idx, idx)]\n'
               'def _sop(w, n):\n'
               '    L = w.size\n'
               '    x = np.arange(L, dtype=float)\n'
               '    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               '    cols = []\n'
               '    for d in range(n):\n'
               '        v = ((x - x.mean()) / L) ** d\n'
               '        for k in range(len(cols) // 2):\n'
               '            a, b = cols[2 * k], cols[2 * k + 1]\n'
               '            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n'
               '        if len(cols) % 2:\n'
               '            v = v / (cols[-1] @ G @ v)\n'
               '        cols.append(v)\n'
               '    return np.array(cols).T\n'
               'def _minors(K):\n'
               '    K = np.asarray(K, dtype=float)\n'
               '    L = K.shape[0] // 2\n'
               '    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n'
               '    r2 = _pf(_blocks(K, [1, L - 2]))\n'
               '    r3 = _pf(_blocks(K, [0, L // 2, L - 1]))\n'
               '    J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])\n'
               '    gap = _pf(J - _blocks(K, [L - 3, L - 2, L - 1]))\n'
               '    asym = float(np.abs(K + K.T).max())\n'
               '    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 3.0 * r3 + 5.0 * gap\n'
               '                 + 1.0e3 * asym + 7.0 * L)\n'
               'w = 0.3 ** (0.5 * np.arange(9.0))\n'
               'R = _sop(w, 4)\n',
      'call': '_minors(assemble_pfaffian_kernel(R.copy(), w.copy()))',
      'gold_call': '_minors(_oracle_assemble_pfaffian_kernel(R.copy(), w.copy()))'},
     {'setup': 'import numpy as np\n'
               'def _pf(A):\n'
               '    A = np.array(A, dtype=float)\n'
               '    n = A.shape[0]\n'
               '    out = 1.0\n'
               '    for k in range(0, n - 1, 2):\n'
               '        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n'
               '        if p != k + 1:\n'
               '            A[[k + 1, p], :] = A[[p, k + 1], :]\n'
               '            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n'
               '            out = -out\n'
               '        if A[k, k + 1] == 0.0:\n'
               '            return 0.0\n'
               '        out *= A[k, k + 1]\n'
               '        t = A[k, k + 2:] / A[k, k + 1]\n'
               '        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + '
               '1], t)\n'
               '    return float(out)\n'
               'def _blocks(K, sites):\n'
               '    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n'
               '    return K[np.ix_(idx, idx)]\n'
               'def _sop(w, n):\n'
               '    L = w.size\n'
               '    x = np.arange(L, dtype=float)\n'
               '    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               '    cols = []\n'
               '    for d in range(n):\n'
               '        v = ((x - x.mean()) / L) ** d\n'
               '        for k in range(len(cols) // 2):\n'
               '            a, b = cols[2 * k], cols[2 * k + 1]\n'
               '            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n'
               '        if len(cols) % 2:\n'
               '            v = v / (cols[-1] @ G @ v)\n'
               '        cols.append(v)\n'
               '    return np.array(cols).T\n'
               'def _minors(K):\n'
               '    K = np.asarray(K, dtype=float)\n'
               '    L = K.shape[0] // 2\n'
               '    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n'
               '    r2 = _pf(_blocks(K, [1, L - 2]))\n'
               '    r3 = _pf(_blocks(K, [0, L // 2, L - 1]))\n'
               '    J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])\n'
               '    gap = _pf(J - _blocks(K, [L - 3, L - 2, L - 1]))\n'
               '    asym = float(np.abs(K + K.T).max())\n'
               '    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 3.0 * r3 + 5.0 * gap\n'
               '                 + 1.0e3 * asym + 7.0 * L)\n'
               'w = 0.5 ** (0.5 * np.arange(12.0))\n'
               'R = _sop(w, 6)\n',
      'call': '_minors(assemble_pfaffian_kernel(R.copy(), w.copy()))',
      'gold_call': '_minors(_oracle_assemble_pfaffian_kernel(R.copy(), w.copy()))'},
     {'setup': 'import numpy as np\n'
               'def _pf(A):\n'
               '    A = np.array(A, dtype=float)\n'
               '    n = A.shape[0]\n'
               '    out = 1.0\n'
               '    for k in range(0, n - 1, 2):\n'
               '        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n'
               '        if p != k + 1:\n'
               '            A[[k + 1, p], :] = A[[p, k + 1], :]\n'
               '            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n'
               '            out = -out\n'
               '        if A[k, k + 1] == 0.0:\n'
               '            return 0.0\n'
               '        out *= A[k, k + 1]\n'
               '        t = A[k, k + 2:] / A[k, k + 1]\n'
               '        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + '
               '1], t)\n'
               '    return float(out)\n'
               'def _blocks(K, sites):\n'
               '    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n'
               '    return K[np.ix_(idx, idx)]\n'
               'def _sop(w, n):\n'
               '    L = w.size\n'
               '    x = np.arange(L, dtype=float)\n'
               '    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               '    cols = []\n'
               '    for d in range(n):\n'
               '        v = ((x - x.mean()) / L) ** d\n'
               '        for k in range(len(cols) // 2):\n'
               '            a, b = cols[2 * k], cols[2 * k + 1]\n'
               '            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n'
               '        if len(cols) % 2:\n'
               '            v = v / (cols[-1] @ G @ v)\n'
               '        cols.append(v)\n'
               '    return np.array(cols).T\n'
               'def _minors(K):\n'
               '    K = np.asarray(K, dtype=float)\n'
               '    L = K.shape[0] // 2\n'
               '    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n'
               '    r2 = _pf(_blocks(K, [1, L - 2]))\n'
               '    r3 = _pf(_blocks(K, [0, L // 2, L - 1]))\n'
               '    J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])\n'
               '    gap = _pf(J - _blocks(K, [L - 3, L - 2, L - 1]))\n'
               '    asym = float(np.abs(K + K.T).max())\n'
               '    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 3.0 * r3 + 5.0 * gap\n'
               '                 + 1.0e3 * asym + 7.0 * L)\n'
               'w = 0.4 ** (0.5 * np.arange(7.0))\n'
               'R = _sop(w, 2)\n',
      'call': '_minors(assemble_pfaffian_kernel(R.copy(), w.copy()))',
      'gold_call': '_minors(_oracle_assemble_pfaffian_kernel(R.copy(), w.copy()))'},
     {'setup': 'import numpy as np\n'
               'def _pf(A):\n'
               '    A = np.array(A, dtype=float)\n'
               '    n = A.shape[0]\n'
               '    out = 1.0\n'
               '    for k in range(0, n - 1, 2):\n'
               '        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n'
               '        if p != k + 1:\n'
               '            A[[k + 1, p], :] = A[[p, k + 1], :]\n'
               '            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n'
               '            out = -out\n'
               '        if A[k, k + 1] == 0.0:\n'
               '            return 0.0\n'
               '        out *= A[k, k + 1]\n'
               '        t = A[k, k + 2:] / A[k, k + 1]\n'
               '        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + '
               '1], t)\n'
               '    return float(out)\n'
               'def _blocks(K, sites):\n'
               '    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n'
               '    return K[np.ix_(idx, idx)]\n'
               'def _sop(w, n):\n'
               '    L = w.size\n'
               '    x = np.arange(L, dtype=float)\n'
               '    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               '    cols = []\n'
               '    for d in range(n):\n'
               '        v = ((x - x.mean()) / L) ** d\n'
               '        for k in range(len(cols) // 2):\n'
               '            a, b = cols[2 * k], cols[2 * k + 1]\n'
               '            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n'
               '        if len(cols) % 2:\n'
               '            v = v / (cols[-1] @ G @ v)\n'
               '        cols.append(v)\n'
               '    return np.array(cols).T\n'
               'def _minors(K):\n'
               '    K = np.asarray(K, dtype=float)\n'
               '    L = K.shape[0] // 2\n'
               '    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n'
               '    r2 = _pf(_blocks(K, [1, L - 2]))\n'
               '    r3 = _pf(_blocks(K, [0, L // 2, L - 1]))\n'
               '    J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])\n'
               '    gap = _pf(J - _blocks(K, [L - 3, L - 2, L - 1]))\n'
               '    asym = float(np.abs(K + K.T).max())\n'
               '    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 3.0 * r3 + 5.0 * gap\n'
               '                 + 1.0e3 * asym + 7.0 * L)\n'
               'w = 1.0 / (1.0 + np.arange(10.0)) ** 2\n'
               'R = _sop(w, 4)\n'
               'T = np.array([[2.0, 0.7, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, -0.4], [0.0, '
               '0.0, 0.0, 2.0]])\n',
      'call': '_minors(assemble_pfaffian_kernel(R @ T, w.copy()))',
      'gold_call': '_minors(_oracle_assemble_pfaffian_kernel(R @ T, w.copy()))'},
     {'setup': 'import numpy as np\n'
               'def _pf(A):\n'
               '    A = np.array(A, dtype=float)\n'
               '    n = A.shape[0]\n'
               '    out = 1.0\n'
               '    for k in range(0, n - 1, 2):\n'
               '        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n'
               '        if p != k + 1:\n'
               '            A[[k + 1, p], :] = A[[p, k + 1], :]\n'
               '            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n'
               '            out = -out\n'
               '        if A[k, k + 1] == 0.0:\n'
               '            return 0.0\n'
               '        out *= A[k, k + 1]\n'
               '        t = A[k, k + 2:] / A[k, k + 1]\n'
               '        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + '
               '1], t)\n'
               '    return float(out)\n'
               'def _blocks(K, sites):\n'
               '    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n'
               '    return K[np.ix_(idx, idx)]\n'
               'def _sop(w, n):\n'
               '    L = w.size\n'
               '    x = np.arange(L, dtype=float)\n'
               '    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n'
               '    cols = []\n'
               '    for d in range(n):\n'
               '        v = ((x - x.mean()) / L) ** d\n'
               '        for k in range(len(cols) // 2):\n'
               '            a, b = cols[2 * k], cols[2 * k + 1]\n'
               '            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n'
               '        if len(cols) % 2:\n'
               '            v = v / (cols[-1] @ G @ v)\n'
               '        cols.append(v)\n'
               '    return np.array(cols).T\n'
               'def _minors(K):\n'
               '    K = np.asarray(K, dtype=float)\n'
               '    L = K.shape[0] // 2\n'
               '    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n'
               '    r2 = _pf(_blocks(K, [1, L - 2]))\n'
               '    r3 = _pf(_blocks(K, [0, L // 2, L - 1]))\n'
               '    J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])\n'
               '    gap = _pf(J - _blocks(K, [L - 3, L - 2, L - 1]))\n'
               '    asym = float(np.abs(K + K.T).max())\n'
               '    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 3.0 * r3 + 5.0 * gap\n'
               '                 + 1.0e3 * asym + 7.0 * L)\n'
               'w = 0.3 ** (0.5 * np.arange(8.0))\n'
               'R = _sop(w, 4)\n',
      'call': 'float(np.trace(assemble_pfaffian_kernel(R.copy(), w.copy())[0::2, 1::2]))',
      'gold_call': 'float(np.trace(_oracle_assemble_pfaffian_kernel(R.copy(), w.copy())[0::2, 1::2]))'},
     {'setup': 'import numpy as np\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'R = np.ones((6, 3))\n'
               'w = np.ones(6)\n',
      'call': '_status(lambda: assemble_pfaffian_kernel(R.copy(), w.copy()))',
      'gold_call': '_status(lambda: _oracle_assemble_pfaffian_kernel(R.copy(), w.copy()))'}]
