"""
Relative correction-form discrepancy.

Section 6.2 Eqs. (99)-(100): return `max(abs(lambda))` for `(exact-local)*v=lambda*local*v`. Both inputs are form matrices in the same basis and local defines the energy denominator. The quantity is the relative difference between two symmetric bilinear forms. Accept roundoff asymmetry with np.allclose(A,A.T,atol=1e-12,rtol=1e-10). Form `D_rel=C_chol^-1*(exact-local)*C_chol^-T` using the Cholesky factor C_chol from local's lower triangle, then use the eigenvalues of `(D_rel+D_rel.T)/2`. Keep the unrounded scalar; the final step alone rounds it.

Returns
-------
delta : float Unrounded maximum absolute generalised eigenvalue.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def relative_correction_discrepancy(exact: np.ndarray, local: np.ndarray) -> float:
    '''Relative correction-form discrepancy.

    Parameters
    ----------
    exact, local : np.ndarray, shape (N, N)
        Finite nonempty symmetric positive-definite correction forms with matching shapes.

    Returns
    -------
    delta : float
        Unrounded maximum absolute generalised eigenvalue.

    Raises
    ------
    ValueError
        If the inputs are not real-valued, finite, nonempty, matching square symmetric
        positive-definite matrices.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_relative_correction_discrepancy(exact: np.ndarray, local: np.ndarray) -> float:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _relative_discrepancy(exact, local):
        exact, local = _require_spd(exact), _require_spd(local)
        if exact.shape != local.shape:
            raise ValueError("matching shapes required")
        chol = np.linalg.cholesky(local)
        whitened = np.linalg.solve(chol, exact - local)
        whitened = np.linalg.solve(chol, whitened.T).T
        return float(max(abs(np.linalg.eigvalsh((whitened + whitened.T) / 2))))

    return _relative_discrepancy(exact, local)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = 'import numpy as np\n'
    # Seeded SPD pairs; exact = C diag(1+lam) C.T with local = C C.T has generalised eigenvalues lam exactly.
    pair = ("def spd_pair(n, lam, seed, cond=10.):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]\n"
            "    C = Q * np.geomspace(1, cond ** -.5, n)\n"
            "    return C @ np.diag(1 + np.asarray(lam, dtype=float)) @ C.T, C @ C.T\n")
    random_pair = ("def random_pair(n, seed):\n"
                   "    rng = np.random.default_rng(seed)\n"
                   "    X, Z = rng.standard_normal((2, n, n))\n"
                   "    Y = X + 0.1 * Z\n"
                   "    return Y @ Y.T / n + 0.05 * np.eye(n), X @ X.T / n + 0.05 * np.eye(n)\n")
    domain_wrapper = 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n'
    step_call = 'relative_correction_discrepancy(exact,local)'
    call_1 = '#case:normal\n' + step_call
    call_2 = '_oracle_' + step_call
    call_5 = '#case:edge\ncheck(lambda:' + step_call + ')'
    call_6 = 'check(lambda:_oracle_' + step_call + ')'
    base = imports + pair + random_pair
    return [
        {'setup': base + 'exact,local=random_pair(12,1)\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'exact,local=spd_pair(10,np.linspace(.3,-.45,10),2,cond=1e6)\n', 'call': '#case:boundary\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'exact=np.array([[1.5]]);local=np.array([[2.]])\n', 'call': '#case:edge\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'exact,local=spd_pair(8,np.linspace(.4,-.7,8),3)\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'exact,local=random_pair(16,4)\nexact=exact+1e-14*np.triu(np.ones((16,16)),1)\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'exact,local=random_pair(40,5)\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'exact,local=random_pair(6,6)\nlocal=-local\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
        {'setup': base + 'exact,local=random_pair(6,7)\nexact[0,1]+=.01\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
    ]
