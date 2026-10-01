"""
Compute the large-radial-index limit $R_\infty$ of the stationary transfer matrices $X_{n+1}=R_n X_n$, which terminates the downward matrix continued fraction.

In general the large-$n$ limit of a matrix continued fraction obeys a quadratic matrix equation whose root selection is the usual difficulty in terminating the fraction. The paper proves that for this Liouvillian the termination can be obtained without solving a quadratic problem, uniquely throughout the dissipative regime. Use the paper's own closure for $R_\infty$ (consult its linear-closure proposition); do not use a generic quadratic-matrix solver or a zero terminal matrix.

Returns
-------
np.ndarray (complex) of shape $(D, D)$: $R_\infty$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def asymptotic_transfer_matrix(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Large-radial-index transfer matrix $R_\infty$ that terminates the continued fraction.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: under the same conditions as Step 3.

    Expected return:
        np.ndarray (complex) of shape $(D,D)$: the asymptotic transfer matrix, independent of any
        radial truncation, bounded, and well defined for every $A>0$. Basis ordering as in
        Step 2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: asymptotic_transfer_matrix
def _oracle_asymptotic_transfer_matrix(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Proposition 2, Eqs. (18)-(20) of arXiv:2609.17785.
    # Bulk blocks are affine in n: M_n = n M1 + M0, F_n = n F1 + F0, G_n constant (G1 = 0),
    # so the Riccati closure (18) is linear: R_inf = -(M1)^{-1} F1.
    # M1, F1 extracted by a finite difference at a bulk index (exact, blocks are affine for n >= 1).
    nb = 8
    b1 = _oracle_radial_blocks(nb, N, A, B, C, g, s, Delta, nu)
    b2 = _oracle_radial_blocks(nb + 1, N, A, B, C, g, s, Delta, nu)
    M1 = b2[0] - b1[0]
    F1 = b2[2] - b1[2]
    return -np.linalg.solve(M1, F1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': (
                'import numpy as np\n'
                '\n'
                '\n'
                'def split_re_im(x):\n'
                '    x = np.asarray(x)\n'
                '    return np.stack((x.real, x.imag), axis=0)\n'
            ),
            'call': (
                'split_re_im(asymptotic_transfer_matrix(3, 1.0, 0.7, '
                '0.35, 1.1, 0.9, 0.0, 0.0))'
            ),
            'gold_call': (
                'split_re_im(_oracle_asymptotic_transfer_matrix(3, 1.0,'
                ' 0.7, 0.35, 1.1, 0.9, 0.0, 0.0))'
            ),
            'tol': 1e-10,
        },
        {
            'setup': (
                'import numpy as np\n'
                '\n'
                '\n'
                'def split_re_im(x):\n'
                '    x = np.asarray(x)\n'
                '    return np.stack((x.real, x.imag), axis=0)\n'
            ),
            'call': (
                'split_re_im(asymptotic_transfer_matrix(3, 1.0, 0.7, '
                '0.35, 2.4, 0.05, 0.0, 0.0))'
            ),
            'gold_call': (
                'split_re_im(_oracle_asymptotic_transfer_matrix(3, 1.0,'
                ' 0.7, 0.35, 2.4, 0.05, 0.0, 0.0))'
            ),
            'tol': 1e-10,
        },
        {
            'setup': (
                'import numpy as np\n'
                '\n'
                '\n'
                'def split_re_im(x):\n'
                '    x = np.asarray(x)\n'
                '    return np.stack((x.real, x.imag), axis=0)\n'
            ),
            'call': (
                'split_re_im(asymptotic_transfer_matrix(2, 1.0, 0.7, '
                '0.45, 1.1, 0.6, 0.7, 0.1))'
            ),
            'gold_call': (
                'split_re_im(_oracle_asymptotic_transfer_matrix(2, 1.0,'
                ' 0.7, 0.45, 1.1, 0.6, 0.7, 0.1))'
            ),
            'tol': 1e-10,
        },
        {
            'setup': (
                'import numpy as np\n'
                '\n'
                '\n'
                'def split_re_im(x):\n'
                '    x = np.asarray(x)\n'
                '    return np.stack((x.real, x.imag), axis=0)\n'
            ),
            'call': (
                'split_re_im(asymptotic_transfer_matrix(4, 1.0, 0.8, '
                '0.5, 1.25, 0.88, 0.3, 0.05))'
            ),
            'gold_call': (
                'split_re_im(_oracle_asymptotic_transfer_matrix(4, 1.0,'
                ' 0.8, 0.5, 1.25, 0.88, 0.3, 0.05))'
            ),
            'tol': 1e-10,
        },
        {
            'setup': (
                'def run_model():\n'
                '    try:\n'
                '        asymptotic_transfer_matrix(3, 0.0, 0.7, 0.35, '
                '1.1, 0.6, 0.0, 0.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                '\n'
                '\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_asymptotic_transfer_matrix(\n'
                '            3, 0.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.0\n'
                '        )\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
            ),
            'call': (
                'run_model()'
            ),
            'gold_call': (
                'run_gold()'
            ),
        },
    ]
