"""
Obtain the stationary state of the $K=0$ sector as a downward matrix continued fraction started from the terminal transfer matrix of Step 4 at radial index $n_{\max}$, and return its damping-basis coefficients at every radial level.

Starting from $R_\infty$ at $n_{\max}$, iterate the stationary recurrence downward to obtain every $R_n$, fix $X_0$ from the $n=0$ equation, and rebuild the higher levels. The overall scale is fixed by the paper's trace normalization of the stationary state; consult the paper for which component carries the trace.

Returns
-------
np.ndarray (complex) of shape $(n_{\max}+1, D)$: stationary coefficients $X_0, \ldots, X_{n_{\max}}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def stationary_radial_coefficients(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    r"""Damping-basis coefficients $X_0,\ldots,X_{n_{\max}}$ of the unique stationary state.

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
        n_max (int): radial index $n_{\max}\ge2$ at which the terminal transfer matrix is imposed.

    Raises:
        ValueError: if $n_{\max}<2$ or not an integer, or under the conditions of Step 3.

    Expected return:
        np.ndarray (complex) of shape $(n_{\max}+1,D)$; row $n$ holds $X_n$ in the Step 2
        ordering. Row $0$ has its index-$0$ component equal to exactly $1$. For a converged
        $n_{\max}$ the rows decay rapidly with $n$ in the few-quanta regime and are insensitive
        to $n_{\max}$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: stationary_radial_coefficients
def _oracle_stationary_radial_coefficients(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (16)-(17) + Sec. IV A normalization of arXiv:2609.17785.
    # Downward matrix continued fraction from R_{n_max} = R_inf, then (M_0 + G_0 R_0) X_0 = 0,
    # X_0 normalized so its (N,0,0,0) component (index 0) equals 1, X_{n+1} = R_n X_n.
    if int(n_max) != n_max or n_max < 2:
        raise ValueError("n_max must be an integer >= 2")
    n_max = int(n_max)
    Rs = [None] * (n_max + 1)
    Rs[n_max] = _oracle_asymptotic_transfer_matrix(N, A, B, C, g, s, Delta, nu)
    for n in range(n_max - 1, -1, -1):
        bl = _oracle_radial_blocks(n + 1, N, A, B, C, g, s, Delta, nu)
        Rs[n] = -np.linalg.solve(bl[0] + bl[1] @ Rs[n + 1], bl[2])
    b0 = _oracle_radial_blocks(0, N, A, B, C, g, s, Delta, nu)
    W = b0[0] + b0[1] @ Rs[0]
    _, _, vh = np.linalg.svd(W)
    x = vh[-1].conj()
    x = x / x[0]
    D = x.shape[0]
    X = np.zeros((n_max + 1, D), dtype=complex)
    X[0] = x
    for n in range(n_max):
        X[n + 1] = Rs[n] @ X[n]
    return X

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
                'split_re_im(stationary_radial_coefficients(2, 1.0, '
                '0.7, 0.35, 1.1, 0.3, 0.0, 0.0, 30))'
            ),
            'gold_call': (
                'split_re_im(_oracle_stationary_radial_coefficients(2, '
                '1.0, 0.7, 0.35, 1.1, 0.3, 0.0, 0.0, 30))'
            ),
            'tol': 1e-09,
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
                'split_re_im(stationary_radial_coefficients(3, 1.0, '
                '0.7, 0.35, 1.1, 0.6, 0.0, 0.1, 30))'
            ),
            'gold_call': (
                'split_re_im(_oracle_stationary_radial_coefficients(3, '
                '1.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.1, 30))'
            ),
            'tol': 1e-09,
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
                'split_re_im(stationary_radial_coefficients(3, 1.0, '
                '0.7, 0.35, 1.1, 0.6, 2.0, 0.0, 25))'
            ),
            'gold_call': (
                'split_re_im(_oracle_stationary_radial_coefficients(3, '
                '1.0, 0.7, 0.35, 1.1, 0.6, 2.0, 0.0, 25))'
            ),
            'tol': 1e-09,
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
                'split_re_im(stationary_radial_coefficients(4, 1.0, '
                '0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 30))'
            ),
            'gold_call': (
                'split_re_im(_oracle_stationary_radial_coefficients(4, '
                '1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 30))'
            ),
            'tol': 1e-09,
        },
        {
            'setup': (
                'def run_model():\n'
                '    try:\n'
                '        stationary_radial_coefficients(\n'
                '            2, 1.0, 0.7, 0.35, 1.1, 0.3, 0.0, 0.0, 1\n'
                '        )\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                '\n'
                '\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_stationary_radial_coefficients(\n'
                '            2, 1.0, 0.7, 0.35, 1.1, 0.3, 0.0, 0.0, 1\n'
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
