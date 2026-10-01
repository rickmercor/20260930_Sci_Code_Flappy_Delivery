"""
Assemble the three $D$-by-$D$ blocks that couple radial level $n$ to levels $n-1$, $n$ and $n+1$ in the stationary sector of vanishing total coherence $K=k+(m_+-m_-)=0$, for the full pumped Tavis-Cummings Liouvillian of the model.

Combining the field damping basis (Step 1) with the permutation-invariant atomic basis (Step 2), the paper shows that the Liouvillian restricted to a fixed total-coherence sector is exactly block tridiagonal in the radial index: an uncoupled diagonal part plus interaction contributions that move the radial index by at most one. Build the blocks exactly as the paper assembles them, including which source level feeds which block; consult the paper's block-Jacobi reduction for the precise assembly. Work in the frame in which the cavity frequency drops out of the $K=0$ sector.

Returns
-------
np.ndarray (complex) of shape $(3, D, D)$: $[M_n, G_n, F_n]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def radial_blocks(n: int, N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Block-Jacobi radial blocks $[M_n,G_n,F_n]$ of the stationary ($K=0$) sector.

    Args:
        n (int): radial index, $n\ge0$.
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
        ValueError: if $n<0$, $A\le0$, $B\le0$, $C<B/2$, or the Step 2 domain is violated.

    Expected return:
        np.ndarray (complex) of shape $(3,D,D)$ stacking $[M_n,G_n,F_n]$: $M_n$ is the
        within-level block, $G_n$ carries amplitude from level $n+1$ down to $n$, and $F_n$
        carries amplitude from level $n-1$ up to $n$ and vanishes identically at $n=0$. Basis
        ordering as in Step 2; the field coherence order attached to basis element $\beta$ is
        minus its atomic coherence charge.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: radial_blocks
def _oracle_radial_blocks(n: int, N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (10)-(14) of arXiv:2609.17785, stationary sector K = 0.
    # Returns stacked complex array [M_n, G_n, F_n], each D_N x D_N.
    # Chains Step 1 (_oracle_field_multiplication_coefficients) and Step 2 (_oracle_atomic_multiplication_matrices).
    if int(n) != n or n < 0:
        raise ValueError("radial index n must be a non-negative integer")
    if A <= 0 or B <= 0:
        raise ValueError("cavity rate A and atomic rate B must be positive")
    if C < B / 2:
        raise ValueError("transverse rate C must satisfy C >= B/2")
    n = int(n)
    At = _oracle_atomic_multiplication_matrices(N, s)
    N = int(N)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    comp = np.array(comp, dtype=int)
    D = comp.shape[0]
    q = comp[:, 2] - comp[:, 3]
    kk = -q                                   # K = 0  =>  k_beta = -q_beta

    def V(nsrc, dl):
        Vm = np.zeros((D, D), dtype=complex)
        if nsrc < 0 or nsrc + dl < 0:
            return Vm
        d = dl + 1
        for b in range(D):
            f = _oracle_field_multiplication_coefficients(nsrc, int(kk[b]), nu)
            Vm[:, b] = 0.5j * g * (f[0, 1, d] * At[1][:, b] + f[0, 0, d] * At[0][:, b]
                                   - f[1, 1, d] * At[3][:, b] - f[1, 0, d] * At[2][:, b])
        return Vm

    diag = (-A * (n + np.abs(kk) / 2.0) - B * comp[:, 1]
            - C * (comp[:, 2] + comp[:, 3]) - 1j * Delta * q)
    M = np.diag(diag).astype(complex) + V(n, 0)
    G = V(n + 1, -1)
    F = V(n - 1, +1) if n >= 1 else np.zeros((D, D), dtype=complex)
    return np.stack([M, G, F])

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
                'split_re_im(radial_blocks(0, 2, 1.0, 0.7, 0.35, 1.1, '
                '0.6, 0.4, 0.1))'
            ),
            'gold_call': (
                'split_re_im(_oracle_radial_blocks(0, 2, 1.0, 0.7, '
                '0.35, 1.1, 0.6, 0.4, 0.1))'
            ),
            'tol': 1e-12,
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
                'split_re_im(radial_blocks(1, 3, 1.0, 0.7, 0.35, 1.1, '
                '0.9, 0.0, 0.0))'
            ),
            'gold_call': (
                'split_re_im(_oracle_radial_blocks(1, 3, 1.0, 0.7, '
                '0.35, 1.1, 0.9, 0.0, 0.0))'
            ),
            'tol': 1e-12,
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
                'split_re_im(radial_blocks(5, 4, 1.0, 0.8, 0.5, 1.25, '
                '0.88, 0.3, 0.05))'
            ),
            'gold_call': (
                'split_re_im(_oracle_radial_blocks(5, 4, 1.0, 0.8, 0.5,'
                ' 1.25, 0.88, 0.3, 0.05))'
            ),
            'tol': 1e-12,
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
                'split_re_im(radial_blocks(2, 1, 0.5, 1.0, 0.8, 0.9, '
                '0.3, -0.2, 0.2))'
            ),
            'gold_call': (
                'split_re_im(_oracle_radial_blocks(2, 1, 0.5, 1.0, 0.8,'
                ' 0.9, 0.3, -0.2, 0.2))'
            ),
            'tol': 1e-12,
        },
        {
            'setup': (
                'def run_model():\n'
                '    try:\n'
                '        radial_blocks(1, 2, 1.0, 0.7, 0.3, 1.1, 0.6, '
                '0.0, 0.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                '\n'
                '\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_radial_blocks(1, 2, 1.0, 0.7, 0.3, '
                '1.1, 0.6, 0.0, 0.0)\n'
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
