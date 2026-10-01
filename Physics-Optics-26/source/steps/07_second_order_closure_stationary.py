"""
Compute the stationary mean photon number, excited-state population, atom-field coherence $Z_1$ and pair coherence $C$ predicted by the second-order cumulant closure of the moment hierarchy, as benchmarked in the paper.

The standard approximate treatment of few-emitter lasers closes the hierarchy of moment equations at second order. Implement the closure exactly as the paper benchmarks it (which moments are kept, which third-order objects are factorized, which rates appear) and select its stationary solution by the paper's own prescription; consult the paper's section on the second-order cumulant closure. Emitters are symmetric, so $Z_1$ and $C$ are the same for every emitter and every pair.

Returns
-------
np.ndarray of shape $(5,)$: the mean photon number, $p_e$, $\operatorname{Re} Z_1$, $\operatorname{Im} Z_1$ and $C$, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def second_order_closure_stationary(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Stationary solution of the paper's second-order cumulant closure.

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
        ValueError: if $N<2$, $A\le0$, $B\le0$, $C<B/2$, or $s$ outside $(0,1)$.

    Expected return:
        np.ndarray of shape $(5,)$:
        $[\langle\hat n\rangle,p_e,\operatorname{Re}Z_1,\operatorname{Im}Z_1,C]$ with
        $Z_1=\langle a^\dagger\tau_-^{(j)}\rangle$ and $C=\langle\tau_+^{(i)}\tau_-^{(j)}\rangle$
        ($i\ne j$), the physically selected stationary solution (non-negative photon number,
        population in $[0,1]$).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# Oracle implementation for public function: second_order_closure_stationary
def _oracle_second_order_closure_stationary(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Sec. VI C, Eqs. (31)-(32) of arXiv:2609.17785 (C_r = C).
    # Physical root selected as in the paper: integrate the closed equations of motion from the
    # vacuum (<n> = 0, p_e = 0, Z_1 = 0, C = 0) to stationarity, then polish with a root solve.
    # Returns [<n>, p_e, Re Z_1, Im Z_1, C] with Z_1 = <a^dag tau_-^(j)>, C = <tau_+^(i) tau_-^(j)>.
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2")
    if A <= 0 or B <= 0:
        raise ValueError("cavity rate A and atomic rate B must be positive")
    if C < B / 2:
        raise ValueError("transverse rate C must satisfy C >= B/2")
    if not (0.0 < s < 1.0):
        raise ValueError("pump parameter s must lie strictly between 0 and 1")
    N = int(N)

    def rhs(t, y):
        n, pe, zr, zi, cc = y
        Z = zr + 1j * zi
        dn = -A * (n - nu) - g * N * zi
        dp = B * (s - pe) + g * zi
        dZ = -(A / 2 + C + 1j * Delta) * Z - 0.5j * g * (pe + (N - 1) * cc + n * (2 * pe - 1))
        dc = -2 * C * cc - g * (2 * pe - 1) * zi
        return [dn, dp, dZ.real, dZ.imag, dc]

    T = 400.0 / min(A, B, C)
    sol = solve_ivp(rhs, (0.0, T), [0.0, 0.0, 0.0, 0.0, 0.0], method="LSODA",
                    rtol=1e-12, atol=1e-14)
    y = fsolve(lambda y: rhs(0.0, y), sol.y[:, -1], xtol=1e-15)
    return np.array(y, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    return [
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(second_order_closure_stationary(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0))",
            "gold_call": "as_real(_oracle_second_order_closure_stationary(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(second_order_closure_stationary(2, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0))",
            "gold_call": "as_real(_oracle_second_order_closure_stationary(2, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(second_order_closure_stationary(3, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0))",
            "gold_call": "as_real(_oracle_second_order_closure_stationary(3, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(second_order_closure_stationary(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05))",
            "gold_call": "as_real(_oracle_second_order_closure_stationary(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(second_order_closure_stationary(2, 1.0, 0.7, 0.4, 1.1, 0.3, -0.5, 0.2))",
            "gold_call": "as_real(_oracle_second_order_closure_stationary(2, 1.0, 0.7, 0.4, 1.1, 0.3, -0.5, 0.2))",
            "tol": 1e-08,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        second_order_closure_stationary(3, 1.0, 0.7, 0.35, 1.1, 1.2, 0.0, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_second_order_closure_stationary(3, 1.0, 0.7, 0.35, 1.1, 1.2, 0.0, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
