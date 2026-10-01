"""
Extract the exact stationary mean photon number, zero-delay second-order coherence, single-emitter excited-state population and inter-emitter pair coherence from the damping-basis coefficients of Step 5.

Because of the trace properties of both bases, only a few components of the stationary coefficients enter photon-number moments and one- and two-atom atomic observables. Use the paper's own observable functionals, which also cover a thermal cavity reservoir; consult the paper's observables subsection and its appendix on observable functionals.

Returns
-------
np.ndarray of shape $(4,)$: the mean photon number, $g^{(2)}(0)$, $p_e$ and $C_2^{(N)}$, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def exact_stationary_observables(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    r"""Exact stationary photon statistics and one- and two-atom observables.

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
        n_max (int): radial truncation index, $n_{\max}\ge2$.

    Raises:
        ValueError: if $N<2$, or under the conditions of Steps 3 and 5.

    Expected return:
        np.ndarray of shape $(4,)$: $[\langle\hat n\rangle,g^{(2)}(0),p_e,C_2^{(N)}]$ with
        $\langle\hat n\rangle\ge0$ the mean photon number, $g^{(2)}(0)$ the zero-delay
        second-order coherence, $p_e\in[0,1]$ the excited-state population of one emitter and
        $C_2^{(N)}=\langle\tau_+^{(1)}\tau_-^{(2)}\rangle$ the pair coherence of two distinct
        emitters (real by permutation symmetry; either sign is possible).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: exact_stationary_observables
def _oracle_exact_stationary_observables(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    # ORACLE (hidden). Sec. IV C Eqs. (21)-(22) and Appendix B Eq. (B1) of arXiv:2609.17785.
    # Returns [<n>, g2(0), p_e (atom 1), C_2^(N) = <tau_+^(1) tau_-^(2)>] (all real).
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2 for the pair coherence")
    N = int(N)
    X = _oracle_stationary_radial_coefficients(N, A, B, C, g, s, Delta, nu, n_max)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    idx = {c: i for i, c in enumerate(comp)}
    c1 = X[1, 0]
    c2 = X[2, 0]
    nbar = nu + (1 + nu) * c1
    nn1 = 2 * nu ** 2 + 4 * nu * (1 + nu) * c1 + 2 * (1 + nu) ** 2 * c2
    g2 = nn1 / nbar ** 2
    pe = s * X[0, idx[(N, 0, 0, 0)]] + X[0, idx[(N - 1, 1, 0, 0)]] / N
    C2 = X[0, idx[(N - 2, 0, 1, 1)]] / (N * (N - 1))
    return np.array([np.real(nbar), np.real(g2), np.real(pe), np.real(C2)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    return [
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(exact_stationary_observables(2, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "gold_call": "as_real(_oracle_exact_stationary_observables(2, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "tol": 1e-09,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(exact_stationary_observables(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "gold_call": "as_real(_oracle_exact_stationary_observables(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "tol": 1e-09,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(exact_stationary_observables(3, 1.0, 0.8, 0.55, 1.3, 0.85, 0.35, 0.07, 40))",
            "gold_call": "as_real(_oracle_exact_stationary_observables(3, 1.0, 0.8, 0.55, 1.3, 0.85, 0.35, 0.07, 40))",
            "tol": 1e-09,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(exact_stationary_observables(2, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0, 90))",
            "gold_call": "as_real(_oracle_exact_stationary_observables(2, 0.1, 1.0, 0.5, 0.5, 0.9, 0.0, 0.0, 90))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(exact_stationary_observables(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 40))",
            "gold_call": "as_real(_oracle_exact_stationary_observables(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 40))",
            "tol": 1e-09,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        exact_stationary_observables(1, 1.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.0, 30)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_exact_stationary_observables(1, 1.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.0, 30)\n"
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
