"""
Chain the earlier steps into the full pipeline: build the exact stationary state with the damping-basis continued fraction (Steps 1-5), extract its pair coherence (Step 6), solve the second-order closure (Step 7), and return the closure error of the pair coherence. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

This step reproduces the task's graded configuration end to end. Every earlier convention comes together here: the field and atomic damping-basis normalizations, the block-Jacobi assembly, the paper's asymptotic closure, its observable functionals, and its benchmarked second-order closure with its root-selection rule.

Returns
-------
float: the closure error $\Delta_C=C_2^{(N)}[\mathrm{exact}]-C[\mathrm{closure}]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pair_coherence_closure_error(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> float:
    r"""Exact minus second-order-closure pair coherence.

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
        n_max (int): radial truncation index $n_{\max}\ge2$ for the exact solution.

    Raises:
        ValueError: under the conditions of Steps 6 and 7.

    Expected return:
        float: $C_2^{(N)}[\mathrm{exact}]-C[\mathrm{closure}]$; positive when the closure
        underestimates the pair coherence, including when it predicts the wrong sign.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: pair_coherence_closure_error
def _oracle_pair_coherence_closure_error(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> float:
    # ORACLE (hidden). Orchestrator: exact pair coherence (Step 6, damping-basis continued fraction)
    # minus the second-order cumulant-closure pair coherence (Step 7). Chains _oracle_ steps only.
    exact = _oracle_exact_stationary_observables(N, A, B, C, g, s, Delta, nu, n_max)
    closure = _oracle_second_order_closure_stationary(N, A, B, C, g, s, Delta, nu)
    return float(exact[3] - closure[4])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    return [
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(pair_coherence_closure_error(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "gold_call": "as_real(_oracle_pair_coherence_closure_error(3, 1.0, 0.7, 0.35, 1.1, 0.9, 0.0, 0.0, 40))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(pair_coherence_closure_error(2, 1.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.0, 40))",
            "gold_call": "as_real(_oracle_pair_coherence_closure_error(2, 1.0, 0.7, 0.35, 1.1, 0.6, 0.0, 0.0, 40))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(pair_coherence_closure_error(3, 1.0, 0.8, 0.55, 1.3, 0.85, 0.35, 0.07, 40))",
            "gold_call": "as_real(_oracle_pair_coherence_closure_error(3, 1.0, 0.8, 0.55, 1.3, 0.85, 0.35, 0.07, 40))",
            "tol": 1e-08,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(pair_coherence_closure_error(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 40))",
            "gold_call": "as_real(_oracle_pair_coherence_closure_error(4, 1.0, 0.8, 0.5, 1.25, 0.88, 0.3, 0.05, 40))",
            "tol": 1e-08,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        pair_coherence_closure_error(3, 1.0, 0.7, 0.2, 1.1, 0.6, 0.0, 0.0, 30)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_pair_coherence_closure_error(3, 1.0, 0.7, 0.2, 1.1, 0.6, 0.0, 0.0, 30)\n"
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
