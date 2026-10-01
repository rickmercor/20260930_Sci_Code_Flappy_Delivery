"""
Assemble the |Phi+> Bell-state fidelity of the PPBS CNOT gate from the five retained emission events, given the intrinsic parameters and both arms' source statistics.

Each event's coincidence matrix enters the measured density matrix with its emission probability relative to the one-plus-one event: a pair-plus-vacuum event needs a pair in one arm and vacuum in the other, and a pair-plus-single event needs a pair in one arm and a single photon in the other. The density matrix is normalized once, after all weighted events are summed.

Returns
-------
float: the Bell-state fidelity F = <Phi+|rho_meas|Phi+>.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cnot_gate_fidelity(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                       g2_s: float, nbar_s: float) -> float:
    """Bell-state fidelity of the PPBS CNOT gate with multiphoton events.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: propagated from invalid eta, m_sn or arm statistics.

    Expected return:
        float: F in (0, 1], equal to 1 only for perfectly indistinguishable
        ideal single photons and decreasing as g2 grows.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: cnot_gate_fidelity
def _oracle_cnot_gate_fidelity(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                               g2_s: float, nbar_s: float) -> float:
    w_m, r_m = _oracle_arm_emission_ratios(g2_m, nbar_m)
    w_s, r_s = _oracle_arm_emission_ratios(g2_s, nbar_s)
    table = _oracle_cnot_event_table(eta, m_sn)
    event_weights = np.array([1.0, w_m * r_s, w_s * r_m, w_m, w_s])
    return float((event_weights @ table[:, 1]) / (event_weights @ table[:, 0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '',
            'call': 'round(cnot_gate_fidelity(0.96, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'gold_call': 'round(_oracle_cnot_gate_fidelity(0.96, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(cnot_gate_fidelity(1.0, 1.0, 0.0, 0.5, 0.0, 0.5), 8)',
            'gold_call': 'round(_oracle_cnot_gate_fidelity(1.0, 1.0, 0.0, 0.5, 0.0, 0.5), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(cnot_gate_fidelity(0.85, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'gold_call': 'round(_oracle_cnot_gate_fidelity(0.85, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(cnot_gate_fidelity(0.7, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'gold_call': 'round(_oracle_cnot_gate_fidelity(0.7, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        cnot_gate_fidelity(1.2, 0.25, 0.018, 0.42, 0.041, 0.77)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_cnot_gate_fidelity(1.2, 0.25, 0.018, 0.42, 0.041, 0.77)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
