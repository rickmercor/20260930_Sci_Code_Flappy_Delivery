"""
Tabulate, for each of the five emission events kept in the fidelity bookkeeping, the trace and the target-Bell-state overlap of that event's post-selected coincidence matrix at the output of the PPBS CNOT gate (control in |D>, target in |H>, target Bell state |Phi+>).

Rows are ordered (1,1), (2,0), (0,2), (2,1), (1,2), where (j,k) counts photons emitted into the control (memory) arm and the target (source) arm. Each entry refers to the post-selected coincidence matrix of the normalized event state, so it already carries the gate's post-selection probability. Pair-plus-vacuum events are separable and orthogonal to the target Bell state. The three-photon rows must include the events in which one photon exits an unmonitored PPBS port while the other two still produce a coincidence, and they depend on the noise overlap only through the effective indistinguishability of the previous step. Consult the paper's own event-by-event coincidence matrices rather than reusing the single-photon entries with a reduced overlap.

Returns
-------
np.ndarray of shape (5, 2): column 0 = trace, column 1 = <Phi+|sigma|Phi+>, rows ordered (1,1), (2,0), (0,2), (2,1), (1,2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cnot_event_table(eta: float, m_sn: float) -> "np.ndarray":
    """Trace and Bell overlap of each retained emission event's coincidence matrix.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap, 0 <= m_sn <= 1.

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        np.ndarray of shape (5, 2). Rows (1,1), (2,0), (0,2), (2,1), (1,2);
        column 0 is the trace, column 1 the overlap with |Phi+>. All entries
        are non-negative; the two pair-plus-vacuum rows have zero overlap; the
        two three-photon rows are equal to each other.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: cnot_event_table
def _oracle_cnot_event_table(eta: float, m_sn: float) -> "np.ndarray":
    e_eff = _oracle_effective_indistinguishability(eta, m_sn)
    tr11 = (2.0 - eta) / 9.0
    ov11 = (1.0 + eta) / 18.0
    tr_pair = 2.0 / 9.0
    tr3 = 2.0 * (3.0 - e_eff) / 9.0
    ov3 = (1.0 + e_eff) / 9.0
    return np.array([
        [tr11, ov11],
        [tr_pair, 0.0],
        [tr_pair, 0.0],
        [tr3, ov3],
        [tr3, ov3],
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np',
            'call': 'np.round(cnot_event_table(0.96, 0.25), 8)',
            'gold_call': 'np.round(_oracle_cnot_event_table(0.96, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(cnot_event_table(1.0, 1.0), 8)',
            'gold_call': 'np.round(_oracle_cnot_event_table(1.0, 1.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(cnot_event_table(0.7, 0.0), 8)',
            'gold_call': 'np.round(_oracle_cnot_event_table(0.7, 0.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(cnot_event_table(0.3, 0.6), 8)',
            'gold_call': 'np.round(_oracle_cnot_event_table(0.3, 0.6), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        cnot_event_table(0.9, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_cnot_event_table(0.9, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
