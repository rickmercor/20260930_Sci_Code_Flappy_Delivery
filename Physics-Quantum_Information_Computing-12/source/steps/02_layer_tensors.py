"""
Turn a disentangler block matrix and an isometry block matrix into the MERA tensors U (four
legs) and W (three legs). Returns the tuple (U, W).

Legs carry the qubit value 0 or 1 and matrix indices use 2*q_left + q_right. The
disentangler is the whole block: U[a, b, c, d] = <a b| G_u |c d> with (a, b) the outputs and
(c, d) the inputs. The isometry is the block acting on the renormalised site on the LEFT
input qubit and a freshly prepared |0> on the RIGHT input qubit: W[a, b, c] = <a b| G_w |c
0>. Return float arrays of shapes (2, 2, 2, 2) and (2, 2, 2).

Returns
-------
tuple: (U, W) as float arrays of shapes (2, 2, 2, 2) and (2, 2, 2).
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layer_tensors(gate_u: "np.ndarray", gate_w: "np.ndarray") -> tuple:
    """Turn a disentangler block matrix and an isometry block matrix into the MERA tensors U
    (four legs) and W (three legs). Returns the tuple (U, W).

    Args:
        gate_u: 4x4 real matrix of the disentangler block.
        gate_w: 4x4 real matrix of the isometry block.

    Returns:
        tuple: (U, W) as float arrays of shapes (2, 2, 2, 2) and (2, 2, 2).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _layer_tensors(gate_u, gate_w):
    gu = np.asarray(gate_u, dtype=float)
    gw = np.asarray(gate_w, dtype=float)
    return gu.reshape(2, 2, 2, 2), gw[:, [0, 2]].reshape(2, 2, 2)


def _oracle_layer_tensors(gate_u: "np.ndarray", gate_w: "np.ndarray") -> tuple:
    return _layer_tensors(gate_u, gate_w)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(3); gate_u = np.linalg.qr(rng.normal(size=(4, 4)))[0]; gate_w = np.linalg.qr(rng.normal(size=(4, 4)))[0]',
            'call': 'layer_tensors(*copy.deepcopy((gate_u, gate_w)))[0]',
            'gold_call': '_oracle_layer_tensors(*copy.deepcopy((gate_u, gate_w)))[0]',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(3); gate_u = np.linalg.qr(rng.normal(size=(4, 4)))[0]; gate_w = np.linalg.qr(rng.normal(size=(4, 4)))[0]',
            'call': 'layer_tensors(*copy.deepcopy((gate_u, gate_w)))[1]',
            'gold_call': '_oracle_layer_tensors(*copy.deepcopy((gate_u, gate_w)))[1]',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; gate_u = np.arange(16.0).reshape(4, 4); gate_w = np.arange(16.0).reshape(4, 4)[::-1] + 1.0',
            'call': 'layer_tensors(*copy.deepcopy((gate_u, gate_w)))[1]',
            'gold_call': '_oracle_layer_tensors(*copy.deepcopy((gate_u, gate_w)))[1]',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; gate_u = np.arange(16.0).reshape(4, 4) + 1.0; gate_w = np.eye(4)',
            'call': 'layer_tensors(*copy.deepcopy((gate_u, gate_w)))[0]',
            'gold_call': '_oracle_layer_tensors(*copy.deepcopy((gate_u, gate_w)))[0]',
        },
    ]
