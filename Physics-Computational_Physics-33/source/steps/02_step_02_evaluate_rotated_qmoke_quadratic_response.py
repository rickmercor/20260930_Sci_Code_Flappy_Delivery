"""
Evaluate the rotated cubic-crystal quadratic magneto-optic response.

Construct the source paper's fourth-rank G tensor for a centrosymmetric cubic crystal and rotate it with exactly the same sample-frame conventions as Step 01. Inputs are the observable parameters G_s=G_11-G_12 and two_g44=2G_44. Fix the irrelevant isotropic diagonal gauge by taking G_12=0 and G_11=G_s before rotation. Include the minor-index copies needed for contraction with M_k M_l. Return only off-diagonal quadratic contributions in the same six-pair packet as Step 01; unlike the cubic-H packet, its reverse entries (32,13,21) equal the corresponding forward entries (23,31,12). Magnetization columns are [M_T,M_L,M_P], including arbitrary finite M_P values. Require a nonempty finite magnetization array of shape (N,3), finite scalar coefficients, a finite non-Boolean scalar alpha, and crystal_cut equal to '001' or '111'; raise ValueError otherwise.

Returns
-------
response : np.ndarray, shape (N,6,2), float Pair order (23,31,12,32,13,21), final axis real/imag.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_rotated_qmoke_quadratic_response(magnetization: np.ndarray, g_s: complex, two_g44: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    '''Return the rotated quadratic-G off-diagonal packet.

    Parameters
    ----------
    magnetization : np.ndarray, shape (N,3)
        Nonempty finite rows in M_T, M_L, M_P order.
    g_s, two_g44 : complex
        Finite scalars G_11-G_12 and 2G_44.
    crystal_cut : str
        Reference cut, '001' or '111'.
    alpha : float
        Finite non-Boolean scalar sample angle in radians.

    Returns
    -------
    response : np.ndarray, shape (N,6,2), float
        Pair order (23,31,12,32,13,21), final axis real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or violates a scalar or cut constraint.'''
    return np.empty((0, 6, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_evaluate_rotated_qmoke_quadratic_response__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f'{name} must be a finite scalar')
    return result

def _c_evaluate_rotated_qmoke_quadratic_response__sample_transform(crystal_cut, alpha):
    if crystal_cut not in ('001', '111'):
        raise ValueError('crystal_cut must be "001" or "111"')
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('alpha must be a finite non-Boolean scalar') from exc
    if not np.isfinite(alpha):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    c, s = (np.cos(alpha), np.sin(alpha))
    rotate_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if crystal_cut == '001':
        return rotate_z
    root2, root3, root6 = (np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0))
    to111 = np.array([[-root6 / 3.0, root6 / 6.0, root6 / 6.0], [0.0, -root2 / 2.0, root2 / 2.0], [root3 / 3.0, root3 / 3.0, root3 / 3.0]])
    return rotate_z @ to111

def _c_evaluate_rotated_qmoke_quadratic_response(magnetization: np.ndarray, g_s: complex, two_g44: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    M = np.asarray(magnetization, dtype=float)
    if M.ndim != 2 or M.shape[1] != 3 or M.shape[0] == 0:
        raise ValueError('magnetization must have nonempty shape (N,3)')
    if not np.all(np.isfinite(M)):
        raise ValueError('magnetization must be finite')
    g_s = _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(g_s, 'g_s')
    two_g44 = _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(two_g44, 'two_g44')
    g44 = 0.5 * two_g44
    tensor = np.zeros((3, 3, 3, 3), dtype=complex)
    for i in range(3):
        tensor[i, i, i, i] = g_s
    for i in range(3):
        for j in range(i + 1, 3):
            tensor[i, j, i, j] = g44
            tensor[i, j, j, i] = g44
            tensor[j, i, i, j] = g44
            tensor[j, i, j, i] = g44
    transform = _c_evaluate_rotated_qmoke_quadratic_response__sample_transform(crystal_cut, alpha)
    rotated = np.einsum('ia,jb,kc,ld,abcd->ijkl', transform, transform, transform, transform, tensor, optimize=True)
    epsilon = np.einsum('ijkl,nk,nl->nij', rotated, M, M, optimize=True)
    return _c_evaluate_rotated_qmoke_quadratic_response__packet(np.stack([epsilon[:, i, j] for i, j in ((1, 2), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0))], axis=1))

def _oracle_evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, crystal_cut, alpha):
    return _c_evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, crystal_cut, alpha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    return [
        {
            "setup": '# Case: boundary\n# Coverage: boundary (zero sample angle, (001) cut)\nM=np.array([[0.2,0.6,-0.3],[-0.7,0.1,0.4],[0.3,-0.5,0.8]])',
            "call": "evaluate_rotated_qmoke_quadratic_response(M,0.7,-0.12,'001',0.0)",
            "gold_call": "_oracle_evaluate_rotated_qmoke_quadratic_response(M,0.7,-0.12,'001',0.0)",
        },
        {
            "setup": '# Case: normal\nM=np.array([[0.13,-0.71,0.22],[-0.42,0.33,-0.59],[0.81,0.14,0.07],[0.28,-0.19,0.63]])',
            "call": "evaluate_rotated_qmoke_quadratic_response(M,0.19-0.44j,-0.31+0.09j,'001',0.37)",
            "gold_call": "_oracle_evaluate_rotated_qmoke_quadratic_response(M,0.19-0.44j,-0.31+0.09j,'001',0.37)",
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (zero sample angle, (111) cut)\nM=np.array([[0.2,0.6,-0.3],[-0.7,0.1,0.4],[0.3,-0.5,0.8]])',
            "call": "evaluate_rotated_qmoke_quadratic_response(M,-0.43+0.17j,0.21-0.08j,'111',0.0)",
            "gold_call": "_oracle_evaluate_rotated_qmoke_quadratic_response(M,-0.43+0.17j,0.21-0.08j,'111',0.0)",
        },
        {
            "setup": '# Case: normal\n# Coverage: normal (negative sample angle with nonzero M_P)\nM=np.array([[0.13,-0.71,0.22],[-0.42,0.33,-0.59],[0.81,0.14,0.07],[0.28,-0.19,0.63]])',
            "call": "evaluate_rotated_qmoke_quadratic_response(M,0.19-0.44j,-0.31+0.09j,'111',-0.41)",
            "gold_call": "_oracle_evaluate_rotated_qmoke_quadratic_response(M,0.19-0.44j,-0.31+0.09j,'111',-0.41)",
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": "_value_error_code(evaluate_rotated_qmoke_quadratic_response, np.zeros((1, 3)), 0.1, 0.2, '010', 0.0)",
            "gold_call": "_value_error_code(_oracle_evaluate_rotated_qmoke_quadratic_response, np.zeros((1, 3)), 0.1, 0.2, '010', 0.0)",
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": "_value_error_code(evaluate_rotated_qmoke_quadratic_response, np.zeros((0, 3)), 0.1, 0.2, '001', 0.0)",
            "gold_call": "_value_error_code(_oracle_evaluate_rotated_qmoke_quadratic_response, np.zeros((0, 3)), 0.1, 0.2, '001', 0.0)",
        },
        {
            "setup": '# Case: edge\nM=np.zeros((1,3))\n' + exception_setup,
            "call": "_value_error_code(evaluate_rotated_qmoke_quadratic_response, M, 0.1, 0.2, '001', 'not-an-angle')",
            "gold_call": "_value_error_code(_oracle_evaluate_rotated_qmoke_quadratic_response, M, 0.1, 0.2, '001', 'not-an-angle')",
        },
        {
            "setup": '# Case: edge\nM=np.zeros((1,3))\n' + exception_setup,
            "call": "_value_error_code(evaluate_rotated_qmoke_quadratic_response, M, 0.1, 0.2, '001', np.bool_(False))",
            "gold_call": "_value_error_code(_oracle_evaluate_rotated_qmoke_quadratic_response, M, 0.1, 0.2, '001', np.bool_(False))",
        },
        {
            "setup": '# Case: edge\nM=np.zeros((1,3))\n' + exception_setup,
            "call": "_value_error_code(evaluate_rotated_qmoke_quadratic_response, M, np.inf, 0.2, '001', 0.0)",
            "gold_call": "_value_error_code(_oracle_evaluate_rotated_qmoke_quadratic_response, M, np.inf, 0.2, '001', 0.0)",
        },
    ]
