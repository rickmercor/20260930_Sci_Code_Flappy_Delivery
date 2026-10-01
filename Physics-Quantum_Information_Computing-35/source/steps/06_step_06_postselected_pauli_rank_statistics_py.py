"""
Evaluate postselected phase-insensitive statistics from the compact propagated error.

After the source propagation algorithm removes the non-Clifford protocol gates in favor of an end Clifford, the paper evaluates target fidelity through a Pauli-rank decomposition and postselected stabilizer simulation. This benchmark performs the same contraction exactly for small verification instances while consuming the compact gate list produced by the previous step.

Returns
-------
np.ndarray: length-2 float array [acceptance, acceptance-weighted target overlap].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def postselected_pauli_rank_statistics(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
    '''Return acceptance and acceptance-weighted target overlap for one schedule.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    target_state : np.ndarray
        Normalized k-qubit target state of length 2^k.
    fault_schedule : np.ndarray
        Shape-(r,k+1) pre-gate Pauli schedule.

    Returns
    -------
    np.ndarray
        Real length-2 vector [A,N], where A is postselection acceptance and N
        is the acceptance-weighted phase-insensitive target overlap.
    '''
    return statistics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _psc2_single_paulis():
    eye = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    return eye, x, y, z


def _psc2_code_matrix(code):
    code = np.asarray(code, dtype=int)
    k = (code.size - 1) // 2
    eye, xmat, _, zmat = _psc2_single_paulis()
    out = np.array([[1.0 + 0.0j]])
    for x, z in zip(code[1:1+k], code[1+k:]):
        local = np.linalg.matrix_power(xmat, int(x)) @ np.linalg.matrix_power(zmat, int(z))
        out = np.kron(out, local)
    return (1j ** int(code[0])) * out


def _psc2_canonical_matrix(alpha_power, p_code, q_codes):
    u = np.exp(1j * np.pi * int(alpha_power) / 4.0) * _psc2_code_matrix(p_code)
    for q in np.asarray(q_codes, dtype=int):
        qm = _psc2_code_matrix(q)
        u = u @ ((np.eye(qm.shape[0], dtype=complex) + 1j * qm) / np.sqrt(2.0))
    return u


def _psc2_embed_controlled_target(qmat, control, rounds, k):
    p0 = np.array([[1, 0], [0, 0]], dtype=complex)
    p1 = np.array([[0, 0], [0, 1]], dtype=complex)
    a = np.array([[1.0 + 0.0j]])
    b = np.array([[1.0 + 0.0j]])
    for j in range(rounds):
        a = np.kron(a, p0 if j == control else np.eye(2, dtype=complex))
        b = np.kron(b, p1 if j == control else np.eye(2, dtype=complex))
    return np.kron(a, np.eye(2 ** k, dtype=complex)) + np.kron(b, qmat)


def _psc2_embed_pauli_row(row, rounds, k):
    single = _psc2_single_paulis()
    out = np.array([[1.0 + 0.0j]])
    for j in range(rounds):
        out = np.kron(out, single[int(row[2])] if j == int(row[1]) else single[0])
    return np.kron(out, _psc2_code_matrix(_psc2_row_code(row, k)))


def _psc2_embed_cz(a, b, rounds, k):
    dim = 2 ** (rounds + k)
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        ba = (idx >> (rounds + k - 1 - int(a))) & 1
        bb = (idx >> (rounds + k - 1 - int(b))) & 1
        if ba and bb:
            diag[idx] = -1.0
    return np.diag(diag)


def _psc2_gate_list_matrix(rows, alpha_power, p_code, q_codes, rounds):
    rows = np.asarray(rows, dtype=int)
    k = (rows.shape[1] - 4) // 2
    u = _psc2_canonical_matrix(alpha_power, p_code, q_codes)
    out = np.eye(2 ** (rounds + k), dtype=complex)
    for row in rows:
        kind = int(row[0])
        if kind == 0:
            gate = _psc2_embed_pauli_row(row, rounds, k)
        elif kind == 1:
            gate = _psc2_embed_controlled_target(_psc2_code_matrix(_psc2_row_code(row, k)), int(row[1]), rounds, k)
        elif kind == 2:
            gate = np.kron(np.eye(2 ** rounds, dtype=complex), u)
        else:
            gate = _psc2_embed_cz(int(row[1]), int(row[2]), rounds, k)
        out = out @ gate
    return out


def _oracle_postselected_pauli_rank_statistics(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
    schedule = np.asarray(fault_schedule, dtype=int)
    rounds = schedule.shape[0]
    psi = np.asarray(target_state, dtype=complex)
    k = int(round(np.log2(psi.size)))
    rows = _oracle_compact_psc_schedule_propagation(alpha_power, p_code, q_codes, schedule)
    cprop = _psc2_gate_list_matrix(rows, alpha_power, p_code, q_codes, rounds)

    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    controls = np.array([1.0 + 0.0j])
    for _ in range(rounds):
        controls = np.kron(controls, plus)
    rho_data = np.outer(psi, psi.conj())
    rho_joint = np.kron(np.outer(controls, controls.conj()), rho_data)
    rho_out = cprop @ rho_joint @ cprop.conj().T
    dc, dd = 2 ** rounds, 2 ** k
    reshaped = rho_out.reshape(dc, dd, dc, dd)
    sigma = np.einsum('a,aibj,b->ij', controls.conj(), reshaped, controls)
    acceptance = float(np.real_if_close(np.trace(sigma)))

    single = _psc2_single_paulis()
    beta = []
    expectations = []
    for labels in itertools.product(range(4), repeat=k):
        pm = np.array([[1.0 + 0.0j]])
        for label in labels:
            pm = np.kron(pm, single[int(label)])
        beta.append(float(np.real_if_close(np.vdot(psi, pm @ psi))))
        expectations.append(float(np.real_if_close(np.trace(pm @ sigma))))
    numerator = float(np.dot(np.asarray(beta), np.asarray(expectations)) / (2 ** k))
    if abs(acceptance) < 5e-15:
        acceptance = 0.0
    if abs(numerator) < 5e-15:
        numerator = 0.0
    return np.array([acceptance, numerator], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = '''import numpy as np\nalpha_power=1; p_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int); h=np.array([np.cos(np.pi/8),np.sin(np.pi/8)],complex); target_state=np.kron(h,np.array([1,0],complex))'''
    return [
        {
            'setup': common + '\nfault_schedule=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],int)',
            'call': 'postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'tol': 2e-11,
        },
        {
            'setup': common + '\nfault_schedule=np.array([[1,1,0],[0,0,0],[0,0,0],[0,0,0]],int)',
            'call': 'postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'tol': 2e-11,
        },
        {
            'setup': common + '\nfault_schedule=np.array([[2,1,3],[3,0,1],[1,2,0],[0,3,3]],int)',
            'call': 'postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_postselected_pauli_rank_statistics(alpha_power,p_code.copy(),q_codes.copy(),target_state.copy(),fault_schedule.copy())',
            'tol': 3e-11,
        },
    ]
