#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def _psc2_mul(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = (a.size - 1) // 2
    phase = (int(a[0]) + int(b[0]) + 2 * int(np.dot(a[1+k:], b[1:1+k]) % 2)) % 4
    x = np.bitwise_xor(a[1:1+k], b[1:1+k])
    z = np.bitwise_xor(a[1+k:], b[1+k:])
    return np.concatenate(([phase], x, z)).astype(int)


def _psc2_dagger(a):
    a = np.asarray(a, dtype=int)
    k = (a.size - 1) // 2
    phase = (-int(a[0]) + 2 * int(np.dot(a[1:1+k], a[1+k:]) % 2)) % 4
    return np.concatenate(([phase], a[1:])).astype(int)


def _psc2_symp(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = (a.size - 1) // 2
    return int((np.dot(a[1:1+k], b[1+k:]) + np.dot(a[1+k:], b[1:1+k])) % 2)


def _psc2_labels_to_code(labels, extra_phase=0):
    labels = np.asarray(labels, dtype=int)
    x = np.isin(labels, [1, 2]).astype(int)
    z = np.isin(labels, [2, 3]).astype(int)
    phase = (int(extra_phase) + int(np.sum(labels == 2))) % 4
    return np.concatenate(([phase], x, z)).astype(int)


def _psc2_is_identity(code):
    code = np.asarray(code, dtype=int)
    return int(code[0]) % 4 == 0 and not np.any(code[1:])


def psc_square_pauli(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray") -> "np.ndarray":
    p = np.asarray(p_code, dtype=int)
    qs = np.asarray(q_codes, dtype=int)
    out = np.zeros_like(p)
    out[0] = int(alpha_power) % 4  # alpha^2 = i^a
    out = _psc2_mul(out, _psc2_mul(p, p))
    for q in qs:
        if _psc2_symp(p, q) == 0:
            iq = np.array(q, dtype=int, copy=True)
            iq[0] = (int(iq[0]) + 1) % 4
            out = _psc2_mul(out, iq)
    return out

import numpy as np


def psc_conjugate_pauli(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    p = np.asarray(p_code, dtype=int)
    out = np.array(pauli_code, dtype=int, copy=True)
    for q in np.asarray(q_codes, dtype=int):
        if _psc2_symp(q, out):
            out = _psc2_mul(q, out)
            out[0] = (int(out[0]) + 1) % 4
    if _psc2_symp(p, out):
        out[0] = (int(out[0]) + 2) % 4
    return out

import numpy as np


def _psc2_relation_with_u(p_code, q_codes, pauli_code):
    r = np.asarray(pauli_code, dtype=int)
    image = psc_conjugate_pauli(p_code, q_codes, r)
    if not np.array_equal(image[1:], r[1:]):
        raise ValueError('Pauli is not a +/- eigenoperator of conjugation by this PSC')
    delta = (int(image[0]) - int(r[0])) % 4
    if delta not in (0, 2):
        raise ValueError('unexpected Pauli phase relation')
    return delta // 2


def psc_commutator_signature(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    r = np.asarray(pauli_code, dtype=int)
    ur = psc_conjugate_pauli(p_code, q_codes, r)
    qprime = _psc2_mul(ur, _psc2_dagger(r))
    parity = _psc2_relation_with_u(p_code, q_codes, qprime)
    return np.concatenate((qprime, [parity])).astype(int)

import numpy as np


def _psc2_conjugate_by_pauli(a, b):
    return _psc2_mul(_psc2_dagger(a), _psc2_mul(b, a))


def controlled_psc_fault_factor(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", control_label: int, target_code: "np.ndarray") -> "np.ndarray":
    r = np.asarray(target_code, dtype=int)
    sig = psc_commutator_signature(p_code, q_codes, r)
    qprime = sig[:-1]
    if int(control_label) in (0, 3):
        ptarget = r.copy()
        q = _psc2_conjugate_by_pauli(r, qprime)
        b = 0
    else:
        ptarget = psc_conjugate_pauli(p_code, q_codes, r)
        u2 = psc_square_pauli(alpha_power, p_code, q_codes)
        q = _psc2_mul(qprime, _psc2_dagger(u2))
        b = 1
    return np.concatenate(([int(control_label)], ptarget, q, [b])).astype(int)

import numpy as np


def _psc2_make_row(gtype, k, c1=-1, aux=-1, code=None):
    row = np.full(4 + 2 * k, -1, dtype=int)
    row[0], row[1], row[2] = int(gtype), int(c1), int(aux)
    if code is None:
        row[3:] = 0
    else:
        code = np.asarray(code, dtype=int)
        row[3] = int(code[0])
        row[4:4+k] = code[1:1+k]
        row[4+k:4+2*k] = code[1+k:]
    return row


def _psc2_row_code(row, k):
    row = np.asarray(row, dtype=int)
    return np.concatenate(([row[3]], row[4:4+k], row[4+k:4+2*k])).astype(int)


def compact_psc_schedule_propagation(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
    faults = np.asarray(fault_schedule, dtype=int)
    rounds, width = faults.shape
    k = width - 1
    gates = []
    for t in range(rounds):
        propagated = []
        for row in gates:
            kind = int(row[0])
            if kind == 0:
                r = _psc2_row_code(row, k)
                sig = psc_commutator_signature(p_code, q_codes, r)
                qprime = sig[:-1]
                q = _psc2_conjugate_by_pauli(r, qprime)
                propagated.append(row.copy())
                if not _psc2_is_identity(q):
                    propagated.append(_psc2_make_row(1, k, c1=t, code=q))
            elif kind == 1:
                q = _psc2_row_code(row, k)
                parity = _psc2_relation_with_u(p_code, q_codes, q)
                if parity:
                    propagated.append(_psc2_make_row(3, k, c1=int(row[1]), aux=t))
                propagated.append(row.copy())
            else:
                propagated.append(row.copy())

        target_code = _psc2_labels_to_code(faults[t, 1:])
        fac = controlled_psc_fault_factor(alpha_power, p_code, q_codes, int(faults[t, 0]), target_code)
        ncode = 1 + 2 * k
        ptarget = fac[1:1+ncode]
        q = fac[1+ncode:1+2*ncode]
        b = int(fac[-1])
        local = [_psc2_make_row(0, k, c1=t, aux=int(fac[0]), code=ptarget)]
        if not _psc2_is_identity(q):
            local.append(_psc2_make_row(1, k, c1=t, code=q))
        if b:
            local.append(_psc2_make_row(2, k))
        gates = local + propagated

    if not gates:
        return np.zeros((0, 4 + 2 * k), dtype=int)
    return np.stack(gates).astype(int)

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


def postselected_pauli_rank_statistics(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
    schedule = np.asarray(fault_schedule, dtype=int)
    rounds = schedule.shape[0]
    psi = np.asarray(target_state, dtype=complex)
    k = int(round(np.log2(psi.size)))
    rows = compact_psc_schedule_propagation(alpha_power, p_code, q_codes, schedule)
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

import numpy as np


def ensemble_compact_magic_fidelity(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedules: "np.ndarray", probabilities: "np.ndarray") -> float:
    schedules = np.asarray(fault_schedules, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    total_a = 0.0
    total_n = 0.0
    for schedule, prob in zip(schedules, probs):
        stats = postselected_pauli_rank_statistics(alpha_power, p_code, q_codes, target_state, schedule)
        total_a += float(prob) * float(stats[0])
        total_n += float(prob) * float(stats[1])
    return float(total_n / total_a)
SCICODE_GOLD_EOF
