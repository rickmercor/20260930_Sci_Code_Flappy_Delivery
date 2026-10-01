"""
Propagate a fresh-ancilla PSC measurement schedule as a compact Clifford gate list.

This is the source error-propagation algorithm specialized to repeated controlled-PSC rounds with pre-gate Pauli faults and no ancilla reuse. Existing propagated factors are pushed through each new round; controlled-Pauli factors can create control-control CZ factors when their target Pauli anticommutes with the PSC. The benchmark uses one exact canonical list representation: every round emits its type-0 Pauli row even when that row represents the identity; type-1 rows are emitted only for nonidentity controlled-Q factors; type-2 rows only when b=1; and type-3 rows only for induced CZ factors. Do not delete identity type-0 rows, cancel or merge factors, or reorder algebraically equivalent rows. The representation remains polynomial in the number of rounds and never forms a 2^(r+k)-dimensional propagated matrix.

Returns
-------
np.ndarray: ordered compact end-error gate list with row width 4+2k
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compact_psc_schedule_propagation(
    alpha_power: int,
    p_code: "np.ndarray",
    q_codes: "np.ndarray",
    fault_schedule: "np.ndarray",
) -> "np.ndarray":
    """Return the ordered compact end-error list for a pre-gate fault schedule.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    fault_schedule : np.ndarray
        Integer array of shape (r, k + 1). Row t contains the fresh-control
        Pauli label followed by k target labels, each using
        0=I, 1=X, 2=Y, 3=Z. The row is a Pauli layer immediately before
        controlled-U round t. Controls use zero-based round indices.

    Returns
    -------
    np.ndarray
        Integer array with row width 4 + 2*k. Rows are in left-to-right
        matrix-product order. The complete row formats are:

        Type 0: [0, control, control_label, phase, x..., z...].
            A Pauli on the named control and the k targets. The x and z
            blocks each contain k bits.

        Type 1: [1, control, -1, phase, x..., z...].
            Controlled-Q on the named control and the k targets. The x
            and z blocks each contain k bits.

        Type 2: [2, -1, -1, 0, x..., z...].
            Target-U, with both k-entry x and z blocks identically zero.
            The two unused index fields must be -1; every field after
            them must be zero.

        Type 3: [3, old_control, new_control, 0, x..., z...].
            CZ between the existing factor's control and the fresh
            control for the current round, in that order. Both k-entry
            x and z blocks are identically zero.

        Keep exactly one type-0 row for every measurement round, including
        an identity Pauli row. For round t, prepend local rows in the
        order type 0, optional nonidentity type 1, optional type 2, then
        the propagated older rows. Emit the local type-2 row only when
        the controlled-fault factorization has b=1.

        When propagating an older type-0 row, keep it and place any
        induced nonidentity type-1 row immediately after it. When
        propagating an older type-1 row, place any induced CZ row
        immediately before that type-1 row. Do not simplify, merge,
        cancel, drop, or reorder rows.

        An empty schedule returns an integer array of shape
        (0, 4 + 2*k).
    """
    return gate_rows

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_compact_psc_schedule_propagation(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
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
                sig = _oracle_psc_commutator_signature(p_code, q_codes, r)
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
        fac = _oracle_controlled_psc_fault_factor(alpha_power, p_code, q_codes, int(faults[t, 0]), target_code)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    hs = '''import numpy as np\nalpha_power=1; p_code=np.array([0,1,0,0,0],int); q_codes=np.array([[3,1,0,1,0],[2,0,0,0,1]],int)'''
    long_setup = '''import numpy as np\nalpha_power=3; p_code=np.array([1,1,0,1,0,0,1],int); q_codes=np.array([[2,0,0,0,1,0,0],[0,0,0,0,0,1,0],[2,0,0,0,0,0,1],[0,0,0,0,1,1,0]],int); fault_schedule=np.array([[1,1,0,3],[2,0,1,2],[3,2,2,0],[0,3,1,1],[1,0,2,3],[2,1,3,0],[3,3,0,2],[1,2,1,3],[0,0,3,1],[2,3,2,1],[1,1,3,2],[3,2,0,1]],int)'''
    return [
        {
            'setup': hs + '\nfault_schedule=np.array([[1,1,0]],int)',
            'call': 'compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'tol': 0,
        },
        {
            'setup': hs + '\nfault_schedule=np.array([[1,1,0],[0,0,0],[2,0,3],[3,1,2],[1,2,1]],int)',
            'call': 'compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'tol': 0,
        },
        {
            'setup': long_setup,
            'call': 'compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'tol': 0,
        },
        {
            'setup': hs + '\nfault_schedule=np.zeros((0,3),dtype=int)',
            'call': 'compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'gold_call': '_oracle_compact_psc_schedule_propagation(alpha_power,p_code.copy(),q_codes.copy(),fault_schedule.copy())',
            'tol': 0,
        },
    ]
