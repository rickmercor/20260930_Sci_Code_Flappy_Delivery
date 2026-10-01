"""
Construct the electronic Hamiltonian in a fixed occupation sector.

The supplied spin-orbital integrals are antisymmetrized, so the two-electron term carries a one-quarter prefactor. Creation and annihilation acquire the parity of the occupied positions below the acted-on index.

Returns
-------
Square floating Hamiltonian with dimension len(basis_states).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_active_space_hamiltonian(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    basis_states: np.ndarray,
    nuclear_energy: float,
) -> np.ndarray:
    """Build a fixed-sector spin-orbital Hamiltonian.

    Parameters
    ----------
    one_body : np.ndarray
        Symmetric one-electron matrix with shape (m, m).
    antisym_two_body : np.ndarray
        Antisymmetrized integrals v[p,q,r,s] with shape (m,m,m,m).
    basis_states : np.ndarray
        Distinct bitstrings sharing one electron count.
    nuclear_energy : float
        Scalar additive nuclear energy.

    Returns
    -------
    np.ndarray
        Symmetric sector Hamiltonian in basis_states order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _annihilate(state: int, orbital: int):
    mask = 1 << orbital
    if not state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state ^ mask, sign


def _create(state: int, orbital: int):
    mask = 1 << orbital
    if state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state | mask, sign


def _oracle_build_active_space_hamiltonian(
    one_body,
    antisym_two_body,
    basis_states,
    nuclear_energy,
):
    """Reference explicit second-quantized construction."""
    import numpy as np

    h = np.asarray(one_body, dtype=float)
    v = np.asarray(antisym_two_body, dtype=float)
    basis = np.asarray(basis_states)

    if h.ndim != 2 or h.shape[0] < 1 or h.shape[0] != h.shape[1]:
        raise ValueError("one_body must be square")

    m = h.shape[0]

    if v.shape != (m, m, m, m) or basis.ndim != 1 or basis.size < 1:
        raise ValueError("two-body tensor or basis shape is invalid")
    if (
        not np.all(np.isfinite(h))
        or not np.all(np.isfinite(v))
        or not np.isfinite(nuclear_energy)
    ):
        raise ValueError("Hamiltonian inputs must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-11):
        raise ValueError("one_body must be symmetric")
    if not np.all(np.isfinite(basis)) or not np.all(
        basis == np.floor(basis)
    ):
        raise ValueError("basis states must be finite integers")

    basis = basis.astype(np.int64)

    if np.any(basis < 0) or len(set(map(int, basis))) != basis.size:
        raise ValueError("basis states must be distinct and nonnegative")
    if any(int(state) >> m for state in basis):
        raise ValueError("basis state exceeds the orbital count")
    if len({int(state).bit_count() for state in basis}) != 1:
        raise ValueError("basis states must share one electron count")

    lookup = {int(state): i for i, state in enumerate(basis)}
    matrix = np.eye(basis.size) * float(nuclear_energy)

    for ket_index, raw in enumerate(basis):
        ket = int(raw)

        for q in range(m):
            aq = _annihilate(ket, q)
            if aq is None:
                continue
            state_q, sign_q = aq

            for p in range(m):
                cp = _create(state_q, p)
                if cp is not None:
                    bra, sign_p = cp
                    matrix[lookup[bra], ket_index] += (
                        h[p, q] * sign_q * sign_p
                    )

        for r in range(m):
            ar = _annihilate(ket, r)
            if ar is None:
                continue
            state_r, sign_r = ar

            for s in range(m):
                ass = _annihilate(state_r, s)
                if ass is None:
                    continue
                state_s, sign_s = ass

                for q in range(m):
                    cq = _create(state_s, q)
                    if cq is None:
                        continue
                    state_q, sign_q = cq

                    for p in range(m):
                        cp = _create(state_q, p)
                        if cp is not None and v[p, q, r, s] != 0.0:
                            bra, sign_p = cp
                            matrix[lookup[bra], ket_index] += (
                                0.25
                                * v[p, q, r, s]
                                * sign_r
                                * sign_s
                                * sign_q
                                * sign_p
                            )

    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=2e-10):
        raise ValueError(
            "integrals do not define a Hermitian Hamiltonian"
        )

    return 0.5 * (matrix + matrix.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    one = """import numpy as np
one_body=np.array([[1.,-0.2],[-0.2,2.]])
antisym_two_body=np.zeros((2,2,2,2))
basis_states=np.array([1,2])
nuclear_energy=0.5"""

    pair = """import numpy as np
one_body=np.zeros((2,2))
antisym_two_body=np.zeros((2,2,2,2))
antisym_two_body[0,1,0,1]=0.7
antisym_two_body[1,0,0,1]=-0.7
antisym_two_body[0,1,1,0]=-0.7
antisym_two_body[1,0,1,0]=0.7
basis_states=np.array([3])
nuclear_energy=0.0"""

    vacuum = """import numpy as np
one_body=np.eye(3)
antisym_two_body=np.zeros((3,3,3,3))
basis_states=np.array([0])
nuclear_energy=-1.2"""

    call = (
        "build_active_space_hamiltonian("
        "one_body, antisym_two_body, basis_states, nuclear_energy)"
    )
    gold = (
        "_oracle_build_active_space_hamiltonian("
        "one_body, antisym_two_body, basis_states, nuclear_energy)"
    )

    return [
        {"setup": one, "call": call, "gold_call": gold},
        {"setup": pair, "call": call, "gold_call": gold},
        {"setup": vacuum, "call": call, "gold_call": gold},
    ]
