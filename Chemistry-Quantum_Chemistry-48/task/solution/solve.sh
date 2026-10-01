#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def enumerate_fock_states(n_orbitals: int, n_electrons: int) -> np.ndarray:
    """Reference fixed-particle bitstring enumeration."""
    from itertools import combinations
    import numpy as np

    if not isinstance(n_orbitals, (int, np.integer)):
        raise ValueError("n_orbitals must be an integer")
    if not isinstance(n_electrons, (int, np.integer)):
        raise ValueError("n_electrons must be an integer")

    m = int(n_orbitals)
    n = int(n_electrons)

    if m < 1 or m > 62 or n < 0 or n > m:
        raise ValueError("invalid orbital or electron count")

    states = [
        sum(1 << p for p in occupied)
        for occupied in combinations(range(m), n)
    ]
    return np.asarray(sorted(states), dtype=np.int64)

def expand_slater_determinant(orbital_rows, basis_states):
    """Reference minor-determinant expansion."""
    import numpy as np

    orbitals = np.asarray(orbital_rows, dtype=float)
    basis = np.asarray(basis_states)

    if orbitals.ndim != 2 or orbitals.shape[1] < 1:
        raise ValueError("orbital_rows must be two-dimensional")
    if basis.ndim != 1 or basis.size < 1:
        raise ValueError(
            "basis_states must be nonempty and one-dimensional"
        )
    if not np.all(np.isfinite(orbitals)) or not np.all(np.isfinite(basis)):
        raise ValueError("inputs must be finite")
    if not np.all(basis == np.floor(basis)):
        raise ValueError("basis states must be integers")

    n, m = orbitals.shape
    basis = basis.astype(np.int64)

    if np.any(basis < 0) or len(set(map(int, basis))) != basis.size:
        raise ValueError("basis states must be distinct and nonnegative")

    amplitudes = []
    for raw in basis:
        state = int(raw)
        if state >> m or state.bit_count() != n:
            raise ValueError("basis state belongs to the wrong sector")
        occupied = [p for p in range(m) if state & (1 << p)]
        amplitudes.append(np.linalg.det(orbitals[:, occupied]))

    return np.asarray(amplitudes, dtype=float)

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


def build_active_space_hamiltonian(
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

def transform_orbital_frames(
    determinant_orbitals,
    mixing_matrices,
):
    """Reference state-preserving row transformation."""
    import numpy as np

    orbitals = np.asarray(determinant_orbitals, dtype=float)
    mixing = np.asarray(mixing_matrices, dtype=float)

    if (
        orbitals.ndim != 3
        or orbitals.shape[0] < 1
        or orbitals.shape[1] < 1
    ):
        raise ValueError(
            "determinant_orbitals must be three-dimensional"
        )

    n_det, n, m = orbitals.shape

    if n > m or mixing.shape != (n_det, n, n):
        raise ValueError("mixing matrices do not align")
    if (
        not np.all(np.isfinite(orbitals))
        or not np.all(np.isfinite(mixing))
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(
        np.linalg.det(mixing),
        1.0,
        rtol=0.0,
        atol=2e-10,
    ):
        raise ValueError(
            "every mixing matrix must have determinant one"
        )

    return np.einsum(
        "dij,djm->dim",
        mixing,
        orbitals,
        optimize=True,
    )

def _create(state: int, orbital: int):
    mask = 1 << orbital
    if state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state | mask, sign


def assemble_hole_lift(
    hole_amplitudes,
    hole_basis_states,
    electron_basis_states,
    n_orbitals,
):
    """Reference creation lift with occupation parity."""
    import numpy as np

    amplitudes = np.asarray(hole_amplitudes, dtype=float)
    holes = np.asarray(hole_basis_states)
    electrons = np.asarray(electron_basis_states)

    if (
        not isinstance(n_orbitals, (int, np.integer))
        or int(n_orbitals) < 1
    ):
        raise ValueError("n_orbitals must be positive")

    m = int(n_orbitals)

    if (
        amplitudes.ndim != 2
        or amplitudes.shape[0] < 1
        or amplitudes.shape[1] != holes.size
    ):
        raise ValueError("hole amplitudes and basis do not align")
    if (
        holes.ndim != 1
        or electrons.ndim != 1
        or holes.size < 1
        or electrons.size < 1
    ):
        raise ValueError(
            "basis arrays must be nonempty and one-dimensional"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be finite")
    if (
        not np.all(np.isfinite(holes))
        or not np.all(holes == np.floor(holes))
    ):
        raise ValueError("hole states must be finite integers")
    if (
        not np.all(np.isfinite(electrons))
        or not np.all(electrons == np.floor(electrons))
    ):
        raise ValueError("electron states must be finite integers")

    holes = holes.astype(np.int64)
    electrons = electrons.astype(np.int64)

    hole_counts = {int(x).bit_count() for x in holes}
    electron_counts = {int(x).bit_count() for x in electrons}

    if len(hole_counts) != 1 or len(electron_counts) != 1:
        raise ValueError("each basis must have fixed particle count")
    if next(iter(electron_counts)) != next(iter(hole_counts)) + 1:
        raise ValueError("sectors must differ by one electron")

    lookup = {int(state): i for i, state in enumerate(electrons)}

    if len(lookup) != electrons.size:
        raise ValueError("electron states must be unique")

    lift = np.zeros(
        (electrons.size, amplitudes.shape[0] * m)
    )

    for determinant in range(amplitudes.shape[0]):
        for mu in range(m):
            for j, raw in enumerate(holes):
                created = _create(int(raw), mu)
                if created is not None:
                    state, sign = created
                    if state not in lookup:
                        raise ValueError(
                            "basis is not closed under creation"
                        )
                    lift[
                        lookup[state],
                        determinant * m + mu,
                    ] += sign * amplitudes[determinant, j]

    return lift

def form_eidos_pencil(
    lift_columns,
    sector_hamiltonian,
):
    """Reference effective quadratic forms."""
    import numpy as np

    lift = np.asarray(lift_columns, dtype=float)
    hamiltonian = np.asarray(
        sector_hamiltonian,
        dtype=float,
    )

    if (
        lift.ndim != 2
        or lift.shape[0] < 1
        or lift.shape[1] < 1
    ):
        raise ValueError(
            "lift_columns must be a nonempty matrix"
        )
    if hamiltonian.shape != (
        lift.shape[0],
        lift.shape[0],
    ):
        raise ValueError("Hamiltonian and lift do not align")
    if (
        not np.all(np.isfinite(lift))
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(
        hamiltonian,
        hamiltonian.T,
        rtol=0.0,
        atol=1e-11,
    ):
        raise ValueError("Hamiltonian must be symmetric")

    metric = lift.T @ lift
    effective_hamiltonian = (
        lift.T @ hamiltonian @ lift
    )

    return np.stack(
        [
            0.5
            * (
                effective_hamiltonian
                + effective_hamiltonian.T
            ),
            0.5 * (metric + metric.T),
        ]
    )

def solve_eidos_update(
    matrix_pencil,
    relative_cutoff,
):
    """Reference positive-subspace whitening solution."""
    import numpy as np

    pencil = np.asarray(matrix_pencil, dtype=float)

    if (
        pencil.ndim != 3
        or pencil.shape[0] != 2
        or pencil.shape[1] < 1
        or pencil.shape[1] != pencil.shape[2]
    ):
        raise ValueError(
            "matrix_pencil must have shape (2,n,n)"
        )
    if not np.all(np.isfinite(pencil)):
        raise ValueError("matrix_pencil must be finite")
    if (
        not np.isfinite(relative_cutoff)
        or relative_cutoff <= 0.0
        or relative_cutoff >= 1.0
    ):
        raise ValueError(
            "relative_cutoff must lie between zero and one"
        )

    hamiltonian, metric = pencil

    if not np.allclose(
        hamiltonian,
        hamiltonian.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "effective Hamiltonian must be symmetric"
        )
    if not np.allclose(
        metric,
        metric.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "effective metric must be symmetric"
        )

    values, vectors = np.linalg.eigh(metric)
    scale = float(values[-1])

    if scale <= 0.0 or values[0] < -1e-9 * scale:
        raise ValueError(
            "metric must be positive semidefinite and nonzero"
        )

    keep = values > relative_cutoff * scale

    if not np.any(keep):
        raise ValueError("cutoff removes the full space")

    whitening = (
        vectors[:, keep]
        / np.sqrt(values[keep])[None, :]
    )

    reduced = (
        whitening.T @ hamiltonian @ whitening
    )
    reduced = 0.5 * (reduced + reduced.T)

    energies, reduced_vectors = np.linalg.eigh(reduced)
    energy = float(energies[0])
    vector = whitening @ reduced_vectors[:, 0]

    vector /= np.sqrt(
        float(vector @ metric @ vector)
    )

    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector

    residual = np.linalg.norm(
        hamiltonian @ vector
        - energy * metric @ vector
    )

    return np.concatenate(
        (
            [
                energy,
                float(np.count_nonzero(keep)),
                residual,
            ],
            vector,
        )
    )

def select_orbital_update(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    determinant_orbitals: np.ndarray,
    mixing_candidates: np.ndarray,
    nuclear_energy: float,
    metric_cutoff: float,
) -> int:
    """Compose the seven orbital-update steps and rank candidate frames."""
    import numpy as np

    orbitals = np.asarray(determinant_orbitals, dtype=float)
    candidates = np.asarray(mixing_candidates, dtype=float)

    if (
        orbitals.ndim != 3
        or orbitals.shape[0] < 1
        or orbitals.shape[1] < 1
    ):
        raise ValueError(
            "determinant_orbitals must have shape "
            "(n_det,n_electrons,n_orbitals)"
        )

    n_det, n_electrons, n_orbitals = orbitals.shape
    expected = (n_det, n_electrons, n_electrons)

    if (
        candidates.ndim != 4
        or candidates.shape[0] < 1
        or candidates.shape[1:] != expected
    ):
        raise ValueError(
            "mixing_candidates has an invalid shape"
        )

    hole_basis = enumerate_fock_states(
        n_orbitals,
        n_electrons - 1,
    )
    electron_basis = enumerate_fock_states(
        n_orbitals,
        n_electrons,
    )

    hamiltonian = build_active_space_hamiltonian(
        one_body,
        antisym_two_body,
        electron_basis,
        nuclear_energy,
    )

    ranking = []

    for index, mixing in enumerate(candidates):
        transformed = transform_orbital_frames(
            orbitals,
            mixing,
        )

        hole_amplitudes = np.stack(
            [
                expand_slater_determinant(
                    transformed[i, 1:, :],
                    hole_basis,
                )
                for i in range(n_det)
            ]
        )

        lift = assemble_hole_lift(
            hole_amplitudes,
            hole_basis,
            electron_basis,
            n_orbitals,
        )

        pencil = form_eidos_pencil(
            lift,
            hamiltonian,
        )

        solution = solve_eidos_update(
            pencil,
            metric_cutoff,
        )

        ranking.append(
            (float(solution[0]), index)
        )

    return int(min(ranking)[1] + 1)
SCICODE_GOLD_EOF
