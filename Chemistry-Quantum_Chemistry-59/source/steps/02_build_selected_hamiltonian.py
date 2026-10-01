"""
Construct a determinant-space electronic Hamiltonian.

The electronic Hamiltonian between Slater determinants follows the Slater–Condon rules. Only diagonal, single-excitation and double-excitation matrix elements can be nonzero for a one- plus two-body Hamiltonian. The fermionic phase of every excitation must be obtained from the factorized-spin operator action defined in the preceding step.

Returns
-------
A real symmetric determinant-space Hamiltonian with shape (n, n), retaining the supplied determinant order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_selected_hamiltonian(
    determinants: 'np.ndarray',
    h1: 'np.ndarray',
    g2: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':
    """Build the electronic Hamiltonian on the supplied determinant archive.

    ``g2[p,q,r,s]`` contains real antisymmetrized spin-orbital integrals. Apply
    the source Slater--Condon cases for diagonal, single and double
    excitations. Excitation phase must be obtained with the factorized-spin
    operator action from step 1; excitations above degree two have zero matrix
    element. Retain the supplied determinant order.

    Returns
    -------
    np.ndarray
        Real symmetric Hamiltonian with shape ``(n,n)``.

        Raises
    ------
    ValueError
        If determinant or integral shapes, symmetries or values are invalid, or
        if the same determinant appears more than once.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_selected_hamiltonian(
    determinants: 'np.ndarray',
    h1: 'np.ndarray',
    g2: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':

    if not isinstance(n_orb, (int, np.integer)) or not 1 <= int(n_orb) <= 31:
        raise ValueError("invalid n_orb")

    n_orb = int(n_orb)
    n_so = 2 * n_orb
    dets = np.asarray(determinants)
    one = np.asarray(h1, dtype=float)
    two = np.asarray(g2, dtype=float)

    if dets.ndim != 2 or dets.shape[1] != 2 or dets.shape[0] == 0:
        raise ValueError("determinants must have shape (n,2)")
    if not np.issubdtype(dets.dtype, np.integer):
        raise ValueError("determinants must be integer masks")
    if one.shape != (n_so, n_so) or two.shape != (
        n_so,
        n_so,
        n_so,
        n_so,
    ):
        raise ValueError("integral shape mismatch")
    if np.any(~np.isfinite(one)) or np.any(~np.isfinite(two)):
        raise ValueError("integrals must be finite")
    if not np.allclose(one, one.T, atol=1e-12, rtol=0.0):
        raise ValueError("h1 must be symmetric")
    if not np.allclose(
        two,
        -np.swapaxes(two, 0, 1),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be antisymmetric in its first pair")
    if not np.allclose(
        two,
        -np.swapaxes(two, 2, 3),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be antisymmetric in its second pair")
    if not np.allclose(
        two,
        np.transpose(two, (2, 3, 0, 1)),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be Hermitian under pair exchange")

    keys = [tuple(map(int, d)) for d in dets]
    if len(set(keys)) != len(keys):
        raise ValueError("determinants must be unique")
    if any(
        a < 0
        or b < 0
        or a >= (1 << n_orb)
        or b >= (1 << n_orb)
        for a, b in keys
    ):
        raise ValueError("determinant mask outside n_orb")

    out = np.zeros((len(keys), len(keys)), dtype=float)

    def _occupied_spin_orbitals(det):
        alpha, beta = map(int, det)
        occ = [
            2 * p
            for p in range(n_orb)
            if (alpha >> p) & 1
        ]
        occ += [
            2 * p + 1
            for p in range(n_orb)
            if (beta >> p) & 1
        ]
        return occ

    def _spin_difference(bra_mask, ket_mask):
        holes_mask = ket_mask & ~bra_mask
        particles_mask = bra_mask & ~ket_mask
        holes = [
            p
            for p in range(n_orb)
            if (holes_mask >> p) & 1
        ]
        particles = [
            p
            for p in range(n_orb)
            if (particles_mask >> p) & 1
        ]
        return holes, particles

    for i, bra in enumerate(dets):
        bra_alpha, bra_beta = map(int, bra)

        for j, ket in enumerate(dets):
            ket_alpha, ket_beta = map(int, ket)

            holes_a, parts_a = _spin_difference(
                bra_alpha,
                ket_alpha,
            )
            holes_b, parts_b = _spin_difference(
                bra_beta,
                ket_beta,
            )

            if (
                len(holes_a) != len(parts_a)
                or len(holes_b) != len(parts_b)
            ):
                continue

            degree = len(holes_a) + len(holes_b)
            if degree > 2:
                continue

            if degree == 0:
                occ = _occupied_spin_orbitals(ket)
                value = sum(one[p, p] for p in occ)
                value += sum(
                    two[p, q, p, q]
                    for pos, p in enumerate(occ)
                    for q in occ[:pos]
                )
                out[i, j] = value
                continue

            annihilators = np.array(
                [2 * p for p in holes_a]
                + [2 * p + 1 for p in holes_b],
                dtype=np.int64,
            )
            creators = np.array(
                [2 * p + 1 for p in reversed(parts_b)]
                + [2 * p for p in reversed(parts_a)],
                dtype=np.int64,
            )

            acted = _oracle_apply_fermion_string(
                ket,
                annihilators,
                creators,
                n_orb,
            )
            if (
                acted[2] == 0
                or tuple(map(int, acted[:2]))
                != tuple(map(int, bra))
            ):
                raise ValueError(
                    "excitation analysis did not reproduce the bra"
                )

            phase = float(acted[2])

            if degree == 1:
                if holes_a:
                    hole = 2 * holes_a[0]
                    particle = 2 * parts_a[0]
                else:
                    hole = 2 * holes_b[0] + 1
                    particle = 2 * parts_b[0] + 1

                value = one[particle, hole]
                for occupied in _occupied_spin_orbitals(ket):
                    if occupied != hole:
                        value += two[
                            particle,
                            occupied,
                            hole,
                            occupied,
                        ]

                out[i, j] = phase * value
                continue

            if len(holes_a) == 2:
                h0, h1_index = (
                    2 * p for p in holes_a
                )
                p0, p1 = (
                    2 * p for p in parts_a
                )
            elif len(holes_b) == 2:
                h0, h1_index = (
                    2 * p + 1 for p in holes_b
                )
                p0, p1 = (
                    2 * p + 1 for p in parts_b
                )
            else:
                h0 = 2 * holes_a[0]
                p0 = 2 * parts_a[0]
                h1_index = 2 * holes_b[0] + 1
                p1 = 2 * parts_b[0] + 1

            out[i, j] = (
                phase
                * two[p0, p1, h0, h1_index]
            )

    if not np.allclose(
        out,
        out.T,
        atol=2e-11,
        rtol=0.0,
    ):
        raise ValueError(
            "constructed Hamiltonian is not Hermitian"
        )

    return 0.5 * (out + out.T)


def _small_integrals(n_orb, seed):
    rng = np.random.default_rng(seed)
    n_so = 2 * n_orb

    h = rng.normal(
        scale=0.2,
        size=(n_so, n_so),
    )
    h = 0.5 * (h + h.T)

    raw = rng.normal(
        scale=0.04,
        size=(n_so, n_so, n_so, n_so),
    )
    g = raw - np.swapaxes(raw, 0, 1)
    g = g - np.swapaxes(g, 2, 3)
    g = 0.5 * (
        g + np.transpose(g, (2, 3, 0, 1))
    )

    return h, g

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = "import numpy as np\n"
    return [
        {
            "setup": base + "d=np.array([[1,1],[2,1],[1,2],[2,2]],dtype=np.int64); h,g=_small_integrals(2,11)",
            "call": "build_selected_hamiltonian(d.copy(),h.copy(),g.copy(),2)",
            "gold_call": "_oracle_build_selected_hamiltonian(d,h,g,2)",
        },
        {
            "setup": base + "d=np.array([[3,1],[5,1],[3,2],[6,1],[5,2]],dtype=np.int64); h,g=_small_integrals(3,29)",
            "call": "build_selected_hamiltonian(d.copy(),h.copy(),g.copy(),3)",
            "gold_call": "_oracle_build_selected_hamiltonian(d,h,g,3)",
        },
        {
            "setup": base + "d=np.array([[3,3]],dtype=np.int64); h,g=_small_integrals(3,47)",
            "call": "build_selected_hamiltonian(d.copy(),h.copy(),g.copy(),3)",
            "gold_call": "_oracle_build_selected_hamiltonian(d,h,g,3)",
        },
        {
            "setup": base + "d=np.array([[3,3],[12,12],[5,10],[10,5]],dtype=np.int64); h,g=_small_integrals(4,73)",
            "call": "build_selected_hamiltonian(d.copy(),h.copy(),g.copy(),4)",
            "gold_call": "_oracle_build_selected_hamiltonian(d,h,g,4)",
        },
        {
            "setup": "import numpy as np\nd=np.array([[1,1],[1,1]],dtype=np.int64); h,g=_small_integrals(2,83)\ndef check(fn):\n try: fn(d.copy(),h.copy(),g.copy(),2)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(build_selected_hamiltonian)",
            "gold_call": "check(_oracle_build_selected_hamiltonian)",
        },
    ]
