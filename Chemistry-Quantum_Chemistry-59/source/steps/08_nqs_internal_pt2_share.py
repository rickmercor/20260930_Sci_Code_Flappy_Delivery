"""
Orchestrate deterministic support construction and PT2 error attribution.

The complete workflow converts neural-state amplitudes into probabilities, selects a compact variational support, constructs its determinant Hamiltonian, screens a dynamic perturbative space and separates the Epstein–Nesbet correction into internal and external channels. The requested signed percentage measures how much of the complete second-order correction comes from the residual remaining inside the selected support.

Returns
-------
The signed internal contribution to the total PT2 correction, expressed as a percentage.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nqs_internal_pt2_share(
    determinants: 'np.ndarray' = None,
    h1: 'np.ndarray' = None,
    g2: 'np.ndarray' = None,
    signs: 'np.ndarray' = None,
    logabs: 'np.ndarray' = None,
    mass_fraction: float = None,
    eps1: float = None,
    min_k: int = None,
    max_k: int = None,
    nuclear_repulsion: float = None,
) -> float:
    """Run steps 1--7 and return the internal share of total PT2 in percent.

    With no arguments, run the benchmark fixture from the problem statement.
    Otherwise every argument is required.

    Returns
    -------
    float
        ``100 * delta_internal / (delta_internal + delta_external)``.

    Raises
    ------
    ValueError
        If an earlier step rejects an input or the total correction is zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np

def _oracle_nqs_internal_pt2_share(
    determinants: 'np.ndarray' = None,
    h1: 'np.ndarray' = None,
    g2: 'np.ndarray' = None,
    signs: 'np.ndarray' = None,
    logabs: 'np.ndarray' = None,
    mass_fraction: float = None,
    eps1: float = None,
    min_k: int = None,
    max_k: int = None,
    nuclear_repulsion: float = None,
) -> float:

    supplied = (
        determinants,
        h1,
        g2,
        signs,
        logabs,
        mass_fraction,
        eps1,
        min_k,
        max_k,
        nuclear_repulsion,
    )

    if all(value is None for value in supplied):
        (
            determinants,
            h1,
            g2,
            signs,
            logabs,
            mass_fraction,
            eps1,
            min_k,
            max_k,
            nuclear_repulsion,
        ) = _make_nqs_fixture(
            26090410,
            5,
            28,
            0.86,
            0.012,
            6,
            11,
            1.3275,
        )
    elif any(value is None for value in supplied):
        raise ValueError(
            "either supply every argument or use the benchmark defaults"
        )

    dets = np.asarray(determinants)
    one = np.asarray(h1, dtype=float)
    n_orb = one.shape[0] // 2 if one.ndim == 2 else 0

    h = _oracle_build_selected_hamiltonian(
        dets,
        one,
        g2,
        n_orb,
    )

    probability = _oracle_normalized_configuration_probabilities(
        signs,
        logabs,
    )

    v = _oracle_select_cumulative_support(
        probability,
        mass_fraction,
        min_k,
        max_k,
    )

    psi = np.asarray(signs, dtype=float) * np.sqrt(probability)

    p = _oracle_screen_dynamic_perturbative_space(
        h,
        psi,
        dets,
        v,
        eps1,
    )

    ledger = _oracle_deterministic_energy_diagnostics(
        h,
        psi,
        v,
        p,
    )

    pt2 = _oracle_decomposed_epstein_nesbet_pt2(
        h,
        psi,
        v,
        p,
        ledger[0] + float(nuclear_repulsion),
        nuclear_repulsion,
    )

    if not np.isfinite(pt2[2]) or abs(float(pt2[2])) <= 1e-15:
        raise ValueError(
            "total PT2 correction must be finite and nonzero"
        )

    return float(100.0 * pt2[0] / pt2[2])


def _make_nqs_fixture(
    seed,
    n_orb,
    n_target,
    mass_fraction,
    eps1,
    min_k,
    max_k,
    nuclear_repulsion,
):
    rng = np.random.default_rng(seed)
    n_so = 2 * n_orb

    h_spatial = rng.normal(
        scale=0.18,
        size=(n_orb, n_orb),
    )
    h_spatial = 0.5 * (h_spatial + h_spatial.T)
    h_spatial[np.diag_indices(n_orb)] -= np.linspace(
        1.55,
        0.25,
        n_orb,
    )

    h1 = np.zeros((n_so, n_so), dtype=float)

    for p in range(n_so):
        for q in range(n_so):
            if p % 2 == q % 2:
                h1[p, q] = h_spatial[p // 2, q // 2]

    pairs = [
        (p, q)
        for p in range(n_so)
        for q in range(p + 1, n_so)
    ]

    pair_block = rng.normal(
        scale=0.055,
        size=(len(pairs), len(pairs)),
    )
    pair_block = 0.5 * (pair_block + pair_block.T)

    spin_count = np.array(
        [(p & 1) + (q & 1) for p, q in pairs]
    )
    pair_block *= spin_count[:, None] == spin_count[None, :]

    g2 = np.zeros(
        (n_so, n_so, n_so, n_so),
        dtype=float,
    )

    for a, (p, q) in enumerate(pairs):
        for b, (r, s) in enumerate(pairs):
            value = pair_block[a, b]
            g2[p, q, r, s] = value
            g2[q, p, r, s] = -value
            g2[p, q, s, r] = -value
            g2[q, p, s, r] = value

    alpha_masks = [
        sum(1 << i for i in occ)
        for occ in itertools.combinations(range(n_orb), 2)
    ]

    beta_masks = [
        sum(1 << i for i in occ)
        for occ in itertools.combinations(range(n_orb), 2)
    ]

    archive = np.array(
        list(itertools.product(alpha_masks, beta_masks)),
        dtype=np.int64,
    )

    if not 1 <= int(n_target) <= archive.shape[0]:
        raise ValueError(
            "n_target exceeds the determinant archive"
        )

    determinants = archive[
        rng.permutation(archive.shape[0])[: int(n_target)]
    ]

    orbital_diag = np.diag(h_spatial)
    one_body_diag = np.empty(int(n_target), dtype=float)

    for k, (alpha, beta) in enumerate(determinants):
        occupied = [
            i
            for i in range(n_orb)
            if (int(alpha) >> i) & 1
        ]
        occupied += [
            i
            for i in range(n_orb)
            if (int(beta) >> i) & 1
        ]
        one_body_diag[k] = np.sum(orbital_diag[occupied])

    logabs = -1.15 * (
        one_body_diag - np.min(one_body_diag)
    )
    logabs += rng.normal(
        scale=0.22,
        size=int(n_target),
    )

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=int(n_target),
    )
    signs[np.argmax(logabs)] = 1.0

    return (
        determinants,
        h1,
        g2,
        signs,
        logabs,
        float(mass_fraction),
        float(eps1),
        int(min_k),
        int(max_k),
        float(nuclear_repulsion),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "nqs_internal_pt2_share()",
            "gold_call": "_oracle_nqs_internal_pt2_share()",
        },
        {
            "setup": "import numpy as np\nargs=_make_nqs_fixture(3107,4,18,.82,.010,4,8,.91)",
            "call": "nqs_internal_pt2_share(*tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args))",
            "gold_call": "_oracle_nqs_internal_pt2_share(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_nqs_fixture(902,5,24,.88,.014,5,10,1.6)",
            "call": "nqs_internal_pt2_share(*tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args))",
            "gold_call": "_oracle_nqs_internal_pt2_share(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_nqs_fixture(1441,4,20,.75,.008,3,7,.5)",
            "call": "nqs_internal_pt2_share(*tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args))",
            "gold_call": "_oracle_nqs_internal_pt2_share(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_nqs_fixture(1771,4,16,.8,.01,4,7,.7); args=(args[0],)+tuple(None for _ in range(9))\ndef check(fn):\n try: fn(*tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args))\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(nqs_internal_pt2_share)",
            "gold_call": "check(_oracle_nqs_internal_pt2_share)",
        },
    ]
