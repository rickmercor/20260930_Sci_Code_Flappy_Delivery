"""
Orchestrate one certified deterministic ITHC-AFQMC projection block.

The complete deterministic block first certifies that the supplied ITHC factors reconstruct the reference electron-repulsion tensor. It then propagates every walker through the physical and extended representations, applies the phaseless importance update, evaluates the post-step local energies, and reduces the surviving ensemble to a normalized real mixed-energy estimate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ithc_afqmc_projected_energy(
    one_body: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
    channels: 'np.ndarray',
    eri_reference: 'np.ndarray',
    trial: 'np.ndarray',
    walkers: 'np.ndarray',
    walker_weights: 'np.ndarray',
    auxiliary_fields: 'np.ndarray',
    time_step: float,
    eri_tolerance: float,
) -> float:
    """Run one deterministic phaseless ITHC-AFQMC block.

    Parameters
    ----------
    one_body, half_one_body
        Physical-basis one-body Hamiltonian and its half-step propagator.
    isometry, kernel, channels
        ITHC isometry, interaction kernel, and channel factor.
    eri_reference
        Physical four-index ERI used to certify the supplied ITHC factors.
    trial
        Trial Slater determinant.
    walkers
        Batch of physical-basis walker determinants.
    walker_weights
        Nonnegative initial walker weights.
    auxiliary_fields
        One sampled auxiliary-field vector per walker.
    time_step
        Finite nonnegative propagation interval.
    eri_tolerance
        Finite nonnegative relative ERI-certificate tolerance.

    Notes
    -----
    Certify the ITHC factors first. Then call all seven preceding public
    functions to propagate every supplied walker, update its weight, evaluate
    its local energy, and form the normalized mixed estimate. All inputs are
    required; the function has no hidden built-in fixture mode.

    Returns
    -------
    float
        Normalized post-propagation mixed energy.

    Raises
    ------
    ValueError
        If inputs are missing, incompatible, nonfinite, or fail the ITHC
        certificate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ithc_afqmc_projected_energy(
    one_body: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
    channels: 'np.ndarray',
    eri_reference: 'np.ndarray',
    trial: 'np.ndarray',
    walkers: 'np.ndarray',
    walker_weights: 'np.ndarray',
    auxiliary_fields: 'np.ndarray',
    time_step: float,
    eri_tolerance: float,
) -> float:
    import numpy as np

    supplied = (
        one_body,
        half_one_body,
        isometry,
        kernel,
        channels,
        eri_reference,
        trial,
        walkers,
        walker_weights,
        auxiliary_fields,
        time_step,
        eri_tolerance,
    )
    if any(value is None for value in supplied):
        raise ValueError("all inputs are required")

    h = np.asarray(one_body, dtype=complex)
    half = np.asarray(half_one_body, dtype=complex)
    u = np.asarray(isometry, dtype=float)
    W = np.asarray(kernel, dtype=float)
    w = np.asarray(channels, dtype=float)
    Vref = np.asarray(eri_reference, dtype=float)
    A = np.asarray(trial, dtype=complex)
    batch = np.asarray(walkers, dtype=complex)
    weights = np.asarray(walker_weights, dtype=float)
    fields = np.asarray(auxiliary_fields, dtype=float)
    dt = float(time_step)
    tol = float(eri_tolerance)

    if batch.ndim != 3 or batch.shape[1:] != A.shape or batch.shape[0] == 0:
        raise ValueError("walkers have incompatible shape")
    if weights.shape != (batch.shape[0],):
        raise ValueError("walker_weights have incompatible shape")
    if fields.shape != (batch.shape[0], w.shape[0]):
        raise ValueError("auxiliary_fields have incompatible shape")
    if h.shape != (A.shape[0], A.shape[0]) or half.shape != h.shape:
        raise ValueError("one-body matrices have incompatible shape")
    if (
        w.ndim != 2
        or W.shape != (u.shape[1], u.shape[1])
        or w.shape[1] != u.shape[1]
    ):
        raise ValueError("ITHC channel shapes are incompatible")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("eri_tolerance must be finite and nonnegative")
    if not np.allclose(w.T @ w, W, rtol=1e-10, atol=1e-12):
        raise ValueError("channels do not factor the kernel")

    reconstructed = _oracle_reconstruct_ithc_eri(u, W)
    if Vref.shape != reconstructed.shape:
        raise ValueError("eri_reference has incompatible shape")
    scale = max(1.0, float(np.linalg.norm(Vref)))
    if np.linalg.norm(reconstructed - Vref) > tol * scale:
        raise ValueError("ITHC factors fail the ERI certificate")

    local_energies = []
    importance = []
    for index in range(batch.shape[0]):
        old_walker = batch[index]
        half_walker = half @ old_walker

        green_before = _oracle_extended_mixed_green(A, half_walker, u)
        bias = _oracle_ithc_force_bias(green_before, w, dt)

        new_walker = _oracle_propagate_ithc_walker(
            half_walker,
            half,
            u,
            w,
            fields[index],
            bias,
            dt,
        )

        importance.append(
            _oracle_phaseless_importance(
                A,
                old_walker,
                new_walker,
                fields[index],
                bias,
            )
        )

        green_extended = _oracle_extended_mixed_green(A, new_walker, u)
        green_physical = u @ green_extended @ u.T

        local_energies.append(
            _oracle_ithc_local_energy(
                h,
                W,
                green_physical,
                green_extended,
            )
        )

    updated_weights = weights * np.asarray(importance, dtype=float)

    return _oracle_phaseless_mixed_energy(
        np.asarray(local_energies, dtype=complex),
        updated_weights,
    )


def _make_ithc_fixture(
    seed,
    n_orbitals,
    n_auxiliary,
    n_electrons,
    n_walkers,
    time_step,
):
    import numpy as np

    rng = np.random.default_rng(seed)

    q, r = np.linalg.qr(
        rng.normal(size=(n_auxiliary, n_orbitals))
    )
    q = q * np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    u = q.T

    channels = rng.normal(
        scale=0.48,
        size=(n_auxiliary, n_auxiliary),
    )
    channels /= np.sqrt(n_auxiliary)
    kernel = channels.T @ channels

    raw_h = rng.normal(
        scale=0.18,
        size=(n_orbitals, n_orbitals),
    )
    one_body = (
        0.5 * (raw_h + raw_h.T)
        - 0.55 * np.eye(n_orbitals)
    )

    eigvals, eigvecs = np.linalg.eigh(one_body)
    half = (
        eigvecs * np.exp(-0.5 * time_step * eigvals)
    ) @ eigvecs.T

    trial, trial_r = np.linalg.qr(
        rng.normal(size=(n_orbitals, n_electrons))
    )
    trial = trial * np.where(
        np.diag(trial_r) >= 0.0,
        1.0,
        -1.0,
    )

    walkers = []
    for _ in range(n_walkers):
        candidate = (
            trial
            + 0.11 * rng.normal(size=trial.shape)
            + 0.07j * rng.normal(size=trial.shape)
        )
        walker, walker_r = np.linalg.qr(candidate)
        diagonal = np.diag(walker_r)
        phase = np.where(
            np.abs(diagonal) > 0.0,
            diagonal / np.abs(diagonal),
            1.0,
        )
        walkers.append(walker * phase)

    walkers = np.asarray(walkers)
    weights = 0.4 + rng.random(n_walkers)
    fields = rng.normal(
        scale=0.7,
        size=(n_walkers, n_auxiliary),
    )

    eri_reference = np.einsum(
        "pa,qa,ab,rb,sb->pqrs",
        u,
        u,
        kernel,
        u,
        u,
        optimize=True,
    )

    return (
        one_body,
        half,
        u,
        kernel,
        channels,
        eri_reference,
        trial,
        walkers,
        weights,
        fields,
        time_step,
        1e-11,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nargs=_make_ithc_fixture(509,5,8,2,7,.035)",
            "call": "ithc_afqmc_projected_energy(*args)",
            "gold_call": "_oracle_ithc_afqmc_projected_energy(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_ithc_fixture(91,4,4,2,5,.02)",
            "call": "ithc_afqmc_projected_energy(*args)",
            "gold_call": "_oracle_ithc_afqmc_projected_energy(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_ithc_fixture(307,3,6,1,4,0.)",
            "call": "ithc_afqmc_projected_energy(*args)",
            "gold_call": "_oracle_ithc_afqmc_projected_energy(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_make_ithc_fixture(811,6,9,3,6,.055)",
            "call": "ithc_afqmc_projected_energy(*args)",
            "gold_call": "_oracle_ithc_afqmc_projected_energy(*args)",
        },
    ]
