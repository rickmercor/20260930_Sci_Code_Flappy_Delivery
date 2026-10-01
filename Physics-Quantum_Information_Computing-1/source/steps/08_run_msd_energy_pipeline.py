"""
Run the complete deterministic symmetric time-shift Krylov and moment-Lanczos pipeline, with central-difference degree $J=\texttt{degree}$. The public orchestrator must call each preceding public step in order, use derivative rows through order $2J$, use derivative order $q=1$ for preprocessing and the projected Hamiltonian, solve with the supplied strict overlap cutoff, compute moments through order $2J$, and use maximum Lanczos order $J$; return only the final mitigated energy.

The end-to-end method first minimizes the sampling/truncation balance in a centered symmetry sector, reconstructs a nonorthogonal Krylov projection from measured time-evolution expectations, resolves its stable positive overlap subspace, and reuses the same shifted data for moment-based classical continuation. The final scalar must retain every earlier convention: signed Toeplitz indexing, Hermitian projection, strict overlap retention, physical moment normalization with the overlap metric in the original nonorthogonal Krylov basis, the negative-square stop, and restoration of the energy center.

Returns
-------
float, the final moment-Lanczos mitigated ground-state energy in the original energy origin
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_msd_energy_pipeline(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
    relative_cutoff: float,
) -> float:
    """Run the complete shifted-propagator Krylov energy pipeline.

    Parameters
    ----------
    g_nonnegative : np.ndarray
        Finite complex sequence of nonnegative-time propagator expectations.
    krylov_dimension : int
        Positive Krylov dimension ``n``.
    degree : int
        Positive central-difference degree ``J``.
    stride : int
        Positive integer ratio ``tau / delta_t``.
    total_shots : float
        Positive finite shot count used in time-shift optimization.
    sector_minimum : float
        Finite lower energy bound of the symmetry sector.
    sector_maximum : float
        Finite upper energy bound, strictly larger than ``sector_minimum``.
    relative_cutoff : float
        Finite overlap cutoff strictly between 0 and 1.

    Returns
    -------
    final_energy : float
        Final moment-Lanczos mitigated ground-state energy in the original
        energy origin.

    Raises
    ------
    ValueError
        If any preceding scientific step rejects the supplied inputs or its
        deterministic intermediate result.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_msd_energy_pipeline(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
    relative_cutoff: float,
) -> float:
    import numpy as np

    coefficients = _oracle_compute_central_difference_coefficients(degree, 2 * degree)
    preprocessing = _oracle_compute_msd_preprocessing(
        coefficients[0],
        krylov_dimension,
        degree,
        total_shots,
        sector_minimum,
        sector_maximum,
    )
    propagators = _oracle_build_shifted_propagators(
        g_nonnegative,
        krylov_dimension,
        degree,
        stride,
    )
    overlap, power_matrices = _oracle_reconstruct_projected_matrices(
        propagators,
        coefficients,
        preprocessing[2],
    )
    _, state_coefficients, _, _ = _oracle_solve_resolved_krylov_ground_state(
        overlap,
        power_matrices[0],
        relative_cutoff,
    )
    moments = _oracle_compute_projected_hamiltonian_moments(
        power_matrices,
        overlap,
        state_coefficients,
    )
    final_energy, _, _, _, _, _ = _oracle_run_moment_lanczos(
        moments,
        degree,
        preprocessing[0],
    )
    if not np.isfinite(final_energy):
        raise ValueError("the final mitigated energy must be finite")
    return float(final_energy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    prompt_setup = """import numpy as np
g = np.array([
    1.0+0.0j,
    0.99429356005856029-0.025590728300645980j,
    0.97730685925486915-0.050590494877476529j,
    0.94943860829937488-0.074423809017078665j,
    0.91133730900167487-0.096544241297493519j,
    0.86388714249963516-0.116449649912794870j,
    0.80818080703044082-0.133693203305266430j,
    0.74548930367185484-0.147895607047029390j,
    0.67722726275148382-0.158754282232542730j,
    0.60491163012666282-0.166049446589216800j,
    0.53012217788209992-0.169649646725830630j,
    0.45445491616160788-0.169514288978398760j,
    0.37948284322780795-0.165693225887655530j,
    0.30671045822235593-0.158324864969491470j,
    0.23753537514773893-0.147629912812195420j,
    0.17321244307405181-0.133906119606736620j,
], dtype=np.complex128)
n = 4
degree = 3
stride = 4
shots = 1.0e14
sector_minimum = -1.2
sector_maximum = 1.8
cutoff = 1.0e-4
"""
    return [
        {
            "setup": prompt_setup,
            "call": "run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
            "gold_call": "_oracle_run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
        },
        {
            "setup": """import numpy as np
n = 4
degree = 3
stride = 4
shots = 1.0e14
sector_minimum = -1.2
sector_maximum = 1.8
cutoff = 1.0e-4
delta_t = 0.11838556114185986
energy_shift = 0.3
energies = np.array([-1.2, -0.35, 0.4, 1.2, 1.8]) - energy_shift
weights = np.array([0.08, 0.22, 0.30, 0.25, 0.15])
g = np.array([np.sum(weights * np.exp(-1j * energies * m * delta_t)) for m in range(16)], dtype=complex)
""",
            "call": "run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
            "gold_call": "_oracle_run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
        },
        {
            "setup": prompt_setup + "cutoff = 1.0e-6\n",
            "call": "run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
            "gold_call": "_oracle_run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
        },
        {
            "setup": prompt_setup + "stride = 3\n",
            "call": "run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
            "gold_call": "_oracle_run_msd_energy_pipeline(g, n, degree, stride, shots, sector_minimum, sector_maximum, cutoff)",
        },
        {
            "setup": """import numpy as np
g = np.ones(5, dtype=complex)
def run_model():
    try:
        run_msd_energy_pipeline(g, 4, 3, 4, 1.0e14, -1.2, 1.8, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_msd_energy_pipeline(g, 4, 3, 4, 1.0e14, -1.2, 1.8, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
