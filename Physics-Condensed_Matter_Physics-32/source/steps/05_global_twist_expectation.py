"""
Evaluate the thermal expectation value of the many-body twist (large-gauge translation) operator of the periodic SSH chain in logarithmic polar form.

The phase of the expectation value of Resta's twist operator is the ensemble geometric phase, a finite-temperature generalisation of the Zak phase that remains quantised by inversion symmetry. Its modulus, however, decays exponentially with system size at any nonzero temperature, while the partition function and the underlying determinant grow exponentially with inverse temperature and system size. For long chains at low temperature neither the modulus nor its ingredients are representable in double precision, so the quantity has to be carried as a logarithm and a phase, and any product of thermal transfer factors has to be kept numerically stable.

Returns
-------
value : np.ndarray -- Float array [ln|<T>|, arg <T>].
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def global_twist_expectation(n_cells: int, t1: float, t2: float, t3: float, beta: float) -> "np.ndarray":
    '''Return [ln|<T>|, arg <T>] for the thermal state of the periodic step-01 chain.

    The state is rho = exp(-beta H) / Tr exp(-beta H) for the periodic extended SSH chain
    with N = n_cells cells, and the twist operator is T = exp(i (2 pi / N) sum_j j n_j),
    with n_j = a_j^dag a_j + b_j^dag b_j the occupation of cell j (unit-cell positions
    x_j = j, no intracell offset). Return the natural logarithm of |<T>| and the phase
    arg <T> in (-pi, pi]; a computed phase within 1e-9 of -pi is returned as +pi.
    The result must be accurate to 1e-9 in ln|<T>| and in the phase, and finite,
    for chains of up to N = 20001 cells and for beta * max|e| up to 1e4, where
    |<T>|, Tr exp(-beta H) and the individual transfer factors are not representable in
    double precision. The Bloch spectrum is assumed gapped (h(k) of step 04 never
    vanishes). The momentum mesh must resolve the occupied band: for normalized
    negative-energy eigenvectors u_m of h(k_m), with k_m = 2 pi m / N and u_N = u_0,
    require |u_(m+1)^dag u_m| >= 0.1 at every adjacent pair, including closure.
    Raise ValueError if any overlap magnitude is below 0.1.
    The supported domain also excludes severe cancellation at a twist zero:
    let lambda_1, lambda_2 be the eigenvalues of
    P = exp(-beta h(k_(N-1))) ... exp(-beta h(k_0)), with k_m = 2 pi m / N,
    and s = (-1)^(N-1). Both cancellation ratios
    r_i = |1 + s lambda_i| / (1 + |lambda_i|) must be at least 0.1.
    If either ratio is below 0.1, raise ValueError, even if <T> is nonzero.
    Both restrictions are dimensionless and invariant under eigenvector phase
    choices. They retain this task's large-N, low-temperature regime and beta = 0
    for odd N on a resolved mesh, but exclude even-N cancellation as beta tends to zero.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature, finite and >= 0.

    Returns
    -------
    value : np.ndarray
        Float array [ln|<T>|, arg <T>].

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 2, a hopping amplitude is invalid, or beta is
        negative, NaN or infinite, or an adjacent occupied-band overlap magnitude
        or a transfer cancellation ratio is below 0.1.
    '''
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_global_twist_expectation(n_cells: int, t1: float, t2: float, t3: float, beta: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    n = int(n_cells)
    ks = 2.0 * np.pi * np.arange(n) / n
    _oracle_bloch_hamiltonian(0.0, t1, t2, t3)  # validates the hopping amplitudes
    off = t1 + t2 * np.exp(-1j * ks) + t3 * np.exp(1j * ks)
    hk = np.zeros((n, 2, 2), dtype=complex)
    hk[:, 0, 1] = -off
    hk[:, 1, 0] = -np.conj(off)
    spectra, bases = np.linalg.eigh(hk)
    spectra = np.concatenate([spectra, spectra[:1]])
    bases = np.concatenate([bases, bases[:1]])
    overlaps = np.einsum("mji,mjk->mik", bases[1:].conj(), bases[:-1])
    if np.min(np.abs(overlaps[:, 0, 0])) < 0.1:
        raise ValueError("adjacent occupied-band overlap is below 0.1")
    weights = -beta * spectra[1:]
    tops = weights.max(axis=1)
    factors = np.exp(weights - tops[:, None])[:, :, None] * overlaps
    # Each factor diag(exp(-beta E)) W is divided by exp(beta |e|) (the larger weight, since
    # E = -|e|, +|e|); the same exp(beta |e|) factors appear in Z and cancel exactly, so only
    # O(1) logarithms are ever accumulated.
    product = np.eye(2, dtype=complex)
    log_scales = []
    for m in range(n):
        product = factors[m] @ product
        scale = np.abs(product).max()
        product = product / scale
        log_scales.append(float(np.log(scale)))
    det_overlaps = np.linalg.det(overlaps)
    log_det_abs = np.log(np.abs(det_overlaps)) + weights.sum(axis=1) - 2.0 * tops
    det_phase = math.fsum(np.angle(det_overlaps))
    sign = (-1.0) ** (n - 1)
    eig = np.linalg.eigvals(product)
    lead = eig[np.argmax(np.abs(eig))]
    log_l1 = math.fsum(log_scales) + np.log(abs(lead))
    total_top = beta * math.fsum(np.abs(spectra[:n]).max(axis=1))
    log_l2 = math.fsum(log_det_abs) - log_l1
    z1 = np.exp(1j * np.angle(lead))
    z2 = np.exp(1j * (det_phase - np.angle(lead)))
    # Check the stated conditioning domain without forming exp(beta sum E).
    # Dividing numerator and denominator by max(1, |lambda|) bounds each term.
    for true_log, direction in ((log_l1 + total_top, z1),
                                (log_l2 + total_top, z2)):
        numerator = abs(np.exp(-max(true_log, 0.0))
                        + sign * direction * np.exp(min(true_log, 0.0)))
        denominator = 1.0 + np.exp(-abs(true_log))
        if numerator / denominator < 0.1:
            raise ValueError("transfer cancellation ratio is below 0.1")
    with np.errstate(over="ignore", under="ignore"):
        part1 = log_l1 + np.log(sign * z1) + np.log1p(np.exp(-(log_l1 + total_top)) / (sign * z1))
        true_log_l2 = log_l2 + total_top
        part2 = np.log1p(sign * z2 * np.exp(true_log_l2)) if true_log_l2 < 700.0 else true_log_l2 + np.log(sign * z2)
        positive_energies = np.abs(spectra[:n]).max(axis=1)
        cutoff = 1e-12 * max(1.0, float(positive_energies.max()))
        if np.all(positive_energies > cutoff):
            # For the positive band, step 03 computes the small thermal
            # correction directly, without an extensive cancellation.
            softplus = 2.0 * _oracle_log_partition_function(positive_energies, beta)
        else:
            # Step 05 does not impose step 03's zero-mode replacement.
            softplus = math.fsum(2.0 * np.log1p(np.exp(-beta * positive_energies)))
    real = float(part1.real + part2.real) - softplus
    phase = float(np.angle(np.exp(1j * (part1.imag + part2.imag))))
    if phase <= -np.pi + 1e-9:
        phase = float(np.pi)
    return np.array([real, phase])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
"""
    guard = setup + """
def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
"""
    return [
        {"setup": setup,
         "call": "global_twist_expectation(2, 1.0, 0.2, 0.3, 0.2)",
         "gold_call": "_oracle_global_twist_expectation(2, 1.0, 0.2, 0.3, 0.2)"},
        {"setup": guard,
         "call": "raises_value_error(global_twist_expectation, 2, 1.0, 0.2, 0.3, 1e-18)",
         "gold_call": "raises_value_error(_oracle_global_twist_expectation, 2, 1.0, 0.2, 0.3, 1e-18)"},
        {"setup": guard,
         "call": "raises_value_error(global_twist_expectation, 2, 1.0, 2.0, 0.0, 50.0)",
         "gold_call": "raises_value_error(_oracle_global_twist_expectation, 2, 1.0, 2.0, 0.0, 50.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(5, 1.3, 0.7, 0.0, 0.9)",
         "gold_call": "_oracle_global_twist_expectation(5, 1.3, 0.7, 0.0, 0.9)"},
        {"setup": setup,
         "call": "global_twist_expectation(6, 0.8, 1.5, 0.0, 1.2)",
         "gold_call": "_oracle_global_twist_expectation(6, 0.8, 1.5, 0.0, 1.2)"},
        {"setup": setup,
         "call": "global_twist_expectation(7, 1.0, 0.6, 0.4, 0.7)",
         "gold_call": "_oracle_global_twist_expectation(7, 1.0, 0.6, 0.4, 0.7)"},
        {"setup": setup,
         "call": "global_twist_expectation(151, 1.0, 0.6, 1.5, 3.8)",
         "gold_call": "_oracle_global_twist_expectation(151, 1.0, 0.6, 1.5, 3.8)"},
        {"setup": setup,
         "call": "global_twist_expectation(150, 3.0, 2.0, 0.0, 50.0)",
         "gold_call": "_oracle_global_twist_expectation(150, 3.0, 2.0, 0.0, 50.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(151, 1.0, 2.0, 0.5, 400.0)",
         "gold_call": "_oracle_global_twist_expectation(151, 1.0, 2.0, 0.5, 400.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(20001, 1.0, 0.6, 1.5, 2.0)",
         "gold_call": "_oracle_global_twist_expectation(20001, 1.0, 0.6, 1.5, 2.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(41, 0.5, 0.6, 1.2, 0.0)",
         "gold_call": "_oracle_global_twist_expectation(41, 0.5, 0.6, 1.2, 0.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(20001, 1.0, 0.6, 1.5, 3000.0)",
         "gold_call": "_oracle_global_twist_expectation(20001, 1.0, 0.6, 1.5, 3000.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(20000, 3.0, 2.0, 0.0, 300.0)",
         "gold_call": "_oracle_global_twist_expectation(20000, 3.0, 2.0, 0.0, 300.0)"},
        {"setup": setup,
         "call": "global_twist_expectation(2000, 1.0, 2.0, 0.0, 500.0)",
         "gold_call": "_oracle_global_twist_expectation(2000, 1.0, 2.0, 0.0, 500.0)"},
        {"setup": guard,
         "call": "raises_value_error(global_twist_expectation, 1, 1.0, 1.0, 0.0, 1.0)",
         "gold_call": "raises_value_error(_oracle_global_twist_expectation, 1, 1.0, 1.0, 0.0, 1.0)"},
        {"setup": guard,
         "call": "raises_value_error(global_twist_expectation, 5, 1.0, 1.0, 0.0, np.inf)",
         "gold_call": "raises_value_error(_oracle_global_twist_expectation, 5, 1.0, 1.0, 0.0, np.inf)"},
        {"setup": guard,
         "call": "raises_value_error(global_twist_expectation, 5, 1.0, -1.0, 0.0, 1.0)",
         "gold_call": "raises_value_error(_oracle_global_twist_expectation, 5, 1.0, -1.0, 0.0, 1.0)"},
    ]
