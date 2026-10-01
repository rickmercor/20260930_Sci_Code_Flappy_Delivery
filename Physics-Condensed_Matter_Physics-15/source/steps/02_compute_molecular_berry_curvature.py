"""
Compute the real antisymmetric molecular Berry-curvature matrix.

## As the ions move, the adiabatic electronic ground state evolves and accumulates a

geometric phase. The resulting molecular Berry curvature acts on the nuclei as a

velocity-dependent force, and it is the quantity that can lift a phonon degeneracy when

electronic order breaks time-reversal symmetry. It is assembled from the electron-phonon

couplings between occupied and unoccupied states at each sampled $\mathbf{k}$ point,

weighted by the electronic energy denominators and averaged over the sampled points. The

supplied couplings are already expressed in the mass-weighted phonon-coordinate

convention, so no further mass factor enters here. A correctly constructed curvature

matrix comes out real and antisymmetric; both properties follow from the structure of

the construction rather than being imposed on it.

Returns
-------
curvature : np.ndarray     Real antisymmetric molecular Berry-curvature matrix with shape $(N, N)$.  Raises ------ ValueError     If the energy arrays are not nonempty, real, finite, one-dimensional,     and equal in length; if g_cv is not a finite two-dimensional array with     one row per k point and at least one coordinate; or if any conduction     energy is not strictly greater than its corresponding valence energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_molecular_berry_curvature(
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
) -> "np.ndarray":
    """Compute the real antisymmetric molecular Berry-curvature matrix.

    Parameters
    ----------
    eps_v : np.ndarray
        One-dimensional real valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional real conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex electron-phonon couplings with shape $(N_k, N)$.

    Returns
    -------
    curvature : np.ndarray
        Real antisymmetric molecular Berry-curvature matrix with shape $(N, N)$.

    Raises
    ------
    ValueError
        If the energy arrays are not nonempty, real, finite, one-dimensional,
        and equal in length; if g_cv is not a finite two-dimensional array with
        one row per k point and at least one coordinate; or if any conduction
        energy is not strictly greater than its corresponding valence energy.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_molecular_berry_curvature(
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
) -> "np.ndarray":
    """Compute the real antisymmetric molecular Berry-curvature matrix.

    Parameters
    ----------
    eps_v : np.ndarray
        One-dimensional real valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional real conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex electron-phonon couplings with shape $(N_k, N)$.

    Returns
    -------
    curvature : np.ndarray
        Real antisymmetric molecular Berry-curvature matrix with shape $(N, N)$.

    Raises
    ------
    ValueError
        If the energy arrays are not nonempty, real, finite, one-dimensional,
        and equal in length; if g_cv is not a finite two-dimensional array with
        one row per k point and at least one coordinate; or if any conduction
        energy is not strictly greater than its corresponding valence energy.
    """
    try:
        eps_v_raw = np.asarray(eps_v)
        eps_c_raw = np.asarray(eps_c)
        coupling = np.asarray(g_cv, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies and couplings must be numeric arrays") from exc

    if np.iscomplexobj(eps_v_raw) and np.any(np.imag(eps_v_raw) != 0.0):
        raise ValueError("eps_v must be real-valued")
    if np.iscomplexobj(eps_c_raw) and np.any(np.imag(eps_c_raw) != 0.0):
        raise ValueError("eps_c must be real-valued")
    try:
        valence = np.asarray(eps_v_raw, dtype=float)
        conduction = np.asarray(eps_c_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies must contain real numeric values") from exc

    if valence.ndim != 1 or conduction.ndim != 1 or valence.size == 0:
        raise ValueError("energy arrays must be nonempty and one-dimensional")
    if conduction.shape != valence.shape:
        raise ValueError("energy arrays must have equal length")
    if coupling.ndim != 2 or coupling.shape[0] != valence.size or coupling.shape[1] == 0:
        raise ValueError("g_cv must have shape (N_k, N) with N >= 1")
    if not np.all(np.isfinite(valence)) or not np.all(np.isfinite(conduction)):
        raise ValueError("energies must be finite")
    if not np.all(np.isfinite(coupling.real)) or not np.all(np.isfinite(coupling.imag)):
        raise ValueError("g_cv must be finite")

    gaps = conduction - valence
    if np.any(gaps <= 0.0):
        raise ValueError("every conduction energy must exceed its valence energy")

    scaled_coupling = coupling / gaps[:, None]
    gram = scaled_coupling.conj().T @ scaled_coupling
    raw_curvature = (1j / float(valence.size)) * (gram - gram.conj())
    real_curvature = np.asarray(raw_curvature.real, dtype=float)
    return 0.5 * (real_curvature - real_curvature.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "eps_v=np.array([-1.0,-0.8,-1.2]); eps_c=np.array([1.0,1.3,0.9]); g=np.array([[1.0+0.0j,0.0+1.0j,0.5-0.2j],[0.3+0.4j,-0.7+0.1j,0.2+0.8j],[-0.2+0.5j,0.6-0.3j,1.1+0.0j]])",
            "call": "compute_molecular_berry_curvature(eps_v, eps_c, g)",
            "gold_call": "_oracle_compute_molecular_berry_curvature(eps_v, eps_c, g)",
        },
        {
            "setup": "eps_v=np.array([-1.0]); eps_c=np.array([1.0]); g=np.array([[2.0+3.0j]])",
            "call": "compute_molecular_berry_curvature(eps_v, eps_c, g)",
            "gold_call": "_oracle_compute_molecular_berry_curvature(eps_v, eps_c, g)",
        },
        {
            "setup": "eps_v=np.array([-2.0,-1.0]); eps_c=np.array([0.5,2.0]); g=np.zeros((2,4),dtype=complex)",
            "call": "compute_molecular_berry_curvature(eps_v, eps_c, g)",
            "gold_call": "_oracle_compute_molecular_berry_curvature(eps_v, eps_c, g)",
        },
        {
            "setup": "eps_v=np.array([1.0]); eps_c=np.array([1.000001]); g=np.array([[1.0+0.0j,0.0+1.0j]])",
            "call": "compute_molecular_berry_curvature(eps_v, eps_c, g)",
            "gold_call": "_oracle_compute_molecular_berry_curvature(eps_v, eps_c, g)",
        },
    ]
