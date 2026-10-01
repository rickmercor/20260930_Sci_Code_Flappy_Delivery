"""
Construct the mass-weighted dynamical matrix.

Harmonic lattice dynamics works with the mass-weighted force-constant matrix rather than

with $\Phi$ itself, so that the equation of motion becomes a standard symmetric

eigenvalue problem. Each entry is divided by the geometric mean of the two coordinate

masses it couples, $\widetilde K_{ab}=\Phi_{ab}/\sqrt{m_a m_b}$. The result inherits the

symmetry of $\Phi$, and for a stable structure it is positive semidefinite, with

eigenvalues that are squared frequencies in the units implied by the supplied scaling.

Returns
-------
mass_weighted_matrix : np.ndarray     Real symmetric array with shape $(N, N)$.  Raises ------ ValueError     If either input cannot be represented by a real numeric array; if phi     is not a nonempty square matrix; if masses does not contain exactly one     entry per coordinate; if either input contains a nonfinite value; if     any mass is not strictly positive; or if phi is not symmetric with     rtol=0 and atol=$10^{-12}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_mass_weighted_dynamical_matrix(
    phi: "np.ndarray",
    masses: "np.ndarray",
) -> "np.ndarray":
    """Construct the mass-weighted dynamical matrix.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.

    Returns
    -------
    mass_weighted_matrix : np.ndarray
        Real symmetric array with shape $(N, N)$.

    Raises
    ------
    ValueError
        If either input cannot be represented by a real numeric array; if phi
        is not a nonempty square matrix; if masses does not contain exactly one
        entry per coordinate; if either input contains a nonfinite value; if
        any mass is not strictly positive; or if phi is not symmetric with
        rtol=0 and atol=$10^{-12}$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_mass_weighted_dynamical_matrix(
    phi: "np.ndarray",
    masses: "np.ndarray",
) -> "np.ndarray":
    """Construct the mass-weighted dynamical matrix.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.

    Returns
    -------
    mass_weighted_matrix : np.ndarray
        Real symmetric array with shape $(N, N)$.

    Raises
    ------
    ValueError
        If either input cannot be represented by a real numeric array; if phi
        is not a nonempty square matrix; if masses does not contain exactly one
        entry per coordinate; if either input contains a nonfinite value; if
        any mass is not strictly positive; or if phi is not symmetric with
        rtol=0 and atol=$10^{-12}$.
    """
    try:
        phi_raw = np.asarray(phi)
        masses_raw = np.asarray(masses)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi and masses must be numeric arrays") from exc

    if np.iscomplexobj(phi_raw) and np.any(np.imag(phi_raw) != 0.0):
        raise ValueError("phi must be real-valued")
    if np.iscomplexobj(masses_raw) and np.any(np.imag(masses_raw) != 0.0):
        raise ValueError("masses must be real-valued")

    try:
        phi_array = np.asarray(phi_raw, dtype=float)
        masses_array = np.asarray(masses_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi and masses must contain real numeric values") from exc

    if (
        phi_array.ndim != 2
        or phi_array.shape[0] != phi_array.shape[1]
        or phi_array.shape[0] == 0
    ):
        raise ValueError("phi must be a nonempty square matrix")
    if masses_array.ndim != 1 or masses_array.size != phi_array.shape[0]:
        raise ValueError("masses must have one entry per coordinate")
    if not np.all(np.isfinite(phi_array)) or not np.all(np.isfinite(masses_array)):
        raise ValueError("phi and masses must contain only finite values")
    if np.any(masses_array <= 0.0):
        raise ValueError("all masses must be strictly positive")
    if not np.allclose(phi_array, phi_array.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("phi must be symmetric within atol=1e-12")

    root_masses = np.sqrt(masses_array)
    result = phi_array / np.outer(root_masses, root_masses)
    return 0.5 * (result + result.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "phi=np.array([[4.0,2.0],[2.0,9.0]]); masses=np.array([1.0,4.0])",
            "call": "build_mass_weighted_dynamical_matrix(phi, masses)",
            "gold_call": "_oracle_build_mass_weighted_dynamical_matrix(phi, masses)",
        },
        {
            "setup": "phi=np.array([[9.0]]); masses=np.array([4.0])",
            "call": "build_mass_weighted_dynamical_matrix(phi, masses)",
            "gold_call": "_oracle_build_mass_weighted_dynamical_matrix(phi, masses)",
        },
        {
            "setup": "phi=np.array([[16.0,0.0,2.0],[0.0,25.0,0.0],[2.0,0.0,36.0]]); masses=np.array([1.0,25.0,4.0])",
            "call": "build_mass_weighted_dynamical_matrix(phi, masses)",
            "gold_call": "_oracle_build_mass_weighted_dynamical_matrix(phi, masses)",
        },
        {
            "setup": "phi=np.zeros((2,2)); masses=np.array([1.0,2.0])",
            "call": "build_mass_weighted_dynamical_matrix(phi, masses)",
            "gold_call": "_oracle_build_mass_weighted_dynamical_matrix(phi, masses)",
        },
    ]
