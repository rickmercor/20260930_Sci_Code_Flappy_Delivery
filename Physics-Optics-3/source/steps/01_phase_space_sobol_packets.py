"""
Construct the paper's weighted full-phase-space Gaussian packet ensemble.

Construct the paper's quasi-Monte-Carlo approximation of the coherent-state integral in two dimensions. With diagonal `Gamma` and `Gamma0`, sample `z=(qx,qy,px,py)` from the normalized Gaussian having mean `(q0,p0)` and covariance `hbar*diag(Gamma^-1+Gamma0^-1, Gamma+Gamma0)` using `Sobol(d=4,scramble=True,seed).random_base2(log2_count)` followed by clipped inverse-normal coordinates. For the normalized packet convention `g_z(r)=det(Gamma)^(1/4)/(pi*hbar)^(d/4) * exp[-(r-q)^T Gamma (r-q)/(2*hbar)+i*p^T(r-q)/hbar]`, attach `c_j=<g_zj|psi0>/[N*(2*pi*hbar)^2*rho(z_j)]`, where `rho` is that normalized sampling density. Preserve Sobol order.

Returns
-------
One float array with one full phase-space sample and complex coefficient per row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

def phase_space_sobol_packets(log2_count, seed, q0, p0,
                              initial_gamma_diag, basis_gamma_diag, hbar=1.0):
    """Return a float array of shape (2**log2_count,6).

    Columns are [qx,qy,px,py,coefficient_real,coefficient_imag].

    Raises
    ------
    ValueError
        If dimensions, widths, hbar, seed, or log2_count are invalid.
    """
    return np.empty((2 ** int(log2_count), 6), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _packet_overlap(q, p, q0, p0, basis_gamma, initial_gamma, hbar=1.0):
    import numpy as np
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    G = np.diag(np.asarray(basis_gamma, float))
    G0 = np.diag(np.asarray(initial_gamma, float))
    A = G + G0
    invA = np.linalg.inv(A)
    b = q @ G + q0 @ G0 + 1j * (p0 - p)
    const = (-0.5 * np.einsum('ni,ij,nj->n', q, G, q)
             - 0.5 * q0 @ G0 @ q0
             + 1j * (p @ q.T).diagonal()
             - 1j * p0 @ q0)
    quad = 0.5 * np.einsum('ni,ij,nj->n', b, invA, b)
    pref = 2.0 * (np.linalg.det(G) * np.linalg.det(G0)) ** 0.25 / np.sqrt(np.linalg.det(A))
    return pref * np.exp((quad + const) / hbar)


def _oracle_phase_space_sobol_packets(log2_count, seed, q0, p0, initial_gamma_diag, basis_gamma_diag, hbar=1.0):
    import numpy as np
    from scipy.special import ndtri
    from scipy.stats import qmc
    initial_gamma = initial_gamma_diag
    basis_gamma = basis_gamma_diag
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    initial_gamma = np.asarray(initial_gamma, float)
    basis_gamma = np.asarray(basis_gamma, float)
    if isinstance(log2_count, bool) or int(log2_count) != log2_count or not 1 <= int(log2_count) <= 12:
        raise ValueError('log2_count must be an integer from 1 through 12')
    if q0.shape != (2,) or p0.shape != (2,) or initial_gamma.shape != (2,) or basis_gamma.shape != (2,):
        raise ValueError('q0, p0, and both width vectors must have length two')
    if not np.all(np.isfinite(np.r_[q0,p0,initial_gamma,basis_gamma,hbar,seed])) or np.any(initial_gamma <= 0) or np.any(basis_gamma <= 0) or hbar <= 0:
        raise ValueError('packet inputs must be finite with positive widths and hbar')
    G = np.diag(np.asarray(basis_gamma, float))
    G0 = np.diag(np.asarray(initial_gamma, float))
    sigma = np.block([[np.linalg.inv(G) + np.linalg.inv(G0), np.zeros((2,2))],
                      [np.zeros((2,2)), G + G0]])
    cov = hbar * sigma
    u = qmc.Sobol(4, scramble=True, seed=int(seed)).random_base2(int(log2_count))
    u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    z = np.r_[q0, p0] + ndtri(u) @ np.linalg.cholesky(cov).T
    delta = z - np.r_[q0, p0]
    invcov = np.linalg.inv(cov)
    pdf = np.exp(-0.5*np.einsum('ni,ij,nj->n', delta, invcov, delta)) / np.sqrt((2*np.pi)**4*np.linalg.det(cov))
    ov = _packet_overlap(z[:,:2], z[:,2:], q0, p0, basis_gamma, initial_gamma, hbar)
    n = len(z)
    coeff = ov / (n * (2*np.pi*hbar)**2 * pdf)
    return np.column_stack((z, coeff.real, coeff.imag))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np", "call":"phase_space_sobol_packets(2,7,np.array([-1.,.2]),np.array([1.5,-.1]),np.array([.8,1.1]),np.array([1.4,2.]))", "gold_call":"_oracle_phase_space_sobol_packets(2,7,np.array([-1.,.2]),np.array([1.5,-.1]),np.array([.8,1.1]),np.array([1.4,2.]))"},
        {"setup":"import numpy as np", "call":"phase_space_sobol_packets(1,0,np.zeros(2),np.zeros(2),np.ones(2),np.ones(2))", "gold_call":"_oracle_phase_space_sobol_packets(1,0,np.zeros(2),np.zeros(2),np.ones(2),np.ones(2))"},
        {"setup":"import numpy as np", "call":"phase_space_sobol_packets(4,91,np.array([-2.,-.3]),np.array([2.2,.4]),np.array([.5,1.7]),np.array([2.1,.9]),.7)", "gold_call":"_oracle_phase_space_sobol_packets(4,91,np.array([-2.,-.3]),np.array([2.2,.4]),np.array([.5,1.7]),np.array([2.1,.9]),.7)"}
    ]
