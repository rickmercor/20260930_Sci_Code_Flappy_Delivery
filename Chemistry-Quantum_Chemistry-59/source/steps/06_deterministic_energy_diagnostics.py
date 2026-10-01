"""
Evaluate deterministic variational, asymmetric, proxy and diagonal energies.

Deterministic neural-quantum-state analysis compares several energy objectives defined on related configuration domains. The variational and asymmetric estimates use normalization on the selected variational support. The proxy estimate uses the combined variational and perturbative target while retaining only the diagonal external–external block. Diagonalization of the variational Hamiltonian provides the exact selected-space reference and the remaining optimization error.

Returns
-------
A five-element array [E_var, E_asym, E_proxy, E_diag, delta_opt] containing finite diagnostic energies in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def deterministic_energy_diagnostics(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
) -> 'np.ndarray':
    """Return the five-energy diagnostic ledger for selected configuration sets.

    The ledger is ``[E_var, E_asym, E_proxy, E_diag, delta_opt]``. ``E_var``
    and ``E_asym`` use normalization on the variational support. ``E_proxy``
    uses normalization on the concatenated target and replaces the external-
    external block by its diagonal. ``E_diag`` is the lowest eigenvalue of the
    variational block and ``delta_opt = E_var - E_diag``.

    Returns
    -------
    np.ndarray
        Five finite diagnostic values in the stated order.

    Raises
    ------
    ValueError
        If shapes, index sets, symmetry, values or normalization are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deterministic_energy_diagnostics(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
) -> 'np.ndarray':

    h = np.asarray(hamiltonian, dtype=float)
    psi = np.asarray(amplitudes, dtype=float)
    v = np.asarray(variational_indices)
    p = np.asarray(perturbative_indices)

    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must be a nonempty square matrix")

    n = h.shape[0]

    if psi.shape != (n,) or v.ndim != 1 or p.ndim != 1 or v.size == 0:
        raise ValueError("incompatible input shapes")

    if not np.issubdtype(v.dtype, np.integer) or not np.issubdtype(
        p.dtype, np.integer
    ):
        raise ValueError("indices must be integral")

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(psi)):
        raise ValueError("hamiltonian and amplitudes must be finite")

    if not np.allclose(h, h.T, atol=2e-11, rtol=0.0):
        raise ValueError("hamiltonian must be symmetric")

    if np.any(v < 0) or np.any(v >= n) or np.unique(v).size != v.size:
        raise ValueError("invalid variational indices")

    if np.any(p < 0) or np.any(p >= n) or np.unique(p).size != p.size:
        raise ValueError("invalid perturbative indices")

    if np.intersect1d(v, p).size:
        raise ValueError(
            "variational and perturbative sets must be disjoint"
        )

    norm_v = float(np.linalg.norm(psi[v]))
    if norm_v == 0.0:
        raise ValueError(
            "variational amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm_v
    hvv = h[np.ix_(v, v)]
    e_var = float(psi_v @ hvv @ psi_v)

    target = np.concatenate((v, p)).astype(np.int64)
    psi_target = psi[target]

    e_asym = float(
        psi_v @ h[np.ix_(v, target)] @ psi_target / norm_v
    )

    norm_target = float(np.linalg.norm(psi_target))
    if norm_target == 0.0:
        raise ValueError("target amplitudes must have nonzero norm")

    proxy = h[np.ix_(target, target)].copy()

    if p.size:
        k = v.size
        proxy[k:, k:] = np.diag(np.diag(proxy[k:, k:]))

    psi_t = psi_target / norm_target
    e_proxy = float(psi_t @ proxy @ psi_t)
    e_diag = float(np.linalg.eigvalsh(hvv)[0])

    return np.array(
        [e_var, e_asym, e_proxy, e_diag, e_var - e_diag],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.2,.3],[.2,-.4,.1],[.3,.1,.5]]); psi=np.array([2.,-1.,.5]); v=np.array([0,1]); p=np.array([2])",
            "call": "deterministic_energy_diagnostics(h.copy(),psi.copy(),v.copy(),p.copy())",
            "gold_call": "_oracle_deterministic_energy_diagnostics(h,psi,v,p)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-2.,.4],[.4,-1.]]); psi=np.array([1.,2.]); v=np.array([1,0]); p=np.array([],dtype=np.int64)",
            "call": "deterministic_energy_diagnostics(h.copy(),psi.copy(),v.copy(),p.copy())",
            "gold_call": "_oracle_deterministic_energy_diagnostics(h,psi,v,p)",
        },
        {
            "setup": "import numpy as np\nh=np.diag(np.array([-3.,-2.,.2,.7])); psi=np.array([1.,-1.,2.,-2.]); v=np.array([0,1]); p=np.array([3,2])",
            "call": "deterministic_energy_diagnostics(h.copy(),psi.copy(),v.copy(),p.copy())",
            "gold_call": "_oracle_deterministic_energy_diagnostics(h,psi,v,p)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-1.2,.15,.2,-.1],[.15,-.8,.05,.3],[.2,.05,.4,.6],[-.1,.3,.6,.9]]); psi=np.array([.3,-.7,.2,.5]); v=np.array([1,0]); p=np.array([3,2])",
            "call": "deterministic_energy_diagnostics(h.copy(),psi.copy(),v.copy(),p.copy())",
            "gold_call": "_oracle_deterministic_energy_diagnostics(h,psi,v,p)",
        },
        {
            "setup": "import numpy as np\nh=np.eye(3); psi=np.ones(3); v=np.array([0,1]); p=np.array([1,2])\ndef check(fn):\n try: fn(h.copy(),psi.copy(),v.copy(),p.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(deterministic_energy_diagnostics)",
            "gold_call": "check(_oracle_deterministic_energy_diagnostics)",
        },
    ]
