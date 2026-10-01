"""
Decompose the Epstein–Nesbet correction into internal and external terms.

Epstein–Nesbet second-order perturbation theory can separate two sources of missing correlation. The internal term measures the residual left by imperfect optimization within the selected variational support, while the external term recovers coupling to screened determinants outside that support. Both channels require electronic-energy denominators, so nuclear repulsion must be removed from the total reference energy before evaluating them.

Returns
-------
A four-element array [delta_internal, delta_external, delta_total, n_external].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decomposed_epstein_nesbet_pt2(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
    reference_total_energy: float,
    nuclear_repulsion: float,
) -> 'np.ndarray':
    """Evaluate internal-residual and screened-external PT2 corrections.

    The selected amplitudes are normalized on the variational support. For
    every supplied perturbative index, form the external residual from the
    complete Hamiltonian row over the selected support; do not apply another
    elementwise screening threshold. Convert the total reference energy to
    its electronic value before forming either Epstein-Nesbet denominator.

    Returns
    -------
    np.ndarray
        ``[delta_internal, delta_external, delta_total, n_external]``.

    Raises
    ------
    ValueError
        If inputs are incompatible, non-finite, non-Hermitian or singular.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_decomposed_epstein_nesbet_pt2(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
    reference_total_energy: float,
    nuclear_repulsion: float,
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

    if not np.isfinite(reference_total_energy) or not np.isfinite(
        nuclear_repulsion
    ):
        raise ValueError("energies must be finite")

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

    norm = float(np.linalg.norm(psi[v]))
    if norm == 0.0:
        raise ValueError(
            "variational amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm
    e_elec = float(reference_total_energy) - float(nuclear_repulsion)

    hvv = h[np.ix_(v, v)]
    residual_v = hvv @ psi_v - e_elec * psi_v
    denom_v = e_elec - np.diag(hvv)

    if np.any(np.abs(denom_v) <= 1e-12):
        raise ValueError(
            "singular internal Epstein-Nesbet denominator"
        )

    delta_internal = float(
        np.sum(residual_v * residual_v / denom_v)
    )

    if p.size:
        residual_p = h[np.ix_(p, v)] @ psi_v
        denom_p = e_elec - np.diag(h)[p]

        if np.any(np.abs(denom_p) <= 1e-12):
            raise ValueError(
                "singular external Epstein-Nesbet denominator"
            )

        delta_external = float(
            np.sum(residual_p * residual_p / denom_p)
        )
    else:
        delta_external = 0.0

    return np.array(
        [
            delta_internal,
            delta_external,
            delta_internal + delta_external,
            float(p.size),
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.2,.3],[.2,-.4,.1],[.3,.1,.5]]); psi=np.array([2.,-1.,.5]); v=np.array([0,1]); p=np.array([2])",
            "call": "decomposed_epstein_nesbet_pt2(h.copy(),psi.copy(),v.copy(),p.copy(),-.5,.2)",
            "gold_call": "_oracle_decomposed_epstein_nesbet_pt2(h,psi,v,p,-.5,.2)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-2.,.3,0.],[.3,-1.,0.],[0.,0.,.4]]); w,u=np.linalg.eigh(h[:2,:2]); psi=np.array([u[0,0],u[1,0],0.]); v=np.array([0,1]); p=np.array([],dtype=np.int64); et=float(w[0]+.7)",
            "call": "decomposed_epstein_nesbet_pt2(h.copy(),psi.copy(),v.copy(),p.copy(),et,.7)",
            "gold_call": "_oracle_decomposed_epstein_nesbet_pt2(h,psi,v,p,et,.7)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-1.5,.25],[.25,-.6]]); psi=np.array([1.,2.]); v=np.array([1,0]); p=np.array([],dtype=np.int64)",
            "call": "decomposed_epstein_nesbet_pt2(h.copy(),psi.copy(),v.copy(),p.copy(),-.8,.4)",
            "gold_call": "_oracle_decomposed_epstein_nesbet_pt2(h,psi,v,p,-.8,.4)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-2.,.1,.25,-.2],[.1,-1.1,.3,.05],[.25,.3,.2,.1],[-.2,.05,.1,.8]]); psi=np.array([-.4,.9,.2,-.1]); v=np.array([0,1]); p=np.array([3,2])",
            "call": "decomposed_epstein_nesbet_pt2(h.copy(),psi.copy(),v.copy(),p.copy(),-1.1,.5)",
            "gold_call": "_oracle_decomposed_epstein_nesbet_pt2(h,psi,v,p,-1.1,.5)",
        },
        {
            "setup": "import numpy as np\nh=np.diag([-1.,.5]); psi=np.array([1.,0.]); v=np.array([0]); p=np.array([1])\ndef check(fn):\n try: fn(h.copy(),psi.copy(),v.copy(),p.copy(),-1.,0.)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(decomposed_epstein_nesbet_pt2)",
            "gold_call": "check(_oracle_decomposed_epstein_nesbet_pt2)",
        },
    ]
