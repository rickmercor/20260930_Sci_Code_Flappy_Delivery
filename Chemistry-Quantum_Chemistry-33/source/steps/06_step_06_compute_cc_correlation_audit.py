"""
Validate T1 shape (O,V), T2 and antisymmetrized integrals shape (O,O,V,V), finite values, and separate occupied- and virtual-index antisymmetry for both rank-four tensors. Form tau=T2+T1_ia*T1_jb-T1_ib*T1_ja. Return total correlation energy, 0.25*sum((T1_ia*T1_jb-T1_ib*T1_ja)*g), 0.25*sum(T2*g), the joint T1/T2 norm, and the weighted C-order checksum of tau*g.

The amplitude-energy identity is source-derived. The joint norm and weighted contraction checksum are disclosed benchmark diagnostics.

Returns
-------
tuple : total energy, singles energy, doubles energy, amplitude norm, and contraction checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cc_correlation_audit(t1: "np.ndarray", t2: "np.ndarray", antisymmetrized_integrals: "np.ndarray") -> tuple:
    """Return correlation-energy components and amplitude diagnostics.

    Both T2 and g must be antisymmetric separately under occupied-index exchange
    and virtual-index exchange. The contraction uses
    tau_ijab=T2_ijab+T1_ia*T1_jb-T1_ib*T1_ja.

    Parameters
    ----------
    t1 : np.ndarray
        Singles amplitudes with shape (O,V).
    t2 : np.ndarray
        Doubles amplitudes with shape (O,O,V,V).
    antisymmetrized_integrals : np.ndarray
        Integrals <ij||ab> with the same shape and antisymmetries as t2.

    Returns
    -------
    correlation_energy : float
        Sum of singles and doubles contributions.
    singles_contribution : float
        0.25*sum((T1_ia*T1_jb-T1_ib*T1_ja)*g_ijab).
    doubles_contribution : float
        0.25*sum(T2_ijab*g_ijab).
    amplitude_norm : float
        Joint Euclidean norm of T1 and T2.
    contraction_checksum : float
        Weighted C-order checksum dot(arange(1,N+1), (tau*g).ravel())/N,
        where N=O*O*V*V and all indices have the order (i,j,a,b).

    Raises
    ------
    ValueError
        If shapes, values, dimensions, or required antisymmetries are invalid.
    """
    return None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_cc_correlation_audit(t1: "np.ndarray", t2: "np.ndarray", antisymmetrized_integrals: "np.ndarray") -> tuple:
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    g = np.asarray(antisymmetrized_integrals, dtype=float)
    if t1.ndim != 2 or min(t1.shape) < 1:
        raise ValueError("t1 must have shape (n_occ,n_virt)")
    no, nv = t1.shape
    if t2.shape != (no, no, nv, nv) or g.shape != t2.shape:
        raise ValueError("t2 and integrals must have shape (n_occ,n_occ,n_virt,n_virt)")
    if not all(np.all(np.isfinite(z)) for z in (t1, t2, g)):
        raise ValueError("all inputs must be finite")
    for name, tensor in (("t2", t2), ("antisymmetrized_integrals", g)):
        if not np.allclose(tensor, -tensor.swapaxes(0, 1), rtol=0.0, atol=1e-12) or not np.allclose(tensor, -tensor.swapaxes(2, 3), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be antisymmetric in occupied and virtual pairs")
    direct = np.einsum("ia,jb->ijab", t1, t1, optimize=True)
    product = direct - direct.swapaxes(2, 3)
    doubles = float(0.25 * np.sum(t2 * g))
    singles = float(0.25 * np.sum(product * g))
    energy = float(doubles + singles)
    norm = float(np.sqrt(np.sum(t1 * t1) + np.sum(t2 * t2)))
    raw = (t2 + product) * g
    flat = raw.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat) / flat.size)
    return energy, singles, doubles, norm, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    setup = lambda seed,no,nv: f'import numpy as np\nr=np.random.default_rng({seed});t1=r.normal(scale=.03,size=({no},{nv}));z=r.normal(scale=.02,size=({no},{no},{nv},{nv}));t2=.25*(z-z.swapaxes(0,1)-z.swapaxes(2,3)+z.transpose(1,0,3,2));q=r.normal(scale=.2,size=t2.shape);g=.25*(q-q.swapaxes(0,1)-q.swapaxes(2,3)+q.transpose(1,0,3,2))'
    return [
        {"tol": 1e-11, "setup": setup(79,3,4), "call": "compute_cc_correlation_audit(t1,t2,g)", "gold_call": "_oracle_compute_cc_correlation_audit(t1,t2,g)"},
        {"tol": 1e-11, "setup": setup(83,1,1), "call": "compute_cc_correlation_audit(t1,t2,g)", "gold_call": "_oracle_compute_cc_correlation_audit(t1,t2,g)"},
        {"tol": 1e-11, "setup": setup(89,5,2) + ";t1[:]=0", "call": "compute_cc_correlation_audit(t1,t2,g)", "gold_call": "_oracle_compute_cc_correlation_audit(t1,t2,g)"},
        {"tol": 0.0, "setup": setup(97,2,3) + "\n" + err + "g=g[:,:,:,:2]", "call": "et(compute_cc_correlation_audit,t1,t2,g)", "gold_call": "et(_oracle_compute_cc_correlation_audit,t1,t2,g)"},
    ]
