"""
Compute the source-derived doubles readout using the supplied orbital states, MP2 amplitudes, and readout maps. Return T2, its correction relative to MP2, and the largest occupied- or virtual-index exchange residual. Use the finite-fixture pair normalization, unscaled scalar coupling, factorized pair product, and normalized exchange projector stated in the problem statement.

The amplitude readout follows the selected molecular-orbital architecture. The supplied analytic maps and diagnostic definitions fix this benchmark instance.

Returns
-------
tuple : T2 amplitudes, learned correction, and exchange residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def readout_double_amplitudes(features: "np.ndarray", occupied_count: int, mp2_amplitudes: "np.ndarray", pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return antisymmetric T2 amplitudes, corrections, and residual.

    Use the orbital normalization, scalar-coupling, factorized pair-product,
    and normalized exchange-projector fixture conventions in the problem statement.
    Recover the source readout method for the supplied maps and MP2 baseline.

    Parameters
    ----------
    features : np.ndarray
        Final orbital states with shape (P,A,K,M).
    occupied_count : int
        Number of occupied orbitals, strictly between zero and P.
    mp2_amplitudes : np.ndarray
        Baseline shape (O,O,V,V), antisymmetric in occupied and virtual pairs.
    pair_weights : np.ndarray
        Channel map with shape (K,K).
    hidden_weights : np.ndarray
        Readout hidden map with shape (K,H).
    output_weights : np.ndarray
        Readout output map with shape (H,).
    epsilon : float
        Positive finite normalization stabilizer.

    Returns
    -------
    t2 : np.ndarray
        Doubles amplitudes with shape (O,O,V,V).
    correction : np.ndarray
        Learned correction with the same shape as t2.
    exchange_residual : float
        Largest occupied- or virtual-pair antisymmetry mismatch.

    Raises
    ------
    ValueError
        If shapes, antisymmetry, counts, values, or epsilon are invalid.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_readout_double_amplitudes(features: "np.ndarray", occupied_count: int, mp2_amplitudes: "np.ndarray", pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(features, dtype=float)
    mp2 = np.asarray(mp2_amplitudes, dtype=float)
    pw = np.asarray(pair_weights, dtype=float)
    wh = np.asarray(hidden_weights, dtype=float)
    wo = np.asarray(output_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if not isinstance(occupied_count, (int, np.integer)) or not 1 <= occupied_count < p:
        raise ValueError("occupied_count must be an integer in [1,P-1]")
    nv = p - occupied_count
    if mp2.shape != (occupied_count, occupied_count, nv, nv):
        raise ValueError("mp2_amplitudes has the wrong shape")
    if not np.allclose(mp2, -mp2.swapaxes(0, 1), rtol=0.0, atol=1e-12) or not np.allclose(mp2, -mp2.swapaxes(2, 3), rtol=0.0, atol=1e-12):
        raise ValueError("mp2_amplitudes must be antisymmetric in occupied and virtual pairs")
    if pw.shape != (k, k) or wh.ndim != 2 or wh.shape[0] != k or wo.shape != (wh.shape[1],):
        raise ValueError("readout weight shapes are inconsistent")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, mp2, pw, wh, wo)):
        raise ValueError("numeric inputs must be finite")
    norms = np.sqrt(np.sum(x * x, axis=(1, 2, 3))) + abs(float(epsilon))
    unit = x / norms[:, None, None, None]
    projected = np.einsum("jk,pakm->pajm", pw, unit, optimize=True)
    occ = projected[:occupied_count]
    virt = projected[occupied_count:]
    pair = np.einsum("iakm,vakm->ivk", occ, virt, optimize=True)
    four = pair[:, None, :, None, :] * pair[None, :, None, :, :]
    hidden = np.tanh(np.einsum("ijabk,kh->ijabh", four, wh, optimize=True))
    raw = np.einsum("ijabh,h->ijab", hidden, wo, optimize=True)
    correction = 0.25 * (raw - raw.swapaxes(0, 1) - raw.swapaxes(2, 3) + raw.transpose(1, 0, 3, 2))
    t2 = mp2 + correction
    residual = float(max(np.max(np.abs(t2 + t2.swapaxes(0, 1))), np.max(np.abs(t2 + t2.swapaxes(2, 3)))))
    return t2, correction, residual

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = lambda call: "(lambda z: [len(z),*z[0].shape,*z[1].shape,*np.concatenate([z[0].ravel(),z[1].ravel(),[z[2]]]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    setup = lambda seed,p,a,k,m,h,o: f'import numpy as np\nr=np.random.default_rng({seed});x=r.normal(size=({p},{a},{k},{m}));o={o};nv={p}-{o};z=r.normal(scale=.02,size=(o,o,nv,nv));mp2=.25*(z-z.swapaxes(0,1)-z.swapaxes(2,3)+z.transpose(1,0,3,2));pw=r.normal(scale=.2,size=({k},{k}));wh=r.normal(scale=.25,size=({k},{h}));wo=r.normal(scale=.12,size={h})'
    return [
        {"tol": 1e-10, "setup": setup(61,7,4,3,3,5,3), "call": pack("readout_double_amplitudes(x,o,mp2,pw,wh,wo)"), "gold_call": pack("_oracle_readout_double_amplitudes(x,o,mp2,pw,wh,wo)")},
        {"tol": 1e-10, "setup": setup(67,2,1,1,1,1,1), "call": pack("readout_double_amplitudes(x,o,mp2,pw,wh,wo,1e-6)"), "gold_call": pack("_oracle_readout_double_amplitudes(x,o,mp2,pw,wh,wo,1e-6)")},
        {"tol": 1e-10, "setup": setup(71,8,3,4,2,6,6), "call": pack("readout_double_amplitudes(x,o,mp2,pw,wh,wo)"), "gold_call": pack("_oracle_readout_double_amplitudes(x,o,mp2,pw,wh,wo)")},
        {"tol": 0.0, "setup": setup(73,5,2,3,3,4,2) + "\n" + err + "mp2[0,1,0,1]+=1", "call": "et(readout_double_amplitudes,x,o,mp2,pw,wh,wo)", "gold_call": "et(_oracle_readout_double_amplitudes,x,o,mp2,pw,wh,wo)"},
    ]
