"""
Compute the source-derived singles readout for the supplied orbital states and maps. Return T1 with shape (O,V), invariant pair features with shape (O,V,K), and the requested signed T1 checksum. Follow the finite-fixture unscaled real-vector scalar-coupling convention stated in the problem statement.

The amplitude readout follows the selected molecular-orbital architecture. The supplied analytic maps and diagnostic definitions fix this benchmark instance

Returns
-------
tuple : T1 amplitudes, pair features, and T1 checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def readout_single_amplitudes(features: "np.ndarray", occupied_count: int, pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return T1 amplitudes, invariant pair features, and a checksum.

    Use the finite-fixture scalar-coupling and checksum conventions in the
    problem statement. Recover the source readout method for the supplied maps.

    Parameters
    ----------
    features : np.ndarray
        Final orbital states with shape (P,A,K,M).
    occupied_count : int
        Number of occupied orbitals, strictly between zero and P.
    pair_weights : np.ndarray
        Shared channel map with shape (K,K).
    hidden_weights : np.ndarray
        Readout hidden map with shape (K,H).
    output_weights : np.ndarray
        Readout output map with shape (H,).
    epsilon : float
        Positive finite normalization stabilizer.

    Returns
    -------
    t1 : np.ndarray
        Singles amplitudes with shape (n_occ,n_virt).
    pair_features : np.ndarray
        Invariant contracted features with shape (n_occ,n_virt,K).
    sign_checksum : float
        Weighted checksum of T1 in C order.

    Raises
    ------
    ValueError
        If shapes, counts, values, or epsilon are invalid.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_readout_single_amplitudes(features: "np.ndarray", occupied_count: int, pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(features, dtype=float)
    pw = np.asarray(pair_weights, dtype=float)
    wh = np.asarray(hidden_weights, dtype=float)
    wo = np.asarray(output_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if not isinstance(occupied_count, (int, np.integer)) or not 1 <= occupied_count < p:
        raise ValueError("occupied_count must be an integer in [1,P-1]")
    if pw.shape != (k, k) or wh.ndim != 2 or wh.shape[0] != k or wo.shape != (wh.shape[1],):
        raise ValueError("readout weight shapes are inconsistent")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, pw, wh, wo)):
        raise ValueError("numeric inputs must be finite")
    norms = np.sqrt(np.sum(x * x, axis=(1, 2, 3))) + abs(float(epsilon))
    unit = x / norms[:, None, None, None]
    projected = np.einsum("jk,pakm->pajm", pw, unit, optimize=True)
    occ = projected[:occupied_count]
    virt = projected[occupied_count:]
    pair_features = np.einsum("iakm,vakm->ivk", occ, virt, optimize=True)
    hidden = np.tanh(np.einsum("ivk,kh->ivh", pair_features, wh, optimize=True))
    t1 = np.einsum("ivh,h->iv", hidden, wo, optimize=True)
    flat = t1.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat))
    return t1, pair_features, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = lambda call: "(lambda z: [len(z),*z[0].shape,*z[1].shape,*np.concatenate([z[0].ravel(),z[1].ravel(),[z[2]]]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    setup = lambda seed,p,a,k,m,h,o: f'import numpy as np\nr=np.random.default_rng({seed});x=r.normal(size=({p},{a},{k},{m}));pw=r.normal(scale=.2,size=({k},{k}));wh=r.normal(scale=.3,size=({k},{h}));wo=r.normal(scale=.2,size={h});o={o}'
    return [
        {"tol": 1e-10, "setup": setup(43,7,4,3,3,5,3), "call": pack("readout_single_amplitudes(x,o,pw,wh,wo)"), "gold_call": pack("_oracle_readout_single_amplitudes(x,o,pw,wh,wo)")},
        {"tol": 1e-10, "setup": setup(47,2,1,1,1,1,1), "call": pack("readout_single_amplitudes(x,o,pw,wh,wo,1e-6)"), "gold_call": pack("_oracle_readout_single_amplitudes(x,o,pw,wh,wo,1e-6)")},
        {"tol": 1e-10, "setup": setup(53,8,3,4,2,6,7), "call": pack("readout_single_amplitudes(x,o,pw,wh,wo)"), "gold_call": pack("_oracle_readout_single_amplitudes(x,o,pw,wh,wo)")},
        {"tol": 0.0, "setup": setup(59,5,2,3,3,4,2) + "\n" + err + "o=5", "call": "et(readout_single_amplitudes,x,o,pw,wh,wo)", "gold_call": "et(_oracle_readout_single_amplitudes,x,o,pw,wh,wo)"},
    ]
