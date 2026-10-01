"""
Form the measurement as the sum over nodes of each node's latent image blurred by that node's kernel, after normalising each kernel to unit sum. Kernels are square with odd size K and centre c = (K-1)/2; kernels[gy, gx, c+dy, c+dx] is the share a point in pixel (row, col) sends to pixel (row+dy, col+dx). The output has the sensor's shape; light leaving the sensor is lost. Raise ValueError unless kernels are (Gy, Gx, K, K) with K odd and matching the node grid.

The approximation is accurate when the PSF varies slowly across the spacing between nodes.

Returns
-------
return measurement
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sum_of_convolutions(weighted: "np.ndarray", kernels: "np.ndarray") -> "np.ndarray":
    """Sum over nodes of each latent image blurred by its unit-sum kernel.

    Args:
        weighted: float (Gy, Gx, H, W), per-node latent images (Step 6).
        kernels: float (Gy, Gx, K, K), K odd, not necessarily normalised; kernels[gy, gx, c+dy, c+dx]
            (c = (K-1)//2) is the share a point in pixel (row, col) sends to (row+dy, col+dx).

    Returns:
        float (H, W), measurement.

    Raises:
        ValueError: on inconsistent kernel shape or even K.
    """
    return measurement

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sum_of_convolutions(weighted: "np.ndarray", kernels: "np.ndarray") -> "np.ndarray":
    """Measurement as the sum over nodes of (weighted latent image) * (unit-sum kernel)."""
    import numpy as np
    from scipy.signal import fftconvolve
    Wt = np.asarray(weighted, dtype=np.float64)
    Kt = np.asarray(kernels, dtype=np.float64)
    gy, gx, H, W = Wt.shape
    K = Kt.shape[-1]
    if K % 2 == 0 or Kt.shape[-2] != K or Kt.shape[:2] != (gy, gx):
        raise ValueError("kernels must have shape (Gy, Gx, K, K) with K odd")
    out = np.zeros((H, W))
    for i in range(gy):
        for j in range(gx):
            h = Kt[i, j] / Kt[i, j].sum()
            out += fftconvolve(Wt[i, j], h, mode="same")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
rng = np.random.default_rng(11)
wt = rng.random((2, 3, 20, 24))
def raises(f):
    try:
        f()
    except ValueError:
        return True
    return False
"""
    return [
        {"setup": c + """delta = np.zeros((2, 3, 5, 5)); delta[..., 2, 2] = 3.0
def check(m):
    return bool(np.allclose(np.asarray(m, dtype=float), wt.sum(axis=(0, 1)), rtol=0, atol=1e-12))
""",  # 7.1
         "call": 'check(sum_of_convolutions(wt, delta))',
         "gold_call": 'check(_oracle_sum_of_convolutions(wt, delta))'},
        {"setup": c + """pt = np.zeros((1, 1, 15, 15)); pt[0, 0, 6, 9] = 2.0
ker = rng.random((1, 1, 5, 5))
expected = np.zeros((15, 15)); expected[4:9, 7:12] = 2.0 * ker[0, 0] / ker.sum()
def check(m):
    return bool(np.allclose(np.asarray(m, dtype=float), expected, rtol=0, atol=1e-12))
""",  # 7.2
         "call": 'check(sum_of_convolutions(pt, ker))',
         "gold_call": 'check(_oracle_sum_of_convolutions(pt, ker))'},
        {"setup": c + """w2 = np.zeros((2, 2, 30, 30)); w2[..., 8:22, 8:22] = rng.random((2, 2, 14, 14))
kk = rng.random((2, 2, 7, 7))
""",  # 7.3
         "call": 'np.asarray(sum_of_convolutions(w2, 5.0 * kk), dtype=float) / w2.sum()',
         "gold_call": '_oracle_sum_of_convolutions(w2, kk) / w2.sum()'},
        {"setup": c + """ker = rng.random((2, 3, 9, 9)) * np.linspace(0.1, 1.0, 9)[None, None, None, :]
""",  # 7.4
         "call": 'np.asarray(sum_of_convolutions(wt, ker), dtype=float)',
         "gold_call": '_oracle_sum_of_convolutions(wt, ker)'},
        {"setup": c,  # 7.5
         "call": 'raises(lambda: sum_of_convolutions(wt, np.ones((2, 3, 4, 4))))',
         "gold_call": 'raises(lambda: _oracle_sum_of_convolutions(wt, np.ones((2, 3, 4, 4))))'},
    ]
