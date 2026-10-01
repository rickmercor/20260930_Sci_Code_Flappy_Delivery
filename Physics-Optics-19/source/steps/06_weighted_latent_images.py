"""
Split the scene into per-node latent images. The nodes form a separable sub-grid: node_rows are strictly increasing scene row indices from 0 to Hs-1, node_cols likewise for Ws (a single node on an axis, any index, has weight 1 everywhere along it; any other invalid list raises ValueError). Each sample is weighted, for every node, by the node's piecewise-linear weight along rows times that along columns, evaluated at the sample's own indices (weights sum to one per sample), and added into the pixel where its chief ray lands: column floor((x-x0)/p+W/2), row floor((y-y0)/p+H/2). Samples with NaN or off-sensor landings are dropped.

Over a small field patch the system is nearly shift-invariant, so a PSF may be interpolated between a few sampled ones.

Returns
-------
return weighted
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_latent_images(scene: "np.ndarray", landing: "np.ndarray", node_rows: list, node_cols: list,
                           pixel_pitch: float, sensor_shape: tuple,
                           sensor_center: tuple) -> "np.ndarray":
    """Per-node weighted latent images on the sensor grid.

    Args:
        scene: float (Hs, Ws), scene intensity on the field-direction grid.
        landing: float (Hs, Ws, 2), chief-ray landing [x, y] of each sample, mm.
        node_rows, node_cols: lists of Gy / Gx node indices into the scene rows / columns.
        pixel_pitch: pixel pitch p, mm.
        sensor_shape: (H, W).
        sensor_center: (x0, y0), mm.

    Returns:
        float (Gy, Gx, H, W); [gy, gx] is the latent image of node (node_rows[gy], node_cols[gx]).

    Raises:
        ValueError: if a node list of length > 1 is not strictly increasing from 0 to n-1.
    """
    return weighted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weighted_latent_images(scene: "np.ndarray", landing: "np.ndarray", node_rows: list, node_cols: list,
                                   pixel_pitch: float, sensor_shape: tuple,
                                   sensor_center: tuple) -> "np.ndarray":
    """Per-node weighted latent images: scene samples weighted by the node's
    interpolation weight, then deposited at their chief-ray landing pixel."""
    import numpy as np
    b = np.asarray(scene, dtype=np.float64)
    hs, ws = b.shape
    L = np.asarray(landing, dtype=np.float64)
    H, W = int(sensor_shape[0]), int(sensor_shape[1])
    weights_1d = []
    for nodes, n in ((np.asarray(node_rows, dtype=int), hs), (np.asarray(node_cols, dtype=int), ws)):
        if len(nodes) == 1:
            weights_1d.append(np.ones((1, n)))
            continue
        if nodes[0] != 0 or nodes[-1] != n - 1 or np.any(np.diff(nodes) <= 0):
            raise ValueError("nodes must increase strictly from 0 to n-1")
        idx = np.arange(n)
        weights_1d.append(np.stack([np.interp(idx, nodes, np.eye(len(nodes))[g]) for g in range(len(nodes))]))
    wy, wx = weights_1d
    with np.errstate(invalid="ignore"):
        col = np.floor((L[..., 0] - sensor_center[0]) / pixel_pitch + W / 2.0)
        row = np.floor((L[..., 1] - sensor_center[1]) / pixel_pitch + H / 2.0)
    inside = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
    r_i, c_i = row[inside].astype(int), col[inside].astype(int)
    out = np.zeros((wy.shape[0], wx.shape[0], H, W))
    for gy in range(wy.shape[0]):
        for gx in range(wx.shape[0]):
            vals = (b * wy[gy][:, None] * wx[gx][None, :])[inside]
            np.add.at(out[gy, gx], (r_i, c_i), vals)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
g = np.random.default_rng(7)
scene = g.random((17, 9))
TX, TY = np.meshgrid(np.linspace(-0.05, 0.05, 9), np.linspace(0.0, 0.2, 17))
r2 = TX ** 2 + TY ** 2
L = 4.5 * np.stack([TX, TY], axis=-1) * (1.0 - 0.8 * r2)[..., None]   # barrel-distorted landing
L[16, 8] = np.nan
def raises(f):
    try:
        f()
    except ValueError:
        return True
    return False
"""
    return [
        {"setup": c + """gg = np.arange(3) * 0.01 - 0.01 + 0.004
land = np.stack(np.meshgrid(gg, gg), axis=-1)
sc = np.arange(1.0, 10.0).reshape(3, 3)
args = (sc, land, [0, 2], [0, 2], 0.01, (3, 3), (0.0, 0.0))
""",  # 6.1
         "call": 'np.asarray(weighted_latent_images(*args), dtype=float)',
         "gold_call": '_oracle_weighted_latent_images(*args)', "tol": 1e-14},
        {"setup": c,  # 6.2
         "call": 'np.asarray(weighted_latent_images(scene, L, [0, 8, 16], [0, 4, 8], 0.02, (60, 30), (0.0, 0.5)), dtype=float).sum(axis=(0, 1))',
         "gold_call": '_oracle_weighted_latent_images(scene, L, [3], [5], 0.02, (60, 30), (0.0, 0.5))[0, 0]'},
        {"setup": c + """land = np.array([[[0.00999999, 0.0], [0.0195, 0.0], [np.nan, np.nan], [0.05, 0.0], [0.0105, 0.0]]])
args = (np.array([[1.0, 2.0, 4.0, 8.0, 16.0]]), land, [0], [0, 4], 0.01, (1, 4), (0.0, 0.0))
""",  # 6.3
         "call": 'np.asarray(weighted_latent_images(*args), dtype=float)',
         "gold_call": '_oracle_weighted_latent_images(*args)', "tol": 1e-14},
        {"setup": c + """args = (scene, L, [0, 3, 16], [0, 6, 8], 0.02, (40, 20), (0.01, 0.45))
""",  # 6.4
         "call": 'np.asarray(weighted_latent_images(*args), dtype=float)',
         "gold_call": '_oracle_weighted_latent_images(*args)'},
        {"setup": c + """args = (scene, L, [1, 16], [0, 8], 0.02, (60, 30), (0.0, 0.5))
""",  # 6.5
         "call": 'raises(lambda: weighted_latent_images(*args))',
         "gold_call": 'raises(lambda: _oracle_weighted_latent_images(*args))'},
    ]
