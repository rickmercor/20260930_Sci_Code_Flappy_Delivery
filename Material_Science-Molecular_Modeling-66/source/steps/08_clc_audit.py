"""
Assembles the audit from all seven earlier steps, cross-checks three aspect-ratio residuals and macroscopic boundary values, and reduces their contamination, instability and curvature to J.

A valid end-to-end result must survive independent direct, sequence and extrapolation paths at three aspect ratios. The flattened, intermediate and cell-cubic fits each produce an asymptotic boundary, a finite-size correction and an uncertainty pair. Their three normalized quality scores and the curvature of the asymptotic boundary across shape form the final audit. No single finite crystal or single extrapolation is sufficient.

Returns
-------
A float64 array of shape (22,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def clc_audit(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    r"""Assemble the complete lattice-sum audit from the seven preceding public functions.

    cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec and span follow the earlier contracts. p is an integer at least six and pz is an integer from one through $p-2$. Let $z_{\mathrm{mid}} = \lfloor(p+p_z+1)/2\rfloor$. Evaluate flattened, intermediate and cell-cubic aspect ratios $p_z/p$, $z_{\mathrm{mid}}/p$ and $1$ with fit_sizes $[4,8,12,16,20,24]$.

    For each aspect $i$, call clc_boundary_decomposition and define $q_i = \lvert\mathrm{correction}_i\rvert/(\lvert B_i\rvert+\lvert\mathrm{correction}_i\rvert) + \mathrm{rmse}_i/(\mathrm{jackknife}_i+\mathrm{rmse}_i)$. Define $\mathrm{curvature} = \lvert B_{\mathrm{flat}}-2B_{\mathrm{mid}}+B_{\mathrm{cube}}\rvert/(\lvert B_{\mathrm{flat}}\rvert+\lvert B_{\mathrm{mid}}\rvert+\lvert B_{\mathrm{cube}}\rvert)$. Define $J = \operatorname{median}([q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}}]) + \operatorname{std}([q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}}],\mathrm{ddof}=0) + \mathrm{curvature}$. Cross-check every reconstructed residual against both direct-minus-bulk and clc_shape_sequence.

    Use $z_{\mathrm{mid}} = \lfloor(p + p_z + 1)/2\rfloor$ for the intermediate thickness. Run the Step 7 decomposition independently at $p_z$, $z_{\mathrm{mid}}$ and $p$.

    Return a numpy float64 array of shape $(22,)$ in this order: cell-cubic direct potential, flattened direct potential, Ewald bulk potential, flattened minus cell-cubic contrast, nearest-neighbour distance, cell-cubic Madelung constant, spherical direct potential, flattened $B$, flattened correction, flattened reconstructed residual, flattened fit RMSE, flattened jackknife spread, intermediate $B$, cell-cubic $B$, $q_{\mathrm{flat}}$, $q_{\mathrm{mid}}$, $q_{\mathrm{cube}}$, curvature, analytic macroscopic boundary values for flat, intermediate and cell-cubic physical aspect ratios, and $J$.

    Obtain the analytic macroscopic values from column 1 of clc_shape_sequence. They
    are independent source-method checks, not replacements for the prescribed
    finite-sample regression intercepts or for the final $J$ definition.

    The sixth value is the signed finite direct diagnostic $M_{\mathrm{cell}} = d_{\mathrm{nn}}\,\phi_{\mathrm{cube}}$ for the supplied ref_index, following Step 7. Retain the potential's sign, with no absolute value or additional reference-charge factor; the benchmark's value at $q_{\mathrm{ref}} = -1$ is positive.

    The seventh value is clc_direct_potential for the same cell and reference ion with half widths $(p,p,p)$ and spherical=True. Its spherical region is defined in the integer cell-index space by Step 1.

    Raises:
        ValueError: if an earlier contract fails, p is below six, pz is outside one through $p-2$, a cross-step identity fails, a denominator is zero, or a result is non-finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_clc_audit(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    if isinstance(p, bool) or isinstance(pz, bool) or not np.isscalar(p) or not np.isscalar(pz):
        raise ValueError("p and pz must be integer scalars")
    pp, zz = float(p), float(pz)
    if not np.isfinite(pp) or not np.isfinite(zz) or pp != int(pp) or zz != int(zz):
        raise ValueError("p and pz must be finite integers")
    pp, zz = int(pp), int(zz)
    if pp < 6 or zz < 1 or zz > pp - 2:
        raise ValueError("require p at least six and 1 <= pz <= p-2")
    midz = (pp + zz + 1) // 2

    off = _oracle_clc_shape_offsets(pp, pp, pp, False)
    if off.shape[0] != (2 * pp + 1) ** 3 - 1:
        raise ValueError("the cell-cubic summation region has the wrong number of cells")

    cube = float(_oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, pp, False)[0])
    flat = float(_oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, zz, False)[0])
    middle = float(_oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, midz, False)[0])
    bulk = float(_oracle_clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)[0])
    dnn = float(_oracle_clc_nearest_neighbour(cell_edges, basis_pos, ref_index, span)[0])
    sph = float(_oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, pp, True)[0])

    zs = (zz, midz, pp)
    seq = _oracle_clc_shape_sequence(cell_edges, basis_pos, basis_q, ref_index, pp,
                                     np.asarray(zs, dtype=np.float64) / float(pp), alpha, n_real, n_rec)
    fit_sizes = np.array([4, 8, 12, 16, 20, 24], dtype=np.float64)
    decs = [_oracle_clc_boundary_decomposition(cell_edges, basis_pos, basis_q, ref_index,
                                                pp, z, fit_sizes, alpha, n_real, n_rec, span)
            for z in zs]
    direct_residuals = np.array([flat - bulk, middle - bulk, cube - bulk], dtype=np.float64)
    q = []
    for k, dec in enumerate(decs):
        B, correction, residual, rmse, jackknife = map(float, dec[:5])
        scale = max(1.0, abs(residual))
        if abs(float(seq[k, 0]) - residual) > 1e-9 * scale or abs(direct_residuals[k] - residual) > 1e-9 * scale:
            raise ValueError("a reconstructed residual disagrees with an independent path")
        if abs((B + correction) - residual) > 1e-9 * scale:
            raise ValueError("boundary plus correction does not reconstruct a residual")
        dc = abs(B) + abs(correction)
        di = jackknife + rmse
        if dc <= 0.0 or di <= 0.0:
            raise ValueError("an aspect score denominator is zero")
        q.append(abs(correction) / dc + rmse / di)

    mad = float(decs[0][5])
    if abs(mad - cube * dnn) > 1e-9 * max(1.0, abs(mad)):
        raise ValueError("the Madelung constant disagrees with the scaled lattice sum")
    Bf, Cf, Rf, Ef, Kf = map(float, decs[0][:5])
    Bm = float(decs[1][0])
    Bc = float(decs[2][0])
    den_curve = abs(Bf) + abs(Bm) + abs(Bc)
    if den_curve <= 0.0:
        raise ValueError("the curvature denominator is zero")
    curvature = abs(Bf - 2.0 * Bm + Bc) / den_curve
    q = np.asarray(q, dtype=np.float64)
    J = float(np.median(q) + np.std(q, ddof=0) + curvature)
    out = np.array([cube, flat, bulk, flat - cube, dnn, mad, sph, Bf, Cf, Rf, Ef, Kf,
                    Bm, Bc, q[0], q[1], q[2], curvature,
                    seq[0, 1], seq[1, 1], seq[2, 1], J], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite audit record")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"tol":1e-7,"setup":"import numpy as np\n# all entries except the aspect scores and J\n","call":"np.delete(clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,4.0,3,9,2),[14,15,16,21])","gold_call":"np.delete(_oracle_clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,4.0,3,9,2),[14,15,16,21])"},
        {"tol":1e-4,"setup":"import numpy as np\n# aspect scores and J: ratios of small fit diagnostics, sensitive to summation order\n","call":"clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,4.0,3,9,2)[[14,15,16,21]]","gold_call":"_oracle_clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,4.0,3,9,2)[[14,15,16,21]]"},
        {"tol":1e-7,"setup":"import numpy as np\n# all entries except the aspect scores and J\n","call":"np.delete(clc_audit(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,4.0,3,8,2),[14,15,16,21])","gold_call":"np.delete(_oracle_clc_audit(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,4.0,3,8,2),[14,15,16,21])"},
        {"tol":1e-4,"setup":"import numpy as np\n# aspect scores and J: ratios of small fit diagnostics, sensitive to summation order\n","call":"clc_audit(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,4.0,3,8,2)[[14,15,16,21]]","gold_call":"_oracle_clc_audit(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,4.0,3,8,2)[[14,15,16,21]]"},
        {"tol":1e-7,"setup":"import numpy as np\n# four-ion basis with a nonzero site boundary term on an elongated cell\n# all entries except the aspect scores and J\n","call":"np.delete(clc_audit(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]),np.array([1.,1,-1,-1]),0,8,4,4.0,3,8,2),[14,15,16,21])","gold_call":"np.delete(_oracle_clc_audit(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]),np.array([1.,1,-1,-1]),0,8,4,4.0,3,8,2),[14,15,16,21])"},
        {"tol":1e-4,"setup":"import numpy as np\n# four-ion basis with a nonzero site boundary term on an elongated cell\n# aspect scores and J: ratios of small fit diagnostics, sensitive to summation order\n","call":"clc_audit(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]),np.array([1.,1,-1,-1]),0,8,4,4.0,3,8,2)[[14,15,16,21]]","gold_call":"_oracle_clc_audit(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]),np.array([1.,1,-1,-1]),0,8,4,4.0,3,8,2)[[14,15,16,21]]"},
        {"tol":1e-7,"setup":"import numpy as np\n# boundary: the thinnest flattened crystal, one cell layer along z\n# all entries except the aspect scores and J\n","call":"np.delete(clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,4.0,3,9,2),[14,15,16,21])","gold_call":"np.delete(_oracle_clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,4.0,3,9,2),[14,15,16,21])"},
        {"tol":1e-4,"setup":"import numpy as np\n# boundary: the thinnest flattened crystal, one cell layer along z\n# aspect scores and J: ratios of small fit diagnostics, sensitive to summation order\n","call":"clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,4.0,3,9,2)[[14,15,16,21]]","gold_call":"_oracle_clc_audit(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,4.0,3,9,2)[[14,15,16,21]]"},
        {"tol":1e-9,"setup":"import numpy as np\n# invalid input: a thickness larger than the width\ndef _probe(fn):\n    try:\n        fn(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,6,7,4.0,3,8,2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(clc_audit)","gold_call":"_probe(_oracle_clc_audit)"},
        {"tol":1e-7,"setup":"import numpy as np\n# zero dipole does not eliminate the site-boundary potential for this basis\n# all entries except the aspect scores and J\n","call":"np.delete(clc_audit(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),0,12,3,4.0,3,9,2),[14,15,16,21])","gold_call":"np.delete(_oracle_clc_audit(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),0,12,3,4.0,3,9,2),[14,15,16,21])"},
        {"tol":1e-4,"setup":"import numpy as np\n# zero dipole does not eliminate the site-boundary potential for this basis\n# aspect scores and J: ratios of small fit diagnostics, sensitive to summation order\n","call":"clc_audit(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),0,12,3,4.0,3,9,2)[[14,15,16,21]]","gold_call":"_oracle_clc_audit(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),0,12,3,4.0,3,9,2)[[14,15,16,21]]"},
    ]
