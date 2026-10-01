"""
Fits the paper-defined boundary and finite-size decomposition across several crystal sizes, then checks its stability by leave-one-size-out refits.

The source separates a finite Coulomb lattice sum into a periodic bulk term, a size-independent boundary contribution fixed by shape, and a finite-size correction that decays as the crystal grows. This step applies a prescribed numerical estimator: it evaluates a rounded sequence of finite crystals approaching the target half-width ratio and extrapolates the residual against inverse size. Rounding means the finite crystals need not have identical physical aspect ratios. The fit and jackknife are disclosed numerical conventions, while the three-way decomposition and the normalization by the nearest-neighbour interaction are source-defined choices.

Returns
-------
A float64 array of shape (6,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def clc_boundary_decomposition(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, fit_sizes: "np.ndarray", alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    r"""Separate a finite flattened-crystal residual into an asymptotic boundary term and a finite-size correction.

    cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec and span follow the earlier contracts. p is an integer at least four and pz is an integer from one through p. fit_sizes is a one-dimensional array of at least six distinct, strictly increasing integers, each at least four.

    Let $\rho=p_z/p$. For every $s$ in fit_sizes, set $s_z=\max(1,\lfloor s\rho+0.5\rfloor)$ and obtain $y(s)$ from clc_boundary_term for the rectangular crystal $(s,s,s_z)$. Fit
        $y(s)=B+c_1/s+c_2/s^2+c_3/s^3$
    by weighted least squares after multiplying row $s$ by $\sqrt{s}$. $B$ is the intercept estimate from this prescribed finite-sample regression, not an exact analytic evaluation of the macroscopic boundary contribution. Evaluate the target residual $y(p)$ separately and define its finite-size correction as $y(p)-B$. Compute the unweighted fit RMSE. Refit after deleting each size and report the largest absolute change in $B$ as the jackknife spread. Finally compute the cell-cubic Madelung constant from clc_direct_potential and clc_nearest_neighbour.

    The sixth value is the signed finite direct diagnostic $M_{\mathrm{cell}} = d_{\mathrm{nn}}\,\phi_{\mathrm{cube}}$ for the supplied ref_index, where $\phi_{\mathrm{cube}}$ is the direct finite potential for half widths $(p,p,p)$. Retain the sign of $\phi_{\mathrm{cube}}$, with no absolute value or additional reference-charge factor. For the benchmark's negative reference ion ($q_{\mathrm{ref}} = -1$), this quantity is positive. It is distinct from the bulk periodic Madelung constant.

    Return a numpy float64 array $[B, y(p)-B, y(p), \mathrm{rmse}, \mathrm{jackknife\_spread}, \mathrm{madelung}]$.

    Raises:
        ValueError: if an upstream contract fails; p or pz is invalid; fit_sizes has the wrong shape, fewer than six values, a non-finite or non-integer value, a value below four, duplicates, or non-increasing order; or a result is non-finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_clc_boundary_decomposition(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, fit_sizes: "np.ndarray", alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    if isinstance(p, bool) or isinstance(pz, bool) or not np.isscalar(p) or not np.isscalar(pz):
        raise ValueError("p and pz must be integer scalars")
    pp = float(p)
    zz = float(pz)
    if not np.isfinite(pp) or not np.isfinite(zz) or pp != int(pp) or zz != int(zz):
        raise ValueError("p and pz must be finite integers")
    pp, zz = int(pp), int(zz)
    if pp < 4 or zz < 1 or zz > pp:
        raise ValueError("require p at least four and 1 <= pz <= p")

    s = np.asarray(fit_sizes, dtype=np.float64)
    if s.ndim != 1 or s.size < 6 or not np.all(np.isfinite(s)):
        raise ValueError("fit_sizes must be a finite one-dimensional array with at least six values")
    if np.any(s != np.floor(s)) or np.any(s < 4) or np.any(np.diff(s) <= 0) or np.unique(s).size != s.size:
        raise ValueError("fit_sizes must be distinct increasing integers at least four")

    rho = float(zz) / float(pp)
    y = []
    for n in s.astype(int):
        nz = max(1, int(np.floor(float(n) * rho + 0.5)))
        y.append(float(_oracle_clc_boundary_term(cell_edges, basis_pos, basis_q, ref_index,
                                                  n, n, nz, False, alpha, n_real, n_rec)[0]))
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack((np.ones(s.size), 1.0 / s, 1.0 / s**2, 1.0 / s**3))
    sw = np.sqrt(s)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    residual = y - X @ beta
    rmse = float(np.sqrt(np.mean(residual**2)))

    loo = []
    for k in range(s.size):
        keep = np.arange(s.size) != k
        b = np.linalg.lstsq(X[keep] * sw[keep, None], y[keep] * sw[keep], rcond=None)[0]
        loo.append(float(b[0]))
    jack = float(np.max(np.abs(np.asarray(loo) - beta[0])))

    target = float(_oracle_clc_boundary_term(cell_edges, basis_pos, basis_q, ref_index,
                                              pp, pp, zz, False, alpha, n_real, n_rec)[0])
    cube = float(_oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index,
                                               pp, pp, pp, False)[0])
    dnn = float(_oracle_clc_nearest_neighbour(cell_edges, basis_pos, ref_index, span)[0])
    mad = cube * dnn
    out = np.array([beta[0], target - beta[0], target, rmse, jack, mad], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite boundary decomposition")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_boundary_decomposition(np.array([1.,1.3,0.42]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,np.array([4,8,12,16,20,24]),4.0,3,9,2)","gold_call":"_oracle_clc_boundary_decomposition(np.array([1.,1.3,0.42]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,4,np.array([4,8,12,16,20,24]),4.0,3,9,2)"},
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_boundary_decomposition(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,np.array([4,8,12,16,20,24]),4.0,3,8,2)","gold_call":"_oracle_clc_boundary_decomposition(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,np.array([4,8,12,16,20,24]),4.0,3,8,2)"},
        {"tol":1e-8,"setup":"import numpy as np\n","call":"clc_boundary_decomposition(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,8,4,np.array([4,6,8,10,12,14]),4.0,3,8,2)","gold_call":"_oracle_clc_boundary_decomposition(np.array([1.,1.,1.6]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,8,4,np.array([4,6,8,10,12,14]),4.0,3,8,2)"},
        {"tol":1e-9,"setup":"import numpy as np\n# boundary: a one-layer target with six independent fit sizes\n","call":"clc_boundary_decomposition(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,np.array([4,8,12,16,20,24]),4.0,3,9,2)","gold_call":"_oracle_clc_boundary_decomposition(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,8,1,np.array([4,8,12,16,20,24]),4.0,3,9,2)"},
        {"tol":1e-9,"setup":"import numpy as np\n# invalid input: duplicate convergence sizes violate the fit contract\ndef _probe(fn):\n    try:\n        fn(np.array([1.,1.,1.]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,3,np.array([4,8,8,12,16,20]),4.0,3,8,2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(clc_boundary_decomposition)","gold_call":"_probe(_oracle_clc_boundary_decomposition)"},
    ]
