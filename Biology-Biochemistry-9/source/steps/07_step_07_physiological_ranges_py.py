"""
Calculate physiological energy intervals.

A physiological energy interval is an extremum over a complete constrained biochemical state. The standard-energy uncertainty, concentration measurements and active flux directions specify that state space.

Returns
-------
A finite ndarray of shape (C,P,1+2N), ordered as feasibility, lower endpoints, upper endpoints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def physiological_ranges(
    stoichiometry: "np.ndarray",
    directions: "np.ndarray",
    standard_mean: "np.ndarray",
    standard_sd: "np.ndarray",
    log_bounds: "np.ndarray",
    contrasts: "np.ndarray",
    contrast_bounds: "np.ndarray",
    gas_constant: float,
    temperature: float,
    driving_floor: float,
) -> "np.ndarray":
    """
    Let stoichiometry have shape (M,N), directions have shape (C,P,N) with entries
    -1,0,+1, and standard_mean and standard_sd have length N in kJ mol^-1. Standard
    uncertainties use the source 99% interval convention; standard_sd is nonnegative.
    log_bounds has shape (C,M,2), and bounds natural logarithms of concentrations
    relative to 1 mol L^-1. contrasts has shape (H,M); contrast_bounds has shape
    (C,H,2). Every last pair is [lower,upper], including equality bounds. Apply the
    source physiological variability analysis, adding the supplied contrast
    inequalities. R, T and driving_floor are positive finite quantities in kJ mol^-1
    K^-1, K and kJ mol^-1. Return shape (C,P,1+2N): feasible flag, N lower endpoints, N
    upper endpoints. If the joint physiological region is empty, return an all-zero row.
    Otherwise the flag is 1 and endpoints are computed for every reaction. Invalid
    shapes, indices, signs, bounds or nonfinite data raise ValueError; an unresolved
    numerical solver status raises RuntimeError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_physiological_ranges(
    stoichiometry: "np.ndarray",
    directions: "np.ndarray",
    standard_mean: "np.ndarray",
    standard_sd: "np.ndarray",
    log_bounds: "np.ndarray",
    contrasts: "np.ndarray",
    contrast_bounds: "np.ndarray",
    gas_constant: float,
    temperature: float,
    driving_floor: float,
) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import linprog

    S = np.asarray(stoichiometry, dtype=float)
    z = np.asarray(directions, dtype=float)
    mu = np.asarray(standard_mean, dtype=float)
    sd = np.asarray(standard_sd, dtype=float)
    lo = np.asarray(log_bounds, dtype=float)
    H = np.asarray(contrasts, dtype=float)
    hb = np.asarray(contrast_bounds, dtype=float)
    R = float(gas_constant)
    T = float(temperature)
    eps = float(driving_floor)
    if S.ndim != 2 or z.ndim != 3 or min(S.shape + z.shape) == 0:
        raise ValueError("Invalid stoichiometry or direction panel")
    m, n = S.shape
    C, P, nz = z.shape
    if (
        nz != n
        or mu.shape != (n,)
        or sd.shape != (n,)
        or (lo.shape != (C, m, 2))
        or (H.ndim != 2)
        or (H.shape[1] != m)
        or (hb.shape != (C, H.shape[0], 2))
        or any((not np.isfinite(x).all() for x in (S, z, mu, sd, lo, H, hb)))
        or np.any(sd < 0)
        or np.any(lo[:, :, 0] > lo[:, :, 1])
        or np.any(hb[:, :, 0] > hb[:, :, 1])
        or (not np.isin(z, [-1, 0, 1]).all())
        or (not np.isfinite([R, T, eps]).all())
        or (min(R, T, eps) <= 0)
    ):
        raise ValueError("Invalid physiological inputs")
    D = np.c_[np.eye(n), R * T * S.T]
    J = np.c_[np.zeros((len(H), n)), H]
    result = np.zeros((C, P, 1 + 2 * n))
    options = {
        "primal_feasibility_tolerance": 1e-09,
        "dual_feasibility_tolerance": 1e-09,
    }
    for c in range(C):
        bounds = list(zip(mu - 2.58 * sd, mu + 2.58 * sd)) + list(map(tuple, lo[c]))
        cache = {}
        for p in range(P):
            key = tuple(z[c, p])
            if key in cache:
                result[c, p] = cache[key]
                continue
            active = np.flatnonzero(z[c, p])
            A = np.vstack([z[c, p, active, None] * D[active], J, -J])
            b = np.r_[np.full(len(active), -eps), hb[c, :, 1], -hb[c, :, 0]]

            def _run(obj):
                return linprog(
                    obj,
                    A_ub=A,
                    b_ub=b,
                    bounds=bounds,
                    method="highs",
                    options=options,
                )

            first = _run(np.zeros(n + m))
            row = np.zeros(1 + 2 * n)
            if first.status == 2:
                cache[key] = row
                continue
            if not first.success:
                raise RuntimeError(first.message)
            row[0] = 1.0
            for j in range(n):
                left = _run(D[j])
                right = _run(-D[j])
                if not left.success or not right.success:
                    raise RuntimeError("Unresolved finite range")
                row[1 + j] = left.fun
                row[1 + n + j] = -right.fun
            result[c, p] = row
            cache[key] = row
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "S = np.array([[-1.0], [1.0]])\n"
            "z = np.ones((1, 1, 1))\n"
            "gam = np.zeros(1)\n"
            "sd = np.zeros(1)\n"
            "lb = np.array([[[-2.0, 0.0], [-2.0, 0.0]]])\n"
            "H = np.empty((0, 2))\n"
            "hb = np.empty((1, 0, 2))",
            "call": "physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
            "gold_call": "_oracle_physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
        },
        {
            "setup": "import numpy as np\n"
            "S = np.array([[-1.0], [1.0]])\n"
            "z = np.ones((1, 1, 1))\n"
            "gam = np.zeros(1)\n"
            "sd = np.zeros(1)\n"
            "lb = np.array([[[-2.0, 0.0], [-2.0, 0.0]]])\n"
            "H = np.empty((0, 2))\n"
            "hb = np.empty((1, 0, 2))\n"
            "z[:] = 0.0\n"
            "gam[:] = 1.0\n"
            "lb[:] = 0.0",
            "call": "physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
            "gold_call": "_oracle_physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
        },
        {
            "setup": "import numpy as np\n"
            "S = np.array([[-1.0], [1.0]])\n"
            "z = np.ones((1, 1, 1))\n"
            "gam = np.zeros(1)\n"
            "sd = np.zeros(1)\n"
            "lb = np.array([[[-2.0, 0.0], [-2.0, 0.0]]])\n"
            "H = np.empty((0, 2))\n"
            "hb = np.empty((1, 0, 2))\n"
            "lb = np.array([[[0.0, 0.0], [1.0, 1.0]]])",
            "call": "physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
            "gold_call": "_oracle_physiological_ranges(S,z,gam,sd,lb,H,hb,1.,1.,.1)",
        },
        {
            "setup": "import numpy as np\n"
            "S = np.array([[-1.0], [1.0]])\n"
            "z = np.ones((1, 1, 1))\n"
            "gam = np.zeros(1)\n"
            "sd = np.zeros(1)\n"
            "lb = np.array([[[-2.0, 0.0], [-2.0, 0.0]]])\n"
            "H = np.empty((0, 2))\n"
            "hb = np.empty((1, 0, 2))\n"
            "sd[:] = -1.0\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(physiological_ranges, (S,z,gam,sd,lb,H,hb,1.0,1.0,0.1,))",
            "gold_call": "_error_check(_oracle_physiological_ranges, "
            "(S,z,gam,sd,lb,H,hb,1.0,1.0,0.1,))",
            "tol": 0.0,
        },
    ]
