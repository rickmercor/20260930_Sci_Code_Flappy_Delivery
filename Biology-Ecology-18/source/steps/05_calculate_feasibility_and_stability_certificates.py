"""
Calculate feasibility and stability certificates.

For every plan, calculate the three finite feasibility thresholds and the three symmetric-part stability boundaries. The plan's lower effort is the buffered maximum of those six numbers and minimum_effort.

Reject misaligned, non-finite, or out-of-contract inputs with ValueError.

Returns
-------
A float64 matrix with three feasibility thresholds, three stability boundaries, and the lower effort per plan.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    """Calculate feasibility and stability certificates.

    Returns
    -------
    A float64 matrix with three feasibility thresholds, three stability boundaries, and the lower effort per plan.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _canonical(k):
    m = k.mean(axis=0)
    return np.vstack([k - m[None, :], m])


def _feasibility(p):
    c, m = p[:-1], p[-1]
    n = c.shape[0]
    # Finite poles reached from uniform forcing only.
    v = np.ones(n, dtype=np.float64)
    krylov = []
    for _ in range(n):
        for q in krylov:
            v = v - q * np.dot(q, v)
        norm = np.linalg.norm(v)
        if norm <= 1e-11:
            break
        q = v / norm
        krylov.append(q)
        v = -c @ q
    Q = np.column_stack(krylov)
    A = Q.T @ (-c) @ Q
    eig = np.linalg.eigvals(A)
    poles = eig.real[np.abs(eig.imag) <= 1e-8]
    bounds = [0.0]
    for pole in poles:
        if pole > 0.0:
            bounds.append(float(pole))
    grid = np.unique(np.array(bounds, dtype=np.float64))
    # Locate the last sign boundary of 1 - (aI+C)^-1 1 dot m.
    def _den(a):
        return float(1.0 - m @ np.linalg.solve(a * np.eye(n) + c, np.ones(n)))
    candidates = list(grid)
    test = np.geomspace(max(1e-8, grid.max(initial=0.0) + 1e-7), 100.0, 300)
    prev_a, prev_v = test[0], _den(test[0])
    for a in test[1:]:
        val = _den(a)
        if np.isfinite(val) and np.isfinite(prev_v) and val * prev_v < 0.0:
            lo, hi = prev_a, a
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _den(lo) * _den(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            candidates.append(0.5 * (lo + hi))
        prev_a, prev_v = a, val
    return float(max(candidates))


def _oracle_plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    D = np.asarray(designs, dtype=np.float64)
    if B.ndim != 3 or D.ndim != 2 or D.shape[1] != B.shape[1] or buffer <= 1.0 or minimum_effort <= 0.0:
        raise ValueError("invalid plan-certificate inputs")
    rows = []
    for d in D:
        f, g = [], []
        for b in B:
            k = b / d[None, :]
            f.append(_feasibility(_canonical(k)))
            g.append(max(0.0, -float(np.linalg.eigvalsh((k + k.T) / 2.0)[0])))
        lo = max(float(minimum_effort), float(buffer) * max(max(f), max(g)))
        rows.append(f + g + [lo])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {

            "setup": "import numpy as np\nresource_matrix=np.array([[7.0, 39.0, 11.0, 9.0, 0.0, 6.0, 0.0], [39.0, 0.0, 25.0, 0.0, 31.0, 4.0, 0.0], [0.0, 0.0, 0.0, 21.0, 37.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0], [0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0], [0.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)\nk=10000.0\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\nbase_competition=np.array([[[0.0, 0.515, 0.858, 1.099, 0.88, 0.244], [0.218, 0.0, 0.792, 0.794, 0.891, 0.134], [1.02, 0.121, 0.0, 0.869, 0.803, 0.199], [0.396, 0.345, 0.452, 0.0, 0.859, 0.543], [0.876, 0.526, 0.397, 0.614, 0.0, 0.387], [0.398, 0.691, 0.511, 0.179, 0.331, 0.0]], [[0.0, 0.685, 1.068, 1.013, 0.914, 0.3], [0.205, 0.0, 0.669, 0.87, 0.757, 0.15], [1.302, 0.093, 0.0, 0.783, 0.697, 0.25], [0.292, 0.416, 0.471, 0.0, 0.952, 0.72], [0.828, 0.54, 0.443, 0.606, 0.0, 0.406], [0.372, 0.636, 0.487, 0.133, 0.421, 0.0]], [[0.0, 0.628, 1.112, 1.418, 1.106, 0.223], [0.226, 0.0, 1.133, 0.992, 1.014, 0.177], [1.352, 0.158, 0.0, 1.025, 0.93, 0.232], [0.337, 0.409, 0.347, 0.0, 1.136, 0.366], [1.115, 0.593, 0.438, 0.666, 0.0, 0.469], [0.465, 0.774, 0.403, 0.219, 0.304, 0.0]]], dtype=float)\nbase_designs=np.array([[0.614, 1.117, 0.817, 1.164, 1.254, 0.973], [1.459, 0.848, 1.019, 0.854, 1.278, 0.643], [0.595, 1.436, 0.855, 0.827, 0.9, 0.781], [0.922, 1.044, 0.85, 1.174, 1.472, 0.922], [1.222, 1.284, 0.928, 0.63, 1.255, 0.553], [1.044, 1.404, 0.966, 0.579, 0.665, 0.716]], dtype=float)\nbase_loadings=np.array([[[0.306, 0.292], [-0.313, 0.23], [-0.395, 0.32], [0.382, 0.177], [0.361, 0.427], [-0.416, -0.493]], [[-0.198, 0.303], [-0.129, 0.568], [-0.205, -0.096], [0.252, 0.326], [-0.297, 0.106], [-0.056, -0.095]], [[-0.008, 0.031], [-0.332, 0.017], [0.395, 0.785], [0.069, 0.69], [0.083, -0.445], [-0.045, -0.223]]], dtype=float)\nbase_reserves=np.array([0.032, 0.045, 0.036, 0.027, 0.041, 0.034], dtype=float)\nupper_efforts=np.array([8.0, 7.6, 8.4, 7.9, 8.2, 7.7], dtype=float)\ncosts=np.array([1.0, 1.0, 1.0, 0.97672, 1.0, 1.0], dtype=float)\npolicies=np.array([[3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05]], dtype=float)\ncoefficients=np.array([1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5], dtype=float)\nbuffer=1.2\nminimum_effort=0.2\nniche=_oracle_noncircular_niche_profile(resource_matrix,k)\ncalibration=_oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)\nsignal=_oracle_petra_residual_signal(reference,disturbed,candidates[int(calibration[0])],max_steps,indices,guild_map)\npacked=_oracle_assemble_coupled_system(base_competition,base_designs,base_loadings,base_reserves,niche,signal,coefficients)\n_s,_q=base_competition.shape[:2]\n_nb=_s*_q*_q\n_nd=base_designs.size\n_nu=base_loadings.size\ncompetition=packed[:_nb].reshape(base_competition.shape)\ndesigns=packed[_nb:_nb+_nd].reshape(base_designs.shape)\nloadings=packed[_nb+_nd:_nb+_nd+_nu].reshape(base_loadings.shape)\nreserves=packed[_nb+_nd+_nu:].reshape(base_reserves.shape)",

            'call': 'plan_certificates(competition,designs,buffer,minimum_effort)',

            'gold_call': '_oracle_plan_certificates(competition,designs,buffer,minimum_effort)',

        },

        {

            "setup": "import numpy as np\nresource_matrix=np.array([[7.0, 39.0, 11.0, 9.0, 0.0, 6.0, 0.0], [39.0, 0.0, 25.0, 0.0, 31.0, 4.0, 0.0], [0.0, 0.0, 0.0, 21.0, 37.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0], [0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0], [0.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)\nk=10000.0\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\nbase_competition=np.array([[[0.0, 0.515, 0.858, 1.099, 0.88, 0.244], [0.218, 0.0, 0.792, 0.794, 0.891, 0.134], [1.02, 0.121, 0.0, 0.869, 0.803, 0.199], [0.396, 0.345, 0.452, 0.0, 0.859, 0.543], [0.876, 0.526, 0.397, 0.614, 0.0, 0.387], [0.398, 0.691, 0.511, 0.179, 0.331, 0.0]], [[0.0, 0.685, 1.068, 1.013, 0.914, 0.3], [0.205, 0.0, 0.669, 0.87, 0.757, 0.15], [1.302, 0.093, 0.0, 0.783, 0.697, 0.25], [0.292, 0.416, 0.471, 0.0, 0.952, 0.72], [0.828, 0.54, 0.443, 0.606, 0.0, 0.406], [0.372, 0.636, 0.487, 0.133, 0.421, 0.0]], [[0.0, 0.628, 1.112, 1.418, 1.106, 0.223], [0.226, 0.0, 1.133, 0.992, 1.014, 0.177], [1.352, 0.158, 0.0, 1.025, 0.93, 0.232], [0.337, 0.409, 0.347, 0.0, 1.136, 0.366], [1.115, 0.593, 0.438, 0.666, 0.0, 0.469], [0.465, 0.774, 0.403, 0.219, 0.304, 0.0]]], dtype=float)\nbase_designs=np.array([[0.614, 1.117, 0.817, 1.164, 1.254, 0.973], [1.459, 0.848, 1.019, 0.854, 1.278, 0.643], [0.595, 1.436, 0.855, 0.827, 0.9, 0.781], [0.922, 1.044, 0.85, 1.174, 1.472, 0.922], [1.222, 1.284, 0.928, 0.63, 1.255, 0.553], [1.044, 1.404, 0.966, 0.579, 0.665, 0.716]], dtype=float)\nbase_loadings=np.array([[[0.306, 0.292], [-0.313, 0.23], [-0.395, 0.32], [0.382, 0.177], [0.361, 0.427], [-0.416, -0.493]], [[-0.198, 0.303], [-0.129, 0.568], [-0.205, -0.096], [0.252, 0.326], [-0.297, 0.106], [-0.056, -0.095]], [[-0.008, 0.031], [-0.332, 0.017], [0.395, 0.785], [0.069, 0.69], [0.083, -0.445], [-0.045, -0.223]]], dtype=float)\nbase_reserves=np.array([0.032, 0.045, 0.036, 0.027, 0.041, 0.034], dtype=float)\nupper_efforts=np.array([8.0, 7.6, 8.4, 7.9, 8.2, 7.7], dtype=float)\ncosts=np.array([1.0, 1.0, 1.0, 0.97672, 1.0, 1.0], dtype=float)\npolicies=np.array([[3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05]], dtype=float)\ncoefficients=np.array([1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5], dtype=float)\nbuffer=1.2\nminimum_effort=0.2\nniche=_oracle_noncircular_niche_profile(resource_matrix,k)\ncalibration=_oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)\nsignal=_oracle_petra_residual_signal(reference,disturbed,candidates[int(calibration[0])],max_steps,indices,guild_map)\npacked=_oracle_assemble_coupled_system(base_competition,base_designs,base_loadings,base_reserves,niche,signal,coefficients)\n_s,_q=base_competition.shape[:2]\n_nb=_s*_q*_q\n_nd=base_designs.size\n_nu=base_loadings.size\ncompetition=packed[:_nb].reshape(base_competition.shape)\ndesigns=packed[_nb:_nb+_nd].reshape(base_designs.shape)\nloadings=packed[_nb+_nd:_nb+_nd+_nu].reshape(base_loadings.shape)\nreserves=packed[_nb+_nd+_nu:].reshape(base_reserves.shape)\nbuffer=1.1",

            'call': 'plan_certificates(competition,designs,buffer,minimum_effort)',

            'gold_call': '_oracle_plan_certificates(competition,designs,buffer,minimum_effort)',

        },

        {

            "setup": "import numpy as np\nresource_matrix=np.array([[7.0, 39.0, 11.0, 9.0, 0.0, 6.0, 0.0], [39.0, 0.0, 25.0, 0.0, 31.0, 4.0, 0.0], [0.0, 0.0, 0.0, 21.0, 37.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0], [0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0], [0.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)\nk=10000.0\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\nbase_competition=np.array([[[0.0, 0.515, 0.858, 1.099, 0.88, 0.244], [0.218, 0.0, 0.792, 0.794, 0.891, 0.134], [1.02, 0.121, 0.0, 0.869, 0.803, 0.199], [0.396, 0.345, 0.452, 0.0, 0.859, 0.543], [0.876, 0.526, 0.397, 0.614, 0.0, 0.387], [0.398, 0.691, 0.511, 0.179, 0.331, 0.0]], [[0.0, 0.685, 1.068, 1.013, 0.914, 0.3], [0.205, 0.0, 0.669, 0.87, 0.757, 0.15], [1.302, 0.093, 0.0, 0.783, 0.697, 0.25], [0.292, 0.416, 0.471, 0.0, 0.952, 0.72], [0.828, 0.54, 0.443, 0.606, 0.0, 0.406], [0.372, 0.636, 0.487, 0.133, 0.421, 0.0]], [[0.0, 0.628, 1.112, 1.418, 1.106, 0.223], [0.226, 0.0, 1.133, 0.992, 1.014, 0.177], [1.352, 0.158, 0.0, 1.025, 0.93, 0.232], [0.337, 0.409, 0.347, 0.0, 1.136, 0.366], [1.115, 0.593, 0.438, 0.666, 0.0, 0.469], [0.465, 0.774, 0.403, 0.219, 0.304, 0.0]]], dtype=float)\nbase_designs=np.array([[0.614, 1.117, 0.817, 1.164, 1.254, 0.973], [1.459, 0.848, 1.019, 0.854, 1.278, 0.643], [0.595, 1.436, 0.855, 0.827, 0.9, 0.781], [0.922, 1.044, 0.85, 1.174, 1.472, 0.922], [1.222, 1.284, 0.928, 0.63, 1.255, 0.553], [1.044, 1.404, 0.966, 0.579, 0.665, 0.716]], dtype=float)\nbase_loadings=np.array([[[0.306, 0.292], [-0.313, 0.23], [-0.395, 0.32], [0.382, 0.177], [0.361, 0.427], [-0.416, -0.493]], [[-0.198, 0.303], [-0.129, 0.568], [-0.205, -0.096], [0.252, 0.326], [-0.297, 0.106], [-0.056, -0.095]], [[-0.008, 0.031], [-0.332, 0.017], [0.395, 0.785], [0.069, 0.69], [0.083, -0.445], [-0.045, -0.223]]], dtype=float)\nbase_reserves=np.array([0.032, 0.045, 0.036, 0.027, 0.041, 0.034], dtype=float)\nupper_efforts=np.array([8.0, 7.6, 8.4, 7.9, 8.2, 7.7], dtype=float)\ncosts=np.array([1.0, 1.0, 1.0, 0.97672, 1.0, 1.0], dtype=float)\npolicies=np.array([[3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05]], dtype=float)\ncoefficients=np.array([1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5], dtype=float)\nbuffer=1.2\nminimum_effort=0.2\nniche=_oracle_noncircular_niche_profile(resource_matrix,k)\ncalibration=_oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)\nsignal=_oracle_petra_residual_signal(reference,disturbed,candidates[int(calibration[0])],max_steps,indices,guild_map)\npacked=_oracle_assemble_coupled_system(base_competition,base_designs,base_loadings,base_reserves,niche,signal,coefficients)\n_s,_q=base_competition.shape[:2]\n_nb=_s*_q*_q\n_nd=base_designs.size\n_nu=base_loadings.size\ncompetition=packed[:_nb].reshape(base_competition.shape)\ndesigns=packed[_nb:_nb+_nd].reshape(base_designs.shape)\nloadings=packed[_nb+_nd:_nb+_nd+_nu].reshape(base_loadings.shape)\nreserves=packed[_nb+_nd+_nu:].reshape(base_reserves.shape)\nbuffer=1.0\ndef _fx_raises_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",

            'call': '_fx_raises_value_error(lambda: plan_certificates(competition,designs,buffer,minimum_effort))',

            'gold_call': '_fx_raises_value_error(lambda: _oracle_plan_certificates(competition,designs,buffer,minimum_effort))',

            'tol': 0.0,

        },

    ]
