"""
The final orchestrator. For each supplied configuration, chain the earlier sub-problem functions to produce the twenty declared columns. Columns 0 to 13 are the reference and relaxed potential energies, classical and quantum coarse-grained free energies, the fixed-geometry free energy from zero relaxation, the classical gap, smallest and largest selected frequency, total classical and quantum backmapping variances, the first and second update displacement norms, and the two strict-inequality flags. With num_iters equal to zero it must reproduce the fixed-geometry rigidification identically. Columns 14 to 17 are the backmapping covariance audit summed over every monomer of the configuration, each monomer contributing its own B atom at the relaxed point: the largest eigenvalue of the quantum covariance, its anisotropy, the signed alignment of its principal axis with that monomer's plane normal, and the quantum to classical trace ratio, with the principal-axis sign fixed by the declared frame and flip convention. Column 19 is the source’s Eq 6 minimum-energy-manifold residual at the relaxed point, summed over every monomer, with the stiff-subspace pseudoinverse and the mass scaling applied exactly as the source defines that manifold condition. Column 18 is the signed frame twist between the reference and relaxed points, also summed over every monomer, under the declared row-stacking, product-order, axis-extraction, and sign conventions.

The audit is the paper’s comparison in miniature: subspace harmonic relaxation against the fixed-shape baseline, on clusters chosen so every convention is load bearing.

Returns
-------
float64 array (n, 20): one audit row per configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def shr_audit(configs, num_iters, T):
    """configs: sequence of float arrays (I_k, 3, 3); num_iters: exact
    relaxation iteration count; T: temperature.

    Returns float64 array (n, 20): one row per configuration with the
    twenty declared audit columns, assembled by calling the earlier
    sub-problem functions. Columns 14 to 17 are the backmapping covariance
    audit summed over every monomer of the configuration, each monomer
    contributing its own B atom at the relaxed point: largest
    eigenvalue of the quantum covariance, its anisotropy, the signed
    alignment of its principal axis with that monomer plane normal under the
    declared sign convention, and the quantum to classical trace ratio. Column 18 is the signed frame twist
    between the reference and relaxed points, also summed over every monomer,
    under the declared row-stacking, product-order, axis-extraction, and
    sign conventions."""
    return np.zeros((len(configs), 20))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 gold: twenty-column SHR audit with per-monomer sums."""

import numpy as np

_LI = 3
_MASSES = (1.0, 16.0, 1.0)


def _audit_frequencies(coords):
    blocks = _oracle_block_mass_scaled_hessian(coords)
    I = blocks.shape[0]
    oms = []
    for i in range(I):
        lam = np.linalg.eigvalsh(blocks[i])
        top = np.sort(lam)[-_LI:]
        if np.any(top <= 0.0) or not np.all(np.isfinite(top)):
            raise ValueError("nonpositive selected eigenvalue")
        oms.extend(np.sqrt(top))
    return np.sort(np.asarray(oms, dtype=np.float64))


def _frame_of(mono):
    """Local frame of one monomer per the frame contract, including the
    near-collinear fallback."""
    A1, B, A2 = mono
    u, w = A1 - B, A2 - B
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    uh, wh = u / nu, w / nw
    bis = uh + wh
    x = bis / np.linalg.norm(bis)
    zc = np.cross(u, w)
    nz = np.linalg.norm(zc)
    if nz / (nu * nw) < 1e-8:
        k = int(np.argmin(np.abs(x)))
        e = np.zeros(3)
        e[k] = 1.0
        z = e - np.dot(e, x) * x
        z = z / np.linalg.norm(z)
    else:
        z = zc / nz
    y = np.cross(z, x)
    return x, y, z


def _cov_audit(coords, T):
    """Backmapping covariance audit summed over every monomer, each using
    its own B atom at the given (relaxed) configuration: [lam_max sum,
    anisotropy sum, signed alignment sum, quantum/classical trace ratio
    sum]."""
    beta = 1.0 / T
    blocks = _oracle_block_mass_scaled_hessian(coords)
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    out = np.zeros(4)
    for i in range(blocks.shape[0]):
        lam, U = np.linalg.eigh(blocks[i])
        Cq = np.zeros((3, 3))
        Cc = np.zeros((3, 3))
        for l in np.argsort(lam)[-_LI:]:
            om = np.sqrt(lam[l])
            wvec = (sm * U[:, l])[3:6]
            Cq += ((1.0 / (2.0 * om)) / np.tanh(beta * om / 2.0)) * np.outer(wvec, wvec)
            Cc += (1.0 / (beta * om * om)) * np.outer(wvec, wvec)
        ev, EV = np.linalg.eigh(Cq)
        tr = float(np.trace(Cq))
        v = EV[:, -1]
        x, y, z = _frame_of(coords[i])
        for comp in (float(np.dot(v, x)), float(np.dot(v, y)), float(np.dot(v, z))):
            if abs(comp) > 1e-8:
                if comp < 0.0:
                    v = -v
                break
        out += [float(ev[-1]), (float(ev[-1]) - float(ev[0])) / tr,
                float(np.dot(v, z)), tr / float(np.trace(Cc))]
    return list(out)


def _frame_twist(r0c, rP):
    """Signed frame twist summed over every monomer: for each monomer the
    frames at the reference and relaxed points stack x, y, z as rows, the
    relative rotation is F_P times F_0 transpose, the axis comes from the
    antisymmetric part, and the sign from its dot product with that
    monomer's reference-point z axis."""
    total = 0.0
    for i in range(r0c.shape[0]):
        x0, y0, z0 = _frame_of(r0c[i])
        xP, yP, zP = _frame_of(rP[i])
        F0 = np.vstack([x0, y0, z0])
        FP = np.vstack([xP, yP, zP])
        R = FP @ F0.T
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
        th = np.arctan2(np.linalg.norm(w), (np.trace(R) - 1.0) / 2.0)
        total += float(th * (1.0 if np.dot(w, z0) >= 0.0 else -1.0))
    return total


def _mem_residual(coords, blocks=None, g=None):
    """Eq 6 minimum-energy-manifold residual, summed over every monomer: the
    stiff-subspace pseudoinverse is rebuilt at this configuration and applied
    once to the mass-scaled gradient."""
    if blocks is None:
        blocks = _oracle_block_mass_scaled_hessian(coords)
    if g is None:
        g = _oracle_cluster_gradient(coords)
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    total = 0.0
    for i in range(blocks.shape[0]):
        lam, U = np.linalg.eigh(blocks[i])
        order = np.argsort(lam)[::-1][:_LI]
        if np.any(lam[order] <= 0.0):
            raise ValueError("nonpositive selected eigenvalue")
        Kt = np.zeros((9, 9), dtype=np.float64)
        for l in order:
            Kt += (1.0 / lam[l]) * np.outer(U[:, l], U[:, l])
        total += float(np.linalg.norm(Kt @ (sm * g[i].reshape(9))))
    return total


def _oracle_shr_audit(configs, num_iters, T):
    if isinstance(num_iters, (bool, np.bool_)) or not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
        raise ValueError("invalid iteration count")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("invalid temperature")
    rows = []
    for cfg in configs:
        r_ref = np.asarray(cfg, dtype=np.float64)
        r0c = _oracle_reference_configuration(r_ref)
        V0 = _oracle_cluster_energy(r0c)
        traj = [r0c]
        for p in range(1, int(num_iters) + 1):
            traj.append(_oracle_shr_relax(r0c, p))
        rP = traj[-1]
        VP = _oracle_cluster_energy(rP)
        om = _audit_frequencies(rP)
        th = _oracle_harmonic_thermo(VP, om, T)
        om_fixed = _audit_frequencies(r0c)
        th_fixed = _oracle_harmonic_thermo(V0, om_fixed, T)
        d1 = float(np.linalg.norm(traj[1] - traj[0])) if len(traj) > 1 else 0.0
        d2 = float(np.linalg.norm(traj[2] - traj[1])) if len(traj) > 2 else 0.0
        cov = _cov_audit(rP, T)
        tw = _frame_twist(r0c, rP)
        blocks_P = _oracle_block_mass_scaled_hessian(rP)
        grad_P = _oracle_cluster_gradient(rP)
        mem = _mem_residual(rP, blocks_P, grad_P)
        rows.append([
            V0, VP, th[0], th[1], th_fixed[0], th[0] - th_fixed[0],
            float(om[0]), float(om[-1]), th[2], th[3], d1, d2,
            1.0 if VP < V0 else 0.0,
            1.0 if th[0] < th_fixed[0] else 0.0,
            cov[0], cov[1], cov[2], cov[3], tw, mem,
        ])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'configs = [np.array(c) for c in [[[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]], [[-0.937176, -3.241764, 0.301703], [0.003554, -3.168499, -0.057999], [0.319276, -4.039705, -0.194912]]], [[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]], [[-0.349612, -0.40276, -1.453091], [0.236177, 0.140533, -2.300676], [-0.612218, 0.189077, -2.936322]]], [[[0.78118, 0.6165, -0.461584], [0.007969, 0.095973, 0.140201], [-0.161187, 0.192377, 0.902405]], [[-3.347254, -0.707507, -0.702641], [-2.704236, -0.004769, 0.031733], [-2.151768, -0.663321, 0.680483]], [[-0.905117, -2.782927, 0.189408], [-0.0752, -2.911689, 0.048541], [0.539791, -2.353142, 0.77435]], [[-0.561311, 0.065038, -3.446814], [-0.033451, -0.029028, -2.777209], [-0.374238, 0.424573, -2.065868]]], [[[0.202603, 0.389625, 0.851976], [-0.026597, -0.025761, -0.019083], [0.092733, 0.775856, -0.650099]], [[-3.067929, 0.397487, 0.903875], [-3.146453, 0.004114, 0.04078], [-2.330466, -0.432468, -0.296686]], [[-0.082103, -4.183185, -0.326684], [-0.005234, -3.206419, -0.004281], [0.810262, -2.781893, -0.358398]], [[0.823373, 0.527608, -3.439266], [-0.018188, 0.008185, -3.21766], [-0.63673, 0.61384, -2.658203]], [[0.116225, 0.872441, 2.80348], [-0.057454, -0.028002, 3.193326], [0.362577, -0.638241, 2.644926]]]]]\nnum_iters = 2\nT = 0.35', "call": "shr_audit(configs, num_iters, T)", "gold_call": "_oracle_shr_audit(configs, num_iters, T)", "tol": 1e-07},
        {"setup": 'configs = [np.array(c) for c in [[[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]], [[-0.937176, -3.241764, 0.301703], [0.003554, -3.168499, -0.057999], [0.319276, -4.039705, -0.194912]]], [[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]], [[-0.349612, -0.40276, -1.453091], [0.236177, 0.140533, -2.300676], [-0.612218, 0.189077, -2.936322]]], [[[0.78118, 0.6165, -0.461584], [0.007969, 0.095973, 0.140201], [-0.161187, 0.192377, 0.902405]], [[-3.347254, -0.707507, -0.702641], [-2.704236, -0.004769, 0.031733], [-2.151768, -0.663321, 0.680483]], [[-0.905117, -2.782927, 0.189408], [-0.0752, -2.911689, 0.048541], [0.539791, -2.353142, 0.77435]], [[-0.561311, 0.065038, -3.446814], [-0.033451, -0.029028, -2.777209], [-0.374238, 0.424573, -2.065868]]], [[[0.202603, 0.389625, 0.851976], [-0.026597, -0.025761, -0.019083], [0.092733, 0.775856, -0.650099]], [[-3.067929, 0.397487, 0.903875], [-3.146453, 0.004114, 0.04078], [-2.330466, -0.432468, -0.296686]], [[-0.082103, -4.183185, -0.326684], [-0.005234, -3.206419, -0.004281], [0.810262, -2.781893, -0.358398]], [[0.823373, 0.527608, -3.439266], [-0.018188, 0.008185, -3.21766], [-0.63673, 0.61384, -2.658203]], [[0.116225, 0.872441, 2.80348], [-0.057454, -0.028002, 3.193326], [0.362577, -0.638241, 2.644926]]]]]\nnum_iters = 0\nT = 0.35', "call": "shr_audit(configs, num_iters, T)", "gold_call": "_oracle_shr_audit(configs, num_iters, T)", "tol": 1e-07},
        {"setup": 'configs = [np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]])]\nnum_iters = 1\nT = 0.5', "call": "shr_audit(configs, num_iters, T)", "gold_call": "_oracle_shr_audit(configs, num_iters, T)", "tol": 1e-07},
    ]
