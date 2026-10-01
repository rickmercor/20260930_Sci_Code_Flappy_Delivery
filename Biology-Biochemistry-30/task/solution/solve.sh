#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def reactant_state(species_energies: "np.ndarray", species_h: "np.ndarray",
                   species_z: "np.ndarray", rt: float) -> "np.ndarray":
    import numpy as np
    from scipy.special import logsumexp
    e = np.asarray(species_energies, dtype=float)
    h, z = np.asarray(species_h, dtype=float), np.asarray(species_z, dtype=float)
    if (e.ndim != 4 or e.shape[1] != 2 or min(e.shape) == 0
            or h.shape != (e.shape[-1],) or z.shape != h.shape
            or not np.isfinite(e).all() or not np.isfinite(h).all()
            or not np.isfinite(z).all() or not np.isfinite(rt) or rt <= 0):
        raise ValueError('Finite aligned species data and positive RT required')
    log_partition = logsumexp(-e / rt, axis=-1)
    fractions = np.exp(-e / rt - log_partition[..., None])
    result = np.concatenate(((-rt * log_partition)[..., None],
                             (fractions @ h)[..., None],
                             (fractions @ z)[..., None], fractions), axis=-1)
    return result

def transport_state(reactants: "np.ndarray", transports: "np.ndarray",
                    species_h: "np.ndarray", species_z: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = np.asarray(reactants, dtype=float)
    t = np.asarray(transports)
    h, z = np.asarray(species_h, dtype=float), np.asarray(species_z, dtype=float)
    if (a.ndim != 4 or a.shape[1] != 2 or t.ndim != 2 or t.shape[1] != 6
            or a.shape[-1] != 3 + h.size or z.shape != h.shape
            or not np.isfinite(a).all() or not np.isfinite(t).all()
            or not np.equal(t, np.floor(t)).all()):
        raise ValueError('Aligned reactants and integer transport rows required')
    t = t.astype(int)
    if (np.any(t[:, 0] < 0) or len(set(t[:, 0])) != len(t)
            or np.any((t[:, 1] < -1) | (t[:, 1] >= a.shape[2]))
            or np.any((t[:, 2:4] < 0) | (t[:, 2:4] > 1))
            or np.any(t[:, 2] == t[:, 3]) or np.any(t[:, 4] < 0)
            or np.any((t[:, 5] < -1) | (t[:, 5] >= h.size))):
        raise ValueError('Invalid transport index, compartment or species')
    out = np.zeros((a.shape[0], len(t), 4))
    for p in range(a.shape[0]):
        for r, (_, compound, src, dst, k, override) in enumerate(t):
            if compound == -1:
                out[p, r] = [0., 0., k, -1.]
            else:
                index = override if override >= 0 else int(np.argmax(a[p, 1, compound, 3:]))
                out[p, r] = [h[index], z[index], k, index]
    return out

def balanced_reactions(stoich: "np.ndarray", reactants: "np.ndarray",
                       transports: "np.ndarray", transported: "np.ndarray") -> "np.ndarray":
    import numpy as np
    s, a = np.asarray(stoich, dtype=float), np.asarray(reactants, dtype=float)
    t, u = np.asarray(transports, dtype=int), np.asarray(transported, dtype=float)
    if (s.ndim != 2 or a.ndim != 4 or s.shape[0] != 2 * a.shape[2]
            or a.shape[1] != 2 or t.ndim != 2 or t.shape[1] != 6
            or u.shape != (a.shape[0], t.shape[0], 4)
            or np.any((t[:, 0] < 0) | (t[:, 0] >= s.shape[1]))
            or not np.isfinite(s).all() or not np.isfinite(u).all()):
        raise ValueError('Aligned reaction, reactant and transport dimensions required')
    p_count, m, r_count = a.shape[0], a.shape[2], s.shape[1]
    out = np.zeros((p_count, 2*m + 3, r_count))
    out[:, :2*m] = s
    for p in range(p_count):
        protons = np.stack([-a[p, c, :, 1] @ s[c*m:(c+1)*m] for c in range(2)])
        for row, (j, compound, src, dst, k, override) in enumerate(t):
            carried = u[p, row, 0] + u[p, row, 2]
            protons[src, j] -= carried
            protons[dst, j] += carried
        out[p, 2*m:2*m+2] = protons
        out[p, -1] = a[p, 1, :, 2] @ s[m:] + protons[1]
    out[np.abs(out) < 1e-13] = 0.
    return out

def energy_operator(stoich: "np.ndarray", reactants: "np.ndarray",
                    balanced: "np.ndarray", transports: "np.ndarray",
                    conditions: "np.ndarray", loadings: "np.ndarray",
                    rt: float, faraday: float) -> "np.ndarray":
    import numpy as np
    s, a, bal = map(lambda x: np.asarray(x, dtype=float), (stoich, reactants, balanced))
    t = np.asarray(transports, dtype=int)
    cond, load = np.asarray(conditions, dtype=float), np.asarray(loadings, dtype=float)
    p_count, m = a.shape[0], s.shape[0]
    if (cond.shape != (p_count, 2, 2) or load.ndim != 2 or load.shape[0] != m
            or bal.shape != (p_count, m+3, s.shape[1])
            or not np.isfinite(cond).all() or not np.isfinite(load).all()
            or not np.isfinite(rt) or rt <= 0 or not np.isfinite(faraday) or faraday <= 0):
        raise ValueError('Aligned finite energy data and positive constants required')
    offset = a[..., 0].reshape(p_count, m) @ s
    dph = cond[:, 1, 0] - cond[:, 0, 0]
    dpsi = cond[:, 1, 1] - cond[:, 0, 1]
    offset += faraday * dpsi[:, None] * bal[:, -1, :]
    for j, compound, src, dst, k, override in t:
        offset[:, j] -= k * (dst-src) * rt * np.log(10.) * dph
    out = np.empty((p_count, s.shape[1], 1 + m + load.shape[1]))
    out[:, :, 0] = offset
    out[:, :, 1:1+m] = rt * s.T
    out[:, :, 1+m:] = s.T @ load
    return out

def phenotype_panel(balanced: "np.ndarray", demand: "np.ndarray",
                    weights: "np.ndarray", factors: "np.ndarray",
                    oneway: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import null_space, qr
    from itertools import combinations
    bal, b, g, fac = map(lambda x: np.asarray(x, dtype=float),
                         (balanced, demand, weights, factors))
    one = np.asarray(oneway, dtype=int)
    if (bal.ndim != 3 or g.ndim != 2 or fac.shape != (bal.shape[0], bal.shape[2])
            or b.shape != (bal.shape[1]-3,) or g.shape[1] != bal.shape[2]
            or g.shape[0] == 0 or np.any(g <= 0) or np.any(fac <= 0)
            or any(not np.isfinite(x).all() for x in (bal, b, g, fac))
            or one.ndim != 1 or np.any((one < 0) | (one >= g.shape[1]))
            or len(np.unique(one)) != len(one)):
        raise ValueError('Aligned positive phenotype inputs required')
    out = np.empty((g.shape[0], fac.shape[0], g.shape[1]))
    for p in range(fac.shape[0]):
        full = np.vstack([bal[p, :-3], bal[p, -1]])
        rhs = np.r_[b, 0.]
        initial = np.linalg.lstsq(full, rhs, rcond=1e-12)[0]
        if np.max(np.abs(full @ initial - rhs)) > 1e-8:
            raise ValueError('Inconsistent source balances')
        _, _, piv = qr(full.T, pivoting=True, mode='economic')
        rank = np.linalg.matrix_rank(full, tol=1e-10)
        basis, target = full[piv[:rank]], rhs[piv[:rank]]
        for c in range(g.shape[0]):
            scale = 2. * g[c] * fac[p] / np.e
            best, best_cost = None, np.inf
            for size in range(one.size+1):
                for active in combinations(one, size):
                    a = np.vstack([basis, np.eye(g.shape[1])[list(active)]])
                    d = np.r_[target, np.zeros(size)]
                    v = np.linalg.lstsq(a, d, rcond=1e-12)[0]
                    if np.max(np.abs(a @ v-d), initial=0.) > 1e-8:
                        continue
                    null = null_space(a, rcond=1e-12)
                    def _objective(n):
                        return float(np.sum(n*np.arcsinh(n/scale)-np.hypot(n, scale)))
                    converged = False
                    for iteration in range(150):
                        gradient = null.T @ np.arcsinh(v/scale)
                        if np.max(np.abs(gradient), initial=0.) < 2e-11:
                            converged = True
                            break
                        hessian = (null.T / np.hypot(v, scale)) @ null
                        direction = np.linalg.solve(hessian, gradient)
                        step, alpha, old = -null @ direction, 1., _objective(v)
                        while (_objective(v+alpha*step) > old-1e-4*alpha*(gradient@direction)+2e-13
                               and alpha > 1e-10):
                            alpha *= .5
                        v += alpha * step
                    if not converged:
                        raise RuntimeError('Entropy face solve did not converge')
                    if np.min(v[one], initial=0.) < -2e-8:
                        continue
                    cost = _objective(v)
                    if cost < best_cost:
                        best_cost, best = cost, v.copy()
            if best is None:
                raise ValueError('Source balances and one-way evidence are infeasible')
            best[np.abs(best) < 2e-10] = 0.
            out[c, p] = best
    return out

def directional_state(net: "np.ndarray", weights: "np.ndarray",
                      factors: "np.ndarray", rt: float) -> "np.ndarray":
    import numpy as np
    n, g, f = map(lambda x: np.asarray(x, dtype=float), (net, weights, factors))
    if (n.ndim != 3 or g.shape != (n.shape[0], n.shape[2])
            or f.shape != n.shape[1:] or np.any(g <= 0) or np.any(f <= 0)
            or any(not np.isfinite(x).all() for x in (n, g, f))
            or not np.isfinite(rt) or rt <= 0):
        raise ValueError('Positive aligned weights and finite net fluxes required')
    geometric = g[:, None, :] * f[None, :, :] / np.e
    angle = np.arcsinh(n/(2.*geometric))
    forward = np.exp(np.log(geometric)+angle)
    reverse = np.exp(np.log(geometric)-angle)
    return np.stack([forward, reverse, -2.*rt*angle], axis=-1)

def calibration_system(operator: "np.ndarray", net: "np.ndarray",
                       energy: "np.ndarray", assay: "np.ndarray",
                       intervals: "np.ndarray", x_bounds: "np.ndarray",
                       activity: float, drive: float) -> "np.ndarray":
    import numpy as np
    op, n, est, h, bands, xb = map(lambda x: np.asarray(x, dtype=float),
                                  (operator, net, energy, assay, intervals, x_bounds))
    if (op.ndim != 3 or n.shape != op.shape[:2] or est.shape != n.shape
            or h.ndim != 2 or op.shape[2] < 1+h.shape[1]
            or bands.shape != (n.shape[0], h.shape[0], 2)
            or xb.shape != (n.shape[0], h.shape[1], 2)
            or np.any(bands[..., 0] > bands[..., 1]) or np.any(xb[..., 0] > xb[..., 1])
            or any(not np.isfinite(x).all() for x in (op, n, est, h, bands, xb))
            or not np.isfinite(activity) or activity < 0
            or not np.isfinite(drive) or drive <= 0):
        raise ValueError('Aligned finite calibration data and ordered bounds required')
    p_count, r_count = n.shape
    m, k = h.shape[1], op.shape[2]-1-h.shape[1]
    width = p_count*m+k+1
    rows = []
    def _append(coeff, bound):
        rows.append(np.r_[coeff, bound])
    for p in range(p_count):
        for j in range(r_count):
            if abs(n[p, j]) <= activity:
                continue
            row = np.zeros(width)
            row[p*m:(p+1)*m] = op[p, j, 1:1+m]
            row[p_count*m:p_count*m+k] = op[p, j, 1+m:]
            sign, offset = np.sign(n[p, j]), op[p, j, 0]
            _append(sign*row, -drive-sign*offset)
            upper, lower = row.copy(), -row
            upper[-1] = lower[-1] = -1.
            _append(upper, est[p, j]-offset)
            _append(lower, offset-est[p, j])
        for row_id in range(h.shape[0]):
            row = np.zeros(width)
            row[p*m:(p+1)*m] = h[row_id]
            _append(row, bands[p, row_id, 1])
            _append(-row, -bands[p, row_id, 0])
        for i in range(m):
            row = np.zeros(width)
            row[p*m+i] = 1.
            _append(row, xb[p, i, 1])
            _append(-row, -xb[p, i, 0])
    row = np.zeros(width)
    row[-1] = -1.
    _append(row, 0.)
    return np.asarray(rows)

def joint_energy_radius(system: "np.ndarray", n_concentrations: int,
                        n_factors: int, radius: float) -> float:
    import numpy as np
    from scipy.optimize import linprog, minimize, LinearConstraint
    data = np.asarray(system, dtype=float)
    nx, k = int(n_concentrations), int(n_factors)
    if (nx != n_concentrations or k != n_factors or min(nx, k) < 0
            or data.ndim != 2 or data.shape[1] != nx+k+2 or not np.isfinite(data).all()
            or not np.isfinite(radius) or radius < 0):
        raise ValueError('Finite half-spaces, dimensions and nonnegative radius required')
    mat, rhs = data[:, :-1].copy(), data[:, -1].copy()
    width = nx+k+1
    cost = np.zeros(width)
    cost[-1] = 1.
    bounds = [(None, None)]*nx + [(-radius, radius)]*k + [(0., None)]
    lp_options = {'primal_feasibility_tolerance': 1e-10,
                  'dual_feasibility_tolerance': 1e-10}
    upper = np.inf
    for iteration in range(180):
        relaxed = linprog(cost, A_ub=mat, b_ub=rhs, bounds=bounds,
                          method='highs', options=lp_options)
        if relaxed.status == 2:
            return float('inf')
        if not relaxed.success:
            raise RuntimeError('Calibration linear relaxation failed')
        v = relaxed.x
        norm = np.linalg.norm(v[nx:nx+k])
        if norm <= radius + 1e-10:
            return float(max(0., relaxed.fun))
        projected = v[nx:nx+k] * (radius / norm)
        fixed_bounds = [(None, None)]*nx + [(u, u) for u in projected] + [(0., None)]
        witness = linprog(cost, A_ub=mat, b_ub=rhs, bounds=fixed_bounds,
                          method='highs', options=lp_options)
        if witness.success:
            upper = min(upper, float(witness.fun))
        if iteration == 0 or iteration % 12 == 0:
            def _cone(x):
                return radius**2 - x[nx:nx+k] @ x[nx:nx+k]
            def _cone_jac(x):
                out = np.zeros(width)
                out[nx:nx+k] = -2.*x[nx:nx+k]
                return out
            solved = minimize(lambda x: x[-1], v, jac=lambda x: cost,
                bounds=bounds, constraints=[LinearConstraint(mat, -np.inf, rhs),
                    {'type':'ineq', 'fun':_cone, 'jac':_cone_jac}],
                method='SLSQP', options={'ftol':2e-12, 'maxiter':600})
            if (np.max(mat@solved.x-rhs, initial=0.) <= 2e-8
                    and np.linalg.norm(solved.x[nx:nx+k]) <= radius+1e-9):
                upper = min(upper, float(solved.fun))
        if upper - relaxed.fun <= 3e-8:
            return float(max(0., upper))
        cut = np.zeros(width)
        cut[nx:nx+k] = v[nx:nx+k]/norm
        mat = np.vstack([mat, cut])
        rhs = np.r_[rhs, radius]
    raise RuntimeError('Calibration optimality gap unresolved')

def robust_delivery(net: "np.ndarray", calibration: "np.ndarray", route: int,
                    demand: float, limit: float, tie: float) -> "np.ndarray":
    import numpy as np
    n, delta = np.asarray(net, dtype=float), np.asarray(calibration, dtype=float)
    if (n.ndim != 3 or min(n.shape) == 0 or delta.shape != (n.shape[0],)
            or not np.isfinite(n).all() or np.any(np.isnan(delta)) or np.any(delta < 0)
            or int(route) != route or not 0 <= route < n.shape[2]
            or not np.isfinite(demand) or demand <= 0 or not np.isfinite(limit)
            or limit < 0 or not np.isfinite(tie) or tie < 0):
        raise ValueError('Aligned performance data and valid selection settings required')
    scores = np.min(n[:, :, int(route)]/demand, axis=1)
    eligible = np.flatnonzero(delta <= limit)
    if eligible.size == 0:
        raise ValueError('No eligible design')
    best = np.max(scores[eligible])
    c = int(eligible[np.flatnonzero(best-scores[eligible] <= tie)[0]])
    condition = int(np.flatnonzero(n[c, :, int(route)]/demand-scores[c] <= tie)[0])
    return np.array([scores[c], c, condition], dtype=float)

def resolve_transport_panel(species_energies: "np.ndarray", weights: "np.ndarray",
                            assay_intervals: "np.ndarray", radius: float,
                            limit: float, model: dict) -> float:
    import numpy as np
    fixed = model
    required = {'s','transports','conditions','factors','loadings','h','z','assay',
                'demand','oneway','x_bounds','rt','faraday','activity','drive'}
    if not isinstance(fixed, dict) or set(fixed) != required:
        raise ValueError('Model keys do not match the stated contract')
    e, w, bands = map(lambda x: np.asarray(x, dtype=float), (species_energies, weights, assay_intervals))
    if e.shape != (4, 2, 4, 2) or w.ndim != 2 or w.shape[1] != 14 or bands.shape != (4, 3, 2):
        raise ValueError('Benchmark-shaped species, weight and assay inputs required')
    a = reactant_state(e, fixed['h'], fixed['z'], fixed['rt'])
    t = transport_state(a, fixed['transports'], fixed['h'], fixed['z'])
    b = balanced_reactions(fixed['s'], a, fixed['transports'], t)
    op = energy_operator(fixed['s'], a, b, fixed['transports'],
                                fixed['conditions'], fixed['loadings'], fixed['rt'], fixed['faraday'])
    n = phenotype_panel(b, fixed['demand'], w, fixed['factors'], fixed['oneway'])
    directional = directional_state(n, w, fixed['factors'], fixed['rt'])
    radii = []
    for c in range(w.shape[0]):
        system = calibration_system(op, n[c], directional[c, :, :, 2],
                    fixed['assay'], bands, fixed['x_bounds'], fixed['activity'], fixed['drive'])
        radii.append(joint_energy_radius(system, 32, 2, radius))
    selected = robust_delivery(n, radii, 10, 8., limit, 1e-10)
    return float(selected[0])


def _fixed_transport_inputs():
    import numpy as np
    edges = [(0,4), (0,4), (1,5), (6,2), (7,3), (4,5), (4,6),
             (5,6), (5,7), (6,7), (2,3), (1,2), (0,1)]
    s = np.zeros((8,14))
    for j, (src, dst) in enumerate(edges):
        s[src,j], s[dst,j] = -1., 1.
    tr = np.array([[0,0,0,1,0,-1], [1,0,0,1,1,-1], [2,1,0,1,0,-1],
                   [3,2,1,0,0,-1], [4,3,1,0,0,1], [13,-1,0,1,1,-1]])
    cond = np.array([[[6.,0.],[6.8,.038]], [[6.4,0.],[7.3,.044]],
                     [[6.2,0.],[7.,.040]], [[6.5,0.],[7.6,.057]]])
    fac = np.ones((4,14))
    fac[1,[0,8]], fac[2,[3,13]], fac[3,[5,10]] = [.31,1.55], [.37,1.8], [.42,1.65]
    load = np.tile([[1.6,-.8], [-1.2,.8], [.8,1.8], [-1.,-1.2]], (2,1))
    h = np.zeros((3,8))
    h[0,[4,5]], h[1,[6,7]], h[2,[1,5,2,6]] = [1.,-1.], [1.,-1.], [1.,-1.,-1.,1.]
    return dict(s=s, transports=tr, conditions=cond, factors=fac, loadings=load,
                h=np.array([0.,1.]), z=np.array([-1.,0.]), assay=h,
                demand=np.array([-10.,0.,0.,8.,0.,0.,0.,2.]),
                oneway=np.array([7,10,12,13]), x_bounds=np.tile([-10.5,-4.5], (4,8,1)),
                rt=.00831446261815324*298.15, faraday=96.48533212,
                activity=1e-7, drive=.001)


def _panel_inputs():
    import numpy as np
    energies = np.array([[[[0.0, -1.712403], [-2.0, -8.27881], [-1.0, -4.995607],
       [-4.0, -13.132815]],
      [[0.0, 2.854005], [-2.0, -3.712403], [-1.0, -0.429199],
       [-4.0, -8.566408]]],
     [[[0.0, 0.570801], [-2.0, -5.995607], [-1.0, -2.712403],
       [-4.0, -10.849611]],
      [[0.0, 5.70801], [-2.0, -0.858398], [-1.0, 2.424806],
       [-4.0, -5.712403]]],
     [[[0.0, -0.570801], [-2.0, -7.137209], [-1.0, -3.854005],
       [-4.0, -11.991213]],
      [[0.0, 3.995607], [-2.0, -2.570801], [-1.0, 0.712403],
       [-4.0, -7.424806]]],
     [[[0.0, 1.141602], [-2.0, -5.424806], [-1.0, -2.141602],
       [-4.0, -10.27881]],
      [[0.0, 7.420412], [-2.0, 0.854005], [-1.0, 4.137209], [-4.0, -4.0]]]], dtype=float)
    weights = np.array([[5.32, 9.35, 14.2, 16.73, 12.15, 11.54, 12.67, 7.33, 5.78, 6.68, 13.78,
      9.64, 17.74, 0.04],
     [12.4, 11.31, 10.31, 14.13, 11.86, 10.59, 12.43, 14.05, 17.7, 15.02,
      16.57, 11.48, 4.84, 0.08],
     [7.7, 13.65, 15.06, 6.44, 6.92, 9.54, 6.51, 10.02, 9.02, 9.07, 17.85,
      2.42, 7.25, 0.025],
     [2.87, 12.67, 15.81, 16.62, 4.19, 8.77, 9.63, 9.24, 17.8, 13.36, 14.85,
      9.08, 7.47, 0.12],
     [9.83, 9.45, 9.87, 9.71, 10.31, 2.23, 17.36, 9.88, 15.8, 9.29, 15.28,
      13.85, 11.52, 0.055],
     [17.22, 8.89, 5.97, 3.19, 7.9, 4.59, 4.2, 4.63, 14.57, 9.47, 17.93,
      10.34, 17.0, 0.035]], dtype=float)
    bands = np.array([[[0.707, 1.067], [-0.433, -0.073], [1.421, 1.821]],
     [[0.437, 0.797], [0.146, 0.506], [0.625, 1.025]],
     [[-0.496, -0.136], [-0.327, 0.033], [1.604, 2.004]],
     [[-0.66, -0.3], [-0.817, -0.457], [1.197, 1.597]]], dtype=float)
    return energies, weights, bands
SCICODE_GOLD_EOF
