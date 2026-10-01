#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _packet_overlap(q, p, q0, p0, basis_gamma, initial_gamma, hbar=1.0):
    import numpy as np
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    G = np.diag(np.asarray(basis_gamma, float))
    G0 = np.diag(np.asarray(initial_gamma, float))
    A = G + G0
    invA = np.linalg.inv(A)
    b = q @ G + q0 @ G0 + 1j * (p0 - p)
    const = (-0.5 * np.einsum('ni,ij,nj->n', q, G, q)
             - 0.5 * q0 @ G0 @ q0
             + 1j * (p @ q.T).diagonal()
             - 1j * p0 @ q0)
    quad = 0.5 * np.einsum('ni,ij,nj->n', b, invA, b)
    pref = 2.0 * (np.linalg.det(G) * np.linalg.det(G0)) ** 0.25 / np.sqrt(np.linalg.det(A))
    return pref * np.exp((quad + const) / hbar)


def phase_space_sobol_packets(log2_count, seed, q0, p0, initial_gamma_diag, basis_gamma_diag, hbar=1.0):
    import numpy as np
    from scipy.special import ndtri
    from scipy.stats import qmc
    initial_gamma = initial_gamma_diag
    basis_gamma = basis_gamma_diag
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    initial_gamma = np.asarray(initial_gamma, float)
    basis_gamma = np.asarray(basis_gamma, float)
    if isinstance(log2_count, bool) or int(log2_count) != log2_count or not 1 <= int(log2_count) <= 12:
        raise ValueError('log2_count must be an integer from 1 through 12')
    if q0.shape != (2,) or p0.shape != (2,) or initial_gamma.shape != (2,) or basis_gamma.shape != (2,):
        raise ValueError('q0, p0, and both width vectors must have length two')
    if not np.all(np.isfinite(np.r_[q0,p0,initial_gamma,basis_gamma,hbar,seed])) or np.any(initial_gamma <= 0) or np.any(basis_gamma <= 0) or hbar <= 0:
        raise ValueError('packet inputs must be finite with positive widths and hbar')
    G = np.diag(np.asarray(basis_gamma, float))
    G0 = np.diag(np.asarray(initial_gamma, float))
    sigma = np.block([[np.linalg.inv(G) + np.linalg.inv(G0), np.zeros((2,2))],
                      [np.zeros((2,2)), G + G0]])
    cov = hbar * sigma
    u = qmc.Sobol(4, scramble=True, seed=int(seed)).random_base2(int(log2_count))
    u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    z = np.r_[q0, p0] + ndtri(u) @ np.linalg.cholesky(cov).T
    delta = z - np.r_[q0, p0]
    invcov = np.linalg.inv(cov)
    pdf = np.exp(-0.5*np.einsum('ni,ij,nj->n', delta, invcov, delta)) / np.sqrt((2*np.pi)**4*np.linalg.det(cov))
    ov = _packet_overlap(z[:,:2], z[:,2:], q0, p0, basis_gamma, initial_gamma, hbar)
    n = len(z)
    coeff = ov / (n * (2*np.pi*hbar)**2 * pdf)
    return np.column_stack((z, coeff.real, coeff.imag))

def interference_potential_derivatives(points, time, potential_parameters):
    import numpy as np
    t = time
    params = potential_parameters
    points = np.asarray(points, float)
    params = np.asarray(params, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or params.shape != (8,):
        raise ValueError('points must be nonempty (N,2) and parameters length eight')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(params)) or not np.isfinite(t):
        raise ValueError('points, time, and parameters must be finite')
    amplitude, xc, wx, wy, kx, ky, omega, phase = params
    if wx <= 0 or wy <= 0:
        raise ValueError('potential widths must be positive')
    x = points[:, 0]
    y = points[:, 1]
    dx = x - xc
    env = np.exp(-0.5 * ((dx / wx) ** 2 + (y / wy) ** 2))
    arg = kx * x + ky * y - omega * t + phase
    co = np.cos(arg)
    si = np.sin(arg)
    avec = np.column_stack((-dx / wx**2, -y / wy**2))
    kval = np.array([kx, ky])
    value = amplitude * env * co
    grad = amplitude * env[:, None] * (avec * co[:, None] - kval * si[:, None])
    hlog = np.diag([-1.0 / wx**2, -1.0 / wy**2])
    aa = np.einsum('ni,nj->nij', avec, avec)
    ak = np.einsum('ni,j->nij', avec, kval) + np.einsum('i,nj->nij', kval, avec)
    hess = amplitude * env[:, None, None] * (
        (aa + hlog[None, :, :] - np.outer(kval, kval)[None, :, :]) * co[:, None, None]
        - ak * si[:, None, None]
    )
    return np.column_stack((value, grad, hess[:, 0, 0], hess[:, 0, 1], hess[:, 1, 1])).astype(float)

def _potential_tuple(points, time, potential_parameters):
    import numpy as np
    packed = interference_potential_derivatives(points, time, potential_parameters)
    value = packed[:, 0]
    gradient = packed[:, 1:3]
    hessian = np.empty((len(packed), 2, 2), dtype=float)
    hessian[:, 0, 0] = packed[:, 3]
    hessian[:, 0, 1] = packed[:, 4]
    hessian[:, 1, 0] = packed[:, 4]
    hessian[:, 1, 1] = packed[:, 5]
    return value, gradient, hessian

def stormer_verlet_centers_actions(packet_rows, final_time, step_count, potential_parameters, mass=1.0, hbar=1.0):
    import numpy as np
    params = potential_parameters
    packet_rows = np.asarray(packet_rows, float)
    if packet_rows.ndim != 2 or packet_rows.shape[1] != 6 or len(packet_rows) == 0 or not np.all(np.isfinite(packet_rows)):
        raise ValueError('packet_rows must be a nonempty finite (N,6) array')
    if isinstance(step_count, bool) or int(step_count) != step_count or int(step_count) < 1:
        raise ValueError('step_count must be a positive integer')
    if not np.all(np.isfinite([final_time,mass,hbar])) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('time, mass, and hbar must be finite and positive')
    interference_potential_derivatives(packet_rows[:,:2],0.0,params)
    n = len(packet_rows)
    dt = final_time / step_count
    out = np.empty((step_count+1, n, 5))
    q = packet_rows[:,:2].copy()
    p = packet_rows[:,2:4].copy()
    action = np.zeros(n)
    out[0] = np.column_stack((q,p,action))
    for s in range(step_count):
        t = s*dt
        _, grad0, _ = _potential_tuple(q,t,params)
        ph = p - 0.5*dt*grad0
        qn = q + dt*ph/mass
        _, grad1, _ = _potential_tuple(qn,t+dt,params)
        pn = ph - 0.5*dt*grad1
        qm = 0.5*(q+qn)
        vm,_,_ = _potential_tuple(qm,t+0.5*dt,params)
        action += dt*(np.sum(ph*ph,axis=1)/(2*mass)-vm)
        q,p=qn,pn
        out[s+1]=np.column_stack((q,p,action))
    return out

def _width_potential_tuple(points, time, potential_parameters):
    import numpy as np
    packed = interference_potential_derivatives(points, time, potential_parameters)
    value = packed[:, 0]
    gradient = packed[:, 1:3]
    hessian = np.empty((len(packed), 2, 2), dtype=float)
    hessian[:, 0, 0] = packed[:, 3]
    hessian[:, 0, 1] = packed[:, 4]
    hessian[:, 1, 0] = packed[:, 4]
    hessian[:, 1, 1] = packed[:, 5]
    return value, gradient, hessian

def hagedorn_width_evolution(center_trajectory, final_time, potential_parameters, basis_gamma_diag, mass=1.0, hbar=1.0):
    import numpy as np
    params = potential_parameters
    basis_gamma = basis_gamma_diag
    traj=np.asarray(center_trajectory,float)
    basis_gamma=np.asarray(basis_gamma,float)
    if traj.ndim != 3 or traj.shape[2] != 5 or traj.shape[0] < 2 or traj.shape[1] < 1 or not np.all(np.isfinite(traj)):
        raise ValueError('center_trajectory must be finite with shape (steps+1,N,5)')
    if basis_gamma.shape != (2,) or np.any(basis_gamma <= 0) or not np.all(np.isfinite(basis_gamma)):
        raise ValueError('basis_gamma_diag must contain two finite positive values')
    if not np.all(np.isfinite([final_time,mass,hbar])) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('time, mass, and hbar must be finite and positive')
    interference_potential_derivatives(traj[0,:,:2],0.0,params)
    steps=traj.shape[0]-1
    n=traj.shape[1]
    dt=final_time/steps
    G=np.diag(np.asarray(basis_gamma,float))
    Q0=np.sqrt(hbar)*np.diag(1/np.sqrt(np.diag(G))).astype(complex)
    P0=1j*np.sqrt(hbar)*np.diag(np.sqrt(np.diag(G))).astype(complex)
    Q=np.repeat(Q0[None,:,:],n,axis=0)
    P=np.repeat(P0[None,:,:],n,axis=0)
    determinant_phase=np.zeros(n,dtype=float)
    previous_principal_phase=np.angle(np.linalg.det(Q))
    for s in range(steps):
        _,_,H0=_width_potential_tuple(traj[s,:,:2],s*dt,params)
        Ph=P-0.5*dt*np.einsum('nij,njk->nik',H0,Q)
        Qn=Q+dt*Ph/mass
        _,_,H1=_width_potential_tuple(traj[s+1,:,:2],(s+1)*dt,params)
        Pn=Ph-0.5*dt*np.einsum('nij,njk->nik',H1,Qn)
        Q,P=Qn,Pn
        principal_phase=np.angle(np.linalg.det(Q))
        phase_increment=(principal_phase-previous_principal_phase+np.pi)%(2*np.pi)-np.pi
        determinant_phase+=phase_increment
        previous_principal_phase=principal_phase
    det_q=np.linalg.det(Q)
    det_q0=np.linalg.det(Q0)
    gamma0=(np.linalg.det(G)/(np.pi*hbar)**2)**0.25
    gammas=gamma0*np.sqrt(abs(det_q0)/np.abs(det_q))*np.exp(-0.5j*determinant_phase)
    return np.column_stack((Q.real.reshape(n,-1),Q.imag.reshape(n,-1),P.real.reshape(n,-1),P.imag.reshape(n,-1),gammas.real,gammas.imag))

def coherent_tgwp_reconstruction(grid_x, grid_y, packet_rows, final_centers_actions, final_width_rows, hbar=1.0):
    import numpy as np
    final_centers = final_centers_actions
    final_widths = final_width_rows
    x=np.asarray(grid_x,float); y=np.asarray(grid_y,float)
    packet_rows=np.asarray(packet_rows,float); final_centers=np.asarray(final_centers,float); final_widths=np.asarray(final_widths,float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 1 or len(y) < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('grids must be nonempty finite vectors')
    if packet_rows.ndim != 2 or packet_rows.shape[1] != 6 or final_centers.shape != (len(packet_rows),5) or final_widths.shape != (len(packet_rows),18):
        raise ValueError('packet, center, and width shapes are inconsistent')
    if not np.all(np.isfinite(np.r_[packet_rows.ravel(),final_centers.ravel(),final_widths.ravel(),hbar])) or hbar <= 0:
        raise ValueError('reconstruction inputs must be finite and hbar positive')
    X,Y=np.meshgrid(x,y,indexing='xy')
    pts=np.column_stack((X.ravel(),Y.ravel()))
    packets_=np.asarray(packet_rows,float); c=packets_[:,4]+1j*packets_[:,5]
    centers_=np.asarray(final_centers,float)
    widths_=np.asarray(final_widths,float)
    out=np.zeros(len(pts),complex)
    for j in range(len(packets_)):
        Q=widths_[j,0:4].reshape(2,2)+1j*widths_[j,4:8].reshape(2,2)
        P=widths_[j,8:12].reshape(2,2)+1j*widths_[j,12:16].reshape(2,2)
        gamma=widths_[j,16]+1j*widths_[j,17]
        C=P@np.linalg.inv(Q)
        dr=pts-centers_[j,:2]
        quad=0.5*np.einsum('ni,ij,nj->n',dr,C,dr)
        phase=quad+dr@centers_[j,2:4]+centers_[j,4]
        out += c[j]*gamma*np.exp(1j*phase/hbar)
    field = out.reshape(len(y),len(x)); return np.stack((field.real, field.imag), axis=-1).astype(float)

def _initial_grid_gaussian(points, q0, p0, gamma_diag, hbar=1.0):
    import numpy as np
    points = np.asarray(points, float)
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    gamma = np.diag(np.asarray(gamma_diag, float))
    dr = points - q0
    pref = (np.linalg.det(gamma) / (np.pi * hbar) ** 2) ** 0.25
    exponent = -0.5 * np.einsum('ni,ij,nj->n', dr, gamma, dr) / hbar + 1j * (dr @ p0) / hbar
    return pref * np.exp(exponent)

def _grid_potential_tuple(points, time, potential_parameters):
    import numpy as np
    packed = interference_potential_derivatives(points, time, potential_parameters)
    value = packed[:, 0]
    gradient = packed[:, 1:3]
    hessian = np.empty((len(packed), 2, 2), dtype=float)
    hessian[:, 0, 0] = packed[:, 3]
    hessian[:, 0, 1] = packed[:, 4]
    hessian[:, 1, 0] = packed[:, 4]
    hessian[:, 1, 1] = packed[:, 5]
    return value, gradient, hessian

def strang_split_step_reference(grid_x, grid_y, q0, p0, initial_gamma_diag, final_time, step_count, potential_parameters, mass=1.0, hbar=1.0):
    import numpy as np
    initial_gamma = initial_gamma_diag
    steps = step_count
    params = potential_parameters
    x=np.asarray(grid_x,float); y=np.asarray(grid_y,float)
    q0=np.asarray(q0,float); p0=np.asarray(p0,float); initial_gamma=np.asarray(initial_gamma,float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3 or q0.shape != (2,) or p0.shape != (2,) or initial_gamma.shape != (2,):
        raise ValueError('grid and initial-state shapes are invalid')
    if not np.all(np.isfinite(np.r_[x,y,q0,p0,initial_gamma,final_time,mass,hbar])) or np.any(initial_gamma <= 0) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('grid and physical inputs must be finite and positive where required')
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError('step_count must be a positive integer')
    if not np.allclose(np.diff(x),x[1]-x[0]) or not np.allclose(np.diff(y),y[1]-y[0]):
        raise ValueError('grids must be uniform')
    interference_potential_derivatives(np.array([[x[0],y[0]]]),0.0,params)
    X,Y=np.meshgrid(x,y,indexing='xy')
    pts=np.column_stack((X.ravel(),Y.ravel()))
    psi=_initial_grid_gaussian(pts,q0,p0,initial_gamma,hbar).reshape(len(y),len(x))
    dx=x[1]-x[0]; dy=y[1]-y[0]; dt=final_time/steps
    kx=2*np.pi*np.fft.fftfreq(len(x),d=dx)
    ky=2*np.pi*np.fft.fftfreq(len(y),d=dy)
    KX,KY=np.meshgrid(kx,ky,indexing='xy')
    kinetic=np.exp(-1j*hbar*(KX*KX+KY*KY)*dt/(2*mass))
    for s in range(steps):
        tm=(s+0.5)*dt
        V=_grid_potential_tuple(pts,tm,params)[0].reshape(len(y),len(x))
        half=np.exp(-0.5j*dt*V/hbar)
        psi=half*psi
        psi=np.fft.ifft2(kinetic*np.fft.fft2(psi))
        psi=half*psi
    return np.stack((psi.real, psi.imag), axis=-1).astype(float)

def wavefunction_error_diagnostics(tgwp_field, reference_field, dx, dy):
    import numpy as np
    tgwp = tgwp_field
    grid = reference_field
    tgwp=np.asarray(tgwp,float); grid=np.asarray(grid,float)
    if tgwp.shape != grid.shape or tgwp.ndim != 3 or tgwp.shape[2] != 2 or not np.all(np.isfinite(tgwp)) or not np.all(np.isfinite(grid)):
        raise ValueError('fields must be equal finite arrays of shape (Ny,Nx,2)')
    if not np.all(np.isfinite([dx,dy])) or dx <= 0 or dy <= 0:
        raise ValueError('grid spacings must be finite and positive')
    tgwp=tgwp[...,0]+1j*tgwp[...,1]; grid=grid[...,0]+1j*grid[...,1]
    area=dx*dy
    ng=np.sqrt(area*np.sum(np.abs(grid)**2))
    nt=np.sqrt(area*np.sum(np.abs(tgwp)**2))
    if ng == 0 or nt == 0:
        raise ValueError('both fields must have nonzero norm')
    l2=np.sqrt(area*np.sum(np.abs(tgwp-grid)**2))/ng
    ab=np.sqrt(area*np.sum((np.abs(tgwp)-np.abs(grid))**2))/ng
    overlap=area*np.sum(np.conj(grid)*tgwp)/(ng*nt)
    aligned=tgwp*np.exp(-1j*np.angle(overlap))
    aligned_l2=np.sqrt(area*np.sum(np.abs(aligned-grid)**2))/ng
    return np.array([nt,ng,l2,ab,abs(overlap),aligned_l2])

def tgwp_convergence_table(log2_counts, step_counts, amplitudes,
                                   grid_x, grid_y, q0, p0,
                                   initial_gamma_diag, basis_gamma_diag,
                                   final_time, potential_parameters, grid_steps,
                                   sobol_seed, feasibility_limits,
                                   mass=1.0, hbar=1.0):
    import numpy as np
    logs = np.asarray(log2_counts, dtype=int)
    steps = np.asarray(step_counts, dtype=int)
    amplitudes = np.asarray(amplitudes, dtype=float)
    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    limits = np.asarray(feasibility_limits, dtype=float)
    params = np.asarray(potential_parameters, dtype=float)
    if logs.ndim != 1 or steps.shape != logs.shape or len(logs) == 0:
        raise ValueError("candidate arrays must be nonempty one-dimensional arrays of equal length")
    if amplitudes.ndim != 1 or len(amplitudes) == 0 or not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be a nonempty finite vector")
    if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3:
        raise ValueError("both grids need at least three points")
    if params.shape != (8,) or limits.shape != (5,) or np.any(limits <= 0.0):
        raise ValueError("potential_parameters and feasibility_limits have wrong shape or domain")
    references = []
    for amplitude in amplitudes:
        local = params.copy()
        local[0] = amplitude
        references.append(strang_split_step_reference(
            x, y, q0, p0, initial_gamma_diag, final_time, int(grid_steps), local, mass, hbar
        ))

    def simulate(log_count, count, local):
        packet_rows = phase_space_sobol_packets(
            int(log_count), int(sobol_seed), q0, p0, initial_gamma_diag, basis_gamma_diag, hbar
        )
        trajectory = stormer_verlet_centers_actions(
            packet_rows, final_time, int(count), local, mass, hbar
        )
        width_rows = hagedorn_width_evolution(
            trajectory, final_time, local, basis_gamma_diag, mass, hbar
        )
        return coherent_tgwp_reconstruction(
            x, y, packet_rows, trajectory[-1], width_rows, hbar
        )

    table = np.empty((len(logs), 12), dtype=float)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grid_cost = float(grid_steps) * len(x) * len(y) * np.log2(len(x) * len(y))
    for ci, (log_count, count) in enumerate(zip(logs, steps)):
        condition_rows = []
        for amplitude, reference in zip(amplitudes, references):
            local = params.copy()
            local[0] = amplitude
            base = simulate(log_count, count, local)
            time_fine = simulate(log_count, 2 * count, local)
            basis_fine = simulate(log_count + 1, count, local)
            physical = wavefunction_error_diagnostics(base, reference, dx, dy)
            time_error = wavefunction_error_diagnostics(base, time_fine, dx, dy)[2]
            basis_error = wavefunction_error_diagnostics(base, basis_fine, dx, dy)[2]
            condition_rows.append(np.r_[physical, time_error, basis_error])
        condition_rows = np.asarray(condition_rows, dtype=float)
        worst_l2 = float(np.max(condition_rows[:, 2]))
        worst_abs = float(np.max(condition_rows[:, 3]))
        worst_norm_error = float(np.max(np.abs(condition_rows[:, 0] - 1.0)))
        worst_time = float(np.max(condition_rows[:, 6]))
        worst_basis = float(np.max(condition_rows[:, 7]))
        worst_overlap_loss = float(np.max(1.0 - condition_rows[:, 4]))
        speedup = grid_cost / (2.0 ** int(log_count) * int(count))
        score = speedup / (1.0 + worst_l2 + worst_abs + worst_time + worst_basis)
        ratios = np.array([worst_l2, worst_abs, worst_norm_error, worst_time, worst_basis]) / limits
        feasible = float(np.all(np.isfinite(ratios)) and np.all(ratios <= 1.0))
        table[ci] = [log_count, count, worst_l2, worst_abs, worst_norm_error,
                     worst_time, worst_basis, speedup, score, feasible,
                     int(np.argmax(ratios)), worst_overlap_loss]
    return table

def select_meshfree_configuration(log2_counts=(4,5,6,6,6),
                                          step_counts=(32,32,32,48,64),
                                          amplitudes=(.35,.65),
                                          grid_x=None, grid_y=None,
                                          q0=(-3.5,.2), p0=(2.2,.1),
                                          initial_gamma_diag=(.7,1.2),
                                          basis_gamma_diag=(1.4,2.4),
                                          final_time=2.4,
                                          potential_parameters=(0.,0.,1.3,1.,1.6,.65,3.45,.3),
                                          grid_steps=192, sobol_seed=20260902,
                                          feasibility_limits=(.55,.48,.19,.001,.40),
                                          rounding_digits=6, mass=1.0, hbar=1.0):
    import numpy as np
    if grid_x is None:
        grid_x = np.linspace(-8.0, 8.0, 48, endpoint=False)
    if grid_y is None:
        grid_y = np.linspace(-6.0, 6.0, 36, endpoint=False)
    table = tgwp_convergence_table(
        log2_counts, step_counts, amplitudes, grid_x, grid_y, q0, p0,
        initial_gamma_diag, basis_gamma_diag, final_time, potential_parameters,
        grid_steps, sobol_seed, feasibility_limits, mass, hbar
    )
    feasible = np.flatnonzero(table[:, 9] > 0.5)
    if len(feasible) == 0:
        raise ValueError("no TGWP candidate satisfies every convergence and accuracy limit")
    selected = feasible[int(np.argmax(table[feasible, 8]))]

    # Recompute the selected certificate through every private oracle twin.  This
    # makes each scientific step causally necessary to the final answer rather
    # than relying only on the candidate-table call above.
    log_count = int(table[selected, 0])
    step_count = int(table[selected, 1])
    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    params = np.asarray(potential_parameters, dtype=float)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    direct_errors = []
    for amplitude in np.asarray(amplitudes, dtype=float):
        local = params.copy()
        local[0] = amplitude
        packets = phase_space_sobol_packets(
            log_count, int(sobol_seed), q0, p0,
            initial_gamma_diag, basis_gamma_diag, hbar
        )
        derivative_probe = interference_potential_derivatives(
            packets[:, :2], 0.0, local
        )
        trajectory = stormer_verlet_centers_actions(
            packets, final_time, step_count, local, mass, hbar
        )
        widths = hagedorn_width_evolution(
            trajectory, final_time, local, basis_gamma_diag, mass, hbar
        )
        tgwp = coherent_tgwp_reconstruction(
            x, y, packets, trajectory[-1], widths, hbar
        )
        reference = strang_split_step_reference(
            x, y, q0, p0, initial_gamma_diag, final_time,
            int(grid_steps), local, mass, hbar
        )
        diagnostics = wavefunction_error_diagnostics(
            tgwp, reference, dx, dy
        )
        if not np.all(np.isfinite(derivative_probe)):
            raise ValueError("selected-candidate derivative certificate is non-finite")
        direct_errors.append(diagnostics[2:4])
    direct_errors = np.asarray(direct_errors, dtype=float)
    if not np.allclose(
        np.max(direct_errors, axis=0), table[selected, 2:4],
        rtol=2e-12, atol=2e-12
    ):
        raise ValueError("selected-candidate certificate disagrees with the convergence table")
    return float(round(float(table[selected, 8]), int(rounding_digits)))
SCICODE_GOLD_EOF
