"""
Orchestrate the microscopic screening, complete inverse branch set, dielectric witness selection, and full versus head-projected exciton prediction.

Orchestrate the microscopic screening, complete inverse branch set, dielectric witness selection, and full versus head-projected exciton prediction.

This constructed three-orbital benchmark uses the supplied paper’s method and Supplementary Information (SI). Its Hamiltonian, observations and discretization are synthetic. Lengths are in Å, momenta in Å⁻¹, and energies in eV. The lattice is square with \(a=3.2\), cell area \(a^2\), orthogonal point orbitals at \(\boldsymbol\tau/a=((0,0),(0.37,0.12),(0.16,0.43))\), and heights \(z_a/d=(0,0.29,-0.23)\). With \(x=ak_x,y=ak_y\), the upper triangle of the lattice-gauge Bloch Hamiltonian is
\[
H_{00}=-\mu+0.17(\cos x+\cos y),\quad H_{11}=\mu+0.11\cos x-0.08\cos y,\quad H_{22}=\mu+1.15+0.09\cos(x+y),
\]
\[
H_{01}=0.44+0.31e^{-ix}+0.23e^{-iy},\quad H_{02}=0.29-0.19e^{ix}+0.21e^{-i(x+y)},\quad H_{12}=0.16+0.13e^{-iy};\qquad H_{ba}=H_{ab}^*.
\]
Eigenvalues \(e_b\) increase with \(b=0,1,2\); eigenvector columns \(U_{ab}\) are normalized. Occupations are \((1,0,0)\), spin degeneracy is two, temperature is zero, and the vertical Tamm–Dancoff kernel is the attractive direct term. The public eigenvector interface fixes each column's largest-magnitude component to real nonnegative, with the lowest component index resolving ties.

For odd \(n\), \(\mathcal K_n=\{2\pi(i-(n-1)/2,j-(n-1)/2)/(an):0\leq i,j<n\}\), ordered by \(i\), then \(j\). Polarization always uses \(\mathcal K_7\); the exciton uses \(\mathcal K_n\). The reciprocal set is \(\mathcal G_L=\{(2\pi/a)(u,v):u,v\in\mathbb Z,\ u^2+v^2\leq L^2\}\), ordered by \((u^2+v^2,u,v)\), with the zero vector first. The same set enters dielectric inversion and the direct interaction. Every transfer is the raw difference \(\mathbf q=\mathbf k_i-\mathbf k_j\); shifted Bloch Hamiltonians are evaluated at \(\mathbf k+\mathbf q\). Brillouin-zone sums are arithmetic averages.

Write \(Q_G=|\mathbf q+\mathbf G|\), \(c=2\pi(14.3996454784255)/(a^2\kappa)\), and \(v_G=c/Q_G\). The Coulomb constant in parentheses has units eV Å. The two dimensionless slab averages are defined by
\[
F_a(Q,d)=\frac1d\int_{-d/2}^{d/2}e^{-Q|z-z_a|}\,dz,\qquad B(Q,d)=\frac1{d^2}\int_{-d/2}^{d/2}\int_{-d/2}^{d/2}e^{-Q|z-z'|}\,dz\,dz'.
\]
Both extend continuously to one at \(Qd=0\), with fixed fractional heights. The symmetric orbital prescription of SI Eq. S.25 uses vertices
\[
J^{nb}_G(\mathbf k,\mathbf q)=\sum_a U_{an}(\mathbf k)^*U_{ab}(\mathbf k+\mathbf q)e^{-i(\mathbf q+\mathbf G)\cdot\boldsymbol\tau_a}\sqrt{F_a(Q_G,d)}.
\]
The static occupation-difference response and symmetric dielectric matrix are
\[
\chi_{GG'}(\mathbf q)=\frac{2}{49}\sum_{\mathbf k\in\mathcal K_7}\sum_{n,b:f_n\ne f_b}\frac{f_n-f_b}{e_n(\mathbf k)-e_b(\mathbf k+\mathbf q)}J^{nb}_G(J^{nb}_{G'})^*,\qquad \bar\varepsilon=I-\sqrt v\,\chi\sqrt v.
\]
The effective screened potential is \(\bar W=\sqrt{vB}\,\bar\varepsilon^{-1}\sqrt{vB}\); diagonal square roots act on the reciprocal indices. The macroscopic response is \(\varepsilon_M=1/(\bar\varepsilon^{-1})_{00}\). At exactly \(Q_G=0\), use the value zero for the bare factor during matrix construction: the dielectric head is then one and its wings vanish, while the body is computed normally.

The following finite-grid origin prescription fixes the benchmark and extends the paper's first-order circular-cell rule to its pair-averaged Q2D potential. Let \(\delta=0.002/a\) and \(s_\alpha(t)=[\varepsilon_M(t\hat{\boldsymbol\alpha})-1]/t\); use \(r_\alpha=2s_\alpha(\delta/2)-s_\alpha(\delta)\) for \(\alpha=x,y\). With \(q_0=0.35(2\pi/an)\), replace \(\bar W_{00}(0)\) by \(c[2/q_0-(r_x+r_y)/2-d/3]\), set the origin wings to zero, and retain the origin body. This is the stipulated first-order cell prescription at the fixed \(\delta,q_0\); the synthetic targets refer to that prescription.

For the direct term define the unweighted point-orbital vertex
\[
I^{nb}_{ij,G}=\sum_a U_{an}(\mathbf k_j)^*e^{-i(\mathbf k_i-\mathbf k_j+\mathbf G)\cdot\boldsymbol\tau_a}U_{ab}(\mathbf k_i).
\]
With \(N=n^2\), electron bands \(c,c'=1,2\), and pair order \((i,c)\),
\[
D_{ic,jc'}=\frac1N\sum_{GG'}(I^{c'c}_{ij,G})^*\bar W_{GG'}(\mathbf k_i-\mathbf k_j)I^{00}_{ij,G'},\qquad H^X_{ic,jc'}=(e_c(\mathbf k_i)-e_0(\mathbf k_i))\delta_{ij}\delta_{cc'}-D_{ic,jc'}.
\]
\(E_0\leq E_1\leq\cdots\) are the eigenvalues of this Hermitian matrix. \(E_0^{\mathrm{head}}\) uses the same parameters and the same full dielectric inversion, with only the \(G=G'=0\) screened-potential entry retained in the direct term. Full-precision arithmetic is used except for the prescribed six-decimal rounding of fitted parameters. The exact-target equations define all admissible calibration roots; the supplied box has isolated roots. The root calculation has coordinate convergence \(10^{-8}\) in \(\kappa\) and \(d\), together with absolute energy residual below \(10^{-9}\) eV before rounding.

Returns
-------
return result  # real ndarray, shape (7,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def predict_exciton_splitting(observed: np.ndarray, witness: float, bounds = ((6.,9.),(.6,3.2)), prediction = (7,1.35,2)) -> np.ndarray:
    """Parameters
    ----------
    observed : finite real ndarray, shape (2,)
        Exact calibration ground energies in eV at (n,mu,L)=(5,1.1,1) and
        (5,1.6,1), respectively. The isolated-root class of calibration_branches
        applies whenever roots exist.
    witness : finite positive float
        Dimensionless dielectric witness at q_w=(0.27,0.13) inverse angstroms,
        evaluated at the prediction mass and reciprocal cutoff.
    bounds : real array-like, shape (2,2)
        Rows [kappa_min,kappa_max], [d_min,d_max], a nondegenerate subrectangle
        of [6,9] x [0.6,3.2], in dimensionless units and angstroms.
    prediction : tuple (n,mu,L)
        n is 3,5 or 7; mu is in [0.8,2] eV; reciprocal cutoff L is 1 or 2.
    Returns
    -------
    real ndarray, shape (7,)
        Ordered [E1-E0,kappa,d,epsilon_M(q_w),W00(0),E0,E0_head-E0]. Units are
        eV, dimensionless, angstrom, dimensionless, eV, eV, eV. The chosen root
        minimizes witness error after six-decimal rounding, then d, then kappa.
        An empty admissible fit set returns seven entries equal to -1.
        This final public function must call and combine every preceding public
        function, directly or through helpers/callbacks, using their returned
        scientific quantities. Constants, Hamiltonian, grids, approximations,
        origin prescription and certificate are those in the task background.
    Raises
    ------
    ValueError : invalid observation/witness, bounds, or prediction configuration."""
    return np.zeros((7,))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_predict_exciton_splitting(observed, witness, bounds=((6., 9.), (.6, 3.2)),
                                      prediction=(7, 1.35, 2)):
    if np.asarray(observed).shape != (2,) or not np.isfinite(observed).all() or not np.isfinite(witness) or witness <= 0 or len(prediction) != 3:
        raise ValueError('invalid experiment')
    n, mass, cutoff = prediction
    if n not in (3,5,7) or cutoff not in (1,2) or not np.isfinite(mass) or not .8 <= mass <= 2.:
        raise ValueError('prediction requires n=3,5,7; mass in [0.8,2]; cutoff=1,2')
    bb = np.asarray(bounds,dtype=float)
    if bb.shape != (2,2) or not np.isfinite(bb).all() or np.any(bb[:,0] >= bb[:,1]) or bb[0,0] < 6 or bb[0,1] > 9 or bb[1,0] < .6 or bb[1,1] > 3.2:
        raise ValueError('bounds must be a nondegenerate subrectangle of the experiment box')
    a = 3.2
    c0 = 2*np.pi*14.3996454784255/a**2
    tau = a*np.array([[0., 0.], [.37, .12], [.16, .43]])
    z_fractions = np.array([0., .29, -.23])
    delta = 2e-3/a

    def prepare(n, mass, cutoff):
        n, cutoff = int(n), int(cutoff)
        kk = (np.arange(n)-(n-1)/2)*2*np.pi/(a*n)
        k = np.array([(x, y) for x in kk for y in kk])
        indices = [(i, j) for i in range(-cutoff, cutoff+1)
                   for j in range(-cutoff, cutoff+1) if i*i+j*j <= cutoff*cutoff]
        indices.sort(key=lambda ij: (ij[0]**2+ij[1]**2, ij[0], ij[1]))
        gs = np.asarray(indices)*2*np.pi/a
        zz = np.arange(-(n-1), n)
        q = np.array([(x, y) for x in zz for y in zz])*2*np.pi/(a*n)
        zero = (len(q)-1)//2
        q = np.vstack((q, [[delta, 0], [delta/2, 0], [0, delta], [0, delta/2], [.27, .13]]))
        lengths = np.linalg.norm(q[:, None, :]+gs[None, :, :], axis=-1)
        phases = np.exp(-1j*np.einsum('qgi,ai->qga', q[:, None, :]+gs, tau))
        pp = (np.arange(7)-3)*2*np.pi/(a*7)
        kp = np.array([(x, y) for x in pp for y in pp])
        left = _oracle_bloch_frames(kp, mass)
        right = _oracle_bloch_frames((kp[None, :, :]+q[:, None, :]).reshape(-1, 2), mass)
        right = right.reshape(len(q), len(kp), 4, 3)
        frames = _oracle_bloch_frames(k, mass)
        i, j = np.indices((len(k), len(k)))
        qi = (i//n-j//n+n-1)*(2*n-1)+(i % n-j % n+n-1)
        pair_phases = phases[qi].reshape(len(k), len(k), len(gs), 3)
        # Pair phases vary with both i and j. Evaluate each i batch with Q=j,K=1.
        pair = np.empty((len(k), len(k), len(gs), 3, 3), dtype=complex)
        for jj in range(len(k)):
            pair[:, jj] = _oracle_density_vertices(frames[jj:jj+1, 1:], frames[:, None, 1:],
                                                   pair_phases[:, jj],
                                                   np.ones((len(k), len(gs), 3)))[:, 0]
        return dict(n=n, mass=mass, lengths=lengths, phases=phases, zero=zero,
                    left=left, right=right, frames=frames, pair=pair, qi=qi)

    geometries = [prepare(5, 1.1, 1), prepare(5, 1.6, 1)]
    future = prepare(*prediction)

    def evaluate(g, parameters, full=False):
        kappa, thickness = parameters
        averages = _oracle_slab_averages(g['lengths'], thickness, z_fractions)
        vertices = _oracle_density_vertices(g['left'][:, 1:], g['right'][:, :, 1:],
                                            g['phases'], averages[:, :, :3])
        bare = np.zeros_like(g['lengths'])
        np.divide(c0/kappa, g['lengths'], out=bare, where=g['lengths'] > 1e-13)
        eps = _oracle_dielectric_matrices(g['left'][:, 0].real, g['right'][:, :, 0].real,
                                          vertices, bare)
        inv = np.linalg.inv(eps)
        r = _oracle_screening_lengths(eps[-5:-1], delta)
        q0 = .35*2*np.pi/(a*g['n'])
        w = _oracle_screened_potentials(eps, bare, averages[:, :, 3], g['zero'], q0,
                                        r, kappa, thickness)
        h = _oracle_exciton_hamiltonian(g['frames'][:, 0].real, g['pair'], w, g['qi'])
        levels = _oracle_exciton_levels(h)
        if not full:
            return levels[0]
        hhead = _oracle_exciton_hamiltonian(g['frames'][:, 0].real, g['pair'], w, g['qi'], True)
        head_e0 = _oracle_exciton_levels(hhead, 1)[0]
        return levels, 1/inv[-1, 0, 0].real, w[g['zero'], 0, 0].real, head_e0-levels[0]

    def forward(parameters):
        return np.array([evaluate(g, parameters) for g in geometries])

    branches = _oracle_calibration_branches(forward, observed, bounds)
    if not len(branches):
        return np.full(7, -1., dtype=float)
    details = [evaluate(future, parameters, True) for parameters in branches]
    choice = min(range(len(branches)), key=lambda i: (abs(details[i][1]-witness),
                                                    branches[i, 1], branches[i, 0]))
    levels, dielectric, head, bias = details[choice]
    return np.array([levels[1]-levels[0], branches[choice, 0], branches[choice, 1],
                     dielectric, head, levels[0], bias], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n\n', 'call': 'predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.)', 'gold_call': '_oracle_predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.)'}, {'setup': 'import numpy as np\n\n', 'call': 'predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.04225,((8.,9.),(.6,1.2)),(5,1.35,1))', 'gold_call': '_oracle_predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.04225,((8.,9.),(.6,1.2)),(5,1.35,1))'}, {'setup': 'import numpy as np\n\n', 'call': 'predict_exciton_splitting(np.array([20.,30.]),1.04)', 'gold_call': '_oracle_predict_exciton_splitting(np.array([20.,30.]),1.04)'}, {'setup': 'import numpy as np\n\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.04,prediction=(4,1.35,2)))', 'gold_call': '_exception_code(lambda: _oracle_predict_exciton_splitting(np.array([.7754618169,1.6367562578]),1.04,prediction=(4,1.35,2)))'}]
