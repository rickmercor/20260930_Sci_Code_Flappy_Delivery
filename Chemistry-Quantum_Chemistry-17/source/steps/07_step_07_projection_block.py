"""
Replay a finite orbital-relaxed DOCI-guided phaseless projection block.

A fixed field replay isolates the propagation and estimator algebra. The block energy is a mixed estimate for the specified finite ensemble, not an infinite-population ground-state limit. A fixed orbital-relaxation schedule prepares the guiding trial before propagation; its accepted frame is held fixed throughout the projection block.

Returns
-------
return trial_energy, energies, weight_history, phase_history
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def projection_block(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Replay a finite DOCI-guided phaseless projection block.
    
    Packed complex arrays have a final axis [real, imaginary]. All inputs are finite and must not be modified. NumPy and SciPy are available; earlier public functions may be called.
    
    h is a real symmetric (n,n) matrix in Eh, factors is a real symmetric (r,n,n) array in sqrt(Eh), with 1<=n<=6, 1<=r<=5 and 1<=npair<=n. The spatial basis is orthonormal; each spin sector has npair electrons. H=sum(h[p,q] a†[p,s]a[q,s]) + 1/2 sum_l,p,q,r,s,spin,spin' L_l[p,q]L_l[r,s] a†[p,spin]a†[r,spin']a[s,spin']a[q,spin]. No nuclear constant is included. The DOCI basis is the complete set of paired occupations. At each retained trial frame, the lowest DOCI eigenvalue is isolated by more than 1e-10 Eh when the paired space has more than one configuration. A walker is two complex (n,npair) orbital matrices. walkers has packed shape (nw,2,n,npair,2), weights is real (nw,), and fields is a real (nt,nw,r) prescribed replay array with 1<=nw<=6 and 0<=nt<=10. dt>=0 is in inverse Eh and energy_shift in Eh. All live total trial overlaps exceed 1e-12 and initial and propagated QR diagonal magnitudes exceed 1e-12, except dedicated error tests. Do not sample new fields, constrain walkers to seniority zero, reconfigure the population, clip local energies, clip force bias, subtract a mean field, or renormalize weights.
    Parameters
    ----------
    h, factors, npair, walkers, weights, fields, dt, energy_shift
        As specified in the common model contract above. The orbitals and weights describe the initial importance-sampled ensemble.
    n_updates : int, default 4
        Number of orbital-relaxation updates, 0 <= n_updates <= 6,
        using exactly the protocol of relax_doci_trial. Construct this
        trial once before the replay. Its columns remain in the input
        Hamiltonian basis; use them in every overlap, force and local
        energy contraction. The Hamiltonian and walkers retain their
        input basis. Zero updates gives the fixed-frame DOCI trial.
    Returns
    -------
    trial_energy : float
        DOCI variational energy after n_updates orbital updates, Eh.
    energies : real ndarray, shape (nt+1,)
        E_t=sum_w weight_tw*Re(E_local_tw)/sum_w weight_tw, including t=0 and each completed time step.
    weight_history : real ndarray, shape (nt+1,nw)
        Absolute importance weights, including the unchanged initial weights; no normalization/reconfiguration.
    phase_history : real ndarray, shape (nt,nw)
        Physical overlap angles in time-major order. A walker already at zero weight is skipped and its later angle is zero.
        At each step evaluate force on the old walker, propagate, normalize by QR, evaluate the new overlap,
        update the weight, and then measure the energy of the new weighted ensemble.
    Raises
    ------
    ValueError
        If block shapes disagree, a weight or dt is negative, initial total weight is nonpositive,
        total surviving weight is <=1e-14, or an earlier step raises a documented ValueError.
    """
    return trial_energy, energies, weight_history, phase_history

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_projection_block(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    walkers = np.array(walkers, dtype=float, copy=True)
    weights = np.array(weights, dtype=float, copy=True)
    fields = np.asarray(fields, dtype=float)
    n = len(h)
    if (walkers.ndim != 5 or walkers.shape[1:] != (2, n, npair, 2)
            or weights.shape != (len(walkers),) or fields.ndim != 3
            or fields.shape[1:] != (len(walkers), len(factors))
            or np.any(weights < 0) or np.sum(weights) <= 0 or dt < 0):
        raise ValueError('Invalid block dimensions or nonpositive total weight.')
    occ = _oracle_pair_occupations(n, npair)
    trial_energy, coeff, orbitals, _, _ = _oracle_relax_doci_trial(
        h, factors, npair, n_updates)
    cp = np.stack((coeff, np.zeros_like(coeff)), axis=-1)
    up = np.stack((orbitals, np.zeros_like(orbitals)), axis=-1)
    values = np.zeros(len(weights))
    for w in range(len(weights)):
        if weights[w] > 0:
            initial = _oracle_advance_walker(
                occ, cp, walkers[w, 0], walkers[w, 1], h, factors,
                np.zeros(len(factors)), 0.0, up)
            values[w] = initial[4][0]
    trajectory, history, angles = [], [], []

    def _measure():
        total = weights.sum()
        if total <= 1e-14:
            raise ValueError('All walkers have died (total weight <= 1e-14).')
        trajectory.append(float(np.dot(weights, values) / total))
        history.append(weights.copy())

    _measure()
    for row in fields:
        phases = np.zeros(len(weights))
        for w, x in enumerate(row):
            if weights[w] == 0:
                continue
            pa, pb, old, new, _, el, _, lg, gauge = _oracle_advance_walker(
                occ, cp, walkers[w, 0], walkers[w, 1], h, factors, x, dt, up)
            weights[w], phases[w] = _oracle_phaseless_weight(
                old, new, gauge, lg, float(weights[w]), dt, energy_shift)
            walkers[w, 0], walkers[w, 1] = pa, pb
            values[w] = el[0]
        angles.append(phases)
        _measure()
    return (trial_energy, np.array(trajectory), np.array(history),
            np.array(angles).reshape(len(fields), len(weights)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge comparisons against the oracle."""
    checks = '''
def _checked(fn, *args):
    copied = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
    before = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in copied]
    try:
        result = fn(*copied)
    finally:
        for original, current in zip(before, copied):
            if isinstance(original, np.ndarray) and not np.array_equal(original, current):
                raise AssertionError('Input arrays must not be modified')
    expected_shapes = ((), (np.asarray(args[5]).shape[0] + 1,), (np.asarray(args[5]).shape[0] + 1, np.asarray(args[3]).shape[0]), (np.asarray(args[5]).shape[0], np.asarray(args[3]).shape[0]))
    if not isinstance(result, (tuple, list)) or len(result) != len(expected_shapes):
        raise AssertionError('Return structure does not match the documented tuple')
    arrays = [np.asarray(value, dtype=float) for value in result]
    if any(value.shape != shape for value, shape in zip(arrays, expected_shapes)):
        raise AssertionError('Return shapes do not match the documented contract')
    return np.concatenate([value.ravel() for value in arrays])
'''
    setup_0 = '''import numpy as np

def pack(z):
    z=np.asarray(z)
    return np.stack((z.real,z.imag),axis=-1)

H = np.array([[-1.084145,.12,-.08,.05],[.12,-.885553,.11,-.06],[-.08,.11,-.763931,.09],[.05,-.06,.09,-.742439]])
L = np.array([
 [[.58,.07,-.03,.04],[.07,.43,.08,-.02],[-.03,.08,.36,.06],[.04,-.02,.06,.29]],
 [[.16,-.11,.04,.03],[-.11,-.22,.05,.07],[.04,.05,.19,-.08],[.03,.07,-.08,-.14]],
 [[-.12,.03,.09,-.05],[.03,.17,-.06,.02],[.09,-.06,.08,.11],[-.05,.02,.11,-.10]]])
for factor in L:
    d = factor.diagonal().copy()
    factor *= 2.8
    np.fill_diagonal(factor,d)
PHI_A=np.array([[1.,.07],[-.04,.92],[.18,-.14],[-.12,.21]])
PHI_B=np.array([[.95,-.03],[.06,1.02],[-.13,.17],[.16,.09]])
WALKERS=[]
for w in range(4):
    spins=[]
    for s,base in enumerate((PHI_A,PHI_B)):
        p=np.empty((4,2),complex)
        for pidx in range(4):
            for j in range(2):
                p[pidx,j]=base[pidx,j]+.035*w*np.cos((pidx+1)*(j+2)) + 1j*.02*(w+s)*np.sin((pidx+2)*(j+1))
        spins.append(pack(p))
    WALKERS.append(spins)
WALKERS=np.array(WALKERS)
WEIGHTS=np.array([1.,.8,1.2,.9])
FIELDS=np.empty((7,4,3))
for t in range(7):
    for w in range(4):
        for l in range(3):
            FIELDS[t,w,l]=.85*np.sin((t+1)*(w+2)*(l+1))+.35*np.cos((t+2)*(l+2)+w)
DT=.08
ENERGY_SHIFT=-2.1
NPAIR=2
ARGS=(H,L,NPAIR,WALKERS,WEIGHTS,FIELDS,DT,ENERGY_SHIFT)


'''
    setup_1 = '''import numpy as np

def pack(z):
    z=np.asarray(z)
    return np.stack((z.real,z.imag),axis=-1)

H = np.array([[-1.084145,.12,-.08,.05],[.12,-.885553,.11,-.06],[-.08,.11,-.763931,.09],[.05,-.06,.09,-.742439]])
L = np.array([
 [[.58,.07,-.03,.04],[.07,.43,.08,-.02],[-.03,.08,.36,.06],[.04,-.02,.06,.29]],
 [[.16,-.11,.04,.03],[-.11,-.22,.05,.07],[.04,.05,.19,-.08],[.03,.07,-.08,-.14]],
 [[-.12,.03,.09,-.05],[.03,.17,-.06,.02],[.09,-.06,.08,.11],[-.05,.02,.11,-.10]]])
for factor in L:
    d = factor.diagonal().copy()
    factor *= 2.8
    np.fill_diagonal(factor,d)
PHI_A=np.array([[1.,.07],[-.04,.92],[.18,-.14],[-.12,.21]])
PHI_B=np.array([[.95,-.03],[.06,1.02],[-.13,.17],[.16,.09]])
WALKERS=[]
for w in range(4):
    spins=[]
    for s,base in enumerate((PHI_A,PHI_B)):
        p=np.empty((4,2),complex)
        for pidx in range(4):
            for j in range(2):
                p[pidx,j]=base[pidx,j]+.035*w*np.cos((pidx+1)*(j+2)) + 1j*.02*(w+s)*np.sin((pidx+2)*(j+1))
        spins.append(pack(p))
    WALKERS.append(spins)
WALKERS=np.array(WALKERS)
WEIGHTS=np.array([1.,.8,1.2,.9])
FIELDS=np.empty((7,4,3))
for t in range(7):
    for w in range(4):
        for l in range(3):
            FIELDS[t,w,l]=.85*np.sin((t+1)*(w+2)*(l+1))+.35*np.cos((t+2)*(l+2)+w)
DT=.08
ENERGY_SHIFT=-2.1
NPAIR=2
ARGS=(H,L,NPAIR,WALKERS,WEIGHTS,FIELDS,DT,ENERGY_SHIFT)



def _raises(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')
'''
    return [
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, *ARGS, 0)',
            "gold_call": '_checked(_oracle_projection_block, *ARGS, 0)',
            "tol": 2e-7,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, *ARGS)',
            "gold_call": '_checked(_oracle_projection_block, *ARGS)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, H, np.zeros_like(L), NPAIR, WALKERS, WEIGHTS, FIELDS[:3], DT, ENERGY_SHIFT)',
            "gold_call": '_checked(_oracle_projection_block, H, np.zeros_like(L), NPAIR, WALKERS, WEIGHTS, FIELDS[:3], DT, ENERGY_SHIFT)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, H, L, NPAIR, WALKERS[:1], WEIGHTS[:1], FIELDS[:2, :1], DT, ENERGY_SHIFT)',
            "gold_call": '_checked(_oracle_projection_block, H, L, NPAIR, WALKERS[:1], WEIGHTS[:1], FIELDS[:2, :1], DT, ENERGY_SHIFT)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, H, L, NPAIR, WALKERS, WEIGHTS, FIELDS[:0], DT, ENERGY_SHIFT)',
            "gold_call": '_checked(_oracle_projection_block, H, L, NPAIR, WALKERS, WEIGHTS, FIELDS[:0], DT, ENERGY_SHIFT)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(projection_block, H, L, NPAIR, WALKERS, np.array([1.0, 0.0, 0.5, 0.0]), FIELDS[:2], DT, ENERGY_SHIFT)',
            "gold_call": '_checked(_oracle_projection_block, H, L, NPAIR, WALKERS, np.array([1.0, 0.0, 0.5, 0.0]), FIELDS[:2], DT, ENERGY_SHIFT)',
            "tol": 2e-08,
        },
        {
            "setup": setup_1 + checks,
            "call": '_raises(_checked, projection_block, H, L, NPAIR, WALKERS, np.zeros(4), FIELDS, DT, ENERGY_SHIFT)',
            "gold_call": '_raises(_checked, _oracle_projection_block, H, L, NPAIR, WALKERS, np.zeros(4), FIELDS, DT, ENERGY_SHIFT)',
            "tol": 2e-08,
        },
    ]
