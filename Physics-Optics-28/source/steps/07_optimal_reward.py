"""
Optimize the pooled source reward over common or adaptive Gaussian controllers with the supplied inline-squeezer bound.

Maximize $R=sum(P_k)+sum(M_k)/sum(P_k)$ over the stated controller domains.

A common controller has one G for all records; the adaptive mode chooses G

by first count. Both inline squeezers have the supplied rmax bound. Passive

adaptivity is still a count-selected interferometer when rmax=0. Normalize

fidelity only after applying the shared C(H) physical interpretation and

forming the correctly weighted emitted ensemble.

Explicitly, $C(H)$ hermitizes each density, clips negative eigenvalues to

zero, and rescales the positive part to $m=max(0,Re trace(H))$, with zero for

$m=0$.

Returns
-------
Return one finite real optimal value with total absolute error <=0.0004. No policy, seed, optimizer prescription, certificate format or iteration history is returned. Tests observe the scalar budget directly.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_reward(states: object, target: object, counts: object, efficiencies: object, rmax: float, adaptive: bool) -> float:
    """Return the optimal physical pooled reward as one real scalar.

    states, target, counts and efficiencies have the density-input domains
    of policy_statistics. rmax is a finite real numeric scalar in [0,0.5],
    not complex/string/bool. adaptive is bool or numpy.bool_. Each G has
    both mixing angles in [0,pi/2], both squeezing magnitudes in [0,rmax],
    and unrestricted periodic phases. A common G is required for every
    record if adaptive is False, otherwise the G settings may differ.
    Interpret accepted states with shared deterministic C(H) BEFORE any
    normalized fidelity or optimization, preserving their physical masses.
    For each policy, sum the P_k and M_k columns and use R=P+M/P when P>0,
    or zero at exactly P=0. No positive-probability floor is permitted.
    The optimum is the supremum if it is approached only at zero success.
    Return only the value, not coordinates. The absolute error from the true
    bounded-domain optimum, including iterative and evaluation errors
    together, must be <=0.0004. No optimizer, seed, coordinates or valid
    representation is prescribed. Raise ValueError for all inherited invalid
    input conditions, invalid rmax or non-boolean adaptive.
    Returns
    -------
    Return one finite real optimal value with total absolute error <=0.0004. No policy, seed, optimizer prescription, certificate format or iteration history is returned. Tests observe the scalar budget directly.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Bounded optimization of the complete adaptive optical instrument."""
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def _r7_target(value, d):
    target = _r7_numeric(value, (d,))
    if abs(np.vdot(target, target).real-1) > 1e-10:
        raise ValueError('normalized target required')
    return target
def _r7_optimize(states, target, counts, efficiencies, rmax, adaptive):
    class _r7_adjoint:
        def __init__(self, d, factors, target, counts, efficiencies):
            self.d, self.factors, self.counts = d, factors, counts
            a = np.diag(np.sqrt(np.arange(1, d)), 1)
            self.generator = (a@a-a.T@a.T)/2
            self.values, self.vectors = eigh(-1j*self.generator)
            self.vectors_h = self.vectors.conj().T
            self.n = np.arange(d)
            # These factors depend only on the fixed local dimension.
            self.number_grid = np.repeat(self.n, d)[:, None]
            root = np.sqrt(np.arange(1, d))
            self.mixing_weights = root[:, None, None]*root[None, :, None]
            self.blocks = []
            for total in range(2*d-1):
                ids = np.array([i*d+total-i for i in range(d) if 0 <= total-i < d])
                g = np.zeros((len(ids), len(ids)), complex)
                for col, index in enumerate(ids):
                    i, j = divmod(int(index), d)
                    if i+1 < d and j:
                        g[col+1, col] = np.sqrt((i+1)*j)
                        g[col, col+1] = -np.sqrt((i+1)*j)
                val, vec = eigh(-1j*g)
                self.blocks.append((ids, ids//d, val, vec, vec.conj().T))
            self.q = np.zeros((d, d), complex)
            eta = efficiencies[1]
            for ell in range(d):
                v = np.zeros(d, complex)
                for n in range(ell, d):
                    v[n] = target[n-ell]*np.sqrt(math.comb(n, ell)*(1-eta)**ell*eta**(n-ell))
                self.q += np.outer(v, v.conj())
            eta = efficiencies[0]
            self.effects, self.effect_logs, self.relative_logs = [], [], []
            for k in counts:
                logs = np.full(d, -np.inf)
                for n in range(k, d):
                    if (eta == 0 and k != 0) or (eta == 1 and n != k):
                        continue
                    logs[n] = (math.log(math.comb(n, k))
                               + (k*math.log(eta) if k else 0.)
                               + ((n-k)*math.log1p(-eta) if n > k else 0.))
                shift = float(np.max(logs))
                self.effect_logs.append(shift)
                self.relative_logs.append(logs-shift if np.isfinite(shift) else logs)
                self.effects.append(np.exp(logs-shift) if np.isfinite(shift) else np.zeros(d))

        def mix(self, w, theta, phase):
            if theta == 0:
                return w.copy()
            out = np.empty_like(w)
            phases = np.exp(1j*phase)**self.n
            for ids, ns, values, vectors, vectors_h in self.blocks:
                ph = phases[ns]
                z = vectors_h@(w[ids]*ph.conj()[:, None])
                out[ids] = ph[:, None]*(vectors@(np.exp(1j*theta*values)[:, None]*z))
            return out

        def mix_generator(self, w, phase):
            t = w.reshape(self.d, self.d, -1)
            out = np.zeros_like(t)
            factors = self.mixing_weights
            out[1:, :-1] += np.exp(1j*phase)*factors*t[:-1, 1:]
            out[:-1, 1:] -= np.exp(-1j*phase)*factors*t[1:, :-1]
            return out.reshape(w.shape)

        def squeeze(self, r, phase):
            ph = np.exp(.5j*phase)**self.n
            u = np.eye(self.d, dtype=complex) if r == 0 else (self.vectors*np.exp(1j*r*self.values))@self.vectors_h
            u = ph[:, None]*u*ph.conj()[None, :]
            g = ph[:, None]*self.generator*ph.conj()[None, :]
            return u, g@u, .5j*(self.n[:, None]-self.n[None, :])*u

        def forward(self, branch, x):
            d = self.d
            w0 = self.factors[branch]
            w1 = self.mix(w0, *x[:2])
            s0, s1 = self.squeeze(*x[2:4]), self.squeeze(*x[4:6])
            w2 = np.einsum('ai,bj,ijr->abr', s0[0], s1[0], w1.reshape(d, d, -1), optimize=True).reshape(d*d, -1)
            w3 = self.mix(w2, *x[6:8])
            t = w3.reshape(d, d, -1)
            q = np.einsum('ik,kjr->ijr', self.q, t, optimize=True)
            effect = self.effects[branch]
            p = np.einsum('ijr,ijr,j->', t.conj(), t, effect).real
            m = np.einsum('ijr,ijr,j->', t.conj(), q, effect).real
            offset = 0.
            rescaled = False
            # This threshold selects an arithmetic representation; it never
            # discards a positive success or changes the physical objective.
            if p < 1e-150:
                magnitude = np.abs(t)
                nonzero = magnitude > 0
                log_amplitude = np.full(t.shape, -np.inf)
                log_amplitude[nonzero] = np.log(magnitude[nonzero])
                log_amplitude += .5*self.relative_logs[branch][None, :, None]
                shift = float(np.max(log_amplitude))
                if np.isfinite(shift):
                    phase = np.zeros_like(t)
                    phase[nonzero] = (t.real[nonzero]/magnitude[nonzero]
                                     +1j*(t.imag[nonzero]/magnitude[nonzero]))
                    weighted = phase*np.exp(log_amplitude-shift)
                    weighted_q = np.einsum('ik,kjr->ijr', self.q, weighted, optimize=True)
                    p = float(np.vdot(weighted, weighted).real)
                    m = float(np.vdot(weighted, weighted_q).real)
                    offset, rescaled = 2*shift, True
            return np.array([p, m]), (w0, w1, w2, w3, s0, s1, q, effect, rescaled), offset

        def gradient(self, x, cache, alpha, beta):
            d = self.d
            w0, w1, w2, w3, s0, s1, q, effect, _ = cache
            z3 = ((alpha*w3.reshape(d, d, -1)+beta*q)*effect[None, :, None]).reshape(d*d, -1)
            grad = np.zeros(8)
            n = self.number_grid
            pair = lambda z, w: 2*float(np.vdot(z, w).real)
            grad[6] = pair(z3, self.mix_generator(w3, x[7]))
            grad[7] = pair(z3, 1j*(n*w3-self.mix(n*w2, *x[6:8])))
            z2 = self.mix(z3, -x[6], x[7]).reshape(d, d, -1)
            t1 = w1.reshape(d, d, -1)
            for j in (1, 2):
                grad[1+j] = pair(z2, np.einsum('ai,bj,ijr->abr', s0[j], s1[0], t1, optimize=True))
                grad[3+j] = pair(z2, np.einsum('ai,bj,ijr->abr', s0[0], s1[j], t1, optimize=True))
            z1 = np.einsum('ia,jb,abr->ijr', s0[0].conj().T, s1[0].conj().T, z2, optimize=True).reshape(d*d, -1)
            grad[0] = pair(z1, self.mix_generator(w1, x[1]))
            grad[1] = pair(z1, 1j*(n*w1-self.mix(n*w0, *x[:2])))
            return grad

    rmax = _r7_scalar(rmax, 0, .5)
    adaptive = _r7_bool(adaptive)
    raw = _r7_numeric(states)
    if raw.ndim != 3 or raw.shape[0] not in (1, 2):
        raise ValueError('one or two branch states required')
    d = _r7_dimension(raw.shape[1], 2)
    branches = len(raw)
    _oracle_policy_statistics(raw, target, np.zeros((branches, 8)), counts, efficiencies)
    target = _r7_target(target, d)
    counts = _r7_counts(counts, branches)
    efficiencies = _r7_numeric(efficiencies, (2,), True)
    pieces = [_r7_state_parts(rho, d*d) for rho in raw]
    mass_logs = np.array([math.log(mass) if mass > 0 else -np.inf for root, mass in pieces])
    if not np.any(np.isfinite(mass_logs)) or all(k >= d for k in counts):
        return 0., np.zeros((branches, 8)), np.zeros((branches, 2))
    full = [root for root, mass in pieces]
    compressed, omissions = [], []
    for w in full:
        weights = np.sum(np.abs(w)**2, axis=0)
        order = np.argsort(weights)[::-1]
        tail = np.cumsum(weights[order][::-1])[::-1]
        rank = max(1, int(np.count_nonzero(tail > 1e-14*weights.sum())))
        compressed.append(w[:, order[:rank]])
        omissions.append(float(weights[order[rank:]].sum()))
    adjoint = _r7_adjoint(d, compressed, target, counts, efficiencies)
    branch_logs = mass_logs+np.array(adjoint.effect_logs)

    def pool(moments, offsets):
        live = (moments[:, 0] > 0) & np.isfinite(branch_logs)
        if not np.any(live):
            return 0., 0., np.zeros(branches)
        log_contributions = branch_logs[live]+offsets[live]+np.log(moments[live, 0])
        shift = float(np.max(log_contributions))
        relative = np.exp(log_contributions-shift)
        total = float(relative.sum())
        fractions = relative/total
        probability = math.exp(shift+math.log(total))
        fidelity = float(np.dot(fractions, moments[live, 1]/moments[live, 0]))
        coefficients = np.zeros(branches)
        coefficients[live] = fractions/moments[live, 0]
        return probability, fidelity, coefficients

    block = [(0., np.pi/2), (-np.pi, np.pi), (0., rmax), (-np.pi, np.pi),
             (0., rmax), (-np.pi, np.pi), (0., np.pi/2), (-np.pi, np.pi)]
    active = [i for i in range(8) if block[i][1] > block[i][0] and not (rmax == 0 and i in (3, 5))]
    bounds = [block[i] for i in active]
    local = [(None, None) if i in (1, 3, 5, 7) else block[i] for i in active]

    def expand(z):
        result = np.zeros(8)
        result[active] = z
        return result

    rng = np.random.default_rng(716)
    candidates = [[] for k in range(branches)]
    starts_per_support = 16
    for k in range(branches):
        for lam in (.7, .85, .95, 1.02):
            def support(z):
                x = expand(z)
                pm, cache, offset = adjoint.forward(k, x)
                if cache[-1]:
                    # The support is below the support search's absolute
                    # tolerance. Joint refinement still evaluates its ratio.
                    return -100.*math.exp(offset)*float(pm[1]-lam*pm[0]), np.zeros(len(active))
                grad = adjoint.gradient(x, cache, -lam, 1.)
                return -100.*float(pm[1]-lam*pm[0]), -100.*grad[active]
            seeds = [np.zeros(len(active))]+[np.array(c['x'])[active] for c in candidates[k]]
            seeds += [np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for j in range(starts_per_support)]
            results = []
            for seed in seeds:
                fit = minimize(support, seed, method='L-BFGS-B', jac=True, bounds=local,
                               options={'maxiter':600, 'ftol':1e-13, 'gtol':1e-7})
                x = expand(fit.x)
                pm, _, offset = adjoint.forward(k, x)
                results.append({'support': -float(fit.fun), 'x': x, 'pm': pm, 'offset': offset})
            candidates[k].extend(sorted(results, key=lambda t: -t['support'])[:3])
    if adaptive and branches == 2:
        starts = []
        for first in candidates[0]:
            for second in candidates[1]:
                probability, fidelity, _ = pool(np.array([first['pm'], second['pm']]),
                                                  np.array([first['offset'], second['offset']]))
                reward = probability+fidelity
                starts.append((reward, np.r_[first['x'], second['x']]))
        starts = [x for value, x in sorted(starts, key=lambda t: -t[0])[:16]]
    else:
        starts = [c['x'] for row in candidates for c in row]
        starts += [expand(np.array([rng.uniform(lo, hi) for lo, hi in bounds])) for j in range(starts_per_support)]
    starts.append(np.zeros(8*branches if adaptive else 8))
    controllers = branches if adaptive else 1
    chosen = active if controllers == 1 else active+[i+8 for i in active]
    local_joint = local*controllers

    def inflate(z):
        return expand(z) if controllers == 1 else np.r_[expand(z[:len(active)]), expand(z[len(active):])]

    def evaluate(z, exact=False, need_gradient=True):
        x = inflate(z).reshape(controllers, 8)
        if not adaptive:
            x = np.repeat(x, branches, axis=0)
        try:
            adjoint.factors = full if exact else compressed
            forward = [adjoint.forward(k, x[k]) for k in range(branches)]
            moments = np.array([item[0] for item in forward])
            offsets = np.array([item[2] for item in forward])
            probability, fidelity, coefficients = pool(moments, offsets)
            # Bound PSD compression error in units of the accepted mass.
            # Any rare-event rescaling uses full factors before acceptance.
            omitted_ratio = float(np.dot(coefficients, omissions))
            if not exact and (np.any(moments[:, 0] == 0)
                              or any(item[1][-1] for item in forward)
                              or (probability+2)*omitted_ratio > 1e-10):
                adjoint.factors = full
                forward = [adjoint.forward(k, x[k]) for k in range(branches)]
                moments = np.array([item[0] for item in forward])
                offsets = np.array([item[2] for item in forward])
                probability, fidelity, coefficients = pool(moments, offsets)
            physical_moments = np.exp(branch_logs+offsets)[:, None]*moments
            if not np.any(coefficients):
                return 0., np.zeros_like(z), physical_moments
            reward = float(probability+fidelity)
            if not need_gradient:
                return reward, None, physical_moments
            if any(item[1][-1] for item in forward):
                # Finite differences of the stabilized scalar avoid a
                # reciprocal amplitude beyond binary64's exponent range.
                gradient = np.empty_like(z)
                for j in range(len(z)):
                    step = 1e-6
                    plus, minus = z.copy(), z.copy()
                    plus[j] += step
                    minus[j] -= step
                    gradient[j] = (evaluate(plus, True, False)[0]
                                   -evaluate(minus, True, False)[0])/(2*step)
                return reward, gradient, physical_moments
            # dR = dP + (dM-F*dP)/P; the common scale cancels before
            # differentiation. Neither physical P squared nor 1/P is formed.
            gradients = [adjoint.gradient(x[k], item[1],
                         (probability-fidelity)*coefficients[k], coefficients[k])
                         for k, item in enumerate(forward)]
            gradient = np.array(gradients).ravel() if adaptive else np.array(gradients).sum(axis=0)
            return reward, gradient[chosen], physical_moments
        finally:
            adjoint.factors = compressed

    def objective(z):
        value, gradient, _ = evaluate(z)
        return -value, -gradient

    results = []
    for x in starts:
        seed = x[chosen]
        value, _, moments = evaluate(seed, True, need_gradient=False)
        results.append((value, seed, moments))
        fit = minimize(objective, seed, method='L-BFGS-B', jac=True, bounds=local_joint,
                       options={'maxiter':1000, 'ftol':5e-14, 'gtol':1e-8, 'maxls':40})
        value, _, moments = evaluate(fit.x, True, need_gradient=False)
        results.append((value, fit.x, moments))
    value, best, best_moments = max(results, key=lambda t: t[0])
    x = inflate(best).reshape(controllers, 8)
    if not adaptive:
        x = np.repeat(x, branches, axis=0)
    return value, x, best_moments
def _oracle_optimal_reward(states: object, target: object, counts: object, efficiencies: object, rmax: float, adaptive: bool) -> float:
    if states is None:
        raise ValueError('states are required')
    return _r7_optimize(states, target, counts, efficiencies, rmax, adaptive)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([np.diag([0.0, 0.0, 0.28, 0.0])])\n'
           't = np.array([0.0, 1.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [0], [0.8, 0.9], 0.0, True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [0], [0.8, 0.9], 0.0, True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.zeros((2, 9, 9))\n'
           't = np.array([1.0, 0.0, 0.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [1, 2], [1.0, 1.0], 0.3, False))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [1, 2], [1.0, 1.0], 0.3, '
               'False))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([np.diag([-5e-11, 0.0, 1e-10, 0.0])])\n'
           't = np.array([0.0, 1.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [0], [1.0, 1.0], 0.0, True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [0], [1.0, 1.0], 0.0, True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([np.diag([0.0, 0.0, 5e-11, 0.0])])\n'
           't = np.array([0.0, 1.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [0], [1.0, 1.0], 0.0, True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [0], [1.0, 1.0], 0.0, True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'v = np.array([0.1, 0.2j, 0.1, 0.1j, -0.2, 0.1, 0.2, 0.1j, -0.1])\n'
           'w = np.array([0.2, 0.1, 0.1j, -0.1, 0.1j, 0.2, 0.1, -0.2j, 0.1])\n'
           's = np.array([np.outer(v, v.conj()), np.outer(w, w.conj())])\n'
           't = np.array([1.0, 1j, 0.5]) / 1.5\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [0, 1], [0.84, 0.93], 0.2, False))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [0, 1], [0.84, 0.93], 0.2, '
               'False))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'v = np.array([0.1, 0.2j, 0.1, 0.1j, -0.2, 0.1, 0.2, 0.1j, -0.1])\n'
           'w = np.array([0.2, 0.1, 0.1j, -0.1, 0.1j, 0.2, 0.1, -0.2j, 0.1])\n'
           's = np.array([np.outer(v, v.conj()), np.outer(w, w.conj())])\n'
           't = np.array([1.0, 1j, 0.5]) / 1.5\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    # A 1e-9 comparison here represents 0.0004 in the physical reward.\n'
           '    return np.array([float(value) * (1e-9 / 0.0004)])\n',
  'call': '_case_encode(_case_run(optimal_reward, s, t, [0, 1], [0.84, 0.93], 0.2, True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s, t, [0, 1], [0.84, 0.93], 0.2, '
               'True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],np.zeros((2,3,4,5)),.2,True))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],np.zeros((2,3,4,5)),.2,True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],np.full(np.shape([.8,.9]),np.nan),.2,True))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],np.full(np.shape([.8,.9]),np.nan),.2,True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],[.8,.9],-1e-06,True))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],[.8,.9],-1e-06,True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           '_case_bad=np.array([.8,.9],float)\n'
           '_case_bad[0]=-1e-6\n',
  'call': '_case_value_error(lambda: '
          'optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],_case_bad,.2,True))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],_case_bad,.2,True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          "optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],'0.5',.2,True))",
  'gold_call': '_case_value_error(lambda: '
               "_oracle_optimal_reward(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],[1],'0.5',.2,True))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(value)*(1e-9/0.0004)])\n'
           's=np.zeros((1,9,9)); s[0,5,5]=.5\n'
           't=np.array([0.,1.,0.])\n',
  'call': '_case_encode(_case_run(optimal_reward, s,t,[2],[1e-200,1.],0.,True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s,t,[2],[1e-200,1.],0.,True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(value)*(1e-9/0.0004)])\n'
           's=np.zeros((1,9,9)); s[0,4,4]=.5\n'
           't=np.array([0.,1.,0.])\n',
  'call': '_case_encode(_case_run(optimal_reward, s,t,[1],[1e-200,1.],0.,True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward, s,t,[1],[1e-200,1.],0.,True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n'
           's=np.zeros((1,9,9)); s[0,0,0]=.5; s[0,4,4]=np.nextafter(0.,1.)\n'
           't=np.array([0.,1.,0.])\n',
  'call': '_case_encode(_case_run(optimal_reward,s,t,[1],[1.,1.],0.,True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward,s,t,[1],[1.,1.],0.,True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n'
           'd=23; s=np.zeros((1,d*d,d*d)); s[0,2*d-1,2*d-1]=.5\n'
           't=np.zeros(d); t[1]=1.\n',
  'call': '_case_encode(_case_run(optimal_reward,s,t,[0],[np.nextafter(1.,0.),1.],0.,True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward,s,t,[0],[np.nextafter(1.,0.),1.],0.,True))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n'
           's=np.array([np.diag([-5e-11,1e-300,5e-11,0.])]); t=np.array([0.,1.])\n',
  'call': '_case_encode(_case_run(optimal_reward,s,t,[0],[1.,1.],0.,True))',
  'gold_call': '_case_encode(_case_run(_oracle_optimal_reward,s,t,[0],[1.,1.],0.,True))'}]
