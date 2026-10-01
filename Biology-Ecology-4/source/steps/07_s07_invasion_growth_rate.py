"""
Invasion growth rate of a species at the fixed point of a sub-community: the natural logarithm of the spectral radius of the species' projection kernel evaluated at the located boundary equilibrium in the limit of a vanishing population of that species (per-capita acquisition in the rare limit, consumers evaluated at their resident numbers, and no feedback of the rare species on the residents). For a species that belongs to the sub-community the same construction returns its resident rate, which vanishes at the fixed point. The step locates the equilibrium internally, validates the face, the species and the existence of the fixed point, and raises ValueError otherwise. It excludes the census and the permanence criterion.

For structured populations in discrete time the invasion growth rate of an absent species at a boundary equilibrium is the dominant Lyapunov exponent of its linearised dynamics there, which for a fixed point is the natural logarithm of the spectral radius of the invader's kernel evaluated in the rare limit: the invader's own number is taken to zero in every density-dependent term, its acquisition is the per-capita limit of the source's capture model, and the residents' numbers set the consumer pressure it experiences. A resident species, whose distribution is the Perron vector of its own kernel at the fixed point, has rate exactly zero, and the transverse multiplier of an absent species equals the exponential of its rate. Benchmark values at theta = 0.78: the small grazer invades the large-grazer-and-predator community at 0.0230030548 per year; the predator invades the two-grazer community at 0.1770012788 per year; at theta = 0.60 the predator's rate at the two-grazer community is -0.0853629857 per year; the resident small grazer's rate at the small-grazer-and-predator fixed point is 0 to within 1e-13.

Returns
-------
Returns the natural logarithm of the spectral radius of the species' kernel at the face's fixed point (per year) as a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invasion_growth_rate(theta, face, species, overrides=None):
    """Invasion growth rate (per year) of a species at a sub-community's fixed point.

    Parameters
    ----------
    theta : float
        Maximum annual intrinsic survival probability of the predator.
    face : sequence of int
        Indices of the species present in the sub-community.
    species : int
        Index of the species whose rate is requested (an absent species gives its invasion rate in the
        vanishing-density limit; a resident gives its resident rate, which is zero at the fixed point).
    overrides : dict, optional
        Benchmark parameter overrides, each key mapped to a float. Pool keys: 'D', 'phi_BD', 'phi_DB',
        'rho_B', 'phi_DP', 'phi_GD', 'phi_PG', 'rho_G', 'gam3' (gamma_3). Species keys 'species.<i>.<name>'
        for species i, with <name> one of bs0, bs1, bs2 (beta_sn0..2), bb0, bb1, bb2 (beta_b0..2),
        a1, a2 (alpha_1, alpha_2), g1, g2 (gamma_1, gamma_2), sig_g (sigma_g), nu (nu_o), mu_o,
        sig_o (sigma_o), delta, b_dh (beta_dh0). Kill keys 'kills.<i><j>.<k>' set parameter k = 0, 1, 2
        (beta_dp0..2) of the kill sigmoid of consumer j on prey i, e.g. 'kills.02.0'. A key that is
        neither a pool key nor of the species or kills form raises ValueError.


    Returns
    -------
    float
        Natural logarithm of the spectral radius of the species' kernel at the fixed point (per year).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

# ---------------------------------------------------------------------------
# Frozen benchmark configuration (species indices: 0 = H1 small grazer,
# 1 = H2 large grazer, 2 = W predator).  Masses in kg, time in years.
# ---------------------------------------------------------------------------

def _s07_config(theta=0.78, overrides=None):
    cfg = {
        'D': 2.0e8, 'phi_BD': 0.25, 'phi_DB': 0.2, 'rho_B': 54.75, 'phi_DP': 0.175,
        'phi_GD': 0.033, 'phi_PG': 0.0032, 'rho_G': 1.3, 'gam3': 13.3,
        'species': [
            dict(m=50, zmax=500.0, bs0=0.96, bs1=7.9, bs2=0.025, a1=0.60, a2=0.075, g1=2500.0, g2=0.5,
                 sig_g=5.0, bb0=0.91, bb1=180.0, bb2=0.08, nu=0.05, mu_o=15.0, sig_o=3.0, delta=1.0,
                 lam2=1.0, lam4=0.0, b_dh=0.0002),
            dict(m=50, zmax=1000.0, bs0=0.95, bs1=-200.0, bs2=0.005, a1=0.62, a2=0.085, g1=4850.0, g2=0.5,
                 sig_g=10.0, bb0=0.8, bb1=400.0, bb2=0.075, nu=0.045, mu_o=22.5, sig_o=3.0, delta=1.25,
                 lam2=1.0, lam4=0.0, b_dh=0.0003),
            dict(m=50, zmax=100.0, bs0=float(theta), bs1=-40.0, bs2=0.025, a1=0.75, a2=0.08, g1=800.0, g2=0.68,
                 sig_g=3.0, bb0=0.58, bb1=40.0, bb2=0.2, nu=0.15, mu_o=2.0, sig_o=1.0, delta=1.5,
                 lam2=0.0, lam4=1.0, b_dh=0.0),
        ],
        # per-encounter kill probability of prey i by one consumer of species j:
        # p(z) = p0 * sigmoid((z - p1) * p2)
        'kills': {(0, 2): [0.006, 200.0, -0.025], (1, 2): [0.004, 350.0, -0.05], (2, 2): [0.0012, 16.0, -0.1]},
    }
    if overrides:
        for key, val in overrides.items():
            parts = key.split('.')
            if parts[0] == 'species':
                cfg['species'][int(parts[1])][parts[2]] = float(val)
            elif parts[0] == 'kills':
                i, j = int(parts[1][0]), int(parts[1][1])
                cfg['kills'][(i, j)][int(parts[2])] = float(val)
            elif parts[0] in cfg:
                cfg[parts[0]] = float(val)
            else:
                raise ValueError('unknown override key: %s' % key)
    return cfg

# ---------------------------------------------------------------------------
# Elementary functions
# ---------------------------------------------------------------------------
def _s07_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _s07_hh(x, y):
    """Half-harmonic (Michaelis-Menten) function h(x, y) = x y / (x + y)."""
    return x * y / (x + y)

def _s07_mesh(m, zmax):
    dz = zmax / m
    z = (np.arange(m) + 0.5) * dz
    return z, dz

def _s07_norm_columns(E, dz):
    """Column-normalised kernel from log-density E (rows: destination mass, cols: origin)."""
    E = E - E.max(axis=0)[None, :]
    K = np.exp(E)
    return K / (K.sum(axis=0) * dz)[None, :]

def _s07_recruit_column(z, dz, mu_o, sig_o):
    E = -0.5 * ((z - mu_o) / sig_o) ** 2
    return _s07_norm_columns(E[:, None], dz)[:, 0]

def _s07_growth_kernel(z, dz, Zg, sig_g):
    E = -0.5 * ((z[:, None] - Zg[None, :]) / sig_g) ** 2
    return _s07_norm_columns(E, dz)

def _s07_survival(sp, z):
    return sp['bs0'] * _s07_sigmoid((z - sp['bs1']) * sp['bs2'])

def _s07_reproduction(sp, z):
    return sp['bb0'] * _s07_sigmoid((z - sp['bb1']) * sp['bb2'])

def _s07_kill_prob(kp, z):
    return kp[0] * _s07_sigmoid((z - kp[1]) * kp[2])

def _s07_B_star(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return a * ((1.0 - cfg['phi_DP']) * cfg['rho_B'] / cfg['phi_BD'] - 1.0) / cfg['rho_B']

def _s07_nutrient_input(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return cfg['phi_DP'] * _s07_hh(a, cfg['rho_B'] * _s07_B_star(cfg))

# ---------------------------------------------------------------------------
# Step-1 core: grass removed by grazer j (population level or rare-invader per-capita limit)
# ---------------------------------------------------------------------------
def _s07_grass_removed(G, counts, betas, phi_GD, j, per_capita_limit=False):
    counts = np.asarray(counts, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if G < 0 or np.any(counts < 0) or np.any(betas < 0) or np.any(betas >= 1) or not (0 <= phi_GD < 1):
        raise ValueError('invalid grazing inputs')
    if betas[j] <= 0:
        return 0.0
    survive_intrinsic = 1.0 - phi_GD
    if per_capita_limit:
        res = [k for k in range(len(betas)) if k != j and counts[k] > 0 and betas[k] > 0]
        cj = -np.log(1.0 - betas[j])
        if not res:
            return survive_intrinsic * G * cj
        pj = 1.0 - (1.0 - betas[res]) ** counts[res]     # per-species capture probabilities of residents
        ph = 1.0 - np.prod((1.0 - betas[res]) ** counts[res])
        return survive_intrinsic * ph * G * cj / pj.sum()
    active = [k for k in range(len(betas)) if counts[k] > 0 and betas[k] > 0]
    if j not in active:
        return 0.0
    pj = 1.0 - (1.0 - betas[active]) ** counts[active]
    ph = 1.0 - np.prod((1.0 - betas[active]) ** counts[active])
    share = (1.0 - (1.0 - betas[j]) ** counts[j]) / pj.sum()
    return survive_intrinsic * ph * G * share

# ---------------------------------------------------------------------------
# Step-2 core: biomass of prey killed by a (single) consumer species
# ---------------------------------------------------------------------------
def _s07_kill_flux(z, dz, N_prey, p_sn_prey, kill_params, n_pred, per_capita_limit=False):
    """Biomass of prey removed by n_pred consumers; if per_capita_limit, the per-consumer
    flux in the limit of a vanishing consumer population (n_pred ignored)."""
    pd1 = _s07_kill_prob(kill_params, z)
    if per_capita_limit:
        pk = -np.log(1.0 - pd1)
    else:
        pk = 1.0 - (1.0 - pd1) ** n_pred
    return float(np.sum(z * p_sn_prey * pk * N_prey) * dz)

# ---------------------------------------------------------------------------
# Step-3 core: assimilated mass per individual per year
# ---------------------------------------------------------------------------
def _s07_assimilated(Zc, N, a1, a2, g1, g2):
    """Population-level saturating assimilation.  For N > 0, Zc is the total acquired mass of
    the population.  For N == 0 the argument Zc is the per-capita acquired mass of a
    vanishingly rare population (the invasion limit)."""
    if Zc < 0 or N < 0 or a1 < 0 or a2 < 0 or g1 <= 0 or g2 <= 0:
        raise ValueError('invalid assimilation inputs')
    if N == 0:
        if Zc == 0:
            return 0.0
        return a1 * a2 * g1 * g2 * Zc / (g1 + g2 * Zc)
    den = g1 * N + g2 * Zc
    if den == 0:
        return 0.0
    return a1 * a2 * g1 * g2 * Zc / den

# ---------------------------------------------------------------------------
# Step-4 core: projection matrix of one structured species in a given environment
# ---------------------------------------------------------------------------
def _s07_projection_matrix(sp, z, dz, Za, consumers):
    """consumers: list of (count, kill_params) acting on this species."""
    p_s = _s07_survival(sp, z)
    for cnt, kp in consumers:
        if cnt > 0:
            p_s = p_s * (1.0 - _s07_kill_prob(kp, z)) ** cnt
    pb = _s07_reproduction(sp, z)
    n_o = sp['nu'] * sp['bb1'] / sp['mu_o']
    Zr = pb * n_o * sp['mu_o']
    Zm = sp['delta'] * z ** 0.75
    Zg = z + Za - Zr - Zm
    Fg = _s07_growth_kernel(z, dz, Zg, sp['sig_g'])
    Fr = _s07_recruit_column(z, dz, sp['mu_o'], sp['sig_o'])
    return (np.outer(Fr, pb * n_o) + Fg) * p_s[None, :] * dz

def _s07_log_spectral_radius(K):
    w = np.linalg.eigvals(K)
    return float(np.log(np.max(np.abs(w))))

def _s07_perron(K):
    w, V = np.linalg.eig(K)
    k = int(np.argmax(w.real))
    v = np.abs(V[:, k].real)
    return float(w[k].real), v

# ---------------------------------------------------------------------------
# Full ecosystem state and annual map
# ---------------------------------------------------------------------------
class _s07_Eco:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ns = len(cfg['species'])
        self.mesh = [_s07_mesh(sp['m'], sp['zmax']) for sp in cfg['species']]
        self.off = [3]
        for sp in cfg['species']:
            self.off.append(self.off[-1] + sp['m'])
        self.Bs = _s07_B_star(cfg)
        self.I = _s07_nutrient_input(cfg)
        self.grazers = [i for i, sp in enumerate(cfg['species']) if sp['b_dh'] > 0]
        self.preds = [i for i, sp in enumerate(cfg['species']) if sp['lam4'] > 0]

    def split(self, x):
        Ns = [x[self.off[i]:self.off[i + 1]] for i in range(self.ns)]
        return x[0], x[1], x[2], Ns

    def join(self, B, Pn, G, Ns):
        return np.concatenate([[B, Pn, G]] + list(Ns))

    def counts(self, Ns):
        return np.array([Ns[i].sum() * self.mesh[i][1] for i in range(self.ns)])

    def gaussian_state(self, B, Pn, G, spec):
        """spec: list of (count, mean, sd) per species (count 0 -> absent)."""
        Ns = []
        for i, (n0, mu, sd) in enumerate(spec):
            z, dz = self.mesh[i]
            if n0 > 0:
                d = np.exp(-0.5 * ((z - mu) / sd) ** 2)
                d = d / (d.sum() * dz) * n0
            else:
                d = np.zeros(len(z))
            Ns.append(d)
        return self.join(B, Pn, G, Ns)

    def environment(self, x, invader=None, self_flux=None):
        """Per-species environment: Za and consumer list.  If invader is given, that species is
        treated in the vanishing-density limit (its count set to zero, per-capita acquisition).
        self_flux: optional {consumer: value} replacing the computed intraspecific kill flux
        (used by the reduced fixed-point solver)."""
        cfg = self.cfg
        B, Pn, G, Ns = self.split(x)
        cnt = self.counts(Ns)
        if invader is not None:
            cnt = cnt.copy(); cnt[invader] = 0.0
        betas = np.array([sp['b_dh'] for sp in cfg['species']])
        env = []
        for i, sp in enumerate(cfg['species']):
            z, dz = self.mesh[i]
            rare = (invader == i)
            # acquired mass: grazing + kills
            Zc = 0.0
            if sp['lam2'] > 0:
                Zc += sp['lam2'] * _s07_grass_removed(G, cnt, betas, cfg['phi_GD'], i, per_capita_limit=rare)
            if sp['lam4'] > 0:
                for (prey, pred), kp in cfg['kills'].items():
                    if pred != i:
                        continue
                    if cnt[prey] <= 0 and not (rare and prey == i):
                        continue
                    if prey == i and rare:
                        continue          # a vanishing population does not feed on itself
                    if prey == i and self_flux is not None and i in self_flux:
                        Zc += sp['lam4'] * self_flux[i]
                        continue
                    zp, dzp = self.mesh[prey]
                    # total kill probability of prey by all its consumers, and this consumer's share
                    p_sn_prey = _s07_survival(cfg['species'][prey], zp)
                    if rare:
                        flux = _s07_kill_flux(zp, dzp, Ns[prey], p_sn_prey, kp, 0.0, per_capita_limit=True)
                        # partition: rare consumer's share of the total removal (limit form)
                        res = [(q, kq) for (pq, q), kq in cfg['kills'].items() if pq == prey and q != i and cnt[q] > 0]
                        if res:
                            pdp_tot = 1.0 - np.prod([(1.0 - _s07_kill_prob(kq, zp)) ** cnt[q] for q, kq in res], axis=0)
                            sum_res = np.sum([1.0 - (1.0 - _s07_kill_prob(kq, zp)) ** cnt[q] for q, kq in res], axis=0)
                            pk = -np.log(1.0 - _s07_kill_prob(kp, zp))
                            flux = float(np.sum(zp * p_sn_prey * pdp_tot * pk / sum_res * Ns[prey]) * dzp)
                    else:
                        res = [(q, kq) for (pq, q), kq in cfg['kills'].items() if pq == prey and cnt[q] > 0 and q != invader]
                        pdp_tot = 1.0 - np.prod([(1.0 - _s07_kill_prob(kq, zp)) ** cnt[q] for q, kq in res], axis=0)
                        sum_res = np.sum([1.0 - (1.0 - _s07_kill_prob(kq, zp)) ** cnt[q] for q, kq in res], axis=0)
                        pk = 1.0 - (1.0 - _s07_kill_prob(kp, zp)) ** cnt[i]
                        with np.errstate(invalid='ignore', divide='ignore'):
                            share = np.where(sum_res > 0, pdp_tot * pk / sum_res, 0.0)
                        flux = float(np.sum(zp * p_sn_prey * share * Ns[prey]) * dzp)
                    Zc += sp['lam4'] * flux
            Za = _s07_assimilated(Zc, 0.0 if rare else cnt[i], sp['a1'], sp['a2'], sp['g1'], sp['g2'])
            consumers = [(cnt[pred], kp) for (prey, pred), kp in cfg['kills'].items()
                         if prey == i and pred != invader and cnt[pred] > 0]
            env.append((Za, consumers))
        return env, cnt

    def kernels(self, x, invader=None, self_flux=None):
        env, cnt = self.environment(x, invader, self_flux)
        Ks = []
        for i, sp in enumerate(self.cfg['species']):
            z, dz = self.mesh[i]
            Za, consumers = env[i]
            Ks.append(_s07_projection_matrix(sp, z, dz, Za, consumers))
        return Ks, env, cnt

    def self_kill_flux(self, x, i):
        """Biomass of species i killed by its own population (intraspecific predation)."""
        kp = self.cfg['kills'].get((i, i))
        if kp is None:
            return 0.0
        B, Pn, G, Ns = self.split(x)
        cnt = self.counts(Ns)
        z, dz = self.mesh[i]
        others = [(q, kq) for (pq, q), kq in self.cfg['kills'].items() if pq == i and q != i and cnt[q] > 0]
        pk = 1.0 - (1.0 - _s07_kill_prob(kp, z)) ** cnt[i]
        if not others:
            share = pk
        else:
            allc = [(i, kp)] + others
            pdp_tot = 1.0 - np.prod([(1.0 - _s07_kill_prob(kq, z)) ** cnt[q] for q, kq in allc], axis=0)
            sum_res = np.sum([1.0 - (1.0 - _s07_kill_prob(kq, z)) ** cnt[q] for q, kq in allc], axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                share = np.where(sum_res > 0, pdp_tot * pk / sum_res, 0.0)
        return float(np.sum(z * _s07_survival(self.cfg['species'][i], z) * share * Ns[i]) * dz)

    def grazing_total(self, x):
        cfg = self.cfg
        B, Pn, G, Ns = self.split(x)
        cnt = self.counts(Ns)
        betas = np.array([sp['b_dh'] for sp in cfg['species']])
        return sum(_s07_grass_removed(G, cnt, betas, cfg['phi_GD'], j) for j in self.grazers)

    def step(self, x):
        cfg = self.cfg
        B, Pn, G, Ns = self.split(x)
        Ks, env, cnt = self.kernels(x)
        hDB = _s07_hh(cfg['phi_DB'] * cfg['D'], cfg['rho_B'] * B)
        hPG = _s07_hh(cfg['phi_PG'] * Pn, cfg['rho_G'] * G)
        Bn = B + (1.0 - cfg['phi_DP']) * hDB - cfg['phi_BD'] * B
        Pnn = Pn + cfg['phi_DP'] * hDB - hPG
        Gn = G + cfg['gam3'] * hPG - cfg['phi_GD'] * G - self.grazing_total(x)
        Nn = [Ks[i] @ Ns[i] for i in range(self.ns)]
        return self.join(Bn, Pnn, Gn, Nn)

    def residual(self, x):
        F = self.step(x)
        return float(np.max(np.abs(F - x) / (1.0 + np.abs(x))))

    # ---- face machinery -------------------------------------------------
    def face_mask(self, face):
        mask = np.ones(self.off[-1], dtype=bool)
        for i in range(self.ns):
            if i not in face:
                mask[self.off[i]:self.off[i + 1]] = False
        return mask

    def step_face(self, y, face):
        mask = self.face_mask(face)
        x = np.zeros(self.off[-1]); x[mask] = y
        return self.step(x)[mask]

    def embed(self, y, face):
        mask = self.face_mask(face)
        x = np.zeros(self.off[-1]); x[mask] = y
        return x

    def assemble(self, u, face):
        """Reduced unknowns u = (P, G, counts of the face members, intraspecific kill flux of each
        consumer member) -> full state whose structured distributions are the Perron vectors of the
        members' kernels, together with the members' dominant eigenvalues and the recomputed
        intraspecific kill fluxes."""
        Pn, G = u[0], u[1]
        cnt = np.zeros(self.ns)
        for k, i in enumerate(face):
            cnt[i] = u[2 + k]
        preds = [i for i in face if i in self.preds]
        self_flux = {i: u[2 + len(face) + k] for k, i in enumerate(preds)}
        Ns = [np.zeros(sp['m']) for sp in self.cfg['species']]
        for i in face:
            z, dz = self.mesh[i]
            Ns[i] = np.ones(len(z)) * cnt[i] / (len(z) * dz)
        lam = {}
        x = self.join(self.Bs, Pn, G, Ns)
        Ks, env, c = self.kernels(x, self_flux=self_flux)
        for i in face:
            if i in self.grazers:
                w, v = _s07_perron(Ks[i])
                z, dz = self.mesh[i]
                Ns[i] = v / (v.sum() * dz) * cnt[i]
                lam[i] = w
        x = self.join(self.Bs, Pn, G, Ns)
        Ks, env, c = self.kernels(x, self_flux=self_flux)
        for i in preds:
            w, v = _s07_perron(Ks[i])
            z, dz = self.mesh[i]
            Ns[i] = v / (v.sum() * dz) * cnt[i]
            lam[i] = w
        x = self.join(self.Bs, Pn, G, Ns)
        flux = {i: self.self_kill_flux(x, i) for i in preds}
        return x, lam, self_flux, flux

    def reduced_residual(self, u, face):
        cfg = self.cfg
        x, lam, self_flux, flux = self.assemble(u, face)
        Pn, G = u[0], u[1]
        hPG = _s07_hh(cfg['phi_PG'] * Pn, cfg['rho_G'] * G)
        r = [(self.I - hPG) / self.I,
             (cfg['gam3'] * hPG - cfg['phi_GD'] * G - self.grazing_total(x)) / (cfg['gam3'] * self.I)]
        for i in face:
            r.append(lam[i] - 1.0)
        for i in face:
            if i in self.preds:
                r.append((self_flux[i] - flux[i]) / (1.0 + abs(flux[i])))
        return np.array(r)

    def newton_reduced(self, u0, face, tol=1e-13, maxit=40):
        """Damped Newton iteration on the reduced fixed-point system (finite-difference Jacobian)."""
        u = np.array(u0, dtype=float)
        n = len(u)
        npos = 2 + len(face)
        self.n_res = 0
        for it in range(maxit):
            F = self.reduced_residual(u, face); self.n_res += 1
            if np.max(np.abs(F)) < tol:
                return u, True
            J = np.zeros((n, n))
            scale = np.maximum(1.0, np.abs(u))
            for k in range(n):
                hk = 1e-7 * scale[k]
                up = u.copy(); up[k] += hk
                J[:, k] = (self.reduced_residual(up, face) - F) / hk; self.n_res += 1
            try:
                du = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return u, False
            lam_ = 1.0
            while np.any((u + lam_ * du)[:npos] <= 0) and lam_ > 1e-6:
                lam_ *= 0.5
            if np.any((u + lam_ * du)[:npos] <= 0) or not np.all(np.isfinite(du)):
                return u, False
            u = u + lam_ * du
        F = self.reduced_residual(u, face)
        return u, bool(np.max(np.abs(F)) < tol)

    def newton_full(self, x, face, tol=1e-13, maxit=4):
        mask = self.face_mask(face)
        y = x[mask]
        n = len(y)
        for it in range(maxit):
            F = self.step_face(y, face) - y
            if np.max(np.abs(F) / (1.0 + np.abs(y))) < tol:
                break
            J = np.zeros((n, n))
            for k in range(n):
                hk = 1e-6 * max(1.0, abs(y[k]))
                yp = y.copy(); yp[k] += hk
                J[:, k] = (self.step_face(yp, face) - yp - F) / hk
            y = y + np.linalg.solve(J, -F)
        return self.embed(y, face)

    def empty_equilibrium(self):
        cfg = self.cfg
        Gs = cfg['gam3'] * self.I / cfg['phi_GD']
        Ps = self.I * cfg['rho_G'] * Gs / (cfg['phi_PG'] * (cfg['rho_G'] * Gs - self.I))
        return self.join(self.Bs, Ps, Gs, [np.zeros(sp['m']) for sp in cfg['species']])

    def face_equilibrium(self, face, base=None):
        """Locate the fixed point of the annual map restricted to `face` (tuple of species indices,
        sorted).  Returns the full state, or None when the face carries no feasible fixed point."""
        face = tuple(sorted(face))
        if len(face) == 0:
            return self.empty_equilibrium()
        herb = tuple(i for i in face if i in self.grazers)
        pred = tuple(i for i in face if i in self.preds)
        if base is None:
            base = self.face_equilibrium(herb) if herb != face else self.empty_equilibrium()
            if base is None:
                return None
        B, Pn, G, Ns = self.split(base)
        cnt = self.counts(Ns)

        def candidates():
            if pred:
                # consumer face: seed the consumer(s) on the resource sub-face and iterate the
                # face map for a while before Newton.  When no consumer can invade the resource
                # sub-face and the seeded consumers decline, the face carries no fixed point.
                declining = all(self.invasion_rate(base, i) < 0 for i in pred)
                trials = ((60, 2.0),) if declining else ((120, 2.0), (400, 0.5), (60, 40.0))
                for T, seed in trials:
                    Ns2 = [n.copy() for n in Ns]
                    for i in pred:
                        z, dz = self.mesh[i]
                        sp = self.cfg['species'][i]
                        d = np.exp(-0.5 * ((z - sp['bb1']) / (0.25 * sp['bb1'])) ** 2)
                        Ns2[i] = d / (d.sum() * dz) * seed
                    y = self.join(B, Pn, G, Ns2)[self.face_mask(face)]
                    for _ in range(T):
                        y = self.step_face(y, face)
                    xe = self.embed(y, face)
                    c = self.counts(self.split(xe)[3])
                    if declining and all(c[i] < seed for i in pred):
                        return
                    if all(c[i] > 1e-4 for i in face):
                        yield ([self.split(xe)[1], self.split(xe)[2]] + [c[i] for i in face]
                               + [self.self_kill_flux(xe, i) for i in pred])
            else:
                for fh in (1.0, 0.5, 0.25, 2.0):
                    yield [Pn, G] + [max(cnt[i], 1.0) * fh if cnt[i] > 0 else 5000.0 * fh for i in face]

        for u0 in candidates():
            try:
                u, ok = self.newton_reduced(u0, face)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                ok = False
            if not ok:
                continue
            x, lam, sf, fl = self.assemble(u, face)
            if x[1] <= 0 or x[2] <= 0 or any(self.counts(self.split(x)[3])[i] <= 0 for i in face):
                continue
            if self.residual(x) > 1e-13:
                x = self.newton_full(x, face)
            if self.residual(x) <= 1e-12:
                return x
        return None

    def invasion_rate(self, x, i):
        Ks, env, cnt = self.kernels(x, invader=i)
        return _s07_log_spectral_radius(Ks[i])

    def resident_rate(self, x, i):
        Ks, env, cnt = self.kernels(x)
        return _s07_log_spectral_radius(Ks[i])

    def all_faces(self):
        from itertools import combinations
        out = []
        for r in range(self.ns):
            for c in combinations(range(self.ns), r):
                out.append(tuple(c))
        return out

    def census(self):
        """Boundary census: {face: state} for every proper sub-community carrying a fixed point."""
        found = {(): self.empty_equilibrium()}
        for face in self.all_faces():
            if len(face) == 0:
                continue
            herb = tuple(i for i in face if i in self.grazers)
            base = found[()] if herb == face else found.get(herb)
            if base is None:
                continue          # a consumer face cannot exist without its resource sub-face
            x = self.face_equilibrium(face, base=base)
            if x is not None:
                found[face] = x
        return found

    def margin(self):
        """Hofbauer-Schreiber margin: min over census vertices of max over absent species of the
        invasion growth rate; raises if the invasion graph is cyclic."""
        cen = self.census()
        rates = {}
        for face, x in cen.items():
            rates[face] = {i: self.invasion_rate(x, i) for i in range(self.ns) if i not in face}
        # invasion graph on the census vertices (plus the full community as sink)
        verts = list(cen.keys())
        edges = {S: [] for S in verts}
        for S in verts:
            for T in verts:
                if S == T:
                    continue
                if all(rates[S][i] > 0 for i in T if i not in S) and all(rates[T][i] < 0 for i in S if i not in T):
                    edges[S].append(T)
        if _s07_has_cycle(verts, edges):
            raise ValueError('cyclic invasion graph: criterion not applicable')
        m = min(max(rates[S].values()) for S in verts)
        return m, cen, rates

def _s07_has_cycle(verts, edges):
    state = {v: 0 for v in verts}
    def visit(v):
        state[v] = 1
        for w in edges[v]:
            if state[w] == 1:
                return True
            if state[w] == 0 and visit(w):
                return True
        state[v] = 2
        return False
    return any(state[v] == 0 and visit(v) for v in verts)


def _oracle_invasion_growth_rate(theta, face, species, overrides=None):
    cfg = _s07_config(theta, overrides)
    eco = _s07_Eco(cfg)
    face = tuple(sorted(int(i) for i in face))
    species = int(species)
    if any(i not in (0, 1, 2) for i in face) or species not in (0, 1, 2):
        raise ValueError('invalid face or species')
    x = eco.face_equilibrium(face)
    if x is None:
        raise ValueError('the face carries no feasible fixed point')
    if species in face:
        return eco.resident_rate(x, species)
    return eco.invasion_rate(x, species)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "",
         "call": "invasion_growth_rate(0.78, [1, 2], 0)",
         "gold_call": "_oracle_invasion_growth_rate(0.78, [1, 2], 0)"},
        {"setup": "",
         "call": "invasion_growth_rate(0.78, [0, 1], 2)",
         "gold_call": "_oracle_invasion_growth_rate(0.78, [0, 1], 2)"},
        {"setup": "",
         "call": "invasion_growth_rate(0.60, [0, 1], 2)",
         "gold_call": "_oracle_invasion_growth_rate(0.60, [0, 1], 2)"},
        {"setup": "",
         "call": "float(abs(invasion_growth_rate(0.78, [0, 2], 0)) < 1e-10)",
         "gold_call": "float(abs(_oracle_invasion_growth_rate(0.78, [0, 2], 0)) < 1e-10)"},
        {"setup": "",
         "call": "invasion_growth_rate(0.78, [0, 2], 1, overrides={'kills.12.0': 0.006})",
         "gold_call": "_oracle_invasion_growth_rate(0.78, [0, 2], 1, overrides={'kills.12.0': 0.006})"},
    ]
