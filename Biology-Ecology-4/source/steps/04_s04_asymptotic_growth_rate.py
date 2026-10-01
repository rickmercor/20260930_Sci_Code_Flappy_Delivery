"""
Asymptotic annual growth rate (natural logarithm of the dominant eigenvalue) of one structured species in a fixed environment, from the source's projection kernel assembled on the benchmark mesh: intrinsic survival, survival of encounters with a given number of predators, reproduction probability and offspring number, bioenergetic allocation of a given per-capita assimilated mass to maintenance, reproduction and growth, and the column-normalised recruitment and growth kernels. It validates the species index and raises ValueError on an unknown species. It excludes the feedback from the species onto its environment.

The source projects the body-mass density of a structured species through a kernel in which individuals first survive intrinsic and consumer-related mortality and then grow and reproduce (a post-breeding census): the density at mass z one year later is the integral over z' of [f_r(z|z') p_b(z') n_o + f_g(z|z')] p_s(z') N(z'). Survival p_s is the product of a sigmoid intrinsic survival beta_{s,n,0} sigmoid((z - beta_{s,n,1}) beta_{s,n,2}) and, for each consumer species, (1 - p(z))^{N_consumers}. The reproduction probability is the sigmoid beta_{b,0} sigmoid((z - beta_{b,1}) beta_{b,2}), the offspring number is n_o = nu beta_{b,1} / mu_o, the reproductive investment is Z_r = p_b n_o mu_o, maintenance costs Z_m = delta z^{3/4}, and the expected mass after growth is z + Za - Z_r - Z_m with Za the assimilated mass per individual. Recruits are drawn from a normal density of mean mu_o and standard deviation sigma_o, growth from a normal density centred on the expected mass with standard deviation sigma_g; both densities are evaluated at the mesh midpoints and each column rescaled to unit midpoint-rule mass. The asymptotic growth rate is the natural logarithm of the spectral radius of the resulting matrix (which includes the quadrature weight dz). Benchmark values: the small grazer with Za = 80 kg per year facing 46.154855 predators grows at -0.0052364067 per year; the predator with Za = 30 kg per year and 46 conspecific cannibals at 0.0091832863 per year; with reproduction switched off, no maintenance, no assimilation and a mass-independent intrinsic survival of 0.48, the rate is exactly ln(0.48) = -0.7339691751 because the growth kernel is column-stochastic.

Returns
-------
Returns the natural logarithm of the spectral radius of the species' annual projection matrix (per year) as a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def asymptotic_growth_rate(species, Za, consumer_count, overrides=None):
    """Natural logarithm of the dominant eigenvalue of a species' annual projection matrix.

    Parameters
    ----------
    species : int
        Species index (0 small grazer, 1 large grazer, 2 predator).
    Za : float
        Assimilated mass per individual per year (kg) in the environment considered.
    consumer_count : float
        Number of predators acting on this species with the frozen kill parameters of the pair
        (for species 2 this is the number of conspecific cannibals).
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
        Asymptotic annual growth rate (per year).
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

def _s04_config(theta=0.78, overrides=None):
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
def _s04_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _s04_hh(x, y):
    """Half-harmonic (Michaelis-Menten) function h(x, y) = x y / (x + y)."""
    return x * y / (x + y)

def _s04_mesh(m, zmax):
    dz = zmax / m
    z = (np.arange(m) + 0.5) * dz
    return z, dz

def _s04_norm_columns(E, dz):
    """Column-normalised kernel from log-density E (rows: destination mass, cols: origin)."""
    E = E - E.max(axis=0)[None, :]
    K = np.exp(E)
    return K / (K.sum(axis=0) * dz)[None, :]

def _s04_recruit_column(z, dz, mu_o, sig_o):
    E = -0.5 * ((z - mu_o) / sig_o) ** 2
    return _s04_norm_columns(E[:, None], dz)[:, 0]

def _s04_growth_kernel(z, dz, Zg, sig_g):
    E = -0.5 * ((z[:, None] - Zg[None, :]) / sig_g) ** 2
    return _s04_norm_columns(E, dz)

def _s04_survival(sp, z):
    return sp['bs0'] * _s04_sigmoid((z - sp['bs1']) * sp['bs2'])

def _s04_reproduction(sp, z):
    return sp['bb0'] * _s04_sigmoid((z - sp['bb1']) * sp['bb2'])

def _s04_kill_prob(kp, z):
    return kp[0] * _s04_sigmoid((z - kp[1]) * kp[2])

def _s04_B_star(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return a * ((1.0 - cfg['phi_DP']) * cfg['rho_B'] / cfg['phi_BD'] - 1.0) / cfg['rho_B']

def _s04_nutrient_input(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return cfg['phi_DP'] * _s04_hh(a, cfg['rho_B'] * _s04_B_star(cfg))

# ---------------------------------------------------------------------------
# Step-1 core: grass removed by grazer j (population level or rare-invader per-capita limit)
# ---------------------------------------------------------------------------
def _s04_grass_removed(G, counts, betas, phi_GD, j, per_capita_limit=False):
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
def _s04_kill_flux(z, dz, N_prey, p_sn_prey, kill_params, n_pred, per_capita_limit=False):
    """Biomass of prey removed by n_pred consumers; if per_capita_limit, the per-consumer
    flux in the limit of a vanishing consumer population (n_pred ignored)."""
    pd1 = _s04_kill_prob(kill_params, z)
    if per_capita_limit:
        pk = -np.log(1.0 - pd1)
    else:
        pk = 1.0 - (1.0 - pd1) ** n_pred
    return float(np.sum(z * p_sn_prey * pk * N_prey) * dz)

# ---------------------------------------------------------------------------
# Step-3 core: assimilated mass per individual per year
# ---------------------------------------------------------------------------
def _s04_assimilated(Zc, N, a1, a2, g1, g2):
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
def _s04_projection_matrix(sp, z, dz, Za, consumers):
    """consumers: list of (count, kill_params) acting on this species."""
    p_s = _s04_survival(sp, z)
    for cnt, kp in consumers:
        if cnt > 0:
            p_s = p_s * (1.0 - _s04_kill_prob(kp, z)) ** cnt
    pb = _s04_reproduction(sp, z)
    n_o = sp['nu'] * sp['bb1'] / sp['mu_o']
    Zr = pb * n_o * sp['mu_o']
    Zm = sp['delta'] * z ** 0.75
    Zg = z + Za - Zr - Zm
    Fg = _s04_growth_kernel(z, dz, Zg, sp['sig_g'])
    Fr = _s04_recruit_column(z, dz, sp['mu_o'], sp['sig_o'])
    return (np.outer(Fr, pb * n_o) + Fg) * p_s[None, :] * dz

def _s04_log_spectral_radius(K):
    w = np.linalg.eigvals(K)
    return float(np.log(np.max(np.abs(w))))

def _s04_perron(K):
    w, V = np.linalg.eig(K)
    k = int(np.argmax(w.real))
    v = np.abs(V[:, k].real)
    return float(w[k].real), v


def _oracle_asymptotic_growth_rate(species, Za, consumer_count, overrides=None):
    cfg = _s04_config(overrides=overrides)
    species = int(species)
    if species not in (0, 1, 2) or consumer_count < 0:
        raise ValueError('invalid species index or consumer number')
    sp = cfg['species'][species]
    z, dz = _s04_mesh(sp['m'], sp['zmax'])
    consumers = []
    kp = cfg['kills'].get((species, 2))
    if kp is not None and consumer_count > 0:
        consumers.append((float(consumer_count), kp))
    K = _s04_projection_matrix(sp, z, dz, float(Za), consumers)
    return _s04_log_spectral_radius(K)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "",
         "call": "asymptotic_growth_rate(0, 80.0, 46.154855)",
         "gold_call": "_oracle_asymptotic_growth_rate(0, 80.0, 46.154855)"},
        {"setup": "",
         "call": "asymptotic_growth_rate(2, 30.0, 46.0)",
         "gold_call": "_oracle_asymptotic_growth_rate(2, 30.0, 46.0)"},
        {"setup": "",
         "call": "asymptotic_growth_rate(1, 150.0, 0.0, overrides={'species.1.bs0': 0.9})",
         "gold_call": "_oracle_asymptotic_growth_rate(1, 150.0, 0.0, overrides={'species.1.bs0': 0.9})"},
        {"setup": "",
         "call": "asymptotic_growth_rate(0, 0.0, 0.0, overrides={'species.0.bb0': 0.0, 'species.0.delta': 0.0, 'species.0.bs2': 0.0})",
         "gold_call": "_oracle_asymptotic_growth_rate(0, 0.0, 0.0, overrides={'species.0.bb0': 0.0, 'species.0.delta': 0.0, 'species.0.bs2': 0.0})"},
    ]
