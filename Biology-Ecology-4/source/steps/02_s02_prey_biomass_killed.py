"""
Biomass of a structured prey species removed by the predator population under the source's encounter model of consumption. The step discretises the prey's body-mass distribution on the benchmark mesh, evaluates the per-encounter kill probability as a decreasing sigmoid of prey mass with the frozen parameters of the prey-predator pair, raises the survival of one encounter to the number of predators, weights the removal by the prey's intrinsic survival (only individuals that survive intrinsic causes of death are exposed to consumers) and integrates mass times removal probability over the distribution. In the vanishing-predator limit it returns the removal per predator. It validates the prey distribution specification and raises ValueError on inadmissible inputs. It excludes acquisition efficiencies and assimilation.

Predation in the source acts through independent encounters with each of the N_j individual consumers of species j, so a prey individual of mass z survives consumption with probability (1 - p_j(z))^{N_j}, where p_j(z) = beta0 * sigmoid((z - beta1) * beta2) with beta2 < 0 so that larger prey are increasingly resistant. The biomass removed from prey species i by consumer j is the integral over the prey distribution of z times the probability of surviving intrinsic death times the probability of being killed, partitioned among consumers by their population-level kill probabilities (with a single consumer species the share is one). For a vanishing consumer population the kill probability behaves as N_j(-ln(1 - p_j(z))), which gives a finite removal per consumer. Prey distributions are given either as a Gaussian specification (count, mean, standard deviation), discretised at the mesh midpoints and rescaled to the count, or as the vector of densities at the midpoints. Benchmark values at the frozen kill parameters: a small-grazer population of 7510.081675 individuals with a Gaussian mass distribution of mean 220 kg and standard deviation 60 kg loses 1.2444337069e+05 kg per year to 41.354707 predators; a large-grazer population of 7656.236479 individuals (mean 400 kg, sd 80 kg) loses 2.4050632813e+03 kg per year per predator in the vanishing-predator limit; with no predators nothing is removed.

Returns
-------
Returns the kilograms of prey biomass removed per year by the predator population (or per predator in the vanishing-predator limit) as a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prey_biomass_killed(prey, prey_dist, n_consumers, per_capita_limit=False, overrides=None):
    """Biomass (kg per year) of a prey species killed by the predator population.

    Parameters
    ----------
    prey : int
        Index of the prey species (0 small grazer, 1 large grazer, 2 predator itself for intraspecific predation).
    prey_dist : dict
        Either {"count": n, "mean": mu, "sd": sigma} (a Gaussian body-mass distribution discretised on the
        prey's benchmark mesh and rescaled to n individuals) or {"values": [...]} (densities per kg at the
        mesh midpoints of the prey).
    n_consumers : float
        Number of predator individuals.
    per_capita_limit : bool, optional
        If True, return the removal per predator in the limit of a vanishing predator population
        (`n_consumers` is ignored).
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
        Kilograms of prey biomass removed per year by the predator population, or per predator in the limit.
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

def _s02_config(theta=0.78, overrides=None):
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
def _s02_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _s02_hh(x, y):
    """Half-harmonic (Michaelis-Menten) function h(x, y) = x y / (x + y)."""
    return x * y / (x + y)

def _s02_mesh(m, zmax):
    dz = zmax / m
    z = (np.arange(m) + 0.5) * dz
    return z, dz

def _s02_norm_columns(E, dz):
    """Column-normalised kernel from log-density E (rows: destination mass, cols: origin)."""
    E = E - E.max(axis=0)[None, :]
    K = np.exp(E)
    return K / (K.sum(axis=0) * dz)[None, :]

def _s02_recruit_column(z, dz, mu_o, sig_o):
    E = -0.5 * ((z - mu_o) / sig_o) ** 2
    return _s02_norm_columns(E[:, None], dz)[:, 0]

def _s02_growth_kernel(z, dz, Zg, sig_g):
    E = -0.5 * ((z[:, None] - Zg[None, :]) / sig_g) ** 2
    return _s02_norm_columns(E, dz)

def _s02_survival(sp, z):
    return sp['bs0'] * _s02_sigmoid((z - sp['bs1']) * sp['bs2'])

def _s02_reproduction(sp, z):
    return sp['bb0'] * _s02_sigmoid((z - sp['bb1']) * sp['bb2'])

def _s02_kill_prob(kp, z):
    return kp[0] * _s02_sigmoid((z - kp[1]) * kp[2])

def _s02_B_star(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return a * ((1.0 - cfg['phi_DP']) * cfg['rho_B'] / cfg['phi_BD'] - 1.0) / cfg['rho_B']

def _s02_nutrient_input(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return cfg['phi_DP'] * _s02_hh(a, cfg['rho_B'] * _s02_B_star(cfg))

# ---------------------------------------------------------------------------
# Step-1 core: grass removed by grazer j (population level or rare-invader per-capita limit)
# ---------------------------------------------------------------------------
def _s02_grass_removed(G, counts, betas, phi_GD, j, per_capita_limit=False):
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
def _s02_kill_flux(z, dz, N_prey, p_sn_prey, kill_params, n_pred, per_capita_limit=False):
    """Biomass of prey removed by n_pred consumers; if per_capita_limit, the per-consumer
    flux in the limit of a vanishing consumer population (n_pred ignored)."""
    pd1 = _s02_kill_prob(kill_params, z)
    if per_capita_limit:
        pk = -np.log(1.0 - pd1)
    else:
        pk = 1.0 - (1.0 - pd1) ** n_pred
    return float(np.sum(z * p_sn_prey * pk * N_prey) * dz)


def _oracle_prey_biomass_killed(prey, prey_dist, n_consumers, per_capita_limit=False, overrides=None):
    cfg = _s02_config(overrides=overrides)
    prey = int(prey)
    if prey not in (0, 1, 2) or n_consumers < 0:
        raise ValueError('invalid prey index or consumer number')
    sp = cfg['species'][prey]
    z, dz = _s02_mesh(sp['m'], sp['zmax'])
    if 'values' in prey_dist:
        N = np.asarray(prey_dist['values'], dtype=float)
        if N.shape != z.shape or np.any(N < 0):
            raise ValueError('density vector must have one non-negative entry per mesh cell')
    else:
        n0, mu, sd = float(prey_dist['count']), float(prey_dist['mean']), float(prey_dist['sd'])
        if n0 < 0 or sd <= 0:
            raise ValueError('invalid Gaussian specification')
        d = np.exp(-0.5 * ((z - mu) / sd) ** 2)
        N = d / (d.sum() * dz) * n0
    kp = cfg['kills'][(prey, 2)]
    p_sn = _s02_survival(sp, z)
    return float(_s02_kill_flux(z, dz, N, p_sn, kp, float(n_consumers), bool(per_capita_limit)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "",
         "call": "prey_biomass_killed(0, {'count': 7510.081675, 'mean': 220.0, 'sd': 60.0}, 41.354707)",
         "gold_call": "_oracle_prey_biomass_killed(0, {'count': 7510.081675, 'mean': 220.0, 'sd': 60.0}, 41.354707)"},
        {"setup": "",
         "call": "prey_biomass_killed(1, {'count': 7656.236479, 'mean': 400.0, 'sd': 80.0}, 0.0, per_capita_limit=True)",
         "gold_call": "_oracle_prey_biomass_killed(1, {'count': 7656.236479, 'mean': 400.0, 'sd': 80.0}, 0.0, per_capita_limit=True)"},
        {"setup": "vals = [0.0] * 50\nvals[10] = 12.5\nvals[20] = 4.0",
         "call": "prey_biomass_killed(0, {'values': vals}, 46.154855, overrides={'kills.02.0': 0.006})",
         "gold_call": "_oracle_prey_biomass_killed(0, {'values': vals}, 46.154855, overrides={'kills.02.0': 0.006})"},
        {"setup": "",
         "call": "prey_biomass_killed(2, {'count': 46.0, 'mean': 12.0, 'sd': 6.0}, 0.0)",
         "gold_call": "_oracle_prey_biomass_killed(2, {'count': 46.0, 'mean': 12.0, 'sd': 6.0}, 0.0)"},
    ]
