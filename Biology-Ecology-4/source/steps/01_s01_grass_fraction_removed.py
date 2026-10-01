"""
Grass removal by a grazer population under the encounter model of the source, expressed as a fraction of the standing grass biomass. The step builds the population-level capture probability of each grazer species from its per-encounter capture probability and its number of individuals, combines the grazers into the total fraction of grass captured, applies the source's rule that only grass surviving intrinsic mortality can be grazed, and partitions the total removal among the grazer species. It also returns the per-individual removal of a vanishingly rare grazer species in the presence of the resident grazers (the rare-invader limit), which is the quantity that enters an invasion analysis. It validates that the inputs are admissible (non-negative counts, capture probabilities in [0, 1), a grass mortality proportion in [0, 1)) and raises ValueError otherwise. It deliberately excludes the conversion of removed grass into assimilated biomass and any dynamics of the grass pool.

In the source framework an unstructured resource such as grass is captured by each of J grazer species through a sequence of independent encounters, so that the proportion of grass captured by all grazers is 1 minus the product over species of (1 - beta_j)^{N_j}, where beta_j is the per-encounter capture probability and N_j the number of individuals of species j. The removal is partitioned additively among the species in proportion to their own population-level capture probabilities 1 - (1 - beta_j)^{N_j}, and it acts only on the grass that survives intrinsic mortality, so the fraction of the standing crop removed by species j is (1 - phi_GD) times the total captured proportion times the share of species j. For a species whose numbers tend to zero, its population-level capture probability behaves as N_j times -ln(1 - beta_j), so the removal per individual has a finite limit that depends on the residents through the total captured proportion and the sum of their capture probabilities; with no residents present the per-individual limit is (1 - phi_GD)(-ln(1 - beta_j)). Benchmark values (species order: small grazer, large grazer, predator; phi_GD = 0.033): with 11137.017970 small grazers and 4552.645906 large grazers and capture probabilities 0.0002 and 0.0003, the small grazer removes the fraction 0.5125239493 of the standing grass and the large grazer 0.4278850338; a vanishingly rare small grazer facing 7656.236479 resident large grazers removes the fraction 1.9341934258e-04 per individual per year.

Returns
-------
Returns the fraction of the standing grass biomass removed per year by grazer species j (population level), or the removal per individual of a vanishingly rare species j when per_capita_limit is True, as a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def grass_fraction_removed(counts, capture_probs, phi_GD, j, per_capita_limit=False):
    """Fraction of the standing grass biomass removed per year by grazer species j.

    Parameters
    ----------
    counts : sequence of float
        Number of individuals of every structured species, in species order (0 for absent species).
    capture_probs : sequence of float
        Per-encounter capture probability beta of every species (0 for species that do not graze).
    phi_GD : float
        Proportion of the grass that dies of intrinsic causes each year; grazing acts on the survivors.
    j : int
        Index of the grazer species whose removal is requested.
    per_capita_limit : bool, optional
        If True, return the removal per individual of species j in the limit of a vanishing
        population of species j (its own entry in `counts` is ignored); the residents are the other
        species with positive counts.

    Returns
    -------
    float
        Fraction of the standing grass biomass removed by species j per year (population level),
        or per individual of a vanishingly rare species j if `per_capita_limit` is True.
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

def _s01_config(theta=0.78, overrides=None):
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
def _s01_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _s01_hh(x, y):
    """Half-harmonic (Michaelis-Menten) function h(x, y) = x y / (x + y)."""
    return x * y / (x + y)

def _s01_mesh(m, zmax):
    dz = zmax / m
    z = (np.arange(m) + 0.5) * dz
    return z, dz

def _s01_norm_columns(E, dz):
    """Column-normalised kernel from log-density E (rows: destination mass, cols: origin)."""
    E = E - E.max(axis=0)[None, :]
    K = np.exp(E)
    return K / (K.sum(axis=0) * dz)[None, :]

def _s01_recruit_column(z, dz, mu_o, sig_o):
    E = -0.5 * ((z - mu_o) / sig_o) ** 2
    return _s01_norm_columns(E[:, None], dz)[:, 0]

def _s01_growth_kernel(z, dz, Zg, sig_g):
    E = -0.5 * ((z[:, None] - Zg[None, :]) / sig_g) ** 2
    return _s01_norm_columns(E, dz)

def _s01_survival(sp, z):
    return sp['bs0'] * _s01_sigmoid((z - sp['bs1']) * sp['bs2'])

def _s01_reproduction(sp, z):
    return sp['bb0'] * _s01_sigmoid((z - sp['bb1']) * sp['bb2'])

def _s01_kill_prob(kp, z):
    return kp[0] * _s01_sigmoid((z - kp[1]) * kp[2])

def _s01_B_star(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return a * ((1.0 - cfg['phi_DP']) * cfg['rho_B'] / cfg['phi_BD'] - 1.0) / cfg['rho_B']

def _s01_nutrient_input(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return cfg['phi_DP'] * _s01_hh(a, cfg['rho_B'] * _s01_B_star(cfg))

# ---------------------------------------------------------------------------
# Step-1 core: grass removed by grazer j (population level or rare-invader per-capita limit)
# ---------------------------------------------------------------------------
def _s01_grass_removed(G, counts, betas, phi_GD, j, per_capita_limit=False):
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


def _oracle_grass_fraction_removed(counts, capture_probs, phi_GD, j, per_capita_limit=False):
    counts = [float(c) for c in counts]
    betas = [float(b) for b in capture_probs]
    if len(counts) != len(betas) or not (0 <= int(j) < len(counts)):
        raise ValueError('inconsistent species lists or index')
    return float(_s01_grass_removed(1.0, counts, betas, float(phi_GD), int(j), bool(per_capita_limit)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "",
         "call": "grass_fraction_removed([11137.017970, 4552.645906, 0.0], [0.0002, 0.0003, 0.0], 0.033, 0)",
         "gold_call": "_oracle_grass_fraction_removed([11137.017970, 4552.645906, 0.0], [0.0002, 0.0003, 0.0], 0.033, 0)"},
        {"setup": "",
         "call": "grass_fraction_removed([11137.017970, 4552.645906, 0.0], [0.0002, 0.0003, 0.0], 0.033, 1)",
         "gold_call": "_oracle_grass_fraction_removed([11137.017970, 4552.645906, 0.0], [0.0002, 0.0003, 0.0], 0.033, 1)"},
        {"setup": "",
         "call": "grass_fraction_removed([0.0, 7656.236479, 46.154855], [0.0002, 0.0003, 0.0], 0.033, 0, per_capita_limit=True)",
         "gold_call": "_oracle_grass_fraction_removed([0.0, 7656.236479, 46.154855], [0.0002, 0.0003, 0.0], 0.033, 0, per_capita_limit=True)"},
        {"setup": "",
         "call": "grass_fraction_removed([3000.0, 3000.0, 0.0], [0.00025, 0.00025, 0.0], 0.033, 1)",
         "gold_call": "_oracle_grass_fraction_removed([3000.0, 3000.0, 0.0], [0.00025, 0.00025, 0.0], 0.033, 1)"},
    ]
