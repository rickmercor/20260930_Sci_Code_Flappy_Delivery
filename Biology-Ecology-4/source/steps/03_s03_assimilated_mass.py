"""
Net assimilated mass per individual per year from the total mass of resources acquired by a population, under the source's saturating consumption-and-assimilation model, including its vanishing-density form for a rare population. The step validates the inputs (non-negative masses and counts, positive rate constants) and raises ValueError otherwise. It excludes how the acquired mass is obtained and how the assimilated mass is allocated.

The source computes the mass consumed and successfully assimilated by one individual as Za = alpha1 alpha2 gamma1 gamma2 Zc / (gamma1 N + gamma2 Zc), a Michaelis-Menten form at the scale of the whole population: Zc is the total acquired resource mass, N the population size, gamma1 the per-capita consumption capacity, gamma2 the consumable fraction of the acquired mass, alpha1 the assimilation efficiency and alpha2 the conversion efficiency. Consumption per individual therefore saturates at alpha1 alpha2 gamma1 when resources are plentiful and is shared as alpha1 alpha2 gamma2 Zc / N when they are scarce. For a vanishingly rare population whose acquired mass scales with its numbers, Zc/N tends to a finite per-capita acquisition c and Za tends to alpha1 alpha2 gamma1 gamma2 c / (gamma1 + gamma2 c); this step returns that limit when N = 0 and the first argument is then interpreted as c. Benchmark values: with Zc = 3.2e7 kg acquired by 11137.017970 small grazers (alpha1 0.60, alpha2 0.075, gamma1 2500 kg per year, gamma2 0.5) each individual assimilates 41.0560108470 kg per year; a rare small grazer acquiring 12345.6 kg per year assimilates 80.0710266581 kg per year; when gamma2 Zc equals gamma1 N the assimilated mass is exactly alpha1 alpha2 gamma1 / 2.

Returns
-------
Returns the mass assimilated per individual per year in kg, as a float (the vanishing-density limit when N == 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assimilated_mass(Zc, N, alpha1, alpha2, gamma1, gamma2):
    """Mass assimilated per individual per year (kg) from the population's acquired resources.

    Parameters
    ----------
    Zc : float
        Total resource mass acquired by the population per year (kg). If N == 0, the per-capita
        acquired mass of a vanishingly rare population (kg per individual per year).
    N : float
        Population size (number of individuals); 0 selects the vanishing-density limit.
    alpha1, alpha2 : float
        Assimilation and conversion efficiencies.
    gamma1 : float
        Per-capita consumption capacity (kg per individual per year).
    gamma2 : float
        Consumable fraction of the acquired mass.

    Returns
    -------
    float
        Assimilated mass per individual per year (kg).
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

def _s03_config(theta=0.78, overrides=None):
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
def _s03_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _s03_hh(x, y):
    """Half-harmonic (Michaelis-Menten) function h(x, y) = x y / (x + y)."""
    return x * y / (x + y)

def _s03_mesh(m, zmax):
    dz = zmax / m
    z = (np.arange(m) + 0.5) * dz
    return z, dz

def _s03_norm_columns(E, dz):
    """Column-normalised kernel from log-density E (rows: destination mass, cols: origin)."""
    E = E - E.max(axis=0)[None, :]
    K = np.exp(E)
    return K / (K.sum(axis=0) * dz)[None, :]

def _s03_recruit_column(z, dz, mu_o, sig_o):
    E = -0.5 * ((z - mu_o) / sig_o) ** 2
    return _s03_norm_columns(E[:, None], dz)[:, 0]

def _s03_growth_kernel(z, dz, Zg, sig_g):
    E = -0.5 * ((z[:, None] - Zg[None, :]) / sig_g) ** 2
    return _s03_norm_columns(E, dz)

def _s03_survival(sp, z):
    return sp['bs0'] * _s03_sigmoid((z - sp['bs1']) * sp['bs2'])

def _s03_reproduction(sp, z):
    return sp['bb0'] * _s03_sigmoid((z - sp['bb1']) * sp['bb2'])

def _s03_kill_prob(kp, z):
    return kp[0] * _s03_sigmoid((z - kp[1]) * kp[2])

def _s03_B_star(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return a * ((1.0 - cfg['phi_DP']) * cfg['rho_B'] / cfg['phi_BD'] - 1.0) / cfg['rho_B']

def _s03_nutrient_input(cfg):
    a = cfg['phi_DB'] * cfg['D']
    return cfg['phi_DP'] * _s03_hh(a, cfg['rho_B'] * _s03_B_star(cfg))

# ---------------------------------------------------------------------------
# Step-3 core: assimilated mass per individual per year
# ---------------------------------------------------------------------------
def _s03_assimilated(Zc, N, a1, a2, g1, g2):
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


def _oracle_assimilated_mass(Zc, N, alpha1, alpha2, gamma1, gamma2):
    return float(_s03_assimilated(float(Zc), float(N), float(alpha1), float(alpha2), float(gamma1), float(gamma2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "",
         "call": "assimilated_mass(3.2e7, 11137.017970, 0.60, 0.075, 2500.0, 0.5)",
         "gold_call": "_oracle_assimilated_mass(3.2e7, 11137.017970, 0.60, 0.075, 2500.0, 0.5)"},
        {"setup": "",
         "call": "assimilated_mass(12345.6, 0.0, 0.60, 0.075, 2500.0, 0.5)",
         "gold_call": "_oracle_assimilated_mass(12345.6, 0.0, 0.60, 0.075, 2500.0, 0.5)"},
        {"setup": "",
         "call": "assimilated_mass(5.0e6, 1000.0, 0.60, 0.075, 2500.0, 0.5)",
         "gold_call": "_oracle_assimilated_mass(5.0e6, 1000.0, 0.60, 0.075, 2500.0, 0.5)"},
        {"setup": "",
         "call": "assimilated_mass(2.0e5, 46.154855, 0.75, 0.08, 800.0, 0.68)",
         "gold_call": "_oracle_assimilated_mass(2.0e5, 46.154855, 0.75, 0.08, 800.0, 0.68)"},
    ]
