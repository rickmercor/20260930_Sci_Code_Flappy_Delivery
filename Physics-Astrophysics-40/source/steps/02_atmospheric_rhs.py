"""
Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation.

For state `(x,y,z,vx,vy,vz,m)`, set $h=max(z,0)$, `rho=1.225*exp(-h/7200)`, `T=216.65+71*exp(-((h-9000)/8500)^2)`, and sound speed `sqrt(1.4*287*T)`. Wind is $wind_scale*(8+17*exp(-((h-12000)/7000)^2),-3+24*exp(-((h-10000)/6500)^2)-8*exp(-((h-31000)/9000)^2),0)$. With $u=v-wind$, interpolate $Cd$ at `min(|u|/sound,4)` from nodes `(0,.5,.8,1,1.2,2,3,4)` and values `(.47,.48,.55,.92,1.08,.91,.72,.64)`, then multiply by $cd_scale$. For density `3050`, radius $(3m/(4*pi*3050))^(1/3)$, and area $pi*r^2$, use drag `-rho*Cd*A*|u|*u/(2m)`, gravity `(0,0,-9.80665*(6371000/(6371000+h))^2)`, Coriolis `(2*7.2921159e-5*vy,-2*7.2921159e-5*vx,0)`, and mass rate $-sigma_abl*rho*A*|u|^3/2$ only when `|u|>=3000`.

Returns
-------
Return a finite length-16 vector: seven derivatives `(rdot,vdot,mdot)`, then `(rho,sound,wind_x,wind_y,wind_z,|u|,ram_pressure,Cd,area)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation."""
import math
import numpy as np

def atmospheric_rhs(state, wind_scale=1.0, cd_scale=1.0, sigma_abl=8e-08):
    """Evaluate the prescribed spherical-fragment dynamics.

    Parameters
    ----------
    state : array_like, shape (7,)
        Finite (east,north,up,vx,vy,vz,mass), with positive mass.
    wind_scale, cd_scale, sigma_abl : float
        Finite wind multiplier >=0, drag multiplier >0, and ablation
        coefficient >=0 in kg/J. Atmosphere and force definitions are in
        the statement and this step's scientific background.

    Returns
    -------
    ndarray, shape (16,)
        Seven state derivatives, then (rho,sound,wind_x,wind_y,wind_z,
        relative_speed,ram_pressure,Cd,area).

    Raises
    ------
    ValueError
        For wrong state shape, nonfinite input, nonpositive mass or drag
        multiplier, or negative wind multiplier or ablation coefficient.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation."""
import math
import numpy as np
_G0 = 9.80665
_R_EARTH = 6371000.0
_RHO_AIR_0 = 1.225
_H_AIR = 7200.0
_RHO_BODY = 3050.0
_OMEGA = 7.2921159e-05
_MACH_NODES = np.array([0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 4.0])
_CD_NODES = np.array([0.47, 0.48, 0.55, 0.92, 1.08, 0.91, 0.72, 0.64])

def _oracle_atmospheric_rhs(state, wind_scale=1.0, cd_scale=1.0, sigma_abl=8e-08):
    state = np.asarray(state, dtype=float)
    if state.shape != (7,) or not np.all(np.isfinite(state)) or state[6] <= 0.0:
        raise ValueError('state must be a finite seven-vector with positive mass')
    if not np.all(np.isfinite([wind_scale, cd_scale, sigma_abl])):
        raise ValueError('physical scales must be finite')
    if wind_scale < 0.0 or cd_scale <= 0.0 or sigma_abl < 0.0:
        raise ValueError('physical scales are outside their domains')
    position = state[:3]
    velocity = state[3:6]
    altitude = max(float(position[2]), 0.0)
    rho = _RHO_AIR_0 * math.exp(-altitude / _H_AIR)
    temperature = 216.65 + 71.0 * math.exp(-((altitude - 9000.0) / 8500.0) ** 2)
    sound = math.sqrt(1.4 * 287.0 * temperature)
    wind = wind_scale * np.array([8.0 + 17.0 * math.exp(-((altitude - 12000.0) / 7000.0) ** 2), -3.0 + 24.0 * math.exp(-((altitude - 10000.0) / 6500.0) ** 2) - 8.0 * math.exp(-((altitude - 31000.0) / 9000.0) ** 2), 0.0])
    relative = velocity - wind
    speed = float(np.linalg.norm(relative))
    mach = speed / sound
    cd = cd_scale * float(np.interp(min(mach, 4.0), _MACH_NODES, _CD_NODES))
    radius = (3.0 * state[6] / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
    area = math.pi * radius * radius
    drag = -0.5 * rho * cd * area / state[6] * speed * relative
    gravity = np.array([0.0, 0.0, -_G0 * (_R_EARTH / (_R_EARTH + altitude)) ** 2])
    coriolis = np.array([2.0 * _OMEGA * velocity[1], -2.0 * _OMEGA * velocity[0], 0.0])
    mass_rate = -0.5 * sigma_abl * rho * area * speed ** 3 if speed >= 3000.0 else 0.0
    derivative = np.concatenate((velocity, drag + gravity + coriolis, [mass_rate]))
    ram_pressure = 0.5 * rho * speed * speed
    return np.concatenate((derivative, [rho, sound], wind, [speed, ram_pressure, cd, area]))

# =============================================================================
# TEST CASES
# =============================================================================

import math
import numpy as np

def test_cases():
    """Cover luminous flight, dark flight, and rejected mass."""
    return [{'setup': 'state=np.array([0.,0.,40000.,12000.,100.,-5000.,185000.])', 'call': 'atmospheric_rhs(state)', 'gold_call': '_oracle_atmospheric_rhs(state)'}, {'setup': 'state=np.array([20.,-4.,1200.,83.,-12.,-77.,3.4])', 'call': 'atmospheric_rhs(state, 0.7, 1.08, 7.5e-08)', 'gold_call': '_oracle_atmospheric_rhs(state, 0.7, 1.08, 7.5e-08)'}, {'setup': 'def catches(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nstate=np.array([0.,0.,1.,0.,0.,0.,0.])', 'call': 'catches(lambda: atmospheric_rhs(state))', 'gold_call': 'catches(lambda: _oracle_atmospheric_rhs(state))'}, {'setup': 's=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'atmospheric_rhs(s,1.3,.83,1.2e-7)', 'gold_call': '_oracle_atmospheric_rhs(s,1.3,.83,1.2e-7)'}, {'setup': 's=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'atmospheric_rhs(s,0.,1.,0.)', 'gold_call': '_oracle_atmospheric_rhs(s,0.,1.,0.)'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: atmospheric_rhs(np.ones(6)))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(np.ones(6)))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\ns[6]=np.nan\n', 'call': 'rejects(lambda: atmospheric_rhs(s))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\ns[6]=np.inf\n', 'call': 'rejects(lambda: atmospheric_rhs(s))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\ns[6]=-np.inf\n', 'call': 'rejects(lambda: atmospheric_rhs(s))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\ns[6]=0.\n', 'call': 'rejects(lambda: atmospheric_rhs(s))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\ns[6]=-1.\n', 'call': 'rejects(lambda: atmospheric_rhs(s))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,wind_scale=np.nan))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,wind_scale=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,wind_scale=np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,wind_scale=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,wind_scale=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,wind_scale=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,wind_scale=-1.))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,wind_scale=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,cd_scale=np.nan))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,cd_scale=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,cd_scale=np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,cd_scale=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,cd_scale=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,cd_scale=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,cd_scale=0.))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,cd_scale=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,sigma_abl=np.nan))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,sigma_abl=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,sigma_abl=np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,sigma_abl=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,sigma_abl=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,sigma_abl=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ns=np.array([11.,-7.,1300.,4000.,70.,-1000.,4.])\n', 'call': 'rejects(lambda: atmospheric_rhs(s,sigma_abl=-1.))', 'gold_call': 'rejects(lambda: _oracle_atmospheric_rhs(s,sigma_abl=-1.))'}]
