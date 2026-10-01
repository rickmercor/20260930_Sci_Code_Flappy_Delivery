"""
Build an endpoint-smoothed pole and plus-distribution quadrature.

The endpoint subtraction is that of Eqs. (61)-(63). Section IV evaluates the Table V sectors with pySecDec, and that paper's Outlook describes the numerical integration as a Laurent series expansion of the integrand in the Appendix A parametrization with that sectorization. Use `x**(-1-c*eps)=-delta(x)/(c*eps)+sum_{k>=0}(-c*eps)**k/k!*[log(x)**k/x]_+`, with plus action `integral_0^1 log(x)**k*(f(x)-f(0))/x dx` and the sector rates from Step 1. Let `(s_i,w_i)` be the ascending n-point Gauss-Legendre rule on [0,1]. Smooth each singular coordinate using `H(s)=s**3/(s**3+(1-s)**3)` and `Hprime(s)=3*s**2*(1-s)**2/(s**3+(1-s)**3)**2`. Its interior nodes and ordinary weights are `x_i=H(s_i)` and `v_i=w_i*Hprime(s_i)`. For II2 only, split r at 1/2: concatenate `H(s_i)/2` and `(1+H(s_i))/2`, with weights `v_i/2` on both halves. Add zero before the interior nodes of every singular axis. For an interior node x_i of rate c, return the six Laurent coefficients in powers -1,0,...,4: the pole coefficient is zero and the regular coefficients are `v_i/x_i*(-c*log(x_i))**k/k!` for k=0,...,4. At zero the pole is -1/c and each regular coefficient is minus the sum over that axis's interior nodes. The endpoint subtraction is applied in the physical coordinate x; there is no additional pole from the map H. For the angular measure `dchi/sqrt(chi*(1-chi))`, put `U(s)=10*s**3-15*s**4+6*s**5`, `Uprime(s)=30*s**2*(1-s)**2`, and `chi_i=sin(pi*U(s_i)/2)**2`. The angular weight is `pi*w_i*Uprime(s_i)`. The map removes the square-root measure and smooths both angular endpoints without changing the integral. The same order n is used on all five axes. Order rows lexicographically by `(i_t,i_a,i_b,i_r,i_chi)`, last index fastest. Columns 0:5 are `(t,a,b,r,chi)`; columns 5:11, 11:17, 17:23 and 23:29 are the Laurent vectors of t, a, b and r, in that order; column 29 is the angular weight. The complete row count is `(n+1)**4*n` except II2, where it is `(n+1)**3*(2*n+1)*n`. Return only the contiguous flat rows `[start:stop]`, with start default zero and stop default the complete count. Require integer bounds excluding bool and `0<=start<=stop<=count`; equal bounds return shape (0,30). Slices permit bounded-memory integration and do not change any row or weight. Raise ValueError for an invalid sector, order or slice. The order is an integer from one through sixteen, excluding bool.

Returns
-------
rule : ndarray, shape (N,30) Requested rows of product nodes, four Laurent vectors, and angular weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_laurent_sector_rule(sector: str, order: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Build an endpoint-smoothed pole and plus-distribution quadrature.

    Parameters
    ----------
    sector : str
        One of the seven named sectors.
    order : int
        Common order n, one through sixteen; bool is not accepted.
    start : int, optional
        First flat row, default zero.
    stop : int or None, optional
        End of the contiguous flat rows; None selects the complete row count.

    Returns
    -------
    rule : ndarray, shape (N,30)
        Requested rows of product nodes, four Laurent vectors, and angular weights.

    Raises
    ------
    ValueError
        Invalid sector, order outside one through sixteen or not an integer, or invalid slice bounds.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_laurent_sector_rule(sector: str, order: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    import math
    import numpy as np
    rates = {'I1': (4,2,1,2), 'I2': (4,3,2,2), 'I3': (4,1,2,3), 'I4': (4,2,2,2), 'I5': (4,2,2,2), 'II1': (4,3,1,2), 'II2': (4,1,1,3)}
    _integer = lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
    if not isinstance(sector, str) or sector not in rates or not _integer(order) or not 1 <= order <= 16:
        raise ValueError('named sector and integer quadrature order from one through sixteen')
    shape = (order+1, order+1, order+1, 2*order+1 if sector == 'II2' else order+1, order)
    count = math.prod(shape)
    stop = count if stop is None else stop
    if not _integer(start) or not _integer(stop) or not 0 <= start <= stop <= count:
        raise ValueError('integer bounds 0 <= start <= stop <= row count')
    gl, wg = np.polynomial.legendre.leggauss(order)
    s, w = (gl.astype(np.float64)+1)/2, wg.astype(np.float64)/2
    denom = s**3+(1-s)**3
    x, v = s**3/denom, w*3*s**2*(1-s)**2/denom**2
    nodes, axes = [], []
    for axis, rate in enumerate(rates[sector]):
        xx, vv = (np.r_[x/2, (1+x)/2], np.r_[v/2, v/2]) if sector == 'II2' and axis == 3 else (x, v)
        nodes.append(np.r_[np.float64(0), xx])
        coeff = np.zeros((len(xx)+1, 6), dtype=np.float64)
        coeff[0,0] = -np.float64(1)/rate
        for k in range(5):
            coeff[1:,k+1] = vv/xx*(-rate*np.log(xx))**k/math.factorial(k)
            coeff[0,k+1] = -np.sum(coeff[1:,k+1], dtype=np.float64)
        axes.append(coeff)
    u = s**3*(10-15*s+6*s**2)
    nodes.append(np.sin(np.pi*u/2)**2)
    angular = np.pi*w*30*s**2*(1-s)**2
    indices = np.unravel_index(np.arange(start, stop), shape)
    result = np.empty((stop-start, 30), dtype=np.float64)
    for j in range(5):
        result[:,j] = nodes[j][indices[j]]
    for j in range(4):
        result[:,5+6*j:11+6*j] = axes[j][indices[j]]
    result[:,29] = angular[indices[4]]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': "sector='I1'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='I2'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='I3'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='I4'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='I5'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='II1'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': "sector='II2'\norder=2\n", 'call': '#case:normal\nassemble_laurent_sector_rule(sector,order)', 'gold_call': '_oracle_assemble_laurent_sector_rule(sector,order)'},
     {'setup': '', 'call': "#case:boundary\nassemble_laurent_sector_rule('I4',1)", 'gold_call': "_oracle_assemble_laurent_sector_rule('I4',1)"},
     {'setup': '', 'call': "#case:edge\nassemble_laurent_sector_rule('II2',3)", 'gold_call': "_oracle_assemble_laurent_sector_rule('II2',3)"},
     {'setup': 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': "#case:edge\ncheck(lambda:assemble_laurent_sector_rule('I4',0))", 'gold_call': "check(lambda:_oracle_assemble_laurent_sector_rule('I4',0))"},
     {'setup': '', 'call': "#case:normal\nassemble_laurent_sector_rule('II2',16,1031,1054)", 'gold_call': "_oracle_assemble_laurent_sector_rule('II2',16,1031,1054)"},
     {'setup': '', 'call': "#case:boundary\nassemble_laurent_sector_rule('I4',16,1336330,1336336)", 'gold_call': "_oracle_assemble_laurent_sector_rule('I4',16,1336330,1336336)"}]
