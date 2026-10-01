"""
Contract the regular jets with every bulk, face and pole-intersection term.

Multiply the four Laurent polynomials in each rule row by its five-entry regular Taylor series. Sum the coefficients of epsilon powers -4 through zero with the angular weight in column 29. The contraction is algebraic and applies to arbitrary finite real rows in the stated column layout. Coordinate columns 0:5 are unused and have no domain restriction beyond finiteness. Use the supplied angular weight and every polynomial coefficient that can contribute through epsilon power zero. Degree-four entries of an individual Laurent weight cannot contribute at this truncation. Empty aligned arrays return five zeros. It carries out the Laurent expansion of Eqs. (61)-(63) on the Table V sectors. Raise ValueError for complex dtype, invalid shapes, nonfinite values or a row-count mismatch.

Returns
-------
coefficients : ndarray, shape (5,) Coefficients of eps powers -4,-3,-2,-1,0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def combine_laurent_sector_coefficients(rule: np.ndarray, jets: np.ndarray) -> np.ndarray:
    """Contract the regular jets with every bulk, face and pole-intersection term.

    Parameters
    ----------
    rule : array_like, shape (N,30)
        Arbitrary finite real rows in the preceding column layout; coordinate columns 0:5 are unused.
    jets : array_like, shape (N,5)
        Finite real Taylor coefficients aligned with the rule rows.

    Returns
    -------
    coefficients : ndarray, shape (5,)
        Coefficients of eps powers -4,-3,-2,-1,0.

    Raises
    ------
    ValueError
        Complex dtype, invalid shape, nonfinite input or a row-count mismatch.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_combine_laurent_sector_coefficients(rule: np.ndarray, jets: np.ndarray) -> np.ndarray:
    import numpy as np
    rule, jets = np.asarray(rule), np.asarray(jets)
    if rule.dtype.kind not in 'iuf' or jets.dtype.kind not in 'iuf':
        raise ValueError('real numeric rule and jets required')
    dtype = np.result_type(rule.dtype, jets.dtype, float)
    rule, jets = rule.astype(dtype), jets.astype(dtype)
    if rule.ndim != 2 or rule.shape[1] != 30 or jets.shape != (len(rule),5) or not np.all(np.isfinite(rule)) or not np.all(np.isfinite(jets)):
        raise ValueError('aligned finite rule and epsilon jets')
    product = np.zeros((len(rule),5), dtype=dtype)
    product[:,0] = 1
    for start in (5,11,17,23):
        updated = np.zeros_like(product)
        for degree in range(5):
            for j in range(degree+1):
                updated[:,degree] += product[:,j]*rule[:,start+degree-j]
        product = updated
    terms = np.zeros_like(product)
    for degree in range(5):
        for j in range(degree+1):
            terms[:,degree] += product[:,j]*jets[:,degree-j]
    return np.sum(terms*rule[:,29,None], axis=0, dtype=dtype)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import itertools\nimport math\nimport numpy as np\nrates=(4, 2, 2, 2)\nsplit=False\naxes=[]\nfor axis,c in enumerate(rates):\n    nodes=[0.25,0.75] if split and axis==3 else [0.5]\n    weights=[0.5,0.5] if split and axis==3 else [1.0]\n    regular=np.array([[w/x*(-c*math.log(x))**k/math.factorial(k) for k in range(5)] for x,w in zip(nodes,weights)])\n    entries=[(0.0,np.r_[-1/c,-regular.sum(axis=0)])]\n    entries.extend((x,np.r_[0.0,v]) for x,v in zip(nodes,regular))\n    axes.append(entries)\nrows=[]\nfor choices in itertools.product(*axes):\n    point=[v[0] for v in choices]+[0.5]\n    rows.append(point+np.concatenate([v[1] for v in choices]).tolist()+[math.pi])\nrule=np.array(rows)\nt,a,b,r,chi=rule[:,:5].T\namplitude=(1+t)*(2+a)*(3+b)*(4+r)\nell=0.2+t+0.7*a+0.3*b+0.5*r+0.1*chi\njets=amplitude[:,None]*np.array([ell**k/math.factorial(k) for k in range(5)]).T\n', 'call': '#case:boundary\ncombine_laurent_sector_coefficients(rule,jets)', 'gold_call': '_oracle_combine_laurent_sector_coefficients(rule,jets)'},
     {'setup': 'import itertools\nimport math\nimport numpy as np\nrates=(4, 1, 2, 3)\nsplit=False\naxes=[]\nfor axis,c in enumerate(rates):\n    nodes=[0.25,0.75] if split and axis==3 else [0.5]\n    weights=[0.5,0.5] if split and axis==3 else [1.0]\n    regular=np.array([[w/x*(-c*math.log(x))**k/math.factorial(k) for k in range(5)] for x,w in zip(nodes,weights)])\n    entries=[(0.0,np.r_[-1/c,-regular.sum(axis=0)])]\n    entries.extend((x,np.r_[0.0,v]) for x,v in zip(nodes,regular))\n    axes.append(entries)\nrows=[]\nfor choices in itertools.product(*axes):\n    point=[v[0] for v in choices]+[0.5]\n    rows.append(point+np.concatenate([v[1] for v in choices]).tolist()+[math.pi])\nrule=np.array(rows)\nt,a,b,r,chi=rule[:,:5].T\namplitude=(1+t)*(2+a)*(3+b)*(4+r)\nell=0.2+t+0.7*a+0.3*b+0.5*r+0.1*chi\njets=amplitude[:,None]*np.array([ell**k/math.factorial(k) for k in range(5)]).T\n', 'call': '#case:normal\ncombine_laurent_sector_coefficients(rule,jets)', 'gold_call': '_oracle_combine_laurent_sector_coefficients(rule,jets)'},
     {'setup': 'import itertools\nimport math\nimport numpy as np\nrates=(4, 1, 1, 3)\nsplit=True\naxes=[]\nfor axis,c in enumerate(rates):\n    nodes=[0.25,0.75] if split and axis==3 else [0.5]\n    weights=[0.5,0.5] if split and axis==3 else [1.0]\n    regular=np.array([[w/x*(-c*math.log(x))**k/math.factorial(k) for k in range(5)] for x,w in zip(nodes,weights)])\n    entries=[(0.0,np.r_[-1/c,-regular.sum(axis=0)])]\n    entries.extend((x,np.r_[0.0,v]) for x,v in zip(nodes,regular))\n    axes.append(entries)\nrows=[]\nfor choices in itertools.product(*axes):\n    point=[v[0] for v in choices]+[0.5]\n    rows.append(point+np.concatenate([v[1] for v in choices]).tolist()+[math.pi])\nrule=np.array(rows)\nt,a,b,r,chi=rule[:,:5].T\namplitude=(1+t)*(2+a)*(3+b)*(4+r)\nell=0.2+t+0.7*a+0.3*b+0.5*r+0.1*chi\njets=amplitude[:,None]*np.array([ell**k/math.factorial(k) for k in range(5)]).T\n', 'call': '#case:edge\ncombine_laurent_sector_coefficients(rule,jets)', 'gold_call': '_oracle_combine_laurent_sector_coefficients(rule,jets)'},
     {'setup': 'import numpy as np\nrule=np.arange(90,dtype=float).reshape(3,30)/97-.2\njets=np.array([[1,-2,3,-4,5],[.2,.3,.4,.5,.6],[-.1,.7,-.3,.9,-.4]])\n', 'call': '#case:normal\ncombine_laurent_sector_coefficients(rule,jets)', 'gold_call': '_oracle_combine_laurent_sector_coefficients(rule,jets)'},
     {'setup': 'import numpy as np\nrule=np.arange(90,dtype=float).reshape(3,30)/97-.2\njets=np.array([[1,-2,3,-4,5],[.2,.3,.4,.5,.6],[-.1,.7,-.3,.9,-.4]])\njets=jets[:2]\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': '#case:edge\ncheck(lambda:combine_laurent_sector_coefficients(rule,jets))', 'gold_call': 'check(lambda:_oracle_combine_laurent_sector_coefficients(rule,jets))'},
     {'setup': 'import numpy as np\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': '#case:edge\ncheck(lambda:combine_laurent_sector_coefficients(np.zeros((1,30),complex)+1j*.01,np.ones((1,5))))', 'gold_call': 'check(lambda:_oracle_combine_laurent_sector_coefficients(np.zeros((1,30),complex)+1j*.01,np.ones((1,5))))'}]
