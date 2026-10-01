"""
Dispersive two-point Laurent data

The scalar two-point function is defined by the Feynman-parameter integral in the common background. Its endpoint and threshold limits are part of the numerical domain.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pv_laurent(queries: 'np.ndarray', scale: float) -> 'np.ndarray':
    """Dispersive two-point Laurent data.

    Parameters
    ----------
    queries : float ndarray, shape (q,3)
        Rows [p_squared,a,b], respectively in GeV^2, GeV, GeV; all are
        nonnegative and each row has p_squared+a+b > 0.
    scale : float
        Positive subtraction scale in GeV.

    Returns
    -------
    result : float ndarray, shape (q,2)
        In input row order, columns are the Delta coefficient and dispersive
        finite part of B0. Both columns are dimensionless.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pv_laurent(queries: 'np.ndarray', scale: float) -> 'np.ndarray':
    out=[]
    for p2,a,b in np.asarray(queries,float):
        aa,bb=a*a,b*b
        if p2>0:
            linear=aa-bb-p2
            disc=complex(linear*linear-4*p2*bb)
            q=-.5*(linear+(1 if linear>=0 else -1)*np.sqrt(disc))
            roots=[q/p2,bb/q] if q!=0 else [0j,0j]
            def _endpoint(z):
                return 0j if abs(z)==0 else z*np.log(z+0j)
            def _root_integral(r):
                if abs(r)>2:
                    if r.imag==0:r=float(r.real)
                    return np.log(abs(r))+((1-r)*np.log1p(-1/r)).real-1
                return (_endpoint(1-r)-_endpoint(-r)-1).real
            value=2*np.log(scale)-np.log(p2)
            value-=sum(_root_integral(r) for r in roots)
            out.append([1.,value]);continue
        if aa==bb:
            val=2*np.log(scale)-np.log(aa)
        elif min(aa,bb)==0:
            val=1+2*np.log(scale)-np.log(max(aa,bb))
        else:
            large,small=max(aa,bb),min(aa,bb);t=(large-small)/small
            val=1+2*np.log(scale)-np.log(small)-(1+t)*np.log1p(t)/t
        out.append([1.,val])
    return np.array(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nqueries = np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 3.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[2.0, 1.0, 1.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[4.0, 1.0, 1.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[20.0, 1.0, 2.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[1e-10, 1.0, 2.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[7.0, 0.0, 0.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[9.0, 1.0, 2.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}, {'setup': 'import numpy as np\nqueries = np.array([[1.0, 0.000511, 154.0], [1000000.0, 1.0, 2.0]],dtype=float)\nscale = 91.0', 'call': 'pv_laurent(queries=queries,scale=scale)', 'gold_call': '_oracle_pv_laurent(queries=queries,scale=scale)'}]
