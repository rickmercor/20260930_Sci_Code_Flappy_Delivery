"""
Euclidean closest-point return to a positive-axis ellipsoid.

On sum_d (x_d/axes_d)^2 = 1, return the global minimizer of Euclidean distance to trial. The trial may be outside, on, or inside the ellipsoid, including its center. The semiaxes may be unordered or repeated. When multiple points attain the minimum, return the lexicographically greatest point in Cartesian coordinate order. This deterministic convention also applies to symmetric fusion trials.

trial and axes are finite length-three array-like inputs. Every semiaxis and the finite scalar tolerance are strictly positive. tolerance requests absolute positional accuracy in the units of trial. Numerical comparisons use absolute component error 1e-9. Return a finite float array of shape (3,) without changing either input.

Raises ------ ValueError For incorrect input shapes, nonfinite entries, nonpositive axes, or nonpositive/nonfinite tolerance.

Return the global closest point under the module's tie convention.



Interior trials, repeated axes and unordered axes are valid. ValueError

is required for invalid shapes, nonfinite entries, nonpositive axes,

or nonpositive/nonfinite tolerance. Both inputs remain unchanged.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_return(trial: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', tolerance: float = 1e-13) -> 'np.ndarray':
    """Return the global closest point under the module's tie convention.

    Interior trials, repeated axes and unordered axes are valid. ValueError
    is required for invalid shapes, nonfinite entries, nonpositive axes,
    or nonpositive/nonfinite tolerance. Both inputs remain unchanged.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_surface_return(trial: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', tolerance: float = 1e-13) -> 'np.ndarray':
    import numpy as np
    from scipy.optimize import brentq
    y = np.asarray(trial, dtype=float)
    a = np.asarray(axes, dtype=float)
    if (y.shape != (3,) or a.shape != (3,) or not np.isfinite(y).all()
            or not np.isfinite(a).all() or np.any(a <= 0)):
        raise ValueError('invalid projection data')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid projection tolerance')
    b = a*a
    residual0 = float(np.sum((y/a)**2)-1)
    if abs(residual0) <= 2e-15:
        return y.copy()
    options = {'xtol': np.nextafter(0., 1.), 'rtol': 4*np.finfo(float).eps, 'maxiter': 1000}
    if residual0 > 0:
        def _residual(lam):
            return float(np.sum((a*y/(b+lam))**2)-1)
        upper = max(1., float(np.linalg.norm(y)*max(a)))
        while _residual(upper) > 0:
            upper *= 2
        lam = brentq(_residual, 0., upper, **options)
        return b*y/(b+lam)

    # The shifted multiplier avoids subtracting nearly equal values at a pole.
    minimum = float(min(b))
    short = b == minimum
    delta = b-minimum
    if np.all(y[short] == 0):
        point = np.zeros(3)
        point[~short] = b[~short]*y[~short]/delta[~short]
        filled = float(np.sum((point[~short]/a[~short])**2))
        if filled <= 1:
            first = int(np.flatnonzero(short)[0])
            point[first] = a[first]*np.sqrt(max(0., 1-filled))
            return point
        lower = 0.
    else:
        lower = float(np.sqrt(minimum)*np.linalg.norm(y[short]))

    def _residual_shifted(shift):
        denominator = delta+shift
        active = denominator != 0
        return float(np.sum((a[active]*y[active]/denominator[active])**2)-1)

    shift = lower if _residual_shifted(lower) <= 0 else brentq(_residual_shifted, lower, minimum, **options)
    return b*y/(delta+shift)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ny=[1.2, 0.4, 0.2];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.0, 0.0, 0.6];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.4, 0.2, 0.25];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.0, 0.0, 0.0];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.0, 0.0, 0.0];a=[1.0, 1.0, 1.0]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.2, 0.0, 0.0];a=[2.0, 1.0, 1.0]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.3, 0.1, 0.0];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.0, 0.3, 0.1];a=[0.6, 1.0, 0.8]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.3, 0.1, 1e-12];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.3, 0.1, -1e-12];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[1.2, 0.07, 1e-10];a=[4.0, 0.5, 0.125]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[4.2, 0.07, 0.001];a=[4.0, 0.5, 0.125]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\ny=[0.9, 0.0, 0.0];a=[1.0, 0.8, 0.6]',
      'call': 'surface_return(y,a)',
      'gold_call': '_oracle_surface_return(y,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _bad_public():\n'
               '    try: surface_return([0.,0.,0.],[1.,-.8,.6]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_surface_return([0.,0.,0.],[1.,-.8,.6]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
