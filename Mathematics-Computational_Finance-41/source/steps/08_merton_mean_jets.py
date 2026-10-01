"""
Derive and assemble analytic first and second jump-mean derivatives of the compensated differential operator, the quadrature jump operator and the American exterior integral. Return ordinary derivatives in the documented tensor order.

Sections 2--4 provide the Merton generator, rational quadrature and American exterior tail. The jet calculation is an authored analytic differentiation of those numerical objects, not a new result claimed by the paper.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def merton_mean_jets(nodes: 'np.ndarray', degree: 'int', quadrature_order: 'int', r: 'float', sigma: 'float', intensity: 'float', jump_mean: 'float', jump_std: 'float', strike: 'float') -> 'tuple':
    r"""Return the first three jump-mean jets of the finite Merton operators.

    For $q=0,1,2$, the entries in slot $q$ are the ordinary derivatives
    $\partial_{\mu_J}^q D$, $\partial_{\mu_J}^q J$, and
    $\partial_{\mu_J}^q R$. These are derivatives, not Taylor coefficients.
    $D,J$ are exactly the objects defined by merton_operators and $R$ is
    the intensity-free American tail defined by american_boundary_tail.
    Hold nodes, degree, quadrature nodes/weights, strike and every parameter
    except jump_mean fixed. Differentiate the compensated drift as well as
    the normal density and the exterior integral. Compute analytic jets;
    finite differences or complex perturbations of a pricing routine are
    not the requested operator. Slot zero must call the preceding routines.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Global Gauss--Legendre order $Q\geq1$.
    r, sigma, intensity, jump_mean, jump_std : float
        Same domains and meaning as merton_operators: finite rate and mean,
        positive volatility and jump standard deviation, nonnegative intensity.
    strike : float
        Finite positive strike, held fixed in the differentiation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        $(D_{\rm jet},J_{\rm jet},R_{\rm jet})$, shapes $(3,M,M)$,
        $(3,M,M)$ and $(3,M)$. The leading axis is derivative order;
        matrix rows are evaluation nodes and columns are value nodes.

    Raises
    ------
    ValueError
        For violations of the stated numeric domains and shapes.
    ArithmeticError
        For a nonfinite or zero rational denominator.

    The constituent formulas must have finite binary64 values.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def _oracle_merton_mean_jets(nodes: "np.ndarray", degree: int, quadrature_order: int,
                     r: float, sigma: float, intensity: float,
                     jump_mean: float, jump_std: float, strike: float) -> tuple:
    r"""Return the first three jump-mean jets of the finite Merton operators.

    For $q=0,1,2$, the entries in slot $q$ are the ordinary derivatives
    $\partial_{\mu_J}^q D$, $\partial_{\mu_J}^q J$, and
    $\partial_{\mu_J}^q R$. These are derivatives, not Taylor coefficients.
    $D,J$ are exactly the objects defined by merton_operators and $R$ is
    the intensity-free American tail defined by american_boundary_tail.
    Hold nodes, degree, quadrature nodes/weights, strike and every parameter
    except jump_mean fixed. Differentiate the compensated drift as well as
    the normal density and the exterior integral. Compute analytic jets;
    finite differences or complex perturbations of a pricing routine are
    not the requested operator. Slot zero must call the preceding routines.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Global Gauss--Legendre order $Q\geq1$.
    r, sigma, intensity, jump_mean, jump_std : float
        Same domains and meaning as merton_operators: finite rate and mean,
        positive volatility and jump standard deviation, nonnegative intensity.
    strike : float
        Finite positive strike, held fixed in the differentiation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        $(D_{\rm jet},J_{\rm jet},R_{\rm jet})$, shapes $(3,M,M)$,
        $(3,M,M)$ and $(3,M)$. The leading axis is derivative order;
        matrix rows are evaluation nodes and columns are value nodes.

    Raises
    ------
    ValueError
        For violations of the stated numeric domains and shapes.
    ArithmeticError
        For a nonfinite or zero rational denominator.

    The constituent formulas must have finite binary64 values.
    """
    x = np.asarray(nodes, dtype=float)
    D, J = _oracle_merton_operators(x, degree, quadrature_order, r, sigma,
                            intensity, jump_mean, jump_std)
    _, R = _oracle_american_boundary_tail(x, strike, jump_mean, jump_std)
    w = _oracle_floater_hormann_weights(x, degree)
    _, Dx, _ = _oracle_rational_evaluation(x, w, x)
    eta, omega = roots_legendre(int(quadrature_order))
    y = (x[-1]+x[0])/2 + (x[-1]-x[0])*eta/2
    W = (x[-1]-x[0])*omega/2
    E, _, _ = _oracle_rational_evaluation(x, w, y)
    mu, s, lam, K = map(float, (jump_mean, jump_std, intensity, strike))
    g = y[None, :] - x[:, None] - mu
    density = np.exp(-0.5*(g/s)**2)/(np.sqrt(2*np.pi)*s)
    J1 = (density*(g/s**2)*W) @ E
    J2 = (density*(g*g/s**4-1/s**2)*W) @ E
    D1 = -lam*np.exp(mu+s*s/2)*Dx
    z = (x[0]-x-mu)/s
    a = z-s
    pdfz = np.exp(-z*z/2)/np.sqrt(2*np.pi)
    pdfa = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    expterm = np.exp(x+mu+s*s/2)
    R1 = K*(-pdfz/s-expterm*(ndtr(a)-pdfa/s))
    R2 = K*(-z*pdfz/s**2-expterm*(ndtr(a)-2*pdfa/s-a*pdfa/s**2))
    return np.stack((D, D1, D1)), np.stack((J, J1, J2)), np.stack((R, R1, R2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'merton_mean_jets([-1.5,-.7,-.1,.4,1.2],3,23,.04,.21,.35,-.4,.32,100.)', 'gold_call': '_oracle_merton_mean_jets([-1.5,-.7,-.1,.4,1.2],3,23,.04,.21,.35,-.4,.32,100.)', 'tol': 2e-07}, {'setup': '', 'call': 'merton_mean_jets([-1.2,-.3,.2,1.4],1,31,.025,.22,.3,-.35,.3,110.)', 'gold_call': '_oracle_merton_mean_jets([-1.2,-.3,.2,1.4],1,31,.025,.22,.3,-.35,.3,110.)', 'tol': 2e-07}, {'setup': '', 'call': 'merton_mean_jets([-2.,-.7,0.,.5,1.8],2,40,.06,.19,0.,.12,.5,80.)', 'gold_call': '_oracle_merton_mean_jets([-2.,-.7,0.,.5,1.8],2,40,.06,.19,0.,.12,.5,80.)', 'tol': 2e-07}, {'setup': '', 'call': 'merton_mean_jets(clustered_grid(-1.5,1.5,49,.4),3,96,.05,.15,.1,-.9,.45,100.)', 'gold_call': '_oracle_merton_mean_jets(_oracle_clustered_grid(-1.5,1.5,49,.4),3,96,.05,.15,.1,-.9,.45,100.)', 'tol': 2e-07}]
