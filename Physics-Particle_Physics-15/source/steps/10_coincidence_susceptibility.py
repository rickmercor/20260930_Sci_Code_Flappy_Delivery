"""
Phase-space coincidence and local susceptibility

Use gauge_scheme_benchmark for the phase-dependent responses, thereby composing all preceding public functions. The coincidence and UV-norm selection are defined in the main prompt; rows follow flavors and the phase directions multiply the first two Dirac columns.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coincidence_susceptibility(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', box: 'np.ndarray', flavors: 'Sequence[int]'=(0, 1), pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    """Gauge-coincidence susceptibility.

    Parameters
    ----------
    md, mr : complex ndarray, shape (3,3)
        Base Dirac matrix and symmetric sterile matrix, in GeV. Phase theta
        multiplies column 0 of md and eta multiplies column 1. mr stays fixed.
    ml : float ndarray, shape (3,)
        Positive charged-lepton masses in GeV, in active row order.
    ew : float ndarray, shape (3,)
        [mW,mZ,alpha], with 0 < mW < mZ (GeV) and alpha > 0.
    scale : float
        Positive subtraction scale in GeV.
    gauges : float ndarray, shape (2,2)
        Rows a,b and columns xiW,xiZ, nonnegative.
    box : float ndarray, shape (2,2)
        Rows theta,eta and columns lower,upper, in radians within [-pi,pi].
        The rectangle contains isolated coincidence points and a unique
        maximum of their UV-response norms. Takagi masses are distinct and
        positive throughout; both flavor responses u are nonzero at the
        selected coincidence.
    flavors : sequence of two int, default (0,1)
        Distinct active rows whose gauge contrasts vanish, in Jacobian row order.
    pair : sequence of two int, default (0,1)
        Distinct zero-based ascending Takagi columns defining each invariant.

    Returns
    -------
    result : float ndarray, shape (5,)
        [largest Jacobian singular value, theta, eta, UV-response norm,
        smallest Jacobian singular value]. Phases are in radians; the norm
        is dimensionless, and singular values are per radian. The Jacobian
        differentiates both normalized contrasts with respect to theta,eta.
        Compare numerical components with absolute tolerance 1e-6.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from typing import Sequence
import numpy as np

def _oracle_coincidence_susceptibility(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', box: 'np.ndarray', flavors: 'Sequence[int]'=(0, 1), pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    from scipy.optimize import root
    box=np.asarray(box,float)
    def _responses(x):
        shifted=np.array(md,complex,copy=True)
        shifted[:,0]*=np.exp(1j*x[0]);shifted[:,1]*=np.exp(1j*x[1])
        return np.array([_oracle_gauge_scheme_benchmark(shifted,mr,ml,ew,scale,gauges,int(f),pair) for f in flavors])
    def _residual(x):
        vals=_responses(x)
        return 1e6*(vals[:,1]-vals[:,2])
    roots=[]
    for theta in np.linspace(box[0,0],box[0,1],4):
        for eta in np.linspace(box[1,0],box[1,1],5):
            fit=root(_residual,[theta,eta],tol=1e-9,options={'maxfev':110})
            point=(fit.x+np.pi)%(2*np.pi)-np.pi
            if np.all(point>=box[:,0]-1e-9) and np.all(point<=box[:,1]+1e-9) and np.linalg.norm(_residual(point))<1e-8:
                if all(np.linalg.norm(point-prev)>1e-6 for prev in roots):roots.append(point)
    if not roots:raise ValueError('The phase rectangle contains no isolated coincidence point')
    scores=[np.linalg.norm(_responses(point)[:,3]) for point in roots]
    selected=roots[int(np.argmax(scores))]
    h=.001
    normalization=_responses(selected)[:,3]
    def _contrast(x):
        vals=_responses(x)
        return vals[:,1]-vals[:,2]
    jac=np.column_stack([(_contrast(selected-2*h*v)-8*_contrast(selected-h*v)
                          +8*_contrast(selected+h*v)-_contrast(selected+2*h*v))/(12*h)/normalization for v in np.eye(2)])
    singular=np.linalg.svd(jac,compute_uv=False)
    return np.array([singular[0],selected[0],selected[1],max(scores),singular[-1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nmd = np.array([[(13.957202759945423+2.2795813470398185j), (3.908213634388463-3.118632102056945j), (2+1j)], [(2.959402686613067+2.0595960134131546j), (18.961455141028267-1.5694645375964051j), (5+4j)], [(2.0195986800264887-0.9598026732799116j), (6.087286703101412+2.8186771000319886j), (24+2j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.37, 2.4], [3.1, 0.64]],dtype=float)\nbox = np.array([[-0.8200000000000001, -0.02], [-1.47, 0.73]],dtype=float)\nflavors = (0, 1)\npair = (0, 1)', 'call': 'coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'gold_call': '_oracle_coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'tol': 1e-06}, {'setup': 'import numpy as np\nmd = np.array([[(14+2j), (4-3j), (2+1j)], [(3+2j), (19-1j), (5+4j)], [(2-1j), (6+3j), (24+2j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.37, 2.4], [3.1, 0.64]],dtype=float)\nbox = np.array([[-0.7, -0.45], [-1.4, -1.1]],dtype=float)\nflavors = (0, 1)\npair = (0, 1)', 'call': 'coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'gold_call': '_oracle_coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'tol': 1e-06}, {'setup': 'import numpy as np\nmd = np.array([[(14-2j), (4+3j), (2-1j)], [(3-2j), (19+1j), (5-4j)], [(2+1j), (6-3j), (24-2j)]],dtype=complex)\nmr = np.array([[(135-7j), (11+4j), (7-3j)], [(11+4j), (193+9j), (13-5j)], [(7-3j), (13-5j), (271-12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.37, 2.4], [3.1, 0.64]],dtype=float)\nbox = np.array([[-0.0, 0.8], [-0.7, 1.5]],dtype=float)\nflavors = (0, 1)\npair = (0, 1)', 'call': 'coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'gold_call': '_oracle_coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'tol': 1e-06}, {'setup': 'import numpy as np\nmd = np.array([[(14+2j), (4-3j), (2+1j)], [(3+2j), (19-1j), (5+4j)], [(2-1j), (6+3j), (24+2j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.35519999999999996, 2.472], [3.1620000000000004, 0.6208]],dtype=float)\nbox = np.array([[-0.8, 0.0], [-1.5, 0.7]],dtype=float)\nflavors = (0, 1)\npair = (0, 1)', 'call': 'coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'gold_call': '_oracle_coincidence_susceptibility(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,box=box,flavors=flavors,pair=pair)', 'tol': 1e-06}]
