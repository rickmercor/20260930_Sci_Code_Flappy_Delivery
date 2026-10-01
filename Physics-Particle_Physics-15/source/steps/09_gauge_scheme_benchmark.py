"""
Complete gauge-scheme benchmark

Compose and use every preceding public function: seesaw_basis, pv_laurent, charged_gauge, neutrino_w_gauge, neutrino_z_gauge, majorana_diagonal_gauge, restored_uv, and scheme_response. Use their outputs to form the integrated scientific result.

Scientific input for this coding step:

Let m[i] denote the neutrino Takagi masses and l[a] the charged-lepton masses.
All displayed sums run over the complete flavor or neutrino range subject to the stated exclusion. A star is complex conjugation; C = B dagger B.
kappa = alpha / (32*pi*sw2*mW**2), qW = xiW*mW**2, qZ = xiZ*mZ**2.
L[a,k] = B0(l[a]**2; m[k], sqrt(qW)),
W[i,r] = B0(m[i]**2; l[r], sqrt(qW)),
Z[i,k] = B0(m[i]**2; m[k], sqrt(qZ)).
The gauge sector is G = Gell + GW + GZ + Gd, with components

Gell[a,i] = kappa * sum_{b != a,k} B[a,k]*conj(B[b,k])*B[b,i]
    * ((m[k]**2-l[a]**2+qW)*L[a,k] - (m[k]**2-l[b]**2+qW)*L[b,k]).

GW[a,i] = kappa * sum_{j != i,r} B[a,j]*conj(B[r,j])*B[r,i]
    * ((m[j]**2-l[r]**2-qW)*W[j,r] - (m[i]**2-l[r]**2-qW)*W[i,r]).

GZ[a,i] = (kappa/2) * sum_{j != i,k} B[a,j] * (
    (C[j,k]*C[k,i]*(m[j]**2-m[k]**2-qZ)
     - conj(C[j,k])*C[k,i]*(m[k]/m[j])*(m[j]**2-m[k]**2+qZ))*Z[j,k]
    - (C[j,k]*C[k,i]*(m[i]**2-m[k]**2-qZ)
       - C[j,k]*conj(C[k,i])*(m[k]/m[i])*(m[i]**2-m[k]**2+qZ))*Z[i,k]).

Gd[a,i] = -(kappa*B[a,i]/2) * sum_k (m[k]/m[i])
    * (m[i]**2-m[k]**2+qZ) * (C[k,i]**2-C[i,k]**2) * Z[i,k].

GZ is the off-diagonal external-neutrino contribution; Gd is its Majorana diagonal counterpart. These are the source's Eq. (2.19) and the gauge sector of Eq. (2.22).
Each loop or correction has final axis [coefficient of Delta, finite part].
The public APIs specify positive-mass spectra, array orders and comparison conventions.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gauge_scheme_benchmark(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', flavor: int=0, pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    """Complete gauge-scheme benchmark.

    Parameters
    ----------
    md, mr : complex ndarray, shape (n,n), n=2 or 3
        Dirac and symmetric sterile mass matrices in GeV, defining distinct
        positive Takagi masses in the convention of seesaw_basis.
    ml : float ndarray, shape (n,)
        Positive charged-lepton masses in GeV, in md's active row order.
    ew : float ndarray, shape (3,)
        Entries [mW,mZ,alpha], masses in GeV, 0 < mW < mZ and alpha > 0.
    scale : float
        Positive subtraction scale in GeV.
    gauges : float ndarray, shape (2,2)
        Rows a,b; columns xiW,xiZ; all entries nonnegative.
    flavor : int, default 0
        Zero-based active row defining I.
    pair : sequence of two int, default (0,1)
        Distinct zero-based Takagi columns defining I. Inputs have D_I[U]!=0.

    Returns
    -------
    result : float ndarray, shape (5,)
        Dimensionless [R, D_I[F(a)], D_I[F(b)], D_I[U],
        D_I[F_diag(a)-F_diag(b)]]. These are the per-flavor scheme responses.
        Here F(a) and F(b) are the finite scheme-difference matrices,
        U is the retained ultraviolet coefficient, and D_I[H] is the
        first-order shift of I under B -> B+epsilon H for real epsilon.
        R=(D_I[F(a)]-D_I[F(b)])/D_I[U].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from typing import Sequence
import numpy as np
from scipy.linalg import eigh

def _oracle_gauge_scheme_benchmark(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', flavor: int=0, pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    basis=_oracle_seesaw_basis(md,mr);m=basis[0].real;b=basis[1:len(ml)+1]
    mw,mz,alpha=map(float,ew);sw2=1-(mw/mz)**2;n=len(m);l=len(ml)
    totals=[];diagonals=[]
    for xi_w,xi_z in np.asarray(gauges):
        queries=[[x*x,z,np.sqrt(xi_w)*mw] for x in ml for z in m]
        queries.extend([x*x,z,np.sqrt(xi_w)*mw] for x in m for z in ml)
        queries.extend([x*x,z,np.sqrt(xi_z)*mz] for x in m for z in m)
        pv=_oracle_pv_laurent(queries,scale)
        bl=pv[:l*n].reshape(l,n,2);bw=pv[l*n:2*l*n].reshape(n,l,2);bz=pv[2*l*n:].reshape(n,n,2)
        charged=_oracle_charged_gauge(b,m,ml,mw,alpha,sw2,xi_w,bl)
        w=_oracle_neutrino_w_gauge(b,m,ml,mw,alpha,sw2,xi_w,bw)
        z=_oracle_neutrino_z_gauge(b,m,mw,mz,alpha,sw2,xi_z,bz)
        diag=_oracle_majorana_diagonal_gauge(b,m,mw,mz,alpha,sw2,xi_z,bz)
        totals.append((charged+w+z+diag)[:,:,1]);diagonals.append(diag[:,:,1])
    restored=_oracle_restored_uv(b,m,mw,alpha,sw2)
    return _oracle_scheme_response(b,totals[0],totals[1],restored,diagonals[0],diagonals[1],flavor,pair)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nmd = np.array([[(14.98+2.14j), (4.28-3.21j), (2.14+1.07j)], [(3.21+2.14j), (20.330000000000002-1.07j), (5.3500000000000005+4.28j)], [(2.14-1.07j), (6.42+3.21j), (25.68+2.14j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-2j), (7+3j)], [(11-2j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.2, 1.3], [2.7, 0.45]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(13.020000000000001-1.86j), (3.72+2.79j), (1.86-0.93j)], [(2.79-1.86j), (17.67+0.93j), (4.65-3.72j)], [(1.86+0.93j), (5.58-2.79j), (22.32-1.86j)]],dtype=complex)\nmr = np.array([[(135-7j), (11+4j), (7-3j)], [(11+4j), (193+9j), (13-5j)], [(7-3j), (13-5j), (271-12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.61, 1.8], [3.8, 0.27]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(3.36+0.48j), (0.96-0.72j), (0.48+0.24j)], [(0.72+0.48j), (4.56-0.24j), (1.2+0.96j)], [(0.48-0.24j), (1.44+0.72j), (5.76+0.48j)]],dtype=complex)\nmr = np.array([[(162+8.4j), (13.2-4.8j), (8.4+3.5999999999999996j)], [(13.2-4.8j), (231.6-10.799999999999999j), (15.6+6j)], [(8.4+3.5999999999999996j), (15.6+6j), (325.2+14.399999999999999j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.05, 5.0], [4.0, 0.13]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(19.599999999999998+2.8j), (5.6-4.199999999999999j), (2.8+1.4j)], [(4.199999999999999+2.8j), (26.599999999999998-1.4j), (7+5.6j)], [(2.8-1.4j), (8.399999999999999+4.199999999999999j), (33.599999999999994+2.8j)]],dtype=complex)\nmr = np.array([[(78.3+4.06j), (6.38-2.32j), (4.06+1.7399999999999998j)], [(6.38-2.32j), (111.94-5.22j), (7.539999999999999+2.9j)], [(4.06+1.7399999999999998j), (7.539999999999999+2.9j), (157.17999999999998+6.959999999999999j)]],dtype=complex)\nml = np.array([0.000511, 21.5, 47.0],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.09, 0.16], [1.7, 0.36]],dtype=float)\nflavor = 1\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(15.540000000000001+2.22j), (4.44-3.33j), (2.22+1.11j)], [(3.33+2.22j), (21.090000000000003-1.11j), (5.550000000000001+4.44j)], [(2.22-1.11j), (6.66+3.33j), (26.64+2.22j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([0.000511, 0.10566, 1.77686],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.43, 1.6], [0.43, 1.6]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(16.66+2.38j), (4.76-3.57j)], [(3.57+2.38j), (22.61-1.19j)]],dtype=complex)\nmr = np.array([[(109.35000000000001+5.67j), (8.91-3.24j)], [(8.91-3.24j), (156.33-7.290000000000001j)]],dtype=complex)\nml = np.array([0.000511, 0.10566],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.8, 0.7], [2.1, 3.4]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(32.199999999999996+4.6j), (9.2-6.8999999999999995j), (4.6+2.3j)], [(6.8999999999999995+4.6j), (43.699999999999996-2.3j), (11.5+9.2j)], [(4.6-2.3j), (13.799999999999999+6.8999999999999995j), (55.199999999999996+4.6j)]],dtype=complex)\nmr = np.array([[(310.5+16.099999999999998j), (25.299999999999997-9.2j), (16.099999999999998+6.8999999999999995j)], [(25.299999999999997-9.2j), (443.9-20.7j), (29.9+11.5j)], [(16.099999999999998+6.8999999999999995j), (29.9+11.5j), (623.3+27.599999999999998j)]],dtype=complex)\nml = np.array([0.0011752999999999998, 0.24301799999999998, 4.086778],dtype=float)\new = np.array([184.8717, 209.73148, 0.0072973525692838015],dtype=float)\nscale = 146.812036\ngauges = np.array([[0.4, 2.0], [1.9, 0.3]],dtype=float)\nflavor = 0\npair = (0, 1)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}, {'setup': 'import numpy as np\nmd = np.array([[(2.06-1.03j), (6.18+3.09j), (24.72+2.06j)], [(14.42+2.06j), (4.12-3.09j), (2.06+1.03j)], [(3.09+2.06j), (19.57-1.03j), (5.15+4.12j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)\nml = np.array([1.77686, 0.000511, 0.10566],dtype=float)\new = np.array([80.379, 91.1876, 0.0072973525692838015],dtype=float)\nscale = 91.1876\ngauges = np.array([[0.0, 0.19], [0.77, 0.0]],dtype=float)\nflavor = 2\npair = (0, 3)', 'call': 'gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)', 'gold_call': '_oracle_gauge_scheme_benchmark(md=md,mr=mr,ml=ml,ew=ew,scale=scale,gauges=gauges,flavor=flavor,pair=pair)'}]
