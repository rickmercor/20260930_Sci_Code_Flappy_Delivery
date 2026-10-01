"""
Logarithm of the absolute determinant of the closed-chain second-derivative matrix of the discretised action with the crossing modes removed. SPECIFICATION: close path exactly as in the crossing step and take j and the crossing indices from that same step, the second being 2n minus j. Build the matrix whose diagonal two by two block at row i is eps over the mass, the mass being one, times twice the mass over eps times the identity plus eps times the Hessian of the potential at that row, and whose off-diagonal two by two blocks coupling consecutive rows cyclically are minus the identity. At each of the two crossing rows remove exactly one coordinate direction, the one that the projected-flux construction pins there, and leave the other direction of that row and every other row in place. Wherever a local direction of travel along the closed sequence is needed, take the difference between the next and previous rows, cyclically, normalised to unit length. Return the sum of the natural logarithms of the absolute values of the eigenvalues of what remains, as a single real number.

Two directions of a closed trajectory cost no action, sliding the loop along itself and reflecting it, and left in place they would make this determinant vanish. The construction used here keeps the determinant finite by pinning the loop at its two crossings, one coordinate direction per crossing row, while every transverse curvature is preserved. Because the removal leaves the matrix banded apart from the two corner blocks, a sparse factorisation costs time linear in the number of rows, where a dense one would cost the cube of it.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pinned_log_determinant(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> float:
    """Logarithm of the absolute determinant of the closed-chain second-derivative matrix of the discretised action with the crossing modes removed. SPECIFICATION: close path exactly as in the crossing step and take j and the crossing indices from that same step, the second being 2n minus j. Build the matrix whose diagonal two by two block at row i is eps over the mass, the mass being one, times twice the mass over eps times the identity plus eps times the Hessian of the potential at that row, and whose off-diagonal two by two blocks coupling consecutive rows cyclically are minus the identity. At each of the two crossing rows remove exactly one coordinate direction, the one that the projected-flux construction pins there, and leave the other direction of that row and every other row in place. Wherever a local direction of travel along the closed sequence is needed, take the difference between the next and previous rows, cyclically, normalised to unit length. Return the sum of the natural logarithms of the absolute values of the eigenvalues of what remains, as a single real number.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if path does not have shape (n + 1, 2) with n at least two, or if eps is not positive and finite.
    """
    return log_det  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_pinned_log_determinant(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> float:
    mass = 1.0
    X = np.atleast_2d(np.asarray(path, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    N = X.shape[0] - 1
    ring = np.concatenate([X, X[-2:0:-1]], axis=0)
    n, f = ring.shape
    sd = _oracle_surface_data(X, eps, a, c, wy)
    i0 = int(sd[0])
    i1 = 2 * N - i0
    rows = _oracle_potential_and_derivatives(ring, a, c, wy)
    eye = np.eye(f)
    diag = np.empty((n, f, f))
    for i in range(n):
        H = np.array([[rows[i, 3], rows[i, 4]], [rows[i, 4], rows[i, 5]]])
        diag[i] = (eps / mass) * ((2.0 * mass / eps) * eye + eps * H)
    rot = {}
    for bead in (i0, i1):
        t = ring[(bead + 1) % n] - ring[(bead - 1) % n]
        t = t / np.linalg.norm(t)
        e = np.zeros(f)
        e[0] = 1.0
        v = t - e
        nv = np.linalg.norm(v)
        rot[bead] = eye.copy() if nv < 1e-14 else eye - 2.0 * np.outer(v / nv, v / nv)
        diag[bead] = rot[bead] @ diag[bead] @ rot[bead].T
    ii, jj, vv = [], [], []
    for i in range(n):
        for p in range(f):
            for q in range(f):
                ii.append(i * f + p); jj.append(i * f + q); vv.append(diag[i][p, q])
    for i in range(n):
        k = (i + 1) % n
        B = -eye.copy()
        if i in rot:
            B = rot[i] @ B
        if k in rot:
            B = B @ rot[k].T
        for p in range(f):
            for q in range(f):
                if B[p, q] != 0.0:
                    ii += [i * f + p, k * f + q]
                    jj += [k * f + q, i * f + p]
                    vv += [B[p, q], B[p, q]]
    A = sp.csc_matrix((vv, (ii, jj)), shape=(n * f, n * f))
    keep = np.ones(n * f, dtype=bool)
    keep[i0 * f] = False
    keep[i1 * f] = False
    idx = np.flatnonzero(keep)
    lu = spla.splu(A[idx][:, idx].tocsc(), permc_spec="COLAMD", diag_pivot_thresh=0.1)
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nPX = np.array([[-0.9999508782094086, 0.0002043864847183135], [-0.9999448458897766, 0.00022948485890126032], [-0.9999252671863211, 0.0003109438467307444], [-0.9998873326987177, 0.00046876872447476517], [-0.9998217231620699, 0.0007417188437521161], [-0.9997123177531757, 0.0011968252853745355], [-0.9995322270080186, 0.0019458495443158324], [-0.9992371677778257, 0.0031727215460142597], [-0.9987545309686214, 0.005178685306484819], [-0.9979653976720475, 0.008456212169573879], [-0.9966749269040955, 0.013809760234625207], [-0.9945634093052916, 0.022552837927471096], [-0.9911047799958141, 0.03682922472941647], [-0.9854292164007918, 0.06013571929827492], [-0.9760852278423974, 0.0981701988536808], [-0.960589578519377, 0.16019457640284632], [-0.934160194730382, 0.2611063758867904], [-0.8809553633434072, 0.4230136683159063], [-0.6751087641174822, 0.6537762100374194], [0.6751087656561584, 0.6537762055190869], [0.8809553668025316, 0.42301365349084524], [0.9341602019231333, 0.26110634535935556], [0.9605895918158295, 0.1601945205740431], [0.9760852510620086, 0.09817010181707483], [0.985429255800968, 0.06013555498722407], [0.9911048458125418, 0.036828950532991914], [0.9945635182655497, 0.02255238423438732], [0.9966751063326005, 0.013809013342403113], [0.9979656922012231, 0.008454986363178851], [0.998755013499543, 0.005176677249972738], [0.9992379573839546, 0.003169435786586752], [0.9995335181787167, 0.0019404768334226334], [0.9997144281590552, 0.0011880438336213466], [0.9998251716732507, 0.0007273696994754803], [0.9998929668174259, 0.00044532555168660005], [0.9999344711922805, 0.0002726468846240405], [0.9999598808225519, 0.00016692637197777467], [0.9999754371986557, 0.00010220030658642108], [0.9999849612551598, 6.25726152104738e-05], [0.9999907921820775, 3.831108986226201e-05], [0.9999943620695141, 2.3457299563761918e-05], [0.9999965476773346, 1.4363269471944495e-05], [0.9999978857830718, 8.795574533631355e-06], [0.9999987050188773, 5.386830087607926e-06], [0.9999992065843112, 3.2998734063877342e-06], [0.9999995136606699, 2.0221630569761467e-06], [0.9999997016638654, 1.239902593705747e-06], [0.9999998167661958, 7.60974472160148e-07], [0.9999998872359911, 4.6775738327144006e-07], [0.9999999303801372, 2.882393122607228e-07], [0.9999999567945308, 1.7833190703518476e-07], [0.9999999729663588, 1.1104269596029777e-07], [0.9999999828673071, 6.984593056346224e-08], [0.9999999889289777, 4.4623980167246043e-08], [0.9999999926400738, 2.9182514279883463e-08], [0.9999999949120139, 1.972921939213644e-08], [0.9999999963027701, 1.394243394510901e-08], [0.9999999971539025, 1.0400964847714738e-08], [0.999999997674443, 8.23505263467558e-09], [0.9999999979922326, 6.912764767088555e-09], [0.9999999981853182, 6.109356807778371e-09], [0.9999999983011201, 5.62751751162619e-09], [0.9999999983680786, 5.3489105958419095e-09], [0.9999999984026381, 5.205112195790959e-09], [0.9999999984132862, 5.160806453534661e-09]])',
         "call": 'pinned_log_determinant(PX.copy(), 120.0 / 64, 5e-10, 1.6, 0.55)',
         "gold_call": '_oracle_pinned_log_determinant(PX.copy(), 120.0 / 64, 5e-10, 1.6, 0.55)'},   # normal: the converged trajectory at the task couplings
        {"setup": 'import numpy as np\nP0 = np.array([[-0.9999552791404388, 0.00014310355066733394], [-0.9999481007141858, 0.00016607340494800837], [-0.9999242607469788, 0.00024235643137267006], [-0.999876104911062, 0.00039643972461256283], [-0.9997881702013725, 0.000677783560626466], [-0.9996322152211959, 0.0011766948671432687], [-0.9993581362690478, 0.002053304756568508], [-0.9988778282119045, 0.003588934890670359], [-0.9980367323608886, 0.006276289373439959], [-0.9965637018714318, 0.010977260979692717], [-0.9939823610509871, 0.019198505471208095], [-0.9894532351690811, 0.033571672661502384], [-0.9814888255740737, 0.05868749643716083], [-0.9674271638250922, 0.10253549230934117], [-0.9424178882702102, 0.1789576381893081], [-0.8973193299299056, 0.3117088322146323], [-0.8137056580102815, 0.5406129633952877], [-0.648461040823306, 0.9271972456549675], [-0.2473967908476793, 1.5020717246052315], [0.46441512101671584, 1.2549097525936468], [0.7307456534520657, 0.7456172639374619], [0.8543619659100488, 0.4321050099301067], [0.9190339307814042, 0.24860261451597002], [0.9544025785265552, 0.14258514896298033], [0.974148938221183, 0.0816541538600669], [0.985290859427581, 0.046723075725534184], [0.9916137543327092, 0.026723459549103268], [0.9952132754824854, 0.01528085808547619], [0.9972660746103986, 0.008736602289866903], [0.9984379185387803, 0.0049947565183167345], [0.9991072931559426, 0.0028553868201685265], [0.9994896855854362, 0.0016325894533215646], [0.9997082001545899, 0.0009336232698725412], [0.9998328344991207, 0.0005348848919261685], [0.999904231245305, 0.00030644534037698183], [0.999944783592229, 0.00017668762670428607], [0.9999671474726599, 0.0001051263606266062], [0.9999783328695263, 6.933406637248396e-05], [0.9999882224590558, 3.7687909084560546e-05], [0.9999932669769447, 2.1545601243211367e-05], [0.9999961505681816, 1.2318158109891898e-05], [0.999997798911711, 7.0434747730274696e-06], [0.999998741152191, 4.0283104532079506e-06], [0.999999279763898, 2.304754696424993e-06], [0.9999995876498116, 1.319520330866908e-06], [0.9999997636462381, 7.563319487147169e-07], [0.9999998642508479, 4.34397257365049e-07], [0.9999999217593307, 2.503701319156448e-07], [0.9999999546328304, 1.451749392700208e-07], [0.9999999734242666, 8.504234578765591e-08], [0.9999999841659909, 5.066882877713397e-08], [0.999999990306269, 3.101993915066714e-08], [0.9999999938162281, 1.9788070204640464e-08], [0.9999999958226216, 1.3367611018111347e-08], [0.9999999969695336, 9.697492586724366e-09], [0.9999999976251415, 7.599547302561404e-09], [0.9999999979999057, 6.400301799658337e-09], [0.9999999982141318, 5.714778339438454e-09], [0.9999999983365895, 5.32291366539539e-09], [0.9999999984065898, 5.0989125810474435e-09], [0.999999998446604, 4.970867095721588e-09], [0.9999999984694774, 4.897672312154101e-09], [0.9999999984825524, 4.8558323584302345e-09], [0.9999999984900264, 4.8319154899445495e-09], [0.9999999984942988, 4.8182439371657896e-09]])',
         "call": 'pinned_log_determinant(P0.copy(), 120.0 / 64, 5e-10, 1.6, 0.55)',
         "gold_call": '_oracle_pinned_log_determinant(P0.copy(), 120.0 / 64, 5e-10, 1.6, 0.55)'},   # edge: the unconverged starting trajectory
        {"setup": 'import numpy as np\nP0 = np.array([[-0.9999552791404388, 0.00014310355066733394], [-0.9999481007141858, 0.00016607340494800837], [-0.9999242607469788, 0.00024235643137267006], [-0.999876104911062, 0.00039643972461256283], [-0.9997881702013725, 0.000677783560626466], [-0.9996322152211959, 0.0011766948671432687], [-0.9993581362690478, 0.002053304756568508], [-0.9988778282119045, 0.003588934890670359], [-0.9980367323608886, 0.006276289373439959], [-0.9965637018714318, 0.010977260979692717], [-0.9939823610509871, 0.019198505471208095], [-0.9894532351690811, 0.033571672661502384], [-0.9814888255740737, 0.05868749643716083], [-0.9674271638250922, 0.10253549230934117], [-0.9424178882702102, 0.1789576381893081], [-0.8973193299299056, 0.3117088322146323], [-0.8137056580102815, 0.5406129633952877], [-0.648461040823306, 0.9271972456549675], [-0.2473967908476793, 1.5020717246052315], [0.46441512101671584, 1.2549097525936468], [0.7307456534520657, 0.7456172639374619], [0.8543619659100488, 0.4321050099301067], [0.9190339307814042, 0.24860261451597002], [0.9544025785265552, 0.14258514896298033], [0.974148938221183, 0.0816541538600669], [0.985290859427581, 0.046723075725534184], [0.9916137543327092, 0.026723459549103268], [0.9952132754824854, 0.01528085808547619], [0.9972660746103986, 0.008736602289866903], [0.9984379185387803, 0.0049947565183167345], [0.9991072931559426, 0.0028553868201685265], [0.9994896855854362, 0.0016325894533215646], [0.9997082001545899, 0.0009336232698725412], [0.9998328344991207, 0.0005348848919261685], [0.999904231245305, 0.00030644534037698183], [0.999944783592229, 0.00017668762670428607], [0.9999671474726599, 0.0001051263606266062], [0.9999783328695263, 6.933406637248396e-05], [0.9999882224590558, 3.7687909084560546e-05], [0.9999932669769447, 2.1545601243211367e-05], [0.9999961505681816, 1.2318158109891898e-05], [0.999997798911711, 7.0434747730274696e-06], [0.999998741152191, 4.0283104532079506e-06], [0.999999279763898, 2.304754696424993e-06], [0.9999995876498116, 1.319520330866908e-06], [0.9999997636462381, 7.563319487147169e-07], [0.9999998642508479, 4.34397257365049e-07], [0.9999999217593307, 2.503701319156448e-07], [0.9999999546328304, 1.451749392700208e-07], [0.9999999734242666, 8.504234578765591e-08], [0.9999999841659909, 5.066882877713397e-08], [0.999999990306269, 3.101993915066714e-08], [0.9999999938162281, 1.9788070204640464e-08], [0.9999999958226216, 1.3367611018111347e-08], [0.9999999969695336, 9.697492586724366e-09], [0.9999999976251415, 7.599547302561404e-09], [0.9999999979999057, 6.400301799658337e-09], [0.9999999982141318, 5.714778339438454e-09], [0.9999999983365895, 5.32291366539539e-09], [0.9999999984065898, 5.0989125810474435e-09], [0.999999998446604, 4.970867095721588e-09], [0.9999999984694774, 4.897672312154101e-09], [0.9999999984825524, 4.8558323584302345e-09], [0.9999999984900264, 4.8319154899445495e-09], [0.9999999984942988, 4.8182439371657896e-09]])',
         "call": 'pinned_log_determinant(P0.copy(), 2.5, 2.5e-8, 1.0, 0.8)',
         "gold_call": '_oracle_pinned_log_determinant(P0.copy(), 2.5, 2.5e-8, 1.0, 0.8)'},   # edge: different couplings and spacing
        {"setup": 'import numpy as np\nP3 = np.array([[-1.2, 0.3], [0.0, 0.4], [0.9, -0.2]])',
         "call": 'pinned_log_determinant(P3.copy(), 1.0, 5e-10, 1.6, 0.55)',
         "gold_call": '_oracle_pinned_log_determinant(P3.copy(), 1.0, 5e-10, 1.6, 0.55)'},   # boundary: the shortest admissible chain, one interior row
        {"setup": 'import numpy as np\nP3 = np.array([[-1.2, 0.3], [0.0, 0.4], [0.9, -0.2]])\ndef _c():\n    try:\n        pinned_log_determinant(P3.copy(), 0.0, 5e-10, 1.6, 0.55)\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_pinned_log_determinant(P3.copy(), 0.0, 5e-10, 1.6, 0.55)\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: eps not positive
    ]
