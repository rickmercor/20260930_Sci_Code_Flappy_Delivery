"""
Reduce the class-AII localizer to an affine real skew-symmetric pencil.

Main paper Eqs. (1), (2), (9), (17), (20). Layer-first H0=diag(A,A*), H1=[[0,G],[G^dagger,0]], G=diag(profile) tensor sigma_y. Material J=[[0,I_2N],[-I_2N,0]], F=J tensor (i sigma_y), Q=(I-iF)/sqrt(2). Use S(c)=i Q^dagger L(c) Q and the material-first localizer specified in the main background. This explicitly fixes the basis for array comparison.

Returns
-------
return result  # real ndarray (2,8N,8N), ordered S0,S1 with S(c)=S0+c*S1; units t. Roundoff-only symmetric and imaginary components are removed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def aii_pencil(layer: ArrayLike, coords: ArrayLike, profile: ArrayLike, probe: ArrayLike, kappa: float) -> np.ndarray:
    'Reduce the class-AII localizer to an affine real skew-symmetric pencil.\n\nParameters\n----------\nlayer : complex (2N,2N), arbitrary Hermitian single-layer matrix in t.\ncoords : real (N,2), vertex coordinates in a.\nprofile : real (N,), dimensionless interlayer texture.\nprobe : real (3,), ordered x/a,y/a,E/t.\nkappa : nonnegative float in t/a.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. N is positive. The layer is Hermitian within relative max-entry tolerance 1e-12. Coordinates have exactly shape (N,2).\n\nReturns\n-------\nreal ndarray (2,8N,8N), ordered S0,S1 with S(c)=S0+c*S1; units t. Roundoff-only symmetric and imaginary components are removed.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_aii_pencil(layer: ArrayLike, coords: ArrayLike, profile: ArrayLike, probe: ArrayLike, kappa: float) -> np.ndarray:
    def _checked_coords(coords, distinct=False):
        r = _checked_numeric(coords, 'coords')
        if r.ndim != 2 or r.shape[1] != 2 or len(r) == 0:
            raise ValueError('coords must have shape (N,2), N>0')
        if distinct and len(np.unique(r, axis=0)) != len(r):
            raise ValueError('vertex coordinates must be distinct')
        return r

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_vector(value, n, name, scalar=False):
        a = _checked_numeric(value, name)
        if scalar and a.shape == ():
            return np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(name + ' has an incompatible shape')
        return a

    r=_checked_coords(coords);n=len(r);a=_checked_numeric(layer,'layer',real=False)
    profile=_checked_vector(profile,n,'profile');probe=_checked_vector(probe,3,'probe')
    kappa=_checked_scalar(kappa,'kappa',minimum=0)
    if a.shape!=(2*n,2*n):raise ValueError('layer must have shape (2N,2N)')
    scale=float(np.max(abs(a)))
    if scale and np.max(abs(a/scale-a.conj().T/scale))>1e-12:
        raise ValueError('layer must be Hermitian to relative max-entry tolerance 1e-12')
    m=2*n;z=np.zeros_like(a)
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.])
    h0=np.block([[a,z],[z,a.conj()]]);c=np.kron(np.diag(profile),sy)
    h1=np.block([[z,c],[c.conj().T,z]])
    j=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
    f=np.kron(j,1j*sy);q=(np.eye(4*m)-1j*f)/np.sqrt(2)
    x=np.tile(np.repeat(r[:,0],2),2);y=np.tile(np.repeat(r[:,1],2),2)
    l0=np.kron(h0-probe[2]*np.eye(2*m),sz)+kappa*np.kron(np.diag(x-probe[0]),sx)+kappa*np.kron(np.diag(y-probe[1]),sy)
    result=np.array([1j*q.conj().T@l@q for l in (l0,np.kron(h1,sz))]).real
    return (result-result.transpose(0,2,1))/2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nh=np.array([[-0.6, 0.0], [0.0, 0.6]],dtype=complex)\nr=np.array([[0.0, 0.0]],dtype=float)\ng=np.array([0.0],dtype=float)\nz=np.array([0.0, 0.0, 0.0],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[0.4, 0.0], [0.0, 0.4]],dtype=complex)\nr=np.array([[1.0102163274947373, -0.5848526891465718]],dtype=float)\ng=np.array([0.7],dtype=float)\nz=np.array([0.13, -0.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[(0.3448049316296533+0j), (-0.532773660661569-0.07201063177197503j)], [(-0.532773660661569+0.07201063177197503j), (2.1934602581081695+0j)]],dtype=complex)\nr=np.array([[-0.6648943421861485, -0.5219255872430351]],dtype=float)\ng=np.array([-1.2],dtype=float)\nz=np.array([0.13, -0.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[(0.08859604015914473+0j), (0.0053147377861438305-0.012470427632158276j), (0.6026767279019885-0.6705070098241143j), (-0.33501888419419334-0.9257486908492936j)], [(0.0053147377861438305+0.012470427632158276j), (1.4881286269831215+0j), (-0.08425022481239508+0.32735521658012867j), (0.48903584182100834-0.029216097285901133j)], [(0.6026767279019885+0.6705070098241143j), (-0.08425022481239508-0.32735521658012867j), (-1.4694155121530101+0j), (0.34762475456136027+0.5236039889384505j)], [(-0.33501888419419334+0.9257486908492936j), (0.48903584182100834+0.029216097285901133j), (0.34762475456136027-0.5236039889384505j), (-0.8708765020715598+0j)]],dtype=complex)\nr=np.array([[-0.3794476653024333, -0.536672440923374], [0.1337791358797726, -0.9615838552238102]],dtype=float)\ng=np.array([0.0, 1.0],dtype=float)\nz=np.array([0.13, -0.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[-0.7232617254025817, -0.042085370634878355, 0.5467063218511657, -0.39906644229353927], [-0.042085370634878355, -2.5549805310722506, -0.10511163398593558, -0.29669786057567193], [0.5467063218511657, -0.10511163398593558, -0.04476069104538298, -1.3303251261305147], [-0.39906644229353927, -0.29669786057567193, -1.3303251261305147, 1.1603091072771687]],dtype=complex)\nr=np.array([[-0.05563022220303466, 0.9644851876788971], [-0.5326114332281371, -2.0830448908189685]],dtype=float)\ng=np.array([0.7794313332438252, 0.07940076706347311],dtype=float)\nz=np.array([0.13, -0.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[(-0.5231065714894412+0j), (-0.08331829860334539+0.5587278607362438j), (-0.4276427130496661+0.8611300407645446j), (-0.27172341643982045+1.3990243519445031j)], [(-0.08331829860334539-0.5587278607362438j), (-1.629616419906959+0j), (-2.3840819116543823+1.0441994025911232j), (-0.12235777262272657-0.5894784860224139j)], [(-0.4276427130496661-0.8611300407645446j), (-2.3840819116543823-1.0441994025911232j), (0.7652940477370748+0j), (0.02289355329995124+0.3989760050347774j)], [(-0.27172341643982045-1.3990243519445031j), (-0.12235777262272657+0.5894784860224139j), (0.02289355329995124-0.3989760050347774j), (0.16566435637741272+0j)]],dtype=complex)\nr=np.array([[1.0378608781716467, -0.03705205946527093], [-1.3112932522521712, -0.8960552259577095]],dtype=float)\ng=np.array([0.8433562094024715, 1.4072786148199141],dtype=float)\nz=np.array([1.0378608781716467, -0.03705205946527093, -0.8],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[(1.2896328005913782+0j), (-0.25089026023976574+0.3124008033901249j), (0.2911356824418871+0.6351690883053377j), (1.3311734196646006+2.2031542017207046j)], [(-0.25089026023976574-0.3124008033901249j), (-0.32993301174224077+0j), (-0.5424318042456298-1.379877544770457j), (-0.22542618823780364+0.3541025429156405j)], [(0.2911356824418871-0.6351690883053377j), (-0.5424318042456298+1.379877544770457j), (-0.5476708508624543+0j), (0.17741346821022103+1.1964973359917987j)], [(1.3311734196646006-2.2031542017207046j), (-0.22542618823780364-0.3541025429156405j), (0.17741346821022103-1.1964973359917987j), (-0.6809580315542588+0j)]],dtype=complex)\nr=np.array([[5.871508761786642, -4.509277094217932], [3.064481682915968, -2.598294106997409]],dtype=float)\ng=np.array([-0.4243694631549752, -0.8275024551592931],dtype=float)\nz=np.array([4.13, -3.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.08)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.08)'}, {'setup': 'import numpy as np\nh=np.array([[(-0.4234124192924312+0j), (0.01568131856563565+0.5956663813116412j), (0.014304555785403239-0.49972236463866176j), (0.393854044572149-0.8042700384989994j), (-0.1617634178750756+0.998206557525123j), (-0.6710243669218643-0.6036904519154176j)], [(0.01568131856563565-0.5956663813116412j), (0.09687540240289225+0j), (-0.759659847205415-0.3667255321519783j), (0.6551521460390186-0.5829769614342007j), (-0.3920851209274153-0.10678020806007354j), (-0.2892443341411841+0.3681056955832476j)], [(0.014304555785403239+0.49972236463866176j), (-0.759659847205415+0.3667255321519783j), (0.10979146279811673+0j), (-0.5511736032331929-0.21160359328068482j), (-0.44616960367516667+0.7353166653262144j), (-0.28648815001720296+0.2090049298657769j)], [(0.393854044572149+0.8042700384989994j), (0.6551521460390186+0.5829769614342007j), (-0.5511736032331929+0.21160359328068482j), (1.666940525241549+0j), (0.5895322823723207-1.3016040393614667j), (0.0725409933889204-0.6134397192182691j)], [(-0.1617634178750756-0.998206557525123j), (-0.3920851209274153+0.10678020806007354j), (-0.44616960367516667-0.7353166653262144j), (0.5895322823723207+1.3016040393614667j), (-0.6334126185334202+0j), (-0.7404327629441617-0.11208813361282416j)], [(-0.6710243669218643+0.6036904519154176j), (-0.2892443341411841-0.3681056955832476j), (-0.28648815001720296-0.2090049298657769j), (0.0725409933889204+0.6134397192182691j), (-0.7404327629441617+0.11208813361282416j), (1.0854181919274863+0j)]],dtype=complex)\nr=np.array([[-0.13035328108451738, -2.2163611738543603], [-0.28747423155361324, -0.3004210694061028], [-0.3136287181317955, -1.1584551949381137]],dtype=float)\ng=np.array([-2.3832352880552934, 0.16644234971023186, -0.544949410198958],dtype=float)\nz=np.array([0.13, -0.17, 0.05],dtype=float)', 'call': 'aii_pencil(h,r,g,z,0.0)', 'gold_call': '_oracle_aii_pencil(h,r,g,z,0.0)'}]
