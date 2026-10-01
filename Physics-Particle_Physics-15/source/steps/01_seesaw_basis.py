"""
Exact Majorana mass basis

Takagi factorization of the type-I seesaw mass matrix, source Eq. (1.3). The order and column-sign convention are those of the step background.

For code comparisons the sign of each Takagi column is fixed at its largest-modulus entry (smallest row index at a tie): its real part is positive when its magnitude exceeds 1e-12 times that entry's modulus, otherwise its imaginary part is positive.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seesaw_basis(md: 'np.ndarray', mr: 'np.ndarray') -> 'np.ndarray':
    """Exact Majorana mass basis.

    Parameters
    ----------
    md : complex ndarray, shape (n,n)
        Dirac mass matrix in GeV, 1 <= n <= 3.
    mr : complex ndarray, shape (n,n)
        Symmetric sterile Majorana mass matrix in GeV. The assembled mass
        matrix has distinct positive Takagi masses and a nonsingular md.

    Returns
    -------
    result : complex ndarray, shape (2*n+1,2*n)
        Row 0 contains positive ascending masses (GeV) with zero imaginary
        part. Rows 1 through 2*n contain V, ordered as active then sterile
        rows, with M=V diag(m) V.T. Largest-entry column signs follow the
        step background. The first n rows of V form B.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh

def _oracle_seesaw_basis(md: 'np.ndarray', mr: 'np.ndarray') -> 'np.ndarray':
    md,mr=np.asarray(md,complex),np.asarray(mr,complex);n=md.shape[0]
    matrix=np.block([[np.zeros((n,n)),md],[md.T,mr]])
    u,_,_=np.linalg.svd(matrix)
    diag=np.diag(u.conj().T@matrix@u.conj())
    u=u*np.exp(.5j*np.angle(diag))[None,:]
    masses=np.abs(diag);order=np.argsort(masses);masses=masses[order];u=u[:,order]
    for j in range(2*n):
        pivot=np.argmax(np.abs(u[:,j]))
        pivot_value=u[pivot,j]
        sign=pivot_value.real if abs(pivot_value.real)>1e-12*abs(pivot_value) else pivot_value.imag
        if sign<0:u[:,j]*=-1
    return np.vstack([masses,u])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nmd = np.array([[(14.98+2.14j), (4.28-3.21j), (2.14+1.07j)], [(3.21+2.14j), (20.330000000000002-1.07j), (5.3500000000000005+4.28j)], [(2.14-1.07j), (6.42+3.21j), (25.68+2.14j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-2j), (7+3j)], [(11-2j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(13.020000000000001-1.86j), (3.72+2.79j), (1.86-0.93j)], [(2.79-1.86j), (17.67+0.93j), (4.65-3.72j)], [(1.86+0.93j), (5.58-2.79j), (22.32-1.86j)]],dtype=complex)\nmr = np.array([[(135-7j), (11+4j), (7-3j)], [(11+4j), (193+9j), (13-5j)], [(7-3j), (13-5j), (271-12j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(3.36+0.48j), (0.96-0.72j), (0.48+0.24j)], [(0.72+0.48j), (4.56-0.24j), (1.2+0.96j)], [(0.48-0.24j), (1.44+0.72j), (5.76+0.48j)]],dtype=complex)\nmr = np.array([[(162+8.4j), (13.2-4.8j), (8.4+3.5999999999999996j)], [(13.2-4.8j), (231.6-10.799999999999999j), (15.6+6j)], [(8.4+3.5999999999999996j), (15.6+6j), (325.2+14.399999999999999j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(19.599999999999998+2.8j), (5.6-4.199999999999999j), (2.8+1.4j)], [(4.199999999999999+2.8j), (26.599999999999998-1.4j), (7+5.6j)], [(2.8-1.4j), (8.399999999999999+4.199999999999999j), (33.599999999999994+2.8j)]],dtype=complex)\nmr = np.array([[(78.3+4.06j), (6.38-2.32j), (4.06+1.7399999999999998j)], [(6.38-2.32j), (111.94-5.22j), (7.539999999999999+2.9j)], [(4.06+1.7399999999999998j), (7.539999999999999+2.9j), (157.17999999999998+6.959999999999999j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[3.0]],dtype=float)\nmr = np.array([[7.0]],dtype=float)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(16.66+2.38j), (4.76-3.57j)], [(3.57+2.38j), (22.61-1.19j)]],dtype=complex)\nmr = np.array([[(109.35000000000001+5.67j), (8.91-3.24j)], [(8.91-3.24j), (156.33-7.290000000000001j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(32.199999999999996+4.6j), (9.2-6.8999999999999995j), (4.6+2.3j)], [(6.8999999999999995+4.6j), (43.699999999999996-2.3j), (11.5+9.2j)], [(4.6-2.3j), (13.799999999999999+6.8999999999999995j), (55.199999999999996+4.6j)]],dtype=complex)\nmr = np.array([[(310.5+16.099999999999998j), (25.299999999999997-9.2j), (16.099999999999998+6.8999999999999995j)], [(25.299999999999997-9.2j), (443.9-20.7j), (29.9+11.5j)], [(16.099999999999998+6.8999999999999995j), (29.9+11.5j), (623.3+27.599999999999998j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}, {'setup': 'import numpy as np\nmd = np.array([[(2.06-1.03j), (6.18+3.09j), (24.72+2.06j)], [(14.42+2.06j), (4.12-3.09j), (2.06+1.03j)], [(3.09+2.06j), (19.57-1.03j), (5.15+4.12j)]],dtype=complex)\nmr = np.array([[(135+7j), (11-4j), (7+3j)], [(11-4j), (193-9j), (13+5j)], [(7+3j), (13+5j), (271+12j)]],dtype=complex)', 'call': 'seesaw_basis(md=md,mr=mr)', 'gold_call': '_oracle_seesaw_basis(md=md,mr=mr)'}]
