"""
Resolve adjacent-sector densities and adiabatic ground-cluster responses.

Adjacent particle sectors share one quenched Hamiltonian. Here the equal-weight ground-cluster density follows a specified infinitesimal Hamiltonian perturbation. This response is a declared extension of the source method.

Returns
-------
return pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sector_density_pair(model: 'np.ndarray | list | tuple', number: int) -> 'np.ndarray':
    """Parameters
    ----------
    model : array_like, shape (2,2,P,P)
        Finite Hamiltonian jet: first axis is value and derivative, second is h and U, for H=sum_pq h_pq c_p^dagger c_q+sum_(p<q) U_pq n_p n_q. Both orders have Hermitian h and real symmetric zero-diagonal U within absolute tolerance 1e-12; average conjugate-transposed h and transposed real U within that tolerance. 1<=P<=18. The derivative can perturb hopping, potentials and interactions.
    number : int
        Particle number in 1,...,P. Creation operators are ordered by increasing site.
    
    Returns
    -------
    pair : complex ndarray, shape (2,2,P,P)
        First axis is sector number and number-1; second is C and C'. C_pq=Tr[(Pi/m)c_p^dagger c_q], with Pi the entire eigenspace E-E_min<=1e-9 and m its rank. C' differentiates the continuously followed same rank-m spectral cluster of H+alpha*H' at alpha=0, with equal fixed weights 1/m. The vacuum has C=C'=0. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import combinations
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, cg
import numpy as np

@lru_cache(maxsize=64)
def _sector_maps(length, number):
    states=sorted(sum(1<<i for i in c) for c in combinations(range(length),number))
    index={s:i for i,s in enumerate(states)}
    occ=np.array([[(s>>i)&1 for i in range(length)] for s in states],dtype=float)
    lower=sorted(sum(1<<i for i in c) for c in combinations(range(length),number-1))
    lookup={s:i for i,s in enumerate(lower)}
    maps=[]
    for i in range(length):
        src=[];dst=[];sign=[]
        for j,s in enumerate(states):
            if (s>>i)&1:
                src.append(j);dst.append(lookup[s^(1<<i)]);sign.append(1-2*(bin(s&((1<<i)-1)).count('1')%2))
        maps.append((np.array(src,dtype=int),np.array(dst,dtype=int),np.array(sign)))
    hops=[]
    for i in range(length):
        for j in range(i+1,length):
            src=[];dst=[];sign=[]
            for k,s in enumerate(states):
                if not (s>>i)&1 and (s>>j)&1:
                    src.append(k);dst.append(index[s^(1<<i)^(1<<j)])
                    sign.append(1-2*(bin((s>> (i+1))&((1<<(j-i-1))-1)).count('1')%2))
            hops.append((i,j,np.array(src,dtype=int),np.array(dst,dtype=int),np.array(sign)))
    return occ,maps,hops,len(lower)

def _response_hamiltonian(model,occ,hops):
    h,U=model
    diagonal=occ@np.diag(h).real+np.einsum('bi,ij,bj->b',occ,np.triu(U.real,1),occ)
    rows=[np.arange(len(occ))];cols=[np.arange(len(occ))];vals=[diagonal]
    for i,j,src,dst,sign in hops:
        if h[i,j]!=0:
            rows.extend((dst,src));cols.extend((src,dst));vals.extend((h[i,j]*sign,h[j,i]*sign))
    return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(occ),len(occ))).tocsr()

def _oracle_sector_density_pair(model: 'np.ndarray | list | tuple', number: int) -> 'np.ndarray':
    jet=np.asarray(model)
    if jet.ndim!=4 or jet.shape[:2]!=(2,2) or jet.shape[2]!=jet.shape[3] or not 1<=jet.shape[2]<=18 or not np.isfinite(jet).all():
        raise ValueError('finite (2,2,P,P) Hamiltonian jet required')
    jet=jet.astype(complex,copy=True)
    for order in range(2):
        h,U=jet[order]
        if not np.allclose(h,h.conj().T,rtol=0,atol=1e-12) or np.max(abs(U.imag))>1e-12 or not np.allclose(U,U.T,rtol=0,atol=1e-12) or np.max(abs(np.diag(U)))>1e-12:
            raise ValueError('Hermitian h and real symmetric zero-diagonal U required')
        jet[order,0]=(h+h.conj().T)/2;jet[order,1]=(U.real+U.real.T)/2
    if not isinstance(number,(int,np.integer)) or not 1<=number<=jet.shape[-1]:
        raise ValueError('integer particle number required')
    P=jet.shape[-1];out=np.zeros((2,2,P,P),complex)
    for layer,n in enumerate((number,number-1)):
        if n==0:continue
        occ,maps,hops,smaller=_sector_maps(P,n)
        H=_response_hamiltonian(jet[0],occ,hops);W=_response_hamiltonian(jet[1],occ,hops);D=len(occ)
        if np.max(abs(jet[0,0]-np.diag(np.diag(jet[0,0]))))==0:
            energies=H.diagonal().real
            chosen=energies<=np.min(energies)+1e-9;rank=int(np.sum(chosen))
            out[layer,0]=np.diag(occ[chosen].mean(axis=0))
            for i,j,src,dst,sign in hops:
                active=chosen[src]!=chosen[dst]
                a,b=src[active],dst[active]
                w=np.asarray(W[a,b]).ravel()
                v=np.sum(sign[active]*w*(chosen[a].astype(float)-chosen[b])/(energies[a]-energies[b]))/rank
                out[layer,1,i,j]=v;out[layer,1,j,i]=np.conj(v)
            continue
        if D<=100:
            E,Q=np.linalg.eigh(H.toarray());g=np.flatnonzero(E-E[0]<=1e-9)
            R=Q[:,g];outside=np.flatnonzero(E-E[0]>1e-9)
            dR=Q[:,outside]@((Q[:,outside].conj().T@W@R)/(E[g][None,:]-E[outside,None])) if len(outside) else np.zeros_like(R)
        else:
            k=4
            while True:
                E,Q=eigsh(H,k=min(k,D-2),which='SA',tol=2e-13,v0=np.cos(np.arange(D)*np.sqrt(2))+np.sin(np.arange(D)*np.sqrt(3)))
                order=np.argsort(E);E,Q=E[order],Q[:,order]
                if E[-1]-E[0]>1e-9:break
                k*=2
                if k>=D-2:
                    E,Q=np.linalg.eigh(H.toarray());break
            R=np.linalg.qr(Q[:,E-E[0]<=1e-9])[0]
            Eg,T=np.linalg.eigh(R.conj().T@(H@R));R=R@T;dR=np.zeros(R.shape,complex)
            for a,energy in enumerate(Eg):
                rhs=-(W@R[:,a]);rhs-=R@(R.conj().T@rhs)
                op=LinearOperator((D,D),matvec=lambda z:H@z-energy*z+R@(R.conj().T@z),dtype=complex)
                dR[:,a],info=cg(op,rhs,rtol=2e-12,atol=1e-13,maxiter=3000)
                if info:raise RuntimeError('response solve did not converge')
                dR[:,a]-=R@(R.conj().T@dR[:,a])
        for a in range(R.shape[1]):
            A=np.zeros((smaller,P),complex);dA=np.zeros_like(A)
            for i,(src,dst,sign) in enumerate(maps):
                A[dst,i]=sign*R[src,a];dA[dst,i]=sign*dR[src,a]
            out[layer,0]+=A.conj().T@A/R.shape[1]
            out[layer,1]+=(dA.conj().T@A+A.conj().T@dA)/R.shape[1]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# single-site filled vacuum\nh=np.array([[.7]]);U=np.zeros((1,1));n=1\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(110);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# detuned coherent dimer\nh=np.array([[-.4,-1.],[-1.,.8]]);U=np.array([[0.,.9],[.9,0.]]);n=1\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(111);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# filled interacting dimer\nh=np.array([[-.4,-.7],[-.7,.8]]);U=np.array([[0.,2.],[2.,0.]]);n=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(112);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# fully degenerate one-electron manifold\nh=np.zeros((4,4));U=np.zeros_like(h);n=1\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(113);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# fully degenerate adjacent sectors\nh=np.zeros((5,5));U=np.zeros_like(h);n=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(114);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# degenerate occupied-shell mixture\nh=np.diag([-2.,0.,0.,1.]);U=np.zeros_like(h);n=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(115);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# degenerate interaction-only ground manifold\nh=np.zeros((4,4));U=np.ones((4,4))-np.eye(4);n=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(116);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# complex hopping phase convention\nh=np.array([[.2,.3+.8j],[.3-.8j,-.1]]);U=np.array([[0.,1.],[1.,0.]]);n=1\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(117);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# three-site magnetic flux\nh=np.array([[.2,-1.,-.7j],[-1.,-.3,-.8],[.7j,-.8,.4]]);U=np.ones((3,3))-.1;np.fill_diagonal(U,0);n=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(118);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# fermionic exchange on a four-site cycle\nh=np.diag([.1,-.3,.4,.7]);h[0,1]=h[1,0]=h[1,2]=h[2,1]=h[2,3]=h[3,2]=h[0,3]=h[3,0]=-1.;U=.7*(abs(h)>0);np.fill_diagonal(U,0);n=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(119);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# repulsive selected-bond ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=0.4\nn=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(120);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# strong interaction rearrangement\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=3.1\nn=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(121);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# attractive selected-bond ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=-1.3\nn=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(122);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# dilute ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=4.0\nn=1\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(123);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# filled ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=1.7\nn=6\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(124);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# hole-doped ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=1.2\nn=5\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(125);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# noninteracting ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=0.0\nn=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(126);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# hopping-dominated ladder\nh=np.diag([.1,-.8,.6,-.3,.4,.2]);U=np.zeros((6,6))\nfor i,j in [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]:h[i,j]=h[j,i]=-1.\nfor i,j in [(0,2),(1,3),(4,5)]:U[i,j]=U[j,i]=0.1\nn=2\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(127);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# time reversal of a flux ladder\nrng=np.random.default_rng(55);a=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=a+a.conj().T;U=np.ones((6,6))*.2;np.fill_diagonal(U,0);n=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(128);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\n# nonlocal interaction and fermionic parity\nh=np.diag([.3,-.2,.6,-.9,.4]);h+=np.diag([-1.]*4,1)+np.diag([-1.]*4,-1);U=np.zeros((5,5));U[0,4]=U[4,0]=2.;U[1,3]=U[3,1]=-.7;n=3\nargs=(np.stack((h,U)),n)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nm,n=args\nrng=np.random.default_rng(129);a=rng.normal(size=m.shape[1:])+1j*rng.normal(size=m.shape[1:]);dh=(a+a.conj().T)/5\nu=rng.normal(size=m.shape[1:]);du=(u+u.T)/3;np.fill_diagonal(du,0)\nargs=(np.stack((m,np.stack((dh,du)))),n)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# cluster splitting remains fixed rank\nh=np.diag([-2.0, -2.0, 1.0, 3.0]);u=np.zeros_like(h);dh=np.diag([1.,-1.,.2,.4])\nargs=(np.array([[h,u],[dh,u]],dtype=complex),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# cluster rotates into both excited shells\nh=np.diag([-2.0, -2.0, 1.0, 3.0]);u=np.zeros_like(h);dh=np.ones((4,4))\nargs=(np.array([[h,u],[dh,u]],dtype=complex),1)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# complex cluster rotation\nh=np.diag([-2.0, -2.0, 1.0, 3.0]);u=np.zeros_like(h);dh=1j*np.triu(np.ones((4,4)),1)-1j*np.tril(np.ones((4,4)),-1)\nargs=(np.array([[h,u],[dh,u]],dtype=complex),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# threefold Fermi shell\nh=np.diag([-3.0, 0.0, 0.0, 0.0, 2.0]);u=np.zeros_like(h);dh=np.ones((5,5))-.3*np.eye(5)\nargs=(np.array([[h,u],[dh,u]],dtype=complex),3)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# whole Hilbert cluster has zero response\nh=np.diag([0.0, 0.0, 0.0, 0.0]);u=np.zeros_like(h);dh=np.ones((4,4))\nargs=(np.array([[h,u],[dh,u]],dtype=complex),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# near degenerate ground energy resolution\nh=np.diag([-2.0, -1.9999999996, 1.0, 3.0]);u=np.zeros_like(h);dh=np.ones((4,4))\nargs=(np.array([[h,u],[dh,u]],dtype=complex),1)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# resolved narrow spectral gap\nh=np.diag([-2.0, -1.99, 1.0, 3.0]);u=np.zeros_like(h);dh=np.ones((4,4))\nargs=(np.array([[h,u],[dh,u]],dtype=complex),1)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# scalar electrostatic perturbation\nh=np.diag([-2.0, -0.1, 0.4, 3.0]);u=np.zeros_like(h);dh=np.eye(4)*2\nargs=(np.array([[h,u],[dh,u]],dtype=complex),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# empty exterior of filled sector\nh=np.diag([-2.0, -0.1, 0.4, 3.0]);u=np.zeros_like(h);dh=np.ones((4,4))\nargs=(np.array([[h,u],[dh,u]],dtype=complex),4)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# gauge-rotated degenerate coherent dimer 1\nh=np.kron(np.eye(2),np.array([[0.,-1.],[-1.,0.]]));u=np.zeros_like(h);v=np.exp(1j*np.arange(4)*0.3);h=v[:,None]*h*v.conj()[None,:];dh=np.diag([1.,-.2,.7,-.5]);du=np.ones((4,4))*.4;np.fill_diagonal(du,0);args=(np.array([[h,u],[dh,du]]),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# gauge-rotated degenerate coherent dimer 2\nh=np.kron(np.eye(2),np.array([[0.,-1.],[-1.,0.]]));u=np.zeros_like(h);v=np.exp(1j*np.arange(4)*1.1);h=v[:,None]*h*v.conj()[None,:];dh=np.diag([1.,-.2,.7,-.5]);du=np.ones((4,4))*.4;np.fill_diagonal(du,0);args=(np.array([[h,u],[dh,du]]),2)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# physical long-ladder interaction response\nL=8;P=2*L;h=np.zeros((P,P),complex);u=np.zeros((P,P));du=np.zeros((P,P));bonds=[]\nfor i in range(P):\n for j in range(i+1,P):\n  if abs(i//2-j//2)+abs(i%2-j%2)==1:h[i,j]=h[j,i]=-1.\nfor x in range(L):\n if x%4==0:bonds.extend([(2*x,2*x+2),(2*x+1,2*x+3)])\n elif x%4 in (2,3):bonds.append((2*x,2*x+1))\ne=1.9*np.where(np.random.default_rng(49273).uniform(-1,1,8)<0,-1.,1.)\nfor val,(i,j) in zip(e,bonds):h[i,i]=h[j,j]=val;u[i,j]=u[j,i]=2.5;du[i,j]=du[j,i]=1.\ngauge=np.exp(1j*np.arange(P)*0.0);h=gauge[:,None]*h*gauge.conj()[None,:]\nargs=(np.stack((np.stack((h,u)),np.stack((np.zeros_like(h),du)))),P//3)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# physical short-ladder response in a complex site gauge\nL=6;P=2*L;h=np.zeros((P,P),complex);u=np.zeros((P,P));du=np.zeros((P,P));bonds=[]\nfor i in range(P):\n for j in range(i+1,P):\n  if abs(i//2-j//2)+abs(i%2-j%2)==1:h[i,j]=h[j,i]=-1.\nfor x in range(L):\n if x%4==0:bonds.extend([(2*x,2*x+2),(2*x+1,2*x+3)])\n elif x%4 in (2,3):bonds.append((2*x,2*x+1))\ne=1.9*np.where(np.random.default_rng(49273).uniform(-1,1,8)<0,-1.,1.)\nfor val,(i,j) in zip(e,bonds):h[i,i]=h[j,j]=val;u[i,j]=u[j,i]=1.5;du[i,j]=du[j,i]=1.\ngauge=np.exp(1j*np.arange(P)*0.31);h=gauge[:,None]*h*gauge.conj()[None,:]\nargs=(np.stack((np.stack((h,u)),np.stack((np.zeros_like(h),du)))),P//3)', 'call': 'sector_density_pair(*deepcopy(args))', 'gold_call': '_oracle_sector_density_pair(*deepcopy(args))'}]
