#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def detector_matrix(modulus, offsets):
    if isinstance(modulus,(bool,np.bool_)) or not isinstance(modulus,(int,np.integer)) or not 3<=modulus<=16:
        raise ValueError('invalid modulus')
    a=np.asarray(offsets)
    if a.ndim!=2 or a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
        raise ValueError('invalid offsets')
    b,k=a.shape
    if not 1<=b<=4 or not 2<=k<=modulus or b*modulus>64 or np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>=modulus):
        raise ValueError('offset bounds')
    if any(len(set(row))!=k for row in a.tolist()):raise ValueError('duplicate offset')
    H=np.zeros((modulus,b*modulus),dtype=int)
    for f in range(b):
        for q in range(modulus):H[(q+a[f].astype(int))%modulus,f*modulus+q]=1
    return H

import numpy as np


def residual_syndrome(H, syndrome, fixed):
    arrays=[]
    for x in (H,syndrome,fixed):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(np.imag(a)!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if not np.all(np.isfinite(a)):raise ValueError('nonfinite input')
        arrays.append(a)
    h,s,f=arrays
    if h.ndim!=2 or not 1<=h.shape[0]<=32 or not 2<=h.shape[1]<=64:raise ValueError('matrix shape')
    m,n=h.shape
    if s.shape!=(m,) or f.shape!=(n,):raise ValueError('vector shape')
    if np.any((h!=0)&(h!=1)) or np.any((s!=0)&(s!=1)) or np.any((f!=-1)&(f!=0)&(f!=1)):raise ValueError('invalid discrete value')
    if np.any(h[:,f<0].sum(axis=1)<2):raise ValueError('too few live neighbors')
    return ((s+h[:,f==1].sum(axis=1))%2).astype(int)

import numpy as np


def masked_run(H, prior, syndrome, messages, fixed, max_iters):
    from fractions import Fraction
    residual=residual_syndrome(H,syndrome,fixed)
    h=np.asarray(H).real.astype(int);f=np.asarray(fixed).real.astype(int)
    n=h.shape[1]
    vals=[]
    for x,shape,bound in ((prior,(n,),16),(messages,h.shape,1e18)):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(a.imag!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if a.shape!=shape or not np.all(np.isfinite(a)) or np.any(np.abs(a)>bound):raise ValueError('numeric domain')
        vals.append(a.copy())
    L,E=vals
    if np.any(E[h==0]!=0):raise ValueError('nonedge message')
    q=np.asarray(max_iters)
    if q.ndim!=0 or q.dtype.kind not in 'iuf' or not np.isfinite(q) or q!=np.floor(q) or not 1<=q<=32:raise ValueError('iteration budget')
    live=f<0
    L=[Fraction(float(x)) for x in L]
    E=np.array([[Fraction(float(x)) for x in row] for row in E],dtype=object)
    acc=np.array([Fraction(0) for _ in range(n)],dtype=object)
    err=np.zeros(n,dtype=int);ok=False;t=0
    for t in range(1,int(q)+1):
        D=np.full(h.shape,Fraction(0),dtype=object)
        for i in range(h.shape[0]):
            nb=np.flatnonzero(h[i]&live)
            for j in nb:
                v=E[i,nb[nb!=j]]
                sign=-1 if residual[i] else 1
                for x in v:sign*=1 if x>0 else (-1 if x<0 else 0)
                D[i,j]=sign*min(abs(x) for x in v)
        posterior=np.array([Fraction(0) for _ in range(n)],dtype=object);new=E.copy()
        for j in np.flatnonzero(live):
            nb=np.flatnonzero(h[:,j]);posterior[j]=L[j]+sum(D[i,j] for i in nb)
            for i in nb:new[i,j]=L[j]+sum(D[k,j] for k in nb if k!=i)
        E=new;acc+=posterior
        err=((posterior<=0)&live).astype(int)
        ok=np.array_equal((h@err)%2,residual)
        if ok:break
    err[~live]=f[~live]
    return E.astype(float),acc.astype(float),int(t),err,int(ok)

import numpy as np


def path_statistics(posterior_sum, fixed, iterations):
    vals=[]
    for x in (posterior_sum,fixed):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(a.imag!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if a.ndim!=1 or not np.all(np.isfinite(a)):raise ValueError('invalid vector')
        vals.append(a)
    a,f=vals
    if not 2<=len(a)<=64 or f.shape!=a.shape or np.any(np.abs(a)>1e12):raise ValueError('shape or magnitude')
    if np.any((f!=-1)&(f!=0)&(f!=1)) or not np.any(f==-1) or np.any(a[f!=-1]!=0):raise ValueError('mask')
    t=np.asarray(iterations)
    if t.ndim!=0 or t.dtype.kind not in 'iuf' or not np.isfinite(t) or t!=np.floor(t) or not 1<=t<=32:raise ValueError('iterations')
    U=np.flatnonzero(f==-1);j=int(U[np.argmin(np.abs(a[U]))]);A=float(np.sum(np.abs(a[U])))
    return j,A/float(t),A

import numpy as np


def rank_paths(scores, paths, width):
    s=np.asarray(scores);p=np.asarray(paths)
    if s.dtype.kind not in 'iuf' or s.ndim!=1 or not 1<=len(s)<=64 or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(s>1e12):raise ValueError('scores')
    if p.dtype.kind not in 'iuf' or p.ndim!=3 or p.shape[0]!=len(s) or not 0<=p.shape[1]<=6 or p.shape[2]!=2 or not np.all(np.isfinite(p)) or np.any(p!=np.floor(p)):raise ValueError('paths')
    if np.any(p[:,:,0]<0) or np.any(p[:,:,0]>63) or np.any((p[:,:,1]!=0)&(p[:,:,1]!=1)):raise ValueError('path values')
    if any(len(set(a[:,0].tolist()))!=len(a) for a in p):raise ValueError('repeated index')
    if isinstance(width,(bool,np.bool_)) or not isinstance(width,(int,np.integer)) or not 1<=width<=32:raise ValueError('width')
    return np.array(sorted(range(len(s)),key=lambda i:(-float(s[i]),tuple(map(tuple,p[i].tolist())),i))[:width],dtype=int)

import numpy as np


def branch_trajectory(H, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    for x,hi in ((rounds,6),(width,8)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 1<=x<=hi:raise ValueError('search budget')
    for x in (initial_iters,inner_iters):
        a=np.asarray(x)
        if a.ndim!=0 or a.dtype.kind not in 'iuf' or not np.isfinite(a) or a!=np.floor(a) or not 1<=a<=32:raise ValueError('iteration budget')
    h=np.asarray(H);l=np.asarray(prior)
    if h.ndim!=2 or h.dtype.kind not in 'biufc' or np.any(h.imag!=0) or not np.all(np.isfinite(h)) or np.any((h!=0)&(h!=1)):raise ValueError('matrix')
    h=h.real.astype(int)
    if h.shape[1]<=rounds or np.any(h.sum(axis=1)<rounds+2):raise ValueError('degree guard')
    if l.shape!=(h.shape[1],) or l.dtype.kind not in 'biufc' or np.any(l.imag!=0) or not np.all(np.isfinite(l)) or np.any(np.abs(l)>16):raise ValueError('prior')
    l=l.real.astype(float);fixed=np.full(len(l),-1,dtype=int)
    root=masked_run(h,l,syndrome,h*l,fixed,initial_iters)
    j,_,_=path_statistics(root[1],fixed,root[2])
    initial_node=j
    beam=[(root,fixed,(),j,0.,0.)]
    results={tuple(root[3])} if root[4] else set()
    for depth in range(rounds):
        children=[]
        for state,mask,path,nxt,score,A in beam:
            for value in (0,1):
                f=mask.copy();f[nxt]=value
                out=masked_run(h,l,syndrome,state[0],f,inner_iters)
                node,sc,total=path_statistics(out[1],f,out[2])
                if out[4]:results.add(tuple(out[3]))
                children.append((out,f,path+((nxt,value),),node,sc,total))
        indices=rank_paths(np.array([x[4] for x in children]),np.array([x[2] for x in children],dtype=int),width)
        beam=[children[int(i)] for i in indices]
    return np.array([x[4] for x in beam]),np.array([x[2] for x in beam],dtype=int),int(initial_node),int(beam[0][0][2]),float(beam[0][5]),int(len(results))

def terminal_score(modulus, offsets, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    H=detector_matrix(modulus,offsets)
    result=branch_trajectory(H,prior,syndrome,initial_iters,inner_iters,rounds,width)
    return float(result[0][0])

import numpy as np


def trace_score_cell(H, base, direction, syndrome, sample, bounds, initial_iters=4, inner_iters=3, rounds=3, width=2):
    from fractions import Fraction as F
    for val,high in ((initial_iters,6),(inner_iters,6),(rounds,3),(width,2)):
        if isinstance(val,(bool,np.bool_)) or not isinstance(val,(int,np.integer)) or not 1<=val<=high:raise ValueError('budget')
    vals=[]
    for x in (H,base,direction,syndrome):
        a=np.asarray(x)
        if a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):raise ValueError('real numeric array')
        vals.append(a)
    h,b,d,s=vals
    if h.ndim!=2 or not 1<=h.shape[0]<=8 or not 2<=h.shape[1]<=24:raise ValueError('matrix shape')
    m,n=h.shape
    if b.shape!=(n,) or d.shape!=(n,) or s.shape!=(m,):raise ValueError('vector shape')
    if np.any((h!=0)&(h!=1)) or np.any((s!=0)&(s!=1)):raise ValueError('binary')
    if n<=rounds or np.any(h.sum(1)<rounds+2) or np.any(np.abs(b)>16) or np.any(np.abs(d)>4):raise ValueError('bounds')
    def rational_table(v,shape):
        a=np.asarray(v,dtype=object)
        if a.shape!=shape:raise ValueError('rational shape')
        for x in a.flat:
            if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)):raise ValueError('integer pair')
        return a
    q=rational_table(sample,(2,));rr=rational_table(bounds,(2,2))
    if q[1]<=0 or any(row[1]<=0 for row in rr):raise ValueError('denominator')
    x=F(int(q[0]),int(q[1]));lo=F(int(rr[0,0]),int(rr[0,1]));hi=F(int(rr[1,0]),int(rr[1,1]))
    if not -1<=lo<x<hi<=1:raise ValueError('interval')
    cell=[lo,hi];cache={};zero=(F(0),F(0))
    def add(a,b):return a[0]+b[0],a[1]+b[1]
    def sub(a,b):return a[0]-b[0],a[1]-b[1]
    def mul(a,k):return a[0]*k,a[1]*k
    def total(values):
        a=zero
        for v in values:a=add(a,v)
        return a
    def sign(a):
        if a in cache:return cache[a]
        intercept,slope=a
        v=intercept+slope*x
        if not slope:z=(v>0)-(v<0)
        else:
            if v==0:raise ValueError('sample at nonconstant comparison boundary')
            z=1 if v>0 else -1;root=-intercept/slope
            if (z>0)==(slope>0):cell[0]=max(cell[0],root)
            else:cell[1]=min(cell[1],root)
        cache[a]=z
        return z
    def minimum(values):
        best=values[0]
        for v in values[1:]:
            if sign(sub(v,best))<0:best=v
        return best
    def argmin(values):
        best=0
        for k in range(1,len(values)):
            if sign(sub(values[k],values[best]))<0:best=k
        return best
    priors=[(F(float(b[j])),F(float(d[j]))) for j in range(n)]
    Nd=[[j for j in range(n) if h[i,j]] for i in range(m)]
    Ne=[[i for i in range(m) if h[i,j]] for j in range(n)]
    def bp(edge,path,budget):
        fixed=dict(path);res=[int(v) for v in s]
        for j,val in path:
            if val:
                for i in Ne[j]:res[i]^=1
        live=[j for j in range(n) if j not in fixed]
        E=edge.copy();acc=[zero for _ in range(n)]
        for t in range(1,budget+1):
            D={}
            for i in range(m):
                nb=[j for j in Nd[i] if j not in fixed]
                for j in nb:
                    values=[E[k,i] for k in nb if k!=j]
                    signs=[sign(v) for v in values]
                    factor=(-1)**res[i]
                    for z in signs:factor*=z
                    D[i,j]=mul(minimum([mul(v,z) for v,z in zip(values,signs)]),factor)
            new=E.copy();err=[0]*n
            for j in live:
                post=add(priors[j],total(D[i,j] for i in Ne[j]))
                for i in Ne[j]:new[j,i]=add(priors[j],total(D[k,j] for k in Ne[j] if k!=i))
                acc[j]=add(acc[j],post);err[j]=int(sign(post)<=0)
            E=new
            if all(sum(err[j] for j in Nd[i] if j not in fixed)%2==res[i] for i in range(m)):break
        mags=[mul(acc[j],sign(acc[j])) for j in live]
        node=live[argmin(mags)]
        return E,node,mul(total(mags),F(1,t))
    E0={(j,i):priors[j] for j in range(n) for i in Ne[j]}
    E,node,_=bp(E0,(),initial_iters)
    beam=[(E,(),node,zero)]
    for _ in range(rounds):
        children=[]
        for edge,path,j,score in beam:
            for val in (0,1):
                pp=path+((j,val),);ee,jj,ss=bp(edge,pp,inner_iters)
                children.append((ee,pp,jj,ss))
        for i in range(len(children)):
            for j in range(i+1,len(children)):sign(sub(children[i][3],children[j][3]))
        children.sort(key=lambda p:(-(p[3][0]+p[3][1]*x),p[1]));beam=children[:width]
    return [[int(v.numerator),int(v.denominator)] for v in [cell[0],cell[1],*beam[0][3]]]

import numpy as np


def cell_squared_integral(cell, reference):
    from fractions import Fraction as F
    c=np.asarray(cell,dtype=object)
    if c.shape!=(4,2):raise ValueError('cell shape')
    for x in c.flat:
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)):raise ValueError('integer rational pair')
    if any(row[1]<=0 for row in c):raise ValueError('denominator')
    a,b,u,v=[F(int(n),int(d)) for n,d in c]
    q=np.asarray(reference)
    if q.ndim!=0 or q.dtype.kind not in 'iuf' or not np.isfinite(q) or abs(float(q))>1e12:raise ValueError('reference')
    if not -1<=a<b<=1 or abs(u)>1e12 or abs(v)>1e12:raise ValueError('coefficient domain')
    r=F(float(q));za=u+v*a-r;zb=u+v*b-r
    return float((b-a)*(za*za+za*zb+zb*zb)/3)

import numpy as np


def uncertainty_response(modulus, offsets, base, syndrome, uncertain_index=12, lower=(-1,16), upper=(1,16), initial_iters=4, inner_iters=3, rounds=3, width=2):
    from fractions import Fraction as F
    import math
    H=detector_matrix(modulus,offsets);n=H.shape[1]
    if H.shape[0]>8 or n>24:raise ValueError('trace size')
    for x,hi in ((initial_iters,6),(inner_iters,6),(rounds,3),(width,2)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 1<=x<=hi:raise ValueError('schedule')
    if isinstance(uncertain_index,(bool,np.bool_)) or not isinstance(uncertain_index,(int,np.integer)) or not 0<=uncertain_index<n:raise ValueError('uncertain index')
    ends=[]
    for pair in (lower,upper):
        a=np.asarray(pair,dtype=object)
        if a.shape!=(2,) or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in a) or a[1]<=0:raise ValueError('endpoint')
        ends.append(F(int(a[0]),int(a[1])))
    lo,hi=ends
    if not -1<=lo<0<hi<=1:raise ValueError('interval')
    reference=terminal_score(modulus,offsets,base,syndrome,initial_iters,inner_iters,rounds,width)
    raw_base=np.asarray(base)
    if raw_base.dtype.kind not in 'biuf':raise ValueError('real base required')
    direction=np.zeros(n);direction[uncertain_index]=1
    pending=[(lo,hi)];left_parts=[];right_parts=[];count=0
    def rp(x):return [int(x.numerator),int(x.denominator)]
    while pending:
        l,h=pending.pop()
        found=False
        for den in range(2,258):
            x=l+(h-l)/den
            try:out=trace_score_cell(H,base,direction,syndrome,rp(x),[rp(l),rp(h)],initial_iters,inner_iters,rounds,width)
            except ValueError:
                continue
            found=True;break
        if not found:raise ValueError('cannot select cell interior')
        a,b,u,v=[F(int(p),int(q)) for p,q in out]
        if not l<=a<x<b<=h:raise ValueError('invalid cell')
        count+=1
        if count>4096:raise ValueError('partition cap')
        for aa,bb in ((a,min(b,F(0))),(max(a,F(0)),b)):
            if aa>=bb:continue
            val=cell_squared_integral([rp(aa),rp(bb),rp(u),rp(v)],reference)
            (left_parts if bb<=0 else right_parts).append(val)
        if a>l:pending.append((l,a))
        if b<h:pending.append((b,h))
    left=math.fsum(left_parts);right=math.fsum(right_parts)
    return (left+right)/float(hi-lo),reference,left/float(-lo),right/float(hi)
SCICODE_GOLD_EOF
