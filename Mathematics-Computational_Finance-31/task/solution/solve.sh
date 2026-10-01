#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def rbf_fd_weights(
    h: float, o: float, c: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a_minus = -o * (2 * c * c + h * h * o) / (2 * c * c * h * (o + 1))
    a_zero = h * (o - 1) / (2 * c * c) + (o - 1) / (h * o)
    a_plus = (h * h / (c * c) + 2 / o) / (2 * h * (o + 1))
    b_minus = (2 / (h * h) - (o - 3) * o / (c * c)) / (o + 1)
    b_zero = ((o * o - 4 * o + 1) / (c * c) - 2 / (h * h)) / o
    b_plus = (2 * c * c + h * h * (3 * o - 1)) / (c * c * h * h * o * (o + 1))
    return (a_minus, a_zero, a_plus), (b_minus, b_zero, b_plus)

def spatial_derivatives(
    values: tuple[float, float, float],
    first_weights: tuple[float, float, float],
    second_weights: tuple[float, float, float],
) -> tuple[float, float]:
    wx = sum(weight * value for weight, value in zip(first_weights, values))
    wxx = sum(weight * value for weight, value in zip(second_weights, values))
    return wx, wxx

def positive_z_from_x(
    x: float,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[float, float, float]:
    import math

    x = float(x)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError("x must be finite and positive")
    if not math.isfinite(residual_tol) or not 0.0 < residual_tol < 1.0:
        raise ValueError("residual_tol must lie in (0, 1)")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    target = math.sqrt(x)
    coefficients = (
        2.0 / 3.0,
        -8.0 / 15.0,
        16.0 / 35.0,
        -128.0 / 315.0,
        256.0 / 693.0,
        -1024.0 / 3003.0,
        2048.0 / 6435.0,
    )

    def _g_jet(y: float) -> tuple[float, float, float]:
        if y < 0.05:
            g = math.fsum(
                coefficient * y ** (2 * index + 3)
                for index, coefficient in enumerate(coefficients)
            )
            dg = math.fsum(
                (2 * index + 3) * coefficient * y ** (2 * index + 2)
                for index, coefficient in enumerate(coefficients)
            )
            d2g = math.fsum(
                (2 * index + 3)
                * (2 * index + 2)
                * coefficient
                * y ** (2 * index + 1)
                for index, coefficient in enumerate(coefficients)
            )
            return g, dg, d2g

        scale = math.hypot(1.0, y)
        inverse_scale = 1.0 / scale
        ratio = y * inverse_scale
        inverse_hyperbolic = math.asinh(y)
        g = y - inverse_hyperbolic * inverse_scale
        dg = ratio * ratio + inverse_hyperbolic * ratio * inverse_scale * inverse_scale
        d2g = inverse_scale ** 3 * (
            3.0 * ratio
            + inverse_hyperbolic
            * (inverse_scale * inverse_scale - 2.0 * ratio * ratio)
        )
        return g, dg, d2g

    span = max(1.0, math.ulp(target))
    lower, upper = target, target + span
    initial = math.exp((math.log(target) + math.log(1.5)) / 3.0)
    y = min(max(initial, lower), upper)

    for _ in range(max_iter):
        g, dg, d2g = _g_jet(y)
        residual = g - target
        if abs(residual) <= residual_tol * target:
            z = y * y
            elasticity = target / (y * dg)
            curvature = 0.5 * elasticity * (
                elasticity - 1.0 - elasticity * y * d2g / dg
            )
            result = (z, elasticity, curvature)
            if all(math.isfinite(value) for value in result):
                return result
            break

        if residual < 0.0:
            lower = y
        else:
            upper = y

        trial = y - residual / dg if dg > 0.0 else math.nan
        if not (math.isfinite(trial) and lower < trial < upper):
            trial = 0.5 * (lower + upper)
        y = trial

    raise ArithmeticError("positive-root solve did not meet the residual tolerance")

def nonlinear_bs_rhs(
    wx: float,
    wxx: float,
    w0: float,
    X: float,
    r: float,
    q: float,
    sigma0: float,
    z: float,
) -> float:
    diffusion = 0.5 * sigma0 ** 2 * (1 + z) * X ** 2 * wxx
    drift = (r - q) * X * wx
    discount = -r * w0
    return diffusion + drift + discount

def spatial_taylor_jets(values_coeff,h_coeff,o_coeff,c_coeff):
    import math
    def _read(a):
        if len(a)!=3 or any(len(row)!=3 for row in a):
            raise ValueError("each Taylor array must have shape (3,3)")
        out=[float(a[i][j]) for i in range(3) for j in range(3)]
        if not all(math.isfinite(x) for x in out):
            raise ValueError("finite coefficients required")
        return out
    def _rows(a): return [a[0:3],a[3:6],a[6:9]]
    def _const(x): return [float(x)]+[0.0]*8
    def _add(a,b): return [x+y for x,y in zip(a,b)]
    def _neg(a): return [-x for x in a]
    def _sub(a,b): return _add(a,_neg(b))
    def _scale(a,x): return [x*y for y in a]
    def _mul(a,b):
        out=[0.0]*9
        for i in range(3):
            for j in range(3):
                out[3*i+j]=math.fsum(a[3*k+l]*b[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1))
        return out
    def _reciprocal(a):
        if a[0]==0: raise ZeroDivisionError("zero Taylor constant")
        out=[0.0]*9;out[0]=1/a[0]
        for total in range(1,5):
            for i in range(3):
                j=total-i
                if 0<=j<3:
                    rem=math.fsum(a[3*k+l]*out[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1) if (k,l)!=(0,0))
                    out[3*i+j]=-rem/a[0]
        return out
    def _div(a,b): return _mul(a,_reciprocal(b))
    def _compose(a,c):
        delta=a.copy();delta[0]=0.0
        out=_const(c[4])
        for k in (3,2,1,0): out=_add(_mul(out,delta),_const(c[k]))
        return out
    if len(values_coeff)!=3: raise ValueError("three value Taylor arrays required")
    wm,w0,wp=[_read(x) for x in values_coeff]
    h,o,c=[_read(x) for x in (h_coeff,o_coeff,c_coeff)]
    if min(h[0],o[0],c[0])<=0: raise ValueError("positive h,o,c required")
    one=_const(1);two=_const(2);h2=_mul(h,h);c2=_mul(c,c)
    am=_neg(_div(_mul(o,_add(_scale(c2,2),_mul(h2,o))),_scale(_mul(_mul(c2,h),_add(o,one)),2)))
    a0=_add(_div(_mul(h,_sub(o,one)),_scale(c2,2)),_div(_sub(o,one),_mul(h,o)))
    ap=_div(_add(_div(h2,c2),_div(two,o)),_scale(_mul(h,_add(o,one)),2))
    bm=_div(_sub(_div(two,h2),_div(_mul(_sub(o,_const(3)),o),c2)),_add(o,one))
    b0=_div(_sub(_div(_add(_sub(_mul(o,o),_scale(o,4)),one),c2),_div(two,h2)),o)
    bp=_div(_add(_scale(c2,2),_mul(h2,_sub(_scale(o,3),one))),_mul(_mul(_mul(c2,h2),o),_add(o,one)))
    delta=_add(_add(_mul(am,wm),_mul(a0,w0)),_mul(ap,wp))
    gamma=_add(_add(_mul(bm,wm),_mul(b0,w0)),_mul(bp,wp))
    return _rows(delta),_rows(gamma)

def implicit_z_taylor_jet(x_coeff,residual_tol=1e-13,max_iter=24):
    import math
    def _read(a):
        if len(a)!=3 or any(len(row)!=3 for row in a):
            raise ValueError("each Taylor array must have shape (3,3)")
        out=[float(a[i][j]) for i in range(3) for j in range(3)]
        if not all(math.isfinite(x) for x in out):
            raise ValueError("finite coefficients required")
        return out
    def _rows(a): return [a[0:3],a[3:6],a[6:9]]
    def _const(x): return [float(x)]+[0.0]*8
    def _add(a,b): return [x+y for x,y in zip(a,b)]
    def _neg(a): return [-x for x in a]
    def _sub(a,b): return _add(a,_neg(b))
    def _scale(a,x): return [x*y for y in a]
    def _mul(a,b):
        out=[0.0]*9
        for i in range(3):
            for j in range(3):
                out[3*i+j]=math.fsum(a[3*k+l]*b[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1))
        return out
    def _reciprocal(a):
        if a[0]==0: raise ZeroDivisionError("zero Taylor constant")
        out=[0.0]*9;out[0]=1/a[0]
        for total in range(1,5):
            for i in range(3):
                j=total-i
                if 0<=j<3:
                    rem=math.fsum(a[3*k+l]*out[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1) if (k,l)!=(0,0))
                    out[3*i+j]=-rem/a[0]
        return out
    def _div(a,b): return _mul(a,_reciprocal(b))
    def _compose(a,c):
        delta=a.copy();delta[0]=0.0
        out=_const(c[4])
        for k in (3,2,1,0): out=_add(_mul(out,delta),_const(c[k]))
        return out
    def _sqrt(a):
        x=a[0]
        if x<=0: raise ValueError("positive square-root constant required")
        z=math.sqrt(x)
        return _compose(a,[z,1/(2*z),-1/(8*x*z),1/(16*x*x*z),-5/(128*x*x*x*z)])
    def _asinh(a):
        x=a[0];d=1+x*x
        return _compose(a,[math.asinh(x),d**-.5,-.5*x*d**-1.5,(2*x*x-1)*d**-2.5/6,x*(3-2*x*x)*d**-3.5/8])
    x=_read(x_coeff)
    if x[0]<=0: raise ValueError("positive baseline liquidity required")
    z0,elasticity,_=positive_z_from_x(x[0],residual_tol,max_iter)
    slope=x[0]/(z0*elasticity)
    if not math.isfinite(slope) or slope<=0: raise ArithmeticError("invalid implicit slope")
    def _phi(z):
        root=_sqrt(z)
        if root[0]<.05:
            coefficients=(2/3,-8/15,16/35,-128/315,256/693,-1024/3003,2048/6435)
            term=_mul(root,z);g=_const(0)
            for coefficient in coefficients:
                g=_add(g,_scale(term,coefficient));term=_mul(term,z)
        else:
            g=_sub(root,_div(_asinh(root),_sqrt(_add(_const(1),z))))
        return _mul(g,g)
    z=_const(z0)
    for total in range(1,5):
        for i in range(3):
            j=total-i
            if 0<=j<3:
                remainder=_sub(_phi(z),x)
                z[3*i+j]=-remainder[3*i+j]/slope
    if not all(math.isfinite(v) for v in z): raise ArithmeticError("nonfinite inverse Taylor coefficients")
    return _rows(z)

def rk4_taylor_step(p_coeff,psi=.002,residual_tol=1e-13,max_iter=24):
    import math
    def _read(a):
        if len(a)!=3 or any(len(row)!=3 for row in a):
            raise ValueError("each Taylor array must have shape (3,3)")
        out=[float(a[i][j]) for i in range(3) for j in range(3)]
        if not all(math.isfinite(x) for x in out):
            raise ValueError("finite coefficients required")
        return out
    def _rows(a): return [a[0:3],a[3:6],a[6:9]]
    def _const(x): return [float(x)]+[0.0]*8
    def _add(a,b): return [x+y for x,y in zip(a,b)]
    def _neg(a): return [-x for x in a]
    def _sub(a,b): return _add(a,_neg(b))
    def _scale(a,x): return [x*y for y in a]
    def _mul(a,b):
        out=[0.0]*9
        for i in range(3):
            for j in range(3):
                out[3*i+j]=math.fsum(a[3*k+l]*b[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1))
        return out
    def _reciprocal(a):
        if a[0]==0: raise ZeroDivisionError("zero Taylor constant")
        out=[0.0]*9;out[0]=1/a[0]
        for total in range(1,5):
            for i in range(3):
                j=total-i
                if 0<=j<3:
                    rem=math.fsum(a[3*k+l]*out[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1) if (k,l)!=(0,0))
                    out[3*i+j]=-rem/a[0]
        return out
    def _div(a,b): return _mul(a,_reciprocal(b))
    def _compose(a,c):
        delta=a.copy();delta[0]=0.0
        out=_const(c[4])
        for k in (3,2,1,0): out=_add(_mul(out,delta),_const(c[k]))
        return out
    if len(p_coeff)!=12: raise ValueError("12 parameter Taylor arrays required")
    p=[_read(x) for x in p_coeff]
    if not math.isfinite(psi) or psi<=0: raise ValueError("positive finite time step required")
    if not math.isfinite(residual_tol) or not 0<residual_tol<1 or not isinstance(max_iter,int) or max_iter<1:
        raise ValueError("invalid root controls")
    wm,w0,wp,h,o,c,r,q,tau,alpha,sigma,X=p
    if min(h[0],o[0],c[0],alpha[0],sigma[0],X[0])<=0:
        raise ValueError("positive baseline geometry and volatility parameters required")
    if tau[0]<0 or tau[0]+psi>1: raise ValueError("all baseline RK4 stage times must lie in [0,1]")
    def _exp(a):
        base=math.exp(a[0])
        return _compose(a,[base,base,base/2,base/6,base/24])
    stages=[]
    def _rhs(y,shift):
        time=_add(tau,_const(shift))
        delta,gamma=spatial_taylor_jets([_rows(wm),_rows(y),_rows(wp)],_rows(h),_rows(o),_rows(c))
        delta=_read(delta);gamma=_read(gamma)
        first,second=rbf_fd_weights(h[0],o[0],c[0])
        delta[0],gamma[0]=spatial_derivatives((wm[0],y[0],wp[0]),first,second)
        if gamma[0]<=0: raise ValueError("each baseline stage requires positive Gamma")
        x=_mul(_mul(_mul(_exp(_mul(r,time)),_mul(alpha,alpha)),_mul(X,X)),gamma)
        z=_read(implicit_z_taylor_jet(_rows(x),residual_tol,max_iter))
        diffusion=_scale(_mul(_mul(_mul(_mul(sigma,sigma),_add(_const(1),z)),_mul(X,X)),gamma),.5)
        drift=_mul(_mul(_sub(r,q),X),delta)
        rate=_sub(_add(diffusion,drift),_mul(r,y))
        rate[0]=nonlinear_bs_rhs(delta[0],gamma[0],y[0],X[0],r[0],q[0],sigma[0],z[0])
        stages.append((time[0],y[0],delta[0],gamma[0],x[0],z[0],rate[0]))
        return rate
    k1=_rhs(w0,0)
    k2=_rhs(_add(w0,_scale(k1,psi/2)),psi/2)
    k3=_rhs(_add(w0,_scale(k2,psi/2)),psi/2)
    k4=_rhs(_add(w0,_scale(k3,psi)),psi)
    annualized=_scale(_add(_add(k1,_scale(k2,2)),_add(_scale(k3,2),k4)),1/6)
    if not all(math.isfinite(x) for x in annualized): raise ArithmeticError("nonfinite annualized Taylor coefficients")
    return _rows(annualized),tuple(stages)

def solve(base=None,direction_s=None,direction_t=None,psi=.002):
    import math
    if base is None: base=[8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100]
    if direction_s is None: direction_s=[0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4]
    if direction_t is None: direction_t=[-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3]
    if any(len(a)!=12 for a in (base,direction_s,direction_t)):
        raise ValueError("base and two directions must have length 12")
    if not all(math.isfinite(float(x)) for a in (base,direction_s,direction_t) for x in a):
        raise ValueError("finite inputs required")
    p=[]
    for x,ds,dt in zip(base,direction_s,direction_t):
        p.append([[float(x),float(dt),0.0],[float(ds),0.0,0.0],[0.0,0.0,0.0]])
    annualized,_=rk4_taylor_step(p,psi)
    return float(4*annualized[2][2])
SCICODE_GOLD_EOF
