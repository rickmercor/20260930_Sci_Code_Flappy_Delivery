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

def component_point_kinematics(C,y,d,points,B1,B2,L1,L2):
    C=np.asarray(C,float); y=np.asarray(y,float); d=np.asarray(d,float); points=np.asarray(points,float)
    mats=[np.asarray(M,float) for M in (B1,B2,L1,L2)]
    if C.shape!=(2,2): raise ValueError('C must have shape (2,2)')
    if y.shape!=(2,) or d.shape!=(2,): raise ValueError('y and d must have shape (2,)')
    if points.ndim!=2 or points.shape[1]!=5 or points.shape[0]<1: raise ValueError('points must have shape (m,5), m>=1')
    if any(M.shape!=(3,3) for M in mats): raise ValueError('B1, B2, L1, L2 must have shape (3,3)')
    if any(not np.all(np.isfinite(x)) for x in [C,y,d,points]+mats): raise ValueError('all inputs must be finite')
    if np.any(points[:,4]<=0): raise ValueError('weights must be positive')
    z=C@y; B1,B2,L1,L2=mats; m=points.shape[0]
    F_points=np.empty((m,3,3)); phis=np.empty((m,9,2))
    for j,(a,b,e1,e2,w) in enumerate(points):
        F_points[j]=np.eye(3)+a*z[0]*B1+b*z[1]*B2+e1*d[0]*L1+e2*d[1]*L2
        phis[j]=np.column_stack((a*B1.ravel(order='C'),b*B2.ravel(order='C')))
    return F_points,phis

import numpy as np

def mooney_rivlin_state(F,directions):
    F=np.asarray(F,float); directions=np.asarray(directions,float)
    if F.shape!=(3,3): raise ValueError('F must have shape (3,3)')
    if directions.shape!=(2,3,3): raise ValueError('directions must have shape (2,3,3)')
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(directions)): raise ValueError('all inputs must be finite')
    J=float(np.linalg.det(F))
    if J<=0: raise ValueError('det(F) must be positive')
    invF=np.linalg.inv(F); C=F.T@F
    I1=float(np.trace(C)); I2=.5*(I1*I1-float(np.trace(C@C)))
    s1=J**(-2/3); s2=J**(-4/3)
    state=np.array([J,I1,I2,s1,s2],float)
    diff=np.empty((2,5),float)
    for q,H in enumerate(directions):
        tr=float(np.trace(invF@H)); dJ=J*tr
        dC=H.T@F+F.T@H; dI1=float(np.trace(dC)); dI2=I1*dI1-float(np.trace(C@dC))
        diff[q]=[tr,dJ,dI1,dI2,tr]
    return state,diff

import numpy as np

def mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa):
    F=np.asarray(F,float); directions=np.asarray(directions,float); state=np.asarray(state,float); differential_state=np.asarray(differential_state,float)
    vals=np.array([c1,c2,kappa],float)
    if F.shape!=(3,3): raise ValueError('F must have shape (3,3)')
    if directions.shape!=(2,3,3): raise ValueError('directions must have shape (2,3,3)')
    if state.shape!=(5,) or differential_state.shape!=(2,5): raise ValueError('invalid constitutive state shape')
    if any(not np.all(np.isfinite(x)) for x in (F,directions,state,differential_state,vals)): raise ValueError('all inputs must be finite')
    if np.any(vals<=0): raise ValueError('material parameters must be positive')
    J,I1,I2,s1,s2=state
    if J<=0: raise ValueError('J must be positive')
    # Guard against an inconsistent caller-provided state without recomputing the full state pipeline.
    if not np.isclose(J,np.linalg.det(F),rtol=1e-10,atol=1e-12): raise ValueError('state is inconsistent with F')
    invF=np.linalg.inv(F); invFT=invF.T; C=F.T@F; G=I1*F-F@C
    beta=-(2/3)*c1*s1*I1-(4/3)*c2*s2*I2+.5*kappa*(J*J-1)
    P=2*c1*s1*F+2*c2*s2*G+beta*invFT
    dP=np.empty((2,3,3))
    for q,H in enumerate(directions):
        tr,dJ,dI1,dI2,dlogJ=differential_state[q]
        ds1=-(2/3)*s1*dlogJ; ds2=-(4/3)*s2*dlogJ
        dC=H.T@F+F.T@H; dG=dI1*F+I1*H-H@C-F@dC
        dbeta=-(2/3)*c1*(ds1*I1+s1*dI1)-(4/3)*c2*(ds2*I2+s2*dI2)+kappa*J*dJ
        dinvFT=-invFT@H.T@invFT
        dP[q]=2*c1*(ds1*F+s1*H)+2*c2*(ds2*G+s2*dG)+dbeta*invFT+beta*dinvFT
    return P,dP

import numpy as np

def component_hyperreduced_system(phis,weights,stresses,tangent_actions):
    phis=np.asarray(phis,float); weights=np.asarray(weights,float); stresses=np.asarray(stresses,float); tangent_actions=np.asarray(tangent_actions,float)
    if phis.ndim!=3 or phis.shape[1:]!=(9,2) or phis.shape[0]<1: raise ValueError('phis must have shape (m,9,2), m>=1')
    m=phis.shape[0]
    if weights.shape!=(m,) or stresses.shape!=(m,3,3) or tangent_actions.shape!=(m,2,3,3): raise ValueError('point arrays have incompatible shapes')
    if any(not np.all(np.isfinite(x)) for x in (phis,weights,stresses,tangent_actions)): raise ValueError('all inputs must be finite')
    if np.any(weights<=0): raise ValueError('weights must be positive')
    r=np.zeros(2); K=np.zeros((2,2))
    for c in range(m):
        r+=weights[c]*(phis[c].T@stresses[c].ravel(order='C'))
        for q in range(2): K[:,q]+=weights[c]*(phis[c].T@tangent_actions[c,q].ravel(order='C'))
    return r,K

import numpy as np

def global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext):
    component_maps=np.asarray(component_maps,float); local_residuals=np.asarray(local_residuals,float); local_jacobians=np.asarray(local_jacobians,float); f_ext=np.asarray(f_ext,float)
    if component_maps.ndim!=3 or component_maps.shape[1:]!=(2,2) or component_maps.shape[0]<1: raise ValueError('component_maps must have shape (n,2,2), n>=1')
    n=component_maps.shape[0]
    if local_residuals.shape!=(n,2) or local_jacobians.shape!=(n,2,2) or f_ext.shape!=(2,): raise ValueError('incompatible global assembly shapes')
    if any(not np.all(np.isfinite(x)) for x in (component_maps,local_residuals,local_jacobians,f_ext)): raise ValueError('all inputs must be finite')
    r=-f_ext.copy(); K=np.zeros((2,2))
    for i,C in enumerate(component_maps):
        r+=C.T@local_residuals[i]; K+=C.T@local_jacobians[i]@C
    return r,K

import numpy as np

def newton_update_diagnostics(y,r,K):
    y=np.asarray(y,float); r=np.asarray(r,float); K=np.asarray(K,float)
    if y.shape!=(2,) or r.shape!=(2,) or K.shape!=(2,2): raise ValueError('y and r must have shape (2,), K must have shape (2,2)')
    if any(not np.all(np.isfinite(x)) for x in (y,r,K)): raise ValueError('all inputs must be finite')
    try: delta=np.linalg.solve(K,r)
    except np.linalg.LinAlgError as exc: raise ValueError('K must be nonsingular') from exc
    cond=float(np.linalg.cond(K,2))
    if not np.isfinite(cond): raise ValueError('condition number must be finite')
    return y-delta,np.array([np.linalg.norm(r),cond,np.linalg.norm(delta)])

import numpy as np

def first_newton_coordinate(y, d, component_maps, point_sets,
                                    B1, B2, L1, L2, f_ext,
                                    c1, c2, kappa):
    local_residuals = []
    local_jacobians = []

    for C, points in zip(component_maps, point_sets):
        F_points, phis = component_point_kinematics(
            C, y, d, points, B1, B2, L1, L2
        )

        stresses = []
        tangent_actions = []
        for F, point in zip(F_points, points):
            a, b = point[0], point[1]
            directions = np.stack((a * B1, b * B2))
            state, differential_state = mooney_rivlin_state(
                F, directions
            )
            P, dP = mooney_rivlin_constitutive(
                F, directions, state, differential_state, c1, c2, kappa
            )
            stresses.append(P)
            tangent_actions.append(dP)

        r_local, K_local = component_hyperreduced_system(
            phis,
            np.asarray(points)[:, 4],
            np.asarray(stresses),
            np.asarray(tangent_actions)
        )
        local_residuals.append(r_local)
        local_jacobians.append(K_local)

    r, K = global_reduced_system(
        component_maps,
        np.asarray(local_residuals),
        np.asarray(local_jacobians),
        f_ext
    )
    y_new, diagnostics = newton_update_diagnostics(y, r, K)
    return float(y_new[0])
SCICODE_GOLD_EOF
