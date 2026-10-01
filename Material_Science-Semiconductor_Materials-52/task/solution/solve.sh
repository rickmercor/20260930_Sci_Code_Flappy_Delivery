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
from scipy.linalg import expm
from scipy.optimize import least_squares

def polaron_optical_rates(huang_rhys: float, gamma0: float) -> np.ndarray:
    if not (np.isfinite(huang_rhys) and 0<=huang_rhys<=1 and np.isfinite(gamma0) and gamma0>=0):raise ValueError("invalid coupling or population rate")
    return gamma0*np.array([1.,huang_rhys,(1-huang_rhys)**2])

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def polaron_free_generator(omega: float, detuning: float, rates: np.ndarray, gamma_ph: float) -> np.ndarray:
    if np.shape(rates)!=(3,) or not np.all(np.isfinite(rates)) or np.any(rates<0) or not np.all(np.isfinite([omega,detuning,gamma_ph])) or min(omega,gamma_ph)<0:raise ValueError("invalid generator data")
    e=np.array([0.,omega,detuning,detuning+omega])
    h=np.diag(e);eye=np.eye(4)
    generator=-1j*(np.kron(eye,h)-np.kron(h.T,eye))
    # States [0, 0', X, X']; separate decay channels as in the source inflow.
    for dest,origin,rate in [(0,2,rates[0]),(1,2,rates[1]),(0,3,rates[1]),
                             (1,3,rates[2]),(0,1,gamma_ph),(2,3,gamma_ph)]:
        jump=np.zeros((4,4));jump[dest,origin]=np.sqrt(rate)
        loss=jump.T@jump
        generator+=np.kron(jump,jump)-.5*(np.kron(eye,loss)+np.kron(loss.T,eye))
    return generator

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def polaron_pulse_coefficient(huang_rhys: float, phase: float, order: int) -> np.ndarray:
    if not np.all(np.isfinite([huang_rhys,phase])) or not 0<=huang_rhys<=1 or order not in (0,1,2):raise ValueError("invalid pulse data")
    s=np.sqrt(huang_rhys)
    raising=np.zeros((4,4),dtype=complex);raising[2:,:2]=np.array([[1.,s],[s,1.]])
    interaction=np.exp(1j*phase)*raising+np.exp(-1j*phase)*raising.conj().T
    action=-1j*(np.kron(np.eye(4),interaction)-np.kron(interaction.T,np.eye(4)))
    if order==0:return np.eye(16,dtype=complex)
    if order==1:return action
    if order==2:return action@action/2
    raise ValueError('order must be 0, 1 or 2')

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def single_mode_rephasing_echo(huang_rhys: float, omega: float, gamma0: float, gamma_ph: float, tau: float, u: float, detuning: float) -> complex:
    if not np.all(np.isfinite([tau,u])) or min(tau,u)<0:raise ValueError("delays must be finite and nonnegative")
    rates=polaron_optical_rates(huang_rhys,gamma0)
    generator=polaron_free_generator(omega,detuning,rates,gamma_ph)
    first=np.zeros((16,16),dtype=complex);second=first.copy()
    for j in range(5):
        phase=2*np.pi*j/5
        first+=np.exp(1j*phase)*polaron_pulse_coefficient(huang_rhys,phase,1)/5
        second+=np.exp(-2j*phase)*polaron_pulse_coefficient(huang_rhys,phase,2)/5
    rho0=np.zeros(16,dtype=complex);rho0[0]=1
    rho=(expm(generator*u)@second@expm(generator*tau)@first@rho0).reshape((4,4),order='F')
    s=np.sqrt(huang_rhys)
    return complex(rho[2,0]+rho[3,1]+s*(rho[3,0]+rho[2,1]))

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def weak_polaron_echo(times: np.ndarray, omega: float, huang_rhys: float, gamma0: float, gamma_ph: float) -> np.ndarray:
    if np.ndim(times)!=1 or not np.all(np.isfinite(times)) or np.any(times<0) or not np.all(np.isfinite([omega,huang_rhys,gamma0,gamma_ph])) or min(omega,gamma0,gamma_ph)<0 or not 0<=huang_rhys<=.2:raise ValueError("invalid calibration-kernel data")
    t=np.asarray(times);s=huang_rhys
    return np.exp(-(1+s)*gamma0*t)*(1+s*np.exp(-(gamma_ph-s*gamma0)*t)
        +s*np.cos(omega*t)*np.exp(-.5*(gamma_ph-2*s*gamma0)*t)
          *(2+np.exp(-s*gamma0*t)+np.exp(-gamma_ph*t))
        +s*np.cos(2*omega*t)*np.exp(-(gamma_ph-s*gamma0)*t))

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def recover_intrinsic_echo(channels: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    if np.ndim(channels)!=2 or channels.shape[0]!=2 or np.shape(mixing)!=(2,2) or not np.all(np.isfinite(channels)) or not np.all(np.isfinite(mixing)):raise ValueError("invalid detector data")
    try:intrinsic=np.linalg.solve(mixing,channels)
    except np.linalg.LinAlgError as exc:raise ValueError("singular detector response") from exc
    denominator=intrinsic[0]+2*intrinsic[1]
    if np.any(denominator==0):raise ValueError("undefined polarization contrast")
    return np.stack([denominator/2,(intrinsic[0]-3*intrinsic[1])/denominator])

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def fit_polaron_bath(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float) -> np.ndarray:
    if np.ndim(times)!=1 or np.shape(channels)!=(2,len(times)) or np.shape(omega)!=(2,) or np.shape(weights)!=(2,) or not np.all(np.isfinite(omega)) or not np.all(np.isfinite(weights)) or min(omega)<=0 or min(weights)<=0 or not np.isclose(sum(weights),1.,rtol=0,atol=1e-12) or omega[0]==omega[1] or not np.isfinite(gamma0) or gamma0<=0:raise ValueError("invalid two-mode fit data")
    target=recover_intrinsic_echo(channels,mixing)[0]
    def residual(z):
        prediction=sum(weights[i]*weak_polaron_echo(times,omega[i],z[i],gamma0,z[2+i]) for i in range(2))
        return prediction-target
    candidates=[]
    for start in [(.06,.035,.11,.16),(.14,.075,.23,.075)]:
        result=least_squares(residual,start,bounds=([.01,.005,.04,.04],[.18,.12,.35,.35]),
          xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
        candidates.append((float(result.fun@result.fun),tuple(result.x)))
    candidates.sort()
    return np.round(np.array(candidates[0][1]),7)

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def predict_polaron_echo(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float, tau: float, gate: np.ndarray,
                         detunings: np.ndarray, detuning_weights: np.ndarray, diagnostic: int=0) -> float:
    if np.shape(gate)!=(2,) or not np.all(np.isfinite(gate)) or not 0<=gate[0]<gate[1] or not np.isfinite(tau) or tau<0 or np.ndim(detunings)!=1 or np.shape(detuning_weights)!=np.shape(detunings) or not np.all(np.isfinite(detunings)) or not np.all(np.isfinite(detuning_weights)) or np.any(detuning_weights<0) or not np.isclose(sum(detuning_weights),1.,rtol=0,atol=1e-12) or diagnostic not in range(7):raise ValueError("invalid prediction data")
    fitted=fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)
    def amplitude(u,reference=False):
        return sum(weights[i]*detuning_weights[j]*single_mode_rephasing_echo(
              0. if reference else fitted[i],omega[i],gamma0,fitted[2+i],tau,u,delta)
              for i in range(2) for j,delta in enumerate(detunings))
    nodes,quadrature=np.polynomial.legendre.leggauss(12)
    mid=.5*(gate[0]+gate[1]);half=.5*(gate[1]-gate[0])
    numerator=sum(q*abs(amplitude(mid+half*x))**2 for x,q in zip(nodes,quadrature))
    denominator=sum(q*abs(amplitude(mid+half*x,True))**2 for x,q in zip(nodes,quadrature))
    center=amplitude(mid)/amplitude(mid,True)
    values=np.concatenate([[numerator/denominator],fitted,[center.real,center.imag]])
    return float(values[int(diagnostic)])
SCICODE_GOLD_EOF
