# p4.py, ECE 5040-ES Final Project, Problem 4
# Note: Modified version of original source 1D_FDTD.py of Project 1. 
#
# Description: 1D FDTD domain containing a vacuum-to-silicon (n~=3.48) 
# interface. Excite the vacuum region with a broadband pulse centered
# around 192 THz (lambda_0 = 1.55*mu meters). Calculate the numerical
# reflection and transmission coefficients and compare against the
# analytical Fresnel equations. 
#
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.fft import fft, fftfreq

# Simulation Constants & Parameters ---
tpause=0.001
nx = 500 # Number of cells
interface_position=nx//2

nt = 700 
N=nt
c0 = 299792458.0
eps0 = 8.854e-12
mu0 = 4*np.pi*1e-7
boundary = 'PEC' # PEC or ABC, etc.

n1=1.0 # index of refraction for vacuum
n2=n_silicon=3.48 # index of refraction for silicon
#n2=n1 # index of refraction for silicon
epsr_silicon=n2**2 # relative permittivity of silicon

eta_0=np.sqrt(mu0/eps0)
eta_1=eta_silicon=np.sqrt(mu0/(epsr_silicon*eps0))
R_ana=(eta_1-eta_0)/(eta_1+eta_0)
T_ana=(2*eta_1)/(eta_1+eta_0)

f=192e12
lambda_0=c0/f
dx = 0.99*lambda_0/20 # 0.99 to ensure slightly below Courant limit
dt = dx/(np.sqrt(2)*c0) 

### Source parameters
source_position=10
t1 = 0  # heaviside step time 1
t2 = 100 # heavisdide step time 2 
t0 = 40 # time of pulse peak
A0=2 # pulse amplitude
f0=f # carrier frequency
omega=2*np.pi*f0
sigma = 1/f # pulse width
sigma= 40

### Configure ce
ce_vacuum = dt/(eps0*dx)
ce_silicon = dt/(epsr_silicon*eps0*dx)
ce=np.ones(nx)
ce[:interface_position]*=ce_vacuum
ce[interface_position:]*=ce_silicon

ch = dt/(mu0*dx)

# Initialize Fields ---
# Ez at centers ( nx ) , Hy at edges ( nx + 1)
Ez = np.zeros(nx)
Hy = np.zeros(nx + 1)

# Setup Visualization ---
plt.ion()
fig, ax = plt.subplots ( figsize =(10 , 5) )
line, = ax.plot(Ez,color='blue', lw=1.5, label=r'$E(i)$')
ax.plot(np.linspace(0,nx-1,nx),np.sqrt(1/(ce*dx/dt*eps0)),label=r'$n(i)=\sqrt{\epsilon_r}$')
ax.set_ylim([-2.0, 4])
ax.set_xlabel(' Cell Index (i) ')
ax.set_ylabel(r'Electric Field ($E_z$ at center ) ')
ax.legend()
plt.grid(visible=True, which='major', axis='both')

point_list1=[(0,1.0),(interface_position,1.0),(interface_position,1.0),(nx,1.0)]
point_list2=[(0,-1.0),(interface_position,-1.0),(interface_position,-1.0),(nx,-1.0)]
point_list3=[(interface_position,-1.0),(interface_position,1.0)]
original1=LineString(point_list1)
original2=LineString(point_list2)
original3=LineString(point_list3)

# Main Time - Stepping Loop ---
for tn in range(nt):

 # Update Magnetic Field ( Hy ) at edges
 # Hy [ i ] depends on Ez [ i ] and Ez [i -1]
    Hy[1:-1] += ch*(Ez[1:] - Ez[:-1])

 # Store old Ez values for simple ABC
    ez_low_old = Ez[1]
    ez_high_old = Ez[-2]

# Update Electric Field ( Ez ) at centers
    Ez[:] += ce*(Hy[1:] - Hy[: -1])

## ABC Boundary Condition
#    Ez[0] = ez_low_old
#    Ez[-1] = ez_high_old

## PEC Boundary Condition
    Ez[0]=0
    Ez[-1]=0

# --- 6. Soft Source Injection ---
   
    if (tn==0): 
        print('t0=%4.3e, f0=%4.3e, A0=%4.3e' % (t0,f0,A0))
    #step_fcn=np.heaviside(tn-t1,1)-np.heaviside(tn-t2,1)
    #harmonic_fcn=np.sin(omega*(tn-t0))
    pulse = A0*np.exp (-0.5 * (( tn - t0 ) / sigma )**2)
    
    Ez[ source_position ] += pulse # original configuration

    if tn % 10 == 0:
        line.set_ydata(Ez)
        plt.draw()
        plt.pause(tpause)

plt.ioff()
plt.show()

## Analysis

### Run a seperate simulation with a continuous, homogenous background medium 
### to record the incident-only field E_inc
###E_inc=Ez
###np.savez("p4-fdtd_reference_data.npz", E_inc_data=E_inc)
###print("Reference field sucessfully saved to 'p4-fdtd_reference_data.npz'.")

loaded_data=np.load("p4-fdtd_reference_data.npz")
E_inc=loaded_data["E_inc_data"]

E_tot=Ez # total field
E_tran=np.zeros(len(Ez)) 
E_refl=np.zeros(len(Ez)) 
E_refl[0:interface_position]=Ez[0:interface_position] # reflected field
E_tran[interface_position:]=Ez[interface_position:] # transmitted field

E_inc_fd=np.fft.fft(E_inc)
E_refl_fd=np.fft.fft(E_refl)
E_tran_fd=np.fft.fft(E_tran)
freqs=np.fft.fftfreq(len(E_inc),d=dt)
window_mask=(freqs>=0)&(freqs<=300e12)
freqs_windowed=freqs[window_mask]

S11=E_refl_fd/E_inc_fd
S11_windowed=S11[window_mask]
S21=E_tran_fd/E_inc_fd
S21_windowed=S21[window_mask]

t=np.arange(len(E_tot))
fig, ax=plt.subplots()
plt.plot(t,E_tot,'b',label='$E_{tot}$')
plt.plot(t,E_refl,'g',label='$E_{refl}$')
plt.plot(t,E_tran,'r',label='$E_{tran}$')
plt.xlabel('time (n)')
plt.ylabel(r'$E$-field')
plt.title('Time-domain Signals')
plt.grid()
plt.legend()
#plt.savefig('fig-p4-2.eps')

fig, ax=plt.subplots()
plt.plot(freqs_windowed/1e12,np.abs(S11_windowed), 'b', label='$S_{11}$')
plt.plot(freqs_windowed/1e12,np.abs(S21_windowed), 'r', label='$S_{21}$')
plt.xlabel('Freq (THz)')
plt.ylabel(r'$|S|$')
plt.grid()
plt.legend()
ax.set_xlim(180,200)
ax.set_ylim(0,1.0)
#plt.savefig('fig-p4-3.eps')

plt.show()
