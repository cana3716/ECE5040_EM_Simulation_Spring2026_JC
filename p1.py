# p1.py, ECE 5040-ES Final Project, Problem 1
# Modified version of original source 1D_FDTD.py of Project 1. 
#
# Implements a 1D FDTD simulation of a plane wave propagating in 
# free space, strikinga PEC boundary. 
#
import numpy as np
import matplotlib.pyplot as plt

 # --- Simulation Constants & Parameters ---
tpause=0.001
nx = 400 # number of cells
nt = 520
c0 = 299792458.0
eps0 = 8.854e-12
epsr = 4.0
mu0 = 4*np.pi*1e-7
boundary = 'PEC' # PEC or ABC, etc.
f0=f=10e9
lambda_0=c0/f
sigma = 10 # pulse width
t0 = 40 # time of pulse peak
dx = 0.99*lambda_0/20 # 0.99 to ensure slighly below Courant limit
dt = dx/(np.sqrt(2)*c0) 
ce = dt/(eps0*dx)
ch = dt/(mu0*dx)
print('dx=%4.3e, dt=%4.3e'%(dx,dt))

# --- Initialize Fields---
# Ez at centers ( nx ) , Hy at edges ( nx + 1)
Ez = np.zeros(nx)
Hy = np.zeros(nx + 1)

# -- -Setup Visualization ---
plt.ion()
fig, ax = plt.subplots ( figsize =(10 , 5) )
line, = ax.plot(Ez,color='blue', lw=1.5)
ax.set_ylim([-1.2, 1.2])
ax.set_xlabel(' Cell Index (i) ')
ax.set_ylabel('Electric Field (Ez at center ) ')
plt.grid(visible=True, which='major', axis='both')

# --- Main Time - Stepping Loop ---
for tn in range(nt):

    # Update Magnetic Field ( Hy ) at edges
    Hy[1:-1] += ch*(Ez[1:] - Ez[:-1])

    # Store old Ez values for simple ABC
    ez_low_old = Ez[1]
    ez_high_old = Ez[-2]

    # Update Electric Field ( Ez ) at centers
    Ez[:] += ce*(Hy[1:] - Hy[: -1])

    # ABC Boundary Condition
#    Ez[0] = ez_low_old
#    Ez[-1] = ez_high_old

    # PEC Boundary Condition
    Ez[0]=0
    Ez[-1]=0

    # Source Injection 
    envelope = np.exp (-0.5 * (( tn - t0 ) / sigma )**2)
    carrier = np.cos(2*np.pi*f0*(tn-t0))
    pulse = envelope*carrier
    Ez[ nx // 4] += pulse # original configuration

    if tn % 10 == 0:
        line.set_ydata(Ez)
        plt.draw()
        plt.pause(tpause)

plt.ioff()
plt.show()

## --- Frequency Spectrum (FFT) Analysis of the Source Pulse
N=1024
t=np.arange(N)*dt

envelope = np.exp (-0.5 * (( t - t0 ) / sigma )**2)
carrier = np.cos(2*np.pi*f0*(t-t0))
pulse_td=envelope*carrier
pulse_fd=np.fft.fft(pulse_td) # compute fft
pulse_fd_shifted = np.fft.fftshift(pulse_td) # shift to center 
freqs=np.fft.fftfreq(N,d=dt)
freqs_shifted=np.fft.fftshift(freqs)
magnitude=np.abs(pulse_fd_shifted)
phase_rad=np.angle(pulse_fd_shifted)
phase_deg=np.degrees(phase_rad)
phase_deg[magnitude<1e-4]=np.nan

fig, (ax1,ax2)=plt.subplots(1, 2, figsize=(8,6))
# magnitude spectrum
ax1.plot(freqs_shifted/1e9, magnitude, color='blue', linewidth=2)
ax1.set_title('Magnitude Spectrum', fontsize=14)
ax1.set_xlabel('Frequency (GHz)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(6, 12)              # Zoom into the relevant frequency band
ax1.grid(True, linestyle='--', alpha=0.6)
# Phase Spectrum
ax2.plot(freqs_shifted/1e9, phase_deg, color='red', linewidth=2)
ax2.set_title('Phase Spectrum', fontsize=14)
ax2.set_xlabel('Frequency (GHz)', fontsize=12)
ax2.set_ylabel('Phase (Degrees)', fontsize=12)
ax2.set_xlim(-5, 5)              # Match the zoom window of the magnitude
ax2.set_ylim(-190, 190)          # Show standard wrapping limits
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig.savefig('fig-p1-4.eps')
plt.show()
