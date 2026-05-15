# p5.py, ECE5040-ES Final Project, Problem 5 
#
# Description: Simulate a 2D SOI slab waveguide (silicon, n=3.48)
# surrounded by a cladding (silicon dioxide n=1.44). Calculate
# the fundamental TE transverse mode profile analytically and use
# it as a "hard source" spatial envelope at the input boundary, 
# oscillating at 192 THz. 
#
# Ref. [1] A. Zadehgol: Complex s-Plane Modeling and 2D Characterization 
#          of Stochastic Scattering Loss, IEEE Access, Vol 9, 2021. 
#
# Mapping: TMz: Ez, Hx, Hy (FDTD)
#          TEz: Ey, Hz, Hx (Paper [1])
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString
from scipy.integrate import quad

## Physical Constants and Simulation Parameters
c = 299792458.0              # Speed of light in vacuum (m/s)
mu0 = 4.0 * np.pi * 1e-7     # Permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c**2)    # Permittivity of free space (F/m)
n1=3.48                      # silicon core, index of refraction
n2=1.44                      # silicon dioxide cladding, index of refraction
epsr1=epsr_core = n1**2      # Relative permittivity of cladding 
epsr2=epsr_clad = n2**2      # Relative permittivity of cladding 
eps1=eps0*epsr_core
eps2=eps0*epsr_clad
print('epsr1=%4.3e, epsr2=%4.3e' % (epsr1,epsr2))

## Source Parameters
f = 192e12                  # Source frequency 
lambda_0 = c / f            # Free-space wavelength 
print('lambda_0=%4.3e' % lambda_0)
kappa_0=2*np.pi/lambda_0    # Free-space wavenumber
omega = 2.0 * np.pi * f
beta1 = omega*np.sqrt(mu0*eps1) # Phase Constant of Core
beta2 = omega*np.sqrt(mu0*eps2) # Phase Constant of Cladding
print('beta1=%4.3e (rad/s), beta2=%4.3e (rad/s)' % (beta1, beta2))

tau = 1/f/5                 # Gaussian pulse width 
t0 = 3.0 * tau              # Pulse delay to ensure a smooth turn-on
source_amplitude = 1.0      # Peak amplitude of the source
print('tau=%4.3e, t0=%4.3e, source_amplitude=%4.3f' % (tau, t0, source_amplitude))

## Grid Parameters
Nx, Ny = 1000, 1000         # Nx x Ny cells grid
dl = lambda_0 / 40         # Spatial resolution (dx = dy). 40 points per wavelength
print('dl=%4.3E (meters)' % (dl))

# SOI Core Geometry
core_target_length=10e-6 # meters
core_target_width=3e-6 # meters
i_core_min,i_core_max=-int(np.ceil(core_target_length/2/dl)),int(np.ceil(core_target_length/2/dl))
j_core_min,j_core_max=-int(np.ceil(core_target_width/2/dl)),int(np.ceil(core_target_width/2/dl))
print('core_length=%4.3e' % ((i_core_max-i_core_min)*dl))
print('core_width=%4.3e' % ((j_core_max-j_core_min)*dl))

# Center coordinates for the infinite line source
ic, jc = Nx // 2, Ny // 2

# Compute Courant Number
def S(N_lambda): 
    return c*dt*N_lambda/lambda_0

# Compute cutoff frequency for TE Modes, eqn. 68 of [1]
def compute_fc_TE(m,d,eps1,eps2):
    fc=m/(4*d*np.sqrt(mu0*eps0)*np.sqrt(n1**2-n2**2))
    lambda_fc=c/fc
    return fc,lambda_fc

# Compute mode 1 (m=1) cutoff frequency (fc_1) 
m=1
#d=2*j_core_max*dl
d=500e-9/2 # 500 nm structure
print('d=%4.3e (meters)' % d)
fc1,lambda_fc1=compute_fc_TE(m,d,eps1,eps2)
print('fc1=%4.3f (THz), lambda_fc=%4.3e (meters)' % (fc1/1e12, lambda_fc1))

# Courant stability condition for 2D FDTD
# dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
dt = (dl / (np.sqrt(2.0) * c)) * 0.99  # Time step slightly below Courant limit
print('dt=',dt)

# FDTD update coefficients
Ch = dt / (mu0 * dl)         # Magnetic field multiplier
Ce = dt / (eps0 * dl)        # Electric field multiplier
Cre_clad = Ce/epsr_clad                # Relative Electric field mulitplier 
Cre_core = Ce/epsr_core                # Relative Electric field mulitplier 
Ce = Cre_clad*np.ones((Nx,Ny))     # Initialize entire region to Ce=dt/(eps0*dl)
Ce[(ic+i_core_min):(ic+i_core_max),(jc+j_core_min):(jc+j_core_max)]=Cre_core    # Initialize desired rectangle to Cre = dt/(epsr * eps0 * dl) 
ABC_coef=(c*dt-dl)/(c*dt+dl) # Coefficient for First-Order Mur Update

n_steps = 1500 
# =========================================
# Grid Initialization
# =========================================
# Yee Grid Configuration:
# Ez[i, j] is at the cell center: (i+0.5, j+0.5)*dl
# Hx[i, j] is at the cell bottom edge: (i+0.5, j)*dl
# Hy[i, j] is at the cell left edge: (i, j+0.5)*dl
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
Ez_old_left = np.zeros((1,Ny))
Ez_old_right = np.zeros((1,Ny))

## Probe Locations 
i_probe1,j_probe1=ic,jc+int(j_core_min)+2
i_probe2,j_probe2=ic,jc+int(j_core_min/2)
i_probe3,j_probe3=ic,jc
i_probe4,j_probe4=ic,jc+int(j_core_max/2)
i_probe5,j_probe5=ic,jc+int(j_core_max)-2

# Track current time step
n = 0

# =========================================
# FDTD Computation Loop (Pre-calculate all frames)
# =========================================
steps_per_frame = 10  # Store 1 frame every 10 steps to save memory
n_frames = n_steps // steps_per_frame

Ez_frames = []
Ez_probe1 = np.zeros(n_steps) 
Ez_probe2 = np.zeros(n_steps) 
Ez_probe3 = np.zeros(n_steps) 
Ez_probe4 = np.zeros(n_steps) 
Ez_probe5 = np.zeros(n_steps) 
Hx_probe1 = np.zeros(n_steps) 
Hx_probe2 = np.zeros(n_steps) 
Hx_probe3 = np.zeros(n_steps) 
Hx_probe4 = np.zeros(n_steps) 
Hx_probe5 = np.zeros(n_steps) 
Hy_probe1 = np.zeros(n_steps) 
Hy_probe2 = np.zeros(n_steps) 
Hy_probe3 = np.zeros(n_steps) 
Hy_probe4 = np.zeros(n_steps) 
Hy_probe5 = np.zeros(n_steps) 

global_max_E = 0.0

print("Running FDTD Simulation (calculating time response)...")
for n in range(n_steps):
    
    # Magnetic Field Updates
    Hx[:, 1:] -= Ch * (Ez[:, 1:] - Ez[:, :-1])

    Hy[1:, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    
    # Electric Field Update
    Ez[1:-1, 1:-1] += Ce[1:-1,1:-1] * ((Hy[2:, 1:-1] - Hy[1:-1, 1:-1]) - 
                            (Hx[1:-1, 2:] - Hx[1:-1, 1:-1]))

    #-------------------------- 
    ## Probes
    #--------------------------
    Ez_probe1[n]=Ez[i_probe1,j_probe1]
    Ez_probe2[n]=Ez[i_probe2,j_probe2]
    Ez_probe3[n]=Ez[i_probe3,j_probe3]
    Ez_probe4[n]=Ez[i_probe4,j_probe4]

    Hx_probe1[n]=Hx[i_probe1,j_probe1] 
    Hx_probe2[n]=Hx[i_probe2,j_probe2] 
    Hx_probe3[n]=Hx[i_probe3,j_probe3] 
    Hx_probe4[n]=Hx[i_probe4,j_probe4] 

    Hy_probe1[n]=Hy[i_probe1,j_probe1] 
    Hy_probe2[n]=Hy[i_probe2,j_probe2] 
    Hy_probe3[n]=Hy[i_probe3,j_probe3] 
    Hy_probe4[n]=Hy[i_probe4,j_probe4] 

    
    # Hard source injection (Modulated Gaussian Pulse)
    t = n * dt
    Ez[ic+i_core_min+2, jc] = source_amplitude * np.cos(omega * t) * np.exp(-((t - t0) / tau)**2)
   
    # ABC (First-Order Mur)
   
    Ez[Nx-1,:]=Ez_old_right+ABC_coef*(Ez[Nx-2,:]-Ez[Nx-1,:])
    Ez_old_right=Ez[Nx-2,:]
    Ez[+1,:]=Ez_old_left+ABC_coef*(Ez[+2,:]-Ez[+1,:])
    Ez_old_left=Ez[+2,:]

    # Save frame and track global maximum
    if n % steps_per_frame == 0:
        Ez_frames.append(Ez.copy()) # Copy the array state into memory
        current_max = np.max(np.abs(Ez))
        if current_max > global_max_E:
            global_max_E = current_max
            
    # Progress feedback in the console
    if n % 100 == 0:
        print(f"Computed step {n}/{n_steps}")

# =========================================
#  Setup Visualization & Animation
# =========================================

print("Setting up visualization...")
fig, ax = plt.subplots(figsize=(7, 6))
point_list1=[(ic+i_core_min,jc+j_core_max),(ic+i_core_max,jc+j_core_max),(ic+i_core_max,jc+j_core_min),(ic+i_core_max,jc+j_core_min),(ic+i_core_min,jc+j_core_min),(ic+i_core_min,jc+j_core_max)] # upper plate boundary
original1=LineString(point_list1)
ax.plot(*original1.xy,color='black')

# Set static vmin and vmax based on the global maximum found during computation
static_vmax = max(global_max_E * 0.1, 1e-9)

# Transpose Ez (Ez.T) so the array maps naturally to x (horizontal) and y (vertical)
im = ax.imshow(Ez_frames[0].T, cmap='bwr', vmin=-static_vmax, vmax=static_vmax, origin='lower')
plt.colorbar(im, ax=ax, label='E_z Field Amplitude (V/m)')
#ax.set_title(f'2D TMz FDTD - 200 THz Gaussian Pulse\nTime Step: 0/{n_steps}')
ax.set_xlabel('x (cells)')
ax.set_ylabel('y (cells)')

def update_plot(frame_idx):
    im.set_array(Ez_frames[frame_idx].T)
    time_step = frame_idx * steps_per_frame
    ax.set_title(f'2D TMz FDTD - 192 GHz Harmonic Pulse Source\nTime Step: {time_step}/{n_steps}')
    return [im]

# Create and run the animation
print("Starting animation...")
#ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
#                              interval=30, blit=False, repeat=False)
ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
                              interval=30, blit=False, repeat=True)
plt.show()
# Display the animation (Will block the script until window is closed)
print("Simulation complete.")

## Analytical transverse E-field amplitude profile applied at source plane
Ae=1
z=0
k0=2*np.pi/lambda_0 
beta=omega*np.sqrt(mu0*eps1) # session 22 notes
kappa=np.sqrt(n1**2*k0**2-beta**2)
x=np.linspace(-0.5*core_target_width,0.5*core_target_width)
Eye=Ae*np.cos(kappa*x)*np.exp(-1j*beta*z) # x<|d|, eqn. 56 of [1]
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(x, Eye, c='green',label=r'$E_{ana}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$|E| \text{V/m}$')
ax.set_ylim(np.min(Eye),np.max(Eye))
ax.legend()
ax.grid()
#plt.savefig('fig-p5-1.eps') 

## Cross-sectional plot of the time-averaged Poynting vector $P_z$ across the waveguide 
## Mapping from FDTD to Ref [1] : Ez, Hy --> Ey, Hx
def compute_S(E_probe,H_probe): 
    t=dt*np.arange(len(E_probe))
    E_fd=np.fft.fft(E_probe)
    H_fd=np.fft.fft(H_probe)
    S_fd=E_fd*np.conj(H_fd) # W/m^2 Poynting Vector
    S_td=0.5*np.real(np.fft.ifft(S_fd))
    return t,S_td 

t1,S1=(Ez_probe1,Hx_probe1) 
t2,S2=(Ez_probe2,Hx_probe2)
t3,S3=(Ez_probe3,Hx_probe3)
t4,S4=(Ez_probe4,Hx_probe4)
t5,S5=(Ez_probe5,Hx_probe5)

fig, ax = plt.subplots(figsize=(7, 6))
t=dt*np.arange(n_steps)
ax.plot(t, S1, label='$P_{z1}$')
ax.plot(t, S2, label='$P_{z2}$') 
ax.plot(t, S3, label='$P_{z3}$')
ax.plot(t, S4, label='$P_{z4}$')
ax.plot(t, S5, label='$P_{z5}$')
#ax.set_title(r'Time-domain signals captured by probes')
ax.legend()
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$P_z$ $W/m^2$')
ax.grid()
fig.tight_layout()
#plt.savefig('fig-p5-4.eps'); 

plt.show()

