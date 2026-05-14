# p2.py, ECE5040-ES Final Project, Problem 2 
#
# Simulation of a parallel-plate PEC waveguide designed
# with a cutoff frequency of 9 GHz for the first
# higher-order mode. Waveguide excited with 11 GHz
# and 7 GHz CW Source. Mur first-degree ABC at the open ends.  
#
# Ref [5]. Nikolova, Numerical Techniques in Electromagnetics, ECE 757, 2009, 
# web: https://www.ece.mcmaster.ca/faculty/bakr/ECE757/FDTD-V.pdf, accessed 
# May 13, 2026

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString

# Physical Constants and Simulation Parameters
c = 299792458.0              # Speed of light in vacuum (m/s)
mu0 = 4.0 * np.pi * 1e-7     # Permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c**2)    # Permittivity of free space (F/m)
epsr = 4.0                   # Relative permittivity of dielectric region 

# Source Parameters
f_lower=7e9
f_upper=11e9
f=f_upper
lambda_0 = c / f             # Free-space wavelength (~1.5 um)
omega = 2.0 * np.pi * f
omega_source_upper = 2.0 * np.pi * f_upper
omega_source_lower = 2.0 * np.pi * f_lower
tau = 10e-15                 # Gaussian pulse width (10 fs)
t0 = 3.0 * tau               # Pulse delay (30 fs) to ensure a smooth turn-on
source_amplitude = 1.0       # Peak amplitude of the source

# Grid Parameters
Nx, Ny = 500, 500            # 500 x 500 cells grid
dl = lambda_0 / 20.0         # Spatial resolution (dx = dy). 20 points per wavelength
print('dl=%4.3E' % (dl))

# Parallel Plate Geometry Paramaters
a=1/(f*np.sqrt(mu0*eps0))
#b=a/2 # b<=a/2 as assumed constraint
b=a/4 # b<=a/2 as assumed constraint
print('a=%4.3f, b=%4.3f, a/dl=%4.3f, b/dl=%4.3f' % (a,b,a/dl,b/dl) )
G=10 # Gain
wg_top=int(G*(b/dl/2))
wg_bot=int(G*(-b/dl/2))
wg_left=int(G*(-a/dl/2))
wg_right=int(G*(a/dl/2))
wg_length=wg_right-wg_left
wg_width=wg_top-wg_bot
print(wg_top,wg_bot,wg_left,wg_right)

# Compute Courant Number
def S(N_lambda): 
    return c*dt*N_lambda/lambda_0

# Courant stability condition for 2D FDTD
# dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
dt = (dl / (np.sqrt(2.0) * c)) * 0.99  # Time step slightly below Courant limit
#print('dt=',dt)

# FDTD update coefficients
Ch = dt / (mu0 * dl)         # Magnetic field multiplier
Ce = dt / (eps0 * dl)        # Electric field multiplier
Cre = Ce/epsr                # Relative Electric field mulitplier 
Ce = Ce*np.ones((Nx,Ny))     # Initialize entire region to Ce=dt/(eps0*dl)
#Ce[350:400,200:300]=Cre     # Initialize desired rectangle to Cre = dt/(epsr * eps0 * dl) 
ABC_coef=(c*dt-dl)/(c*dt+dl) # Coefficient for First-Order Mur Update

n_steps = 3000 
# =========================================
# 2. Grid Initialization
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
Ez_wg_old_left = np.zeros((1,wg_width))
Ez_wg_old_right = np.zeros((1,wg_width))

# Center coordinates for the infinite line source
ic, jc = Nx // 2, Ny // 2

# Probe Locations
i_probe1,j_probe1=ic+wg_right-3,jc
i_probe2,j_probe2=i_probe1+10,j_probe1
# Track current time step
n = 0

# =========================================
# 3. FDTD Computation Loop (Pre-calculate all frames)
# =========================================
steps_per_frame = 10  # Store 1 frame every 10 steps to save memory
n_frames = n_steps // steps_per_frame

Ez_frames = []
Ez_probe1 = np.zeros(n_steps) 
Ez_probe2 = np.zeros(n_steps) 
Hx_probe = np.zeros(n_steps) 
global_max_E = 0.0

print("Running FDTD Simulation (calculating time response)...")
for n in range(n_steps):
    # ----------------------
    # Magnetic Field Updates
    # Hx update (requires Ez above and below)
    Hx[:, 1:] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    # Hy update (requires Ez to the right and left)
    Hy[1:, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    # ----------------------
    # Electric Field Update
    # ----------------------
    # Ez update for the interior core region ONLY.
    # By NOT updating the boundaries (i=0, i=-1, j=0, j=-1), they remain 
    # strictly at 0.0, which naturally enforces the PEC walls perfectly!
    
    #Ez[1:-1, 1:-1] += Ce * ((Hy[2:, 1:-1] - Hy[1:-1, 1:-1]) - 
    #                        (Hx[1:-1, 2:] - Hx[1:-1, 1:-1]))

    Ez[1:-1, 1:-1] += Ce[1:-1,1:-1] * ((Hy[2:, 1:-1] - Hy[1:-1, 1:-1]) - 
                            (Hx[1:-1, 2:] - Hx[1:-1, 1:-1]))
    
    # Probes
    Ez_probe1[n]=Ez[i_probe1,j_probe2]
    Ez_probe2[n]=Ez[i_probe2,j_probe2]
    
    # Waveguide boundaries  
    Ez[(wg_left+ic):(wg_right+ic),(wg_top+jc)]=0.0  # upper plate PEC boundary 
    Ez[(wg_left+ic):(wg_right+ic),(wg_bot+jc)]=0.0  # lower plate PEC boundary 

    Ez[wg_left+ic,(wg_bot+jc):(wg_top+jc)]=Ez_wg_old_left+ABC_coef*(Ez[wg_left+ic+2,(wg_bot+jc):(wg_top+jc)]-Ez[wg_left+ic+1,(wg_bot+jc):(wg_top+jc)])     # left opening of waveguide is ABC
    Ez_wg_old_left=Ez[wg_left+2,(wg_bot+jc):(wg_top+jc)]
    
    Ez[wg_right+ic,(wg_bot+jc):(wg_top+jc)]=Ez_wg_old_right+ABC_coef*(Ez[wg_right+ic-2,(wg_bot+jc):(wg_top+jc)]-Ez[wg_right-1,(wg_bot+jc):(wg_top+jc)])     # right opening of waveguide is ABC
    Ez_wg_old_right=Ez[wg_right+ic-2,(wg_bot+jc):(wg_top+jc)]

   # Ez[wg_right,wg_bot:wg_top]=Ez_wg_old_right+ABC_coef*(Ez[wg_right-2,:]-Ez[wg_right-1,:]) # left opening of waveguide is ABC

    # ----------------------
    # Source Injection
    # ----------------------
    # Hard source injection (Modulated Gaussian Pulse)
    t = n * dt
    #Ez[ic, jc] = source_amplitude * np.exp(-((t - t0) / tau)**2) # hard source
    Ez[ic+wg_left+3, jc] = source_amplitude * np.cos(omega_source_upper * (t-t0)) * np.exp(-((t - t0) / tau)**2)
    
    # Soft Source 
    #Ez[ic, jc] += source_amplitude * np.exp(-((t - t0) / tau)**2) # soft source
    #Ez[ic, jc] = source_amplitude * np.cos(omega * t) * np.exp(-((t - t0) / tau)**2)
   
    # ABC (First-Order Mur)
    #Ez[Nx-1,:]=Ez[Nx-2,:]+ABC_coef*(Ez[Nx-2,:]-Ez[Nx-1,:])
    
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
# 4. Setup Visualization & Animation
# =========================================

print("Setting up visualization...")
fig, ax = plt.subplots(figsize=(7, 6))

point_list1=[(wg_left+ic,wg_top+jc),(wg_right+ic,wg_top+jc)]
point_list2=[(wg_left+ic,wg_bot+jc), (wg_right+ic,wg_bot+jc)]
original1=LineString(point_list1)
original2=LineString(point_list2)

ax.plot(*original1.xy,color='black')
ax.plot(*original2.xy,color='black')

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
    ax.set_title(f'2D TMz FDTD - %d GHz Sinusoidal Source\nTime Step: {time_step}/{n_steps}' % (f/1e9))
    return [im]

# Create and run the animation
print("Starting animation...")
#ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
#                              interval=30, blit=False, repeat=False)
ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
                              interval=30, blit=False, repeat=False)

# Display the animation (Will block the script until window is closed)
plt.show()
print("Simulation complete.")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(np.linspace(0,n_steps,len(Ez_probe1)), Ez_probe1)
ax.set_title(r'Time-domain signal captured by probe, $f_c=%d\text{GHz}$'%(f/1e9))
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$E_z\text{ V/m}$')
ax.grid()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(np.linspace(0,n_steps,len(Ez_probe1)), np.angle(Ez_probe1,deg=True))
ax.set_title(r'Time-domain signal captured by probe, $f_c=%d\text{GHz}$'%(f/1e9))
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$\angle{E_z}^\circ$')
ax.grid()

plt.show()

### Phase Velocity Calculations (see Ref [5]) 

dx_probe = np.abs((i_probe2-i_probe1))*dl  # Distance between probe 1 and probe 2 (meters)
print(f"dx_probe={dx_probe:0.4e}")

#dt = 2e-11        # FDTD time-step (seconds)
target_freq = f_upper

# Compute FFT of both time-domain spatial probe monitors 
freqs = np.fft.fftfreq(len(Ez_probe1), d=dt)
E1_fd = np.fft.fft(Ez_probe1)
E2_fd = np.fft.fft(Ez_probe2)

# Locate the index corresponding to 9 GHz
idx = np.argmin(np.abs(freqs - target_freq))

# Extract and unwrap phases
phase1 = np.unwrap(np.angle(E1_fd))[idx]
phase2 = np.unwrap(np.angle(E2_fd))[idx]
delta_phase = phase2 - phase1

# Calculate physical phase velocity
omega = 2 * np.pi * target_freq
v_p = (omega * dx_probe) / np.abs(delta_phase)
print(f"Simulated phase Velocity at {target_freq/1e9} GHz: {v_p:.4e} m/s, {v_p/c*100:.2f} % velocity of light")

# Analytical phase velocity calculations
dx=dl
k_numerical = 2/dx*np.arcsin(dx/(c*dt)*np.sin(omega*dt/2)) # numerical wave number
v_p_analytical=omega/k_numerical
print(f"Analytical phase Velocity at {target_freq/1e9} GHz: {v_p_analytical:.4e} m/s, {v_p_analytical/c*100:.2f} % velocity of light")
