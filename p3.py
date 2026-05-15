# p3.py, ECE5040-ES Final Project, Problem 3 
#
# Using the 2D TMz code, insert a PEC iris halfway down 
# a rectangular waveguide, excite the waveguide with a broadband
# differentiated Gaussian pulse, record the incident, reflected, and
# transmitted time-domain fields using appropriate probe locations. 
#
# Note: Source base is 2D_TMz.py from Project 2. 
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString

## Physical Constants and Simulation Parameters
c = 299792458.0              # Speed of light in vacuum (m/s)
mu0 = 4.0 * np.pi * 1e-7     # Permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c**2)    # Permittivity of free space (F/m)
epsr = 4.0                   # Relative permittivity of dielectric region 

## Source Parameters
f = 10e9                   # Source frequency 
f_lower=8e9
f_upper=12e9
lambda_0 = c / f             # Free-space wavelength 
omega = 2.0 * np.pi * f
tau = 1/f                 # Gaussian pulse width (10 fs)
t0 = 3.0 * tau               # Pulse delay (30 fs) to ensure a smooth turn-on
source_amplitude = 1.0       # Peak amplitude of the source
print('tau=%4.3f, t0=%4.3f, source_amplitude=%4.3f' % (tau, t0, source_amplitude))
## Grid Parameters
Nx, Ny = 1000, 1000            # 500 x 500 cells grid
dl = lambda_0 / 20.0         # Spatial resolution (dx = dy). 20 points per wavelength
print('dl=%4.3E' % (dl))

## Parallel Plate Geometry Paramaters
a=1/(f*np.sqrt(mu0*eps0))
#b=a/2 # b<=a/2 as assumed constraing
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

n_steps = 800 
#  Grid Initialization
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

# Probe Location
i_probe1,j_probe1=ic+wg_left,jc
i_probe2,j_probe2=ic+wg_right,jc

# Track current time step
n = 0

# FDTD Computation Loop (Pre-calculate all frames)
steps_per_frame = 10  # Store 1 frame every 10 steps to save memory
n_frames = n_steps // steps_per_frame

Ez_frames = []
Ez_probe1 = np.zeros(n_steps) 
Ez_probe2 = np.zeros(n_steps) 
Hx_probe1 = np.zeros(n_steps) 
Hx_probe2 = np.zeros(n_steps) 
global_max_E = 0.0

print("Running FDTD Simulation (calculating time response)...")
T_period=1/f
t_axis=np.arange(n_steps)*dt
step_mask=t_axis<=T_period
steps_per_pulse=int(T_period/dt+0.5)
for n in range(n_steps):
    
    # Magnetic Field Update
    Hx[:, 1:] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[1:, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    # Electric Field Update
    Ez[1:-1, 1:-1] += Ce[1:-1,1:-1] * ((Hy[2:, 1:-1] - Hy[1:-1, 1:-1]) - 
                            (Hx[1:-1, 2:] - Hx[1:-1, 1:-1]))
    
    # Probes
    Ez_probe1[n]=Ez[i_probe1,j_probe1]
    Hx_probe1[n]=Hx[i_probe1,j_probe1]
    Ez_probe2[n]=Ez[i_probe2,j_probe2]
    Hx_probe2[n]=Hx[i_probe2,j_probe2]
    
    # Waveguide boundaries  
    Ez[(wg_left+ic):(wg_right+ic),(wg_top+jc)]=0.0  # upper plate PEC boundary 
    Ez[(wg_left+ic):(wg_right+ic),(wg_bot+jc)]=0.0  # lower plate PEC boundary 
    Ez[ic,jc:(wg_top+jc)]=0.0 # PEC iris

    #Ez[wg_left+ic,(wg_bot+jc):(wg_top+jc)]=Ez_wg_old_left+ABC_coef*(Ez[wg_left+ic+2,(wg_bot+jc):(wg_top+jc)]-Ez[wg_left+ic+1,(wg_bot+jc):(wg_top+jc)])     # left opening of waveguide is ABC
    #Ez_wg_old_left=Ez[wg_left+2,(wg_bot+jc):(wg_top+jc)]
    
   # Ez[wg_right+ic,(wg_bot+jc):(wg_top+jc)]=Ez_wg_old_right+ABC_coef*(Ez[wg_right+ic-2,(wg_bot+jc):(wg_top+jc)]-Ez[wg_right-1,(wg_bot+jc):(wg_top+jc)])     # right opening of waveguide is ABC
   # Ez_wg_old_right=Ez[wg_right+ic-2,(wg_bot+jc):(wg_top+jc)]

   # Ez[wg_right,wg_bot:wg_top]=Ez_wg_old_right+ABC_coef*(Ez[wg_right-2,:]-Ez[wg_right-1,:]) # left opening of waveguide is ABC

    # ----------------------
    # Source Injection
    # ----------------------
    t = n * dt
    # Hard source injection (Modulated Gaussian Pulse)
    #Ez[ic, jc] = source_amplitude * np.exp(-((t - t0) / tau)**2) # hard source
    #Ez[ic+wg_left+3, jc] = source_amplitude * np.exp(-((t - t0) / tau)**2) # hard source
    if t<steps_per_pulse:
       Ez[ic+wg_left-200, jc] += source_amplitude * np.exp(-((t - t0) / tau)**2) # hard source
    #Ez[ic+wg_left+3, jc] = source_amplitude * np.cos(omega_source_upper * t) * np.exp(-((t - t0) / tau)**2)
    
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

point_list1=[(wg_left+ic,wg_top+jc),(wg_right+ic,wg_top+jc)] # upper plate boundary
point_list2=[(wg_left+ic,wg_bot+jc), (wg_right+ic,wg_bot+jc)] # lower plate boundary
point_list3=[(ic,wg_top+jc),(ic,jc)] # PEC iris location 
point_probe1=[(i_probe1,j_probe1)] # probe #1 location
point_probe2=[(i_probe2,j_probe2)] # probe #2 location
original1=LineString(point_list1)
original2=LineString(point_list2)
original3=LineString(point_list3) # PEC iris 

ax.plot(*original1.xy,color='black')
ax.plot(*original2.xy,color='black')
ax.plot(*original3.xy,color='black') # PEC iris

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
    ax.set_title(f'2D TMz FDTD - 10 GHz Pulse Source\nTime Step: {time_step}/{n_steps}')
    return [im]

# Create and run the animation
print("Starting animation...")
#ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
#                              interval=30, blit=False, repeat=False)
ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
                              interval=30, blit=False, repeat=True)

# Display the animation (Will block the script until window is closed)
plt.show()
print("Simulation complete.")

### Run a separate simulation with a continuous, homogeneous background medium (no scatterer) to 
### record the incident-only fields E_inc(t) and H_inc(t) at port 1
#Einc,Hinc=Ez_probe1,Hx_probe1
#np.savez("fdtd_reference_data.npz", Einc_data=Einc,Hinc_data=Hinc)
#print("Reference fields sucessfully saved to 'fdtd_reference_data.npz'.")

loaded_data=np.load("fdtd_reference_data.npz")
Einc=loaded_data["Einc_data"]
Hinc=loaded_data["Hinc_data"]
print(Einc)

### Now, run a simulation with the PEC waveguides in place...
Etot,Htot=Ez_probe1,Hx_probe1 # total fields at port 1
Et,Ht=Ez_probe2,Hx_probe2 # transmitted fields at port 2
print('Etot_max=', np.max(Etot))

### Calculations
Eref=Etot-Einc # calculate the time-domain reflected E-field
Href=Htot-Hinc # calculate the time-domain reflected H-field

print('Eref_max=',np.max(np.abs(Eref)))
Einc_fd=np.fft.fft(Einc)
Hinc_fd=np.fft.fft(Hinc)
Eref_fd=np.fft.fft(Eref)
Href_fd=np.fft.fft(Href)
Et_fd=np.fft.fft(Et)
Ht_fd=np.fft.fft(Ht)

S11_complex=Eref_fd/Einc_fd
S21_complex=Et_fd/Einc_fd

freqs=np.fft.fftfreq(len(Einc_fd),d=dt)

S11_mag=np.abs(S11_complex)
S21_mag=np.abs(S21_complex)

source_peak=np.max(np.abs(Einc_fd)) # find the maximum spectral amplitude
valid_frequencies_mask=np.abs(Einc_fd)>(0.01*source_peak) # create mask to isolate valid source frequencies

S11_valid=np.abs(Eref_fd[valid_frequencies_mask]/Einc_fd[valid_frequencies_mask])
S21_valid=np.abs(Et_fd[valid_frequencies_mask]/Einc_fd[valid_frequencies_mask])
freqs_valid=freqs[valid_frequencies_mask]

fig, ax = plt.subplots(figsize=(7, 6))
tn=np.linspace(0,n_steps,len(Einc))
ax.plot(tn, Einc, c='r', label=r'$E_{inc}$')
ax.plot(tn, Eref, c='g', label=r'$E_{ref}$')
ax.plot(tn, Et, c='b', label=r'$E_{t}$')
ax.set_title(r'Time-domain signals captured by probes')
ax.set_xlabel(r'$t_n$')
ax.set_ylabel(r'$E(t)\text{ V/m}$')
ax.legend()
ax.grid()
plt.savefig('fig-p3-2.eps')

fig, ax = plt.subplots(figsize=(7, 6))
tn=np.linspace(0,n_steps,len(Einc))
ax.plot(tn, Hinc, c='r', linestyle='--',label=r'$H_{inc}$')
ax.plot(tn, Href, c='g',linestyle='--', label=r'$H_{ref}$')
ax.plot(tn, Ht, c='b', linestyle='--',label=r'$H_{t}$')
ax.set_title(r'Time-domain signals captured by probes')
ax.set_xlabel(r'$t_n$')
ax.set_ylabel(r'$H(t)\text{ A/m}$')
ax.legend()
ax.grid()
plt.savefig('fig-p3-3.eps')

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(freqs/1e9,S11_mag, c='b', linestyle='--',label=r'$|S_{11}|$')
ax.plot(freqs/1e9,S21_mag, c='r', linestyle='--',label=r'$|S_{21}|$')
#ax.plot(freqs_valid/1e9,S11_valid, c='b', linestyle='-',label=r'$|S_{11}|$')
#ax.plot(freqs_valid/1e9,S21_valid, c='r', linestyle='-',label=r'$|S_{21}|$')
ax.set_title(r'S-Paramaters')
ax.set_xlabel(r'$f$ (GHz)')
ax.set_ylabel(r'$|S|$')
ax.set_xlim(8,12)
ax.legend()
ax.grid()
plt.savefig('fig-p3-4.eps')

plt.show()



