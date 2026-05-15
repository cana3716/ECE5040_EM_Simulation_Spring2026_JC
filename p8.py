# p8.py, ECE5040-ES Final Project, Problem 8 
#
# Combine waveguide structures and radiation boundaries by
# simulating a parallel-plate waveguide that gradually
# flares outward into a horn. Waveguide is fed with the
# fundamental mode. Surrounding domain is terminated with
# a robust PML. 
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString

## Physical Constants and Simulation Parameters
c = 299792458.0              # Speed of light in vacuum (m/s)
mu0 = 4.0 * np.pi * 1e-7     # Permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c**2)    # Permittivity of free space (F/m)
eta0=np.sqrt(mu0/eps0)

## Source Parameters
f = 10e12                  # Source frequency 
lambda_0 = c / f            # Free-space wavelength 
kappa_0=2*np.pi/lambda_0    # Free-space wavenumber
omega = 2.0 * np.pi * f

tau = 1/f/2                   # Gaussian pulse width (10 fs)
t0 = 3.0 * tau              # Pulse delay to ensure a smooth turn-on
source_amplitude = 1.0      # Peak amplitude of the source

## Grid Parameters
Nx, Ny = 500, 500         # Nx x Ny cells grid
dl = lambda_0 / 20         # Spatial resolution (dx = dy). 20 points per wavelength
dx=dy=dl
print('dx/(2*sqrt(c))=%4.3E ' % (dl/(2*np.sqrt(c))))

### PML Parameters
m=3         # polynomial order (2,3, or 4)
pml_len=100 # thickness of PML in cell
R_0=1e-5    # target reflection coefficient
sigma_max=-(m+1)*eps0*c/(2*pml_len*dx)*np.log(R_0)
sigx=np.zeros((Nx,Ny))
sigy=np.zeros((Nx,Ny))

## Fill PML regions
for i in range(pml_len):
    val=sigma_max*((pml_len-i)/pml_len)**m
    sigx[i,:]=val      # left boundary
    sigx[Nx-1-i,:]=val # right boundary
    sigy[:,i]=val      # bottom boundary
    sigy[:,Ny-1-i]=val #top boundary

### Coefficients for Ez/Dz update
dt=1e-15

ca_x=(1-sigx*dt/(2*eps0))/(1+sigx*dt/(2*eps0))
cb_x=dt/(1+sigx*dt/(2*eps0))
ca_y=(1-sigy*dt/(2*eps0))/(1+sigy*dt/(2*eps0))
cb_y=dt/(1+sigy*dt/(2*eps0))

ca_x_h=(1-sigx*dt/(2*eps0))/(1+sigx*dt/(2*eps0))
cb_x_h=dt/(1+sigx*dt/(2*eps0))
ca_y_h=(1-sigy*dt/(2*eps0))/(1+sigy*dt/(2*eps0))
cb_y_h=dt/(1+sigy*dt/(2*eps0))

# Center coordinates for the infinite line source
ic, jc = Nx // 2, Ny // 2

# Courant stability condition for 2D FDTD
# dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
dt = (dl / (np.sqrt(2.0) * c)) * 0.99  # Time step slightly below Courant limit
print('dt=',dt)

### PEC waveguide with outward flares 
origin_x,origin_y=ic-100,jc
l1,h1,phi_horn=100,10,45*np.pi/180
l2=round(0.5*l1)
h2=l2*np.tan(phi_horn)

p1=(origin_x,origin_y+h1)
p2=(origin_x+l1,origin_y+h1)
p3=(origin_x+l1+l2,origin_y+h1+h2)

p4=(origin_x,origin_y-h1)
p5=(origin_x+l1,origin_y-h1)
p6=(origin_x+l1+l2,origin_y-h1-h2)

n_steps = 2400 

# Grid Initialization
# Yee Grid Configuration:
# Ez[i, j] is at the cell center: (i+0.5, j+0.5)*dl
# Hx[i, j] is at the cell bottom edge: (i+0.5, j)*dl
# Hy[i, j] is at the cell left edge: (i, j+0.5)*dl
Ez = np.zeros((Nx, Ny))
Dz = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
Ez_old_left = np.zeros((1,Ny))
Ez_old_right = np.zeros((1,Ny))
Dz_old=np.zeros((Nx,Ny))

### Probes 
E_probe1 = np.zeros(n_steps)
E_probe2 = np.zeros(n_steps)
r_probe=Nx//2-90
phi_probe=np.linspace(-np.pi/2,np.pi/2,n_steps)
E_probe3 = np.zeros(n_steps)

# Track current time step
n = 0

# FDTD Computation Loop (Pre-calculate all frames)
steps_per_frame = 10  # Store 1 frame every 10 steps to save memory
n_frames = n_steps // steps_per_frame
Ez_frames = []
S_frames = []
global_max_E = 0.0

print("Running FDTD Simulation (calculating time response)...")
for n in range(n_steps):

    # Magnetic Update
    Hx[:, :-1] -= (dt / (mu0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += (dt / (mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])

    # Curl H
    dHy_dx = (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) / dx  
    dHx_dy = (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2]) / dy 
    curl_h = dHy_dx - dHx_dy
    
    # E/D update
    Dz[1:-1, 1:-1] = ca_y[1:-1, 1:-1] * Dz[1:-1, 1:-1] + cb_y[1:-1, 1:-1] * curl_h 
    Ez[1:-1, 1:-1] = ca_x[1:-1, 1:-1] * Ez[1:-1, 1:-1] + (1.0 / eps0) * (Dz[1:-1, 1:-1] - Dz_old[1:-1, 1:-1])
    
    ### PEC Waveguide Boundaries
    Ez[p1[0]:p2[0],p1[1]]=0.0 # upper horizontal boundary 

    # upper horn boundary
    x_horn,y_horn=p2[0],p2[1]
    z_horn=np.sqrt(h2**2+l2**2)
    #dx_horn,dy_horn=z_horn*np.cos(phi_horn),z_horn*np.sin(phi_horn)
    dx_horn,dy_horn=1,1
    for i in range(p3[0]-p2[0]): 
        Ez[x_horn,y_horn]=0.0
        x_horn+=dx_horn
        y_horn+=dy_horn

    Ez[p4[0]:p5[0],p5[1]]=0.0 # lower horizontal boundary

    # lower horn boundary 
    x_horn,y_horn=p5[0],p5[1]
    for i in range(p6[0]-p5[0]): 
        Ez[x_horn,y_horn]=0.0
        x_horn+=1
        y_horn-=1

    ### Measurements
    E_probe1[n]=Ez[p3[0],ic]
    E_probe3[n]=Ez[Nx-50,jc]
    if (n==round(0.96*n_steps)): 
        i=0
        for phi in phi_probe:
            probe_x=round(r_probe*np.cos(phi))
            probe_y=round(r_probe*np.sin(phi))
            E_probe2[i]=Ez[p3[0]+probe_x,jc+probe_y]
            i+=1

    Dz_old[1:-1, 1:-1] = Dz[1:-1, 1:-1].copy()
   
    # Source
    t = n * dt
    Ez[origin_x, origin_y] += source_amplitude * np.exp(-((t - t0) / tau)**2) # += gives soft source

    # Save frame and track global maximum
    if n % steps_per_frame == 0:
        Ez_frames.append(Ez.copy()) # Copy the array state into memory
        current_max = np.max(np.abs(Ez))
        if current_max > global_max_E:
            global_max_E = current_max
            
    # Progress feedback in the console
    if n % 100 == 0:
        print(f"Computed step {n}/{n_steps}")
       # print('dt/eps0=%4.3e' % (dt/eps0))
        print('Ez_max=%4.3e' % np.max(Ez))
        print('Dz_max=%4.3e' % (np.max(Dz)))
        print('E_probe1_max=%4.3e' % (np.max(E_probe1)))
        print('E_probe2_max=%4.3e' % (np.max(E_probe2)))
        print('E_probe3_max=%4.3e' % (np.max(E_probe3)))
       # print('ca_x_max=%4.3e' % (np.max(ca_x)))
       # print('ca_y_max=%4.3e' % (np.max(ca_y)))

# Setup Visualization & Animation
print("Setting up visualization...")
fig, ax = plt.subplots(figsize=(7, 6))
point_list1=[p1,p2,p3] # upper PEC boundary
point_list2=[p4,p5,p6] # lower PEC boundary
original1=LineString(point_list1)
original2=LineString(point_list2)
ax.plot(*original1.xy,color='black')
ax.plot(*original2.xy,color='black')

# Set static vmin and vmax based on the global maximum found during computation
static_vmax = max(global_max_E * 0.1, 1e-9)

# Transpose Ez (Ez.T) so the array maps naturally to x (horizontal) and y (vertical)
im = ax.imshow(Ez_frames[0].T, cmap='bwr', vmin=-static_vmax, vmax=static_vmax, origin='lower')
plt.colorbar(im, ax=ax, label='E_z Field Amplitude (V/m)')
ax.set_xlabel('x (cells)')
ax.set_ylabel('y (cells)')

def update_plot(frame_idx):
    im.set_array(Ez_frames[frame_idx].T)
    time_step = frame_idx * steps_per_frame
    ax.set_title(f'1D TMz FDTD - 10 GHz Harmonic Pulse Source, PML Outer Boundaries \nTime Step: {time_step}/{n_steps}')
    return [im]

# Create and run the animation
print("Starting animation...")
ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
                              interval=30, blit=False, repeat=True)
# Display the animation (Will block the script until window is closed)
print("Simulation complete.")
plt.show()

# Electric field sampled directly across the aperture plane
#fig, ax = plt.subplots(figsize=(8, 6))
#t=np.arange(len(E_probe1))
#plt.plot(t,E_probe1,c='b',label='$E_{probe_1}$')
#ax.set_xlim(180,200)
#ax.set_ylim(0.2,0.65)
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$E_z$ (V/m)')
#ax.legend()
#ax.grid()
#plt.tight_layout()
#plt.savefig('fig-p8-2.eps'); 

# Un-calibrated far-field radiation pattern (polar plot) 
# calculated by sampling the fields in a semi-circle
# just inside the PML region. 
fig, ax = plt.subplots(figsize=(8, 6))
phi=np.linspace(0,180,n_steps)
plt.plot(phi,E_probe2/np.max(E_probe2),c='b',label='$E_{probe_2}$')
ax.set_xlabel(r'$\phi$ (degrees)')
ax.set_ylabel(r'$E_z/Ez_{z_{max}}$')
ax.legend()
ax.grid()
plt.tight_layout()
#plt.savefig('fig-p8-3.eps'); 

# phase distribution of electric field across the horn
fig, ax = plt.subplots(figsize=(8, 6))
freq=np.fft.fftfreq(len(E_probe2),d=1/2/f)
window_mask=(freq>=0)&(freq<=20e9)
freq_windowed=freq[window_mask]
E_probe2_fd=np.fft.fft(E_probe2)
plt.plot(freq_windowed/1e9,np.angle(E_probe2_fd[window_mask],deg=True),c='r',label=r'$\angle{E_{probe_2}}$')
ax.set_xlabel(r'$f$ (GHz)')
ax.set_ylabel(r'$\angle{E_z}$ (degrees)')
ax.legend()
ax.grid()
plt.tight_layout()
#plt.savefig('fig-p8-4.eps'); 

# Electric field sampled at a point deep inside the PML 
fig, ax = plt.subplots(figsize=(8, 6))
t=np.arange(len(E_probe3))
plt.plot(t,E_probe3/np.max(E_probe3),c='g',label='$E_{probe_3}$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$E_z/Ez_{z_{max}}$')
ax.legend()
ax.grid()
plt.tight_layout()

plt.show()

