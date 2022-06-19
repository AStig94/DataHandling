#%% https://wiki.math.uwaterloo.ca/sheets/matlab/html/intro_spectral2d.html
import numpy as np
targ=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/targets.npz')
data_targ=targ["test"]-np.mean(targ["test"])
pred=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/predictions.npz')
data_pred=pred["test"]-np.mean(pred["test"])

Nx = 256
Nz  = 256
Lx  = 12
Lz  = 6

dx=Lx/Nx
dz=Lz/Nz

x_range=np.linspace(1,Nx,Nx)
z_range=np.linspace(1,Nz,Nz)
x=dx*x_range
z=dz*z_range

[xx,zz]=np.meshgrid(x,z)

dkx = 2*np.pi/Lx
dkz = 2*np.pi/Lz

kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

[kkx,kkz]=np.meshgrid(kx,kz)

kkx_norm= np.sqrt(kkx**2)
kkz_norm = np.sqrt(kkz**2)


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

# calculating wavelength in plus units 
Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu

Theta_fluc_targ=data_targ-np.mean(data_targ)
Theta_fluc_pred=data_pred-np.mean(data_pred)

fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
fourier_image_pred = np.fft.fftn(Theta_fluc_pred)


fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)
fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)


pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

import matplotlib.pyplot as plt
import numpy             as np
import matplotlib        as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
fig,ax=plt.subplots(1,1,dpi=1000)
CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
plt.xscale('log')
plt.yscale('log')
ax.set_ylabel(r'$\lambda_{z}^+$')
ax.set_xlabel(r'$\lambda_{x}^+$')
ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')

#%% not meaning:
# import numpy as np
targ=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/targets.npz')
data_targ=targ["test"][0]-np.mean(targ["test"][0])
pred=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/predictions.npz')
data_pred=pred["test"][0]-np.mean(pred["test"][0])

Nx = 256
Nz  = 256
Lx  = 12
Lz  = 6

dx=Lx/Nx
dz=Lz/Nz

x_range=np.linspace(1,Nx,Nx)
z_range=np.linspace(1,Nz,Nz)
x=dx*x_range
z=dz*z_range

[xx,zz]=np.meshgrid(x,z)

dkx = 2*np.pi/Lx
dkz = 2*np.pi/Lz

kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

[kkx,kkz]=np.meshgrid(kx,kz)

kkx_norm= np.sqrt(kkx**2)
kkz_norm = np.sqrt(kkz**2)


Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

# calculating wavelength in plus units 
Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu

Theta_fluc_targ=data_targ-np.mean(data_targ)
Theta_fluc_pred=data_pred-np.mean(data_pred)

fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
fourier_image_pred = np.fft.fftn(Theta_fluc_pred)


fourier_amplitudes_targ = np.abs(fourier_image_targ)**2
fourier_amplitudes_pred = np.abs(fourier_image_pred)**2


pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

import matplotlib.pyplot as plt
import numpy             as np
import matplotlib        as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
fig,ax=plt.subplots(1,1,dpi=1000)
CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
plt.xscale('log')
plt.yscale('log')
ax.set_ylabel(r'$\lambda_{z}^+$')
ax.set_xlabel(r'$\lambda_{x}^+$')
ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')

#%% old 2D to 1D PSD https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
import numpy as np
import scipy.stats as stats
targ=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/targets.npz')
pred=np.load('/home/au567859/DataHandling/models/output/treasured-music-60/y_plus_30-VARS-pr0.71_u_vel_v_vel_w_vel-TARGETS-pr0.71_flux/predictions.npz')



data_targ=targ["test"][0]
data_pred=pred["test"][0]
npix = data_targ.shape[0]



fourier_image_targ = np.fft.fftn(data_targ)
fourier_image_pred = np.fft.fftn(data_pred)
fourier_amplitudes_targ = np.abs(fourier_image_targ)**2
fourier_amplitudes_pred = np.abs(fourier_image_pred)**2

kfreq = np.fft.fftfreq(npix) * npix
# Not taking into account the aspect ratio, maybe this can be done in the function below?
kfreq2D = np.meshgrid(kfreq, kfreq)
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

knrm = knrm.flatten()
fourier_amplitudes_targ = fourier_amplitudes_targ.flatten()
fourier_amplitudes_pred = fourier_amplitudes_pred.flatten()
kbins = np.arange(0.5, npix//2+1, 1.)
kvals = 0.5 * (kbins[1:] + kbins[:-1])

Abins_targ, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_targ,
                                     statistic = "mean",
                                     bins = kbins)
Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_pred,
                                     statistic = "mean",
                                     bins = kbins)

Abins_targ *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
Abins_pred *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

import matplotlib.pyplot as plt
plt.figure(dpi=500)
plt.loglog(kvals, Abins_targ,label="targ")
plt.loglog(kvals, Abins_pred,label="pred")
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.tight_layout()
plt.title(r"$q_w$ at $y_{plus}=15$ and $Pr=0.71$")
plt.legend()
# %%
