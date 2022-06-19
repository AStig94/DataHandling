#%%
def PSD(target_path,prediction_path):
    import numpy as np
    targ=np.load(target_path)
    data_targ=targ["test"]-np.mean(targ["test"])
    pred=np.load(prediction_path)
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


    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)*kkx*kkz
    fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)*kkx*kkz


    # pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
    # pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
    # pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
    # pct100=np.max(fourier_amplitudes_targ*kkx*kkz)
    return fourier_amplitudes_targ, fourier_amplitudes_pred, Lambda_x, Lambda_z

#%%
from DataHandling import utility
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])

models=['winter-firefly-75','unique-night-53','exalted-durian-66','pleasant-snowflake-76','amber-glitter-73','worthy-bush-58','frosty-lion-63','super-pond-78','different-silence-74','devoted-bee-61','glad-oath-62','solar-water-77','clear-donkey-82','earthy-universe-83','effortless-totem-85','deep-terrain-84']

cm = 1/2.54  # centimeters in inches
fig, axs=plt.subplots(nrows=4,ncols=4,figsize=([1.4*21*cm,0.7*21*cm]),dpi=1000,sharex=True,sharey=True,constrained_layout=False)

for model in models:
    path_of_output="/home/au567859/DataHandling/models/output"
    full_dir=os.path.join(path_of_output,model)
    subdirs=os.listdir(full_dir)

    print('This is model ' + model,flush=True)

    for dir in subdirs:
        dir_split = dir.split("-")

        y_plus = int(dir_split[0][-2:])


        index_vars_s = dir_split.index("VARS")
        index_target = dir_split.index("TARGETS")

        var = dir_split[index_vars_s+1:index_target]

        if '|' in dir_split[index_target+1:][0]:
            target=dir_split[index_target+1:][0].split('|')
        else:
            target = dir_split[index_target+1:] 
            
        if "normalized" not in dir_split:
            normalize = False
        else:
            normalize = True
            target = target[:-1]

        if 'flux' in target[0]:
            target_type = "flux"
        elif 'tau_wall' in target[0]:
            target_type = "stress"

        model_path, output_path =utility.model_output_paths(model,y_plus,var,target,normalize)
        prediction_path=os.path.join(output_path,'predictions.npz')
        target_path=os.path.join(output_path,'targets.npz')
        fourier_amplitudes_targ, fourier_amplitudes_pred, Lambda_x, Lambda_z = PSD(target_path,prediction_path)

    pct10=0.1*np.max(fourier_amplitudes_targ)
    pct50=0.5*np.max(fourier_amplitudes_targ)
    pct90=0.9*np.max(fourier_amplitudes_targ)
    pct100=np.max(fourier_amplitudes_targ)

    i = models.index(model)
    row, col = divmod(i,4)
    CP=axs[row,col].contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ,[pct10,pct50,pct90,pct100],cmap=cmap)
    CS=axs[row,col].contour(Lambda_x,Lambda_z,fourier_amplitudes_pred,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
    axs[row,col].set_xscale('log')
    axs[row,col].set_yscale('log')

fig.subplots_adjust(hspace=0.3)
for i in range(4):
    y_plus=[15,30,50,75]
    pr=[0.025,0.2,0.71,1]	
    axs[3,i].set_xlabel(r'$\lambda_x^+$')
    axs[i,0].set_ylabel(r'$\lambda_z^+$')
    axs[0,i].set_title(r'$y^+=$'+str(y_plus[i]))
    ax2 = axs[i,3].twinx()
    ax2.set_ylabel(r'$k_x\ k_z\ \phi_{q_w}$' + "\n" + r'Pr = '+str(pr[i]),fontsize=9,linespacing=2)
    ax2.get_yaxis().set_ticks([])


# %%
