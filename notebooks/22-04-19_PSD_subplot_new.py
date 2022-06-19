#%%
def PSD(target_path,prediction_path):
    import numpy as np
    targ=np.load(target_path)
    data_targ=targ["test"]

    pred=np.load(prediction_path)
    data_pred=pred["test"]

    # Computational box and dimensions of DNS daa
    Nx = 256
    Nz  = 256
    Lx  = 12
    Lz  = 6

    # Wavenumber spacing
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz

    # Creating the wavenumber grid. The fftfreq returns a one dimensional array containing the wave vectors
    # for the fftn in the correct order. Since this is a fraction of 1 returned, we multiply by N to get 
    # a pixel frequency and also dk to get a physical meaning and into wave domain?
    kx = dkx * np.fft.fftfreq(Nx) * Nx
    kz = dkz * np.fft.fftfreq(Nz) * Nz
    [kkx,kkz]=np.meshgrid(kx,kz)


    # We convert to wavelength, however since the DC components creates a division by zero, we ignore the 
    # error and set to zero if the division was zero.
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu

    # calculating wavelength in plus units
    with np.errstate(divide='ignore', invalid='ignore'): 
            Lambda_x=(2*np.pi/kkx)*u_tau/nu
            Lambda_x[Lambda_x==np.inf]=0

            Lambda_z = (2*np.pi/kkz)*u_tau/nu
            Lambda_z[Lambda_z==np.inf]=0

    # It doesn't matter if the mean is subtracted as far as I can tell
    Theta_fluc_targ=data_targ#-np.mean(data_targ)
    Theta_fluc_pred=data_pred#-np.mean(data_pred)

    # We compute the 2 dimensional discrete Fourier Transform
    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    fourier_image_pred = np.fft.fftn(Theta_fluc_pred)

    # The now contains complex valued amplitudes of all Fourier components. We are only interested in
    # the size of the amplitudes. We will further assume that the average amplitude is zero, so 
    # that we only require the square of the amplitudes to compute the variances.
    # We also compute the pre-multiplication with the wavenumber vectors
    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)*kkx*kkz
    fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)*kkx*kkz

    # We remove the DC component (It fucks up the plots), and we remove the negative symmetric part since
    # for real signals signals, the coefficients of positive and negative frequencies become complex conjugates.
    # That means, we do not need both sides of spectrum to represent the signal, a single side will do. This is
    #  known as Single Side Band (SSB) spectrum (We might need to multiply by 2, https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab)
    # however since the do not use the magnitude of the power
    fourier_amplitudes_targ = fourier_amplitudes_targ[1:128,1:128]
    fourier_amplitudes_pred = fourier_amplitudes_pred[1:128,1:128]
    Lambda_x = Lambda_x[1:128,1:128]
    Lambda_z = Lambda_z[1:128,1:128]

    return fourier_amplitudes_targ, fourier_amplitudes_pred, Lambda_x, Lambda_z

#%%
# Theory file:///C:/Users/ander/Downloads/Miller_InvestigationofTurbulentStructureModificationbyMomentum%20(1).pdf
# Still need to get minor ticks on symlog
from DataHandling import utility
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


cmap = mpl.cm.Greys(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[5:,:-1])

#models=['winter-firefly-75']
#models=['winter-firefly-75','unique-night-53','exalted-durian-66','pleasant-snowflake-76','amber-glitter-73','worthy-bush-58','frosty-lion-63','super-pond-78','different-silence-74','devoted-bee-61','glad-oath-62','solar-water-77','clear-donkey-82','earthy-universe-83','effortless-totem-85','deep-terrain-84']
models=['youthful-wave-72','bumbling-shape-52','zany-snowball-67','flowing-planet-79','silvery-dragon-71','glowing-sponge-59','crisp-galaxy-64','rich-moon-81','cool-bird-51','treasured-music-60','skilled-planet-65','driven-gorge-80','true-salad-89','iconic-monkey-87','stellar-bee-88','divine-glitter-86']

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

    pct10=np.max(fourier_amplitudes_targ)*0.1
    pct50=np.max(fourier_amplitudes_targ)*0.5
    pct90=np.max(fourier_amplitudes_targ)*0.9
    pct100=np.max(fourier_amplitudes_targ)*1

    i = models.index(model)
    row, col = divmod(i,4)
    CP=axs[row,col].contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ,[pct10,pct50,pct90,pct100],cmap=cmap)
    CS=axs[row,col].contour(Lambda_x,Lambda_z,fourier_amplitudes_pred,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
    
    # zero freq https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab
    axs[row,col].set_xscale('log')
    #axs[row,col].set_xlim([np.min(Lambda_x[np.nonzero(Lambda_x)]),np.max(Lambda_x)])
    axs[row,col].set_yscale('log')
    #axs[row,col].set_ylim([np.min(Lambda_z[np.nonzero(Lambda_z)]),np.max(Lambda_z)])
    #axs[row,col].minorticks_on()

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

