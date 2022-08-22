import matplotlib
matplotlib.use('Agg')

import yt, gc, sys
import numpy as np
from matplotlib.colors import LogNorm
from yt.visualization.base_plot_types import get_multi_plot
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

yt.set_log_level(40)

from analysis_helpers import *

def _sum_metallicity(field, data):
    return ((data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3')).to('Zsun')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.01295

def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])


outs = [d for d in np.arange(200, 1306, 2)]
louts = np.array(outs)[np.arange(rank, len(outs), size)]
simname = 'phoenix_256_IC1'
simroot = '/scratch3/06429/azton/phoenix'

rsds = yt.load('%s/%s/rockstar_halos/halos_RD%04d.0.bin'%(simroot, simname, outs[-1]))
rsad = rsds.all_data()


if rank == 0:
    print('[%d] Locating largest star forming halo'%rank)
    ds = yt.load('%s/%s/RD%04d/RD%04d'%(simroot, simname, outs[-1], outs[-1]))
    ds.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun',sampling_type='cell')
    ds.add_particle_filter('p2_stars')

    mstarmax = 0
    maxind = -1
    for i, hc in enumerate(rsad['halos','particle_position'].to('unitary')):
        r = rsad['halos','virial_radius'][i].to('unitary')
        sp = ds.sphere(hc, r)
        if sp['p2_stars','particle_mass'].sum() > mstarmax:
            maxind = i
            mstarmax =  sp['p2_stars','particle_mass'].sum()
    print('[%d] Found %d with %e Mstar'%(rank, i, mstarmax.to('Msun')))
else:
    maxind = None

maxind = comm.bcast(maxind, root=0)
comm.Barrier()

print('Rank %d using outputs: %s'%(rank, ','.join(['%s'%d for d in louts])))
print(maxind)
sr = rsad['halos','virial_radius'][maxind].to('unitary')

for d in louts:


    orient = "horizontal"
    ds4 = yt.load('%s/%s/RD%04d/RD%04d'%(simroot, simname, d, d))  # load data
    ds4.add_field(('gas','p3_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')
    
    swidth = ds4.quan(21, 'kpc')
    sc = rsad['halos','particle_position'][maxind].to('unitary')
    

    hcbox = ds4.sphere(sc, swidth*1.5)

    # There's a lot in here:
    #   From this we get a containing figure, a list-of-lists of axes into which we
    #   can place plots, and some axes that we'll put colorbars.
    # We feed it:
    #   Number of plots on the x-axis, number of plots on the y-axis, and how we
    #   want our colorbars oriented.  (This governs where they will go, too.
    #   bw is the base-width in inches, but 4 is about right for most cases.
    fig, axes, colorbars = get_multi_plot(3,1,colorbar=orient, bw=6)

    proj4 = yt.ProjectionPlot(ds4,"z",  
                                fields=[("gas", "density"), ("gas", "temperature"), ("gas", "p3_metallicity")], 
                                weight_field=('gas','density'),
                                center = sc, 
                                data_source = hcbox, 
                                width=(21, 'kpc')
                                )
    proj4.set_antialias(True)

    frb4 = proj4.data_source.to_frb((20, 'kpc'), 1024)

    dens_axes = [axes[0][0]]
    temp_axes = [axes[0][1]]
    meta_axes = [axes[0][2]]

    for ax in dens_axes + temp_axes + meta_axes:

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Converting our Fixed Resolution Buffers to numpy arrays so that matplotlib
    # can render them

    dens4 = np.array(frb4[("gas", "density")])
    temp4 = np.array(frb4[("gas", "temperature")])
    meta4 = np.array(frb4[('gas','p3_metallicity')])


    plots = [
        dens_axes[0].imshow(dens4, origin="lower", norm=LogNorm()),
        temp_axes[0].imshow(temp4, origin="lower", norm=LogNorm()),
        meta_axes[0].imshow(meta4, origin="lower", norm=LogNorm()),
        ]

    for plot in [plots[i] for i in [0]]:
        plot.set_clim(1e-27, 1e-23)
        plot.set_cmap("RdYlBu_r")
    for plot in [plots[i] for i in [1]]:
        plot.set_clim(10, 3e4)
        plot.set_cmap("inferno")
    for plot in [plots[i] for i in [2]]:
        plot.set_clim(5e-6, 1e-2)
        plot.set_cmap("seismic")

    titles = [
        r"$\mathrm{Density}\ (\mathrm{g\ cm^{-3}})$",
        r"$\mathrm{Temperature}\ (\mathrm{K})$",
        r"$\mathrm{Pop III Metallicity}\ (Z_\odot)$",
    ]

    for p, cax, t in zip(plots, colorbars, titles):
        cbar = fig.colorbar(p, cax=cax, orientation=orient)
        cbar.set_label(t)

    # And now we're done!
    # plt.subplots_adjust(wspace = 0.1)

    fig.savefig(f"./images/%s_side-by-side_RD%04d.png"%(simname, d), bbox_inches='tight')
    gc.collect()