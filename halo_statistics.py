''' 
    Reproduces some basic analysis from Xu, 2016 
    -- but focused on the phoenix simulations.  This only 
    makes datafiles, json outputs; plot elsewhere
        * Figure 3 Repro:
            - N_p3, N_p3,remnant, n_halos, n_halos W/ p3 as fn of halo_mass with varying redshift
            - f_halo with p3, p3 per halo as fn of halo mass with varying redshift
        * Figure 4
            - 2d histogram of p3 number - halo mass (at same redshifts as f3)
        
    
'''

import yt, json, sys, os
from mpi4py import MPI
import numpy as np
from analysis_helpers import *

sim = sys.argv[1]
di = int(sys.argv[2]) # first dump to look at
df = int(sys.argv[3]) # last dump to get
n_analyze = int(sys.argv[4]) # how many total outputs to analyze
# dlist = np.linspace(di, df, n_analyze+1, dtype=int)
dlist = [1000, 1080]
simpath = '/scratch3/06429/azton/phoenix'
datapath = simpath + '/%s'%sim
rspath = '%s/%s/rockstar_halos'%(simpath, sim)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_idxs = [0,1]#np.arange(rank, n_analyze, size)

print('Checking ', dlist)
for idx in local_idxs:
    d = dlist[idx]
    label = 'RD%04d'%d

    rdout = '%s/%s/%s'%(datapath, label, label)
    halofile = '%s/halos_%s.0.bin'%(rspath, label)


    if os.path.exists(rdout) and os.path.exists(halofile):
        # each rank has its own output file
        stats = {}
        stats['halo_rs_index'] = []
        stats['halo_mvir'] = []
        stats['halo_mgas'] = []
        stats['halo_mstar'] = []
        stats['halo_mmetal'] = []
        stats['halo_redshift'] = []
        stats['halo_rvir'] = []
        stats['halo_live_p3cnt'] = []
        stats['halo_remnant_p3cnt'] = []
        stats['halo_p2cnt'] = []
        ds = yt.load(rdout)
        ds = add_particle_filters(ds)
        rsds = yt.load(halofile)
        hc = rsds.all_data()
        if not os.path.exists('/scratch3/06429/azton/phoenix_analysis/halo_logs'):
            os.makedirs('/scratch3/06429/azton/phoenix_analysis/halo_logs',exist_ok=True)
        logfile = '/scratch3/06429/azton/phoenix_analysis/halo_logs/%d_%s_%s-%0.2f_halo_stats.json'\
                %(rank, sim, label, ds.current_redshift)
        for i, hcenter in enumerate(hc['particle_position'].to('unitary')):
            stats['halo_rs_index'].append(i)
            rvir = hc['virial_radius'][i].to('unitary')
            sp = ds.sphere(hcenter, rvir)

            stats['halo_mvir'].append(float(
                        sp['cell_mass'].sum().to('Msun') \
                            + sp['particle_mass'].sum().to('Msun')
                    ))
            stats['halo_mgas'].append(float(
                sp['cell_mass'].sum().to('Msun')
            ))
            stats['halo_mmetal'].append(float(
                ((sp['SN_Colour'] + sp['Metal_Density'])\
                    * sp['cell_volume']).sum().to('Msun')
            ))
            stats['halo_mstar'].append(float(
                (sp['p3_stars','particle_mass'].sum()\
                     + sp['p2_stars','particle_mass'].sum()).to('Msun')
            ))
            stats['halo_redshift'].append(float(ds.current_redshift))
            stats['halo_rvir'].append(float(hc['virial_radius'][i].to('kpc')))
            stats['halo_live_p3cnt'].append(len(sp['p3_stars','particle_mass']))
            stats['halo_remnant_p3cnt'].append(len(sp['all_p3','particle_mass'])\
                                                -len(sp['p3_stars','particle_mass']))
            stats['halo_p2cnt'].append(len(sp['p2_stars','particle_mass']))
        
        with open(logfile, 'w') as f:
                print('saving halofile to %s'%logfile)
                json.dump(stats, f, indent=4)
    else:
        print("%d: %s or %s don't exist!"%(rank, rdout, halofile))