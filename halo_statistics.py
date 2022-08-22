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
import matplotlib
matplotlib.use("Agg")
import yt, json, h5py, os
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser as ap
import matplotlib.pyplot as plt
from analysis_helpers import add_particle_filters

def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.01295
# yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.01295
# yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')

argparser = ap()
argparser.add_argument('--sim', type=str, default=None, 
                    help="simulation name")
argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                    help="file path to simulation directory")
argparser.add_argument('--output_dest','-od', type=str, default='./enriched_stats',
                    help='Destination for analysis logs and other output.')
argparser.add_argument('--outputs', type=int, nargs='+', default=None,
                    help="white-space separated list of dumps to analyze")
args = argparser.parse_args()

if not os.path.exists(args.output_dest):
    os.makedirs(args.output_dest, exist_ok = True)


# simpath = '/scratch3/06429/azton/phoenix'
datapath = args.sim_root + '/%s'%args.sim
rspath = '%s/%s/rockstar_halos'%(args.sim_root, args.sim)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_inds = np.arange(rank, len(args.outputs), size)
outputs = [args.outputs[i] for i in local_inds]
print('%d Checking '%rank, outputs)
for i, d in enumerate(outputs):
    label = 'RD%04d'%d

    rdout = '%s/%s/%s'%(datapath, label, label)
    halofile = '%s/halos_%s.0.bin'%(rspath, label)
    profiles = []
    labels = []
    if os.path.exists(rdout) and os.path.exists(halofile):
        # each rank has its own output file
        stats = {}
        stats['halo_rs_index'] = []
        stats['halo_mvir'] = []
        stats['halo_mgas'] = []
        stats['halo_mstar'] = []
        stats['halo_p2mass'] = []
        stats['halo_p3mass'] = []
        stats['halo_mmetal'] = []
        stats['halo_redshift'] = []
        stats['halo_rvir'] = []
        stats['halo_live_p3cnt'] = []
        stats['halo_remnant_p3cnt'] = []
        stats['halo_p2cnt'] = []
        ds = yt.load(rdout)
        ds = add_particle_filters(ds)
        ds.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')
        ds.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units = 'Zsun', sampling_type='cell')

        rsds = yt.load(halofile)
        hc = rsds.all_data()
        if not os.path.exists('%s/halo_logs'%args.output_dest):
            os.makedirs('%s/halo_logs'%args.output_dest,exist_ok=True)
        logfile = '%s/halo_logs/%d_%s_%s-%0.2f_halo_stats.json'\
                %(args.output_dest, rank, args.sim, label, ds.current_redshift)
        for i, hcenter in enumerate(hc['halos','particle_position'].to('unitary')):
            stats['halo_rs_index'].append(i)
            rvir = hc['halos','virial_radius'][i].to('unitary')
            sp = ds.sphere(hcenter, rvir)

            stats['halo_mvir'].append(float(
                        sp['gas','cell_mass'].sum().to('Msun') \
                            + sp['all','particle_mass'].sum().to('Msun')
                    ))
            stats['halo_mgas'].append(float(
                sp['gas','cell_mass'].sum().to('Msun')
            ))
            stats['halo_mmetal'].append(float(
                ((sp['enzo','SN_Colour'] + sp['enzo','Metal_Density'])\
                    * sp['gas','cell_volume']).sum().to('Msun')
            ))
            stats['halo_mstar'].append(float(
                (sp['p3_stars','particle_mass'].sum()\
                     + sp['p2_stars','particle_mass'].sum()).to('Msun')
            ))
            stats['halo_p2mass'].append(float(sp['p2_stars','particle_mass'].to('Msun').sum()))
            stats['halo_p3mass'].append(float(sp['p3_stars','particle_mass'].to('Msun').sum()))
            stats['halo_redshift'].append(float(ds.current_redshift))
            stats['halo_rvir'].append(float(hc['halos','virial_radius'][i].to('kpc')))
            stats['halo_live_p3cnt'].append(len(sp['p3_stars','particle_mass']))
            stats['halo_remnant_p3cnt'].append(len(sp['all_p3','particle_mass'])\
                                                -len(sp['p3_stars','particle_mass']))
            stats['halo_p2cnt'].append(len(sp['p2_stars','particle_mass']))

            # log a profile of mass as fn of metallicity
            prof = yt.create_profile(sp, 
                                        ('gas','sum_metallicity'), 
                                        ('gas','cell_mass'), 
                                        weight_field=None,
                                        extrema={('gas','sum_metallicity'):(1e-8, 100)})
            # prof.set_unit('cell_mass','Msun')

            profiles.append(prof)
            labels.append(float((sp['gas','cell_mass'].sum() + sp['all','particle_mass'].sum()).to('Msun')))

        metal_profiles = h5py.File('%s/%d_%s_%s_%0.2f_profiles.h5'%\
                    (args.output_dest, rank, args.sim, label, ds.current_redshift),'w')
        save_mass = np.array([p.field_data[('gas','cell_mass')] for p in profiles])
        save_metal = np.array([p.x for p in profiles])
        save_hmass = np.array(labels)
        metal_profiles.create_dataset('mass', data=save_mass)
        metal_profiles.create_dataset('metallicity', data=save_metal)
        metal_profiles.create_dataset('halo_mass', data=save_hmass)
        metal_profiles.close()

        # mass_bin_edge = np.logspace(6.2, 7.8, 4)#[10**6.3, 10**6.8, 10**7.5, 10**8]
        # linestyles = ['solid','dotted','dashed','dashdot']
        # colors = ['tab:blue','tab:orange','tab:green','tab:grey']
        # fig, ax = plt.subplots()
        # for e in range(1, len(mass_bin_edge)):
        #     inds = np.logical_and((save_hmass > mass_bin_edge[e-1]), (save_hmass < mass_bin_edge[e]))
        #     for pltind in inds:
        #         ax.plot( save_metal[0], save_mass[pltind]/1.998e33, linestyle=linestyles[e-1], label='$M > 10^{%0.1f} M_\odot$'%np.log10(mass_bin_edge[e-1]), color=colors[e-1], alpha=0.3)
    
        # inds = save_hmass > mass_bin_edge[-1]
        # for pltind in inds:
        #     ax.plot( save_metal[0], save_metal[pltind]/1.998e33, linestyle=linestyles[-1], label = '$M > 10^{%0.1f} M_\odot$'%np.log10(mass_bin_edge[-1]), color=colors[-1], alpha=0.3)
        # ax.legend()
        # ax.set_xlim(100, 10**-8)           
        # ax.set_ylabel('$M~[M_\odot$]')
        # ax.set_xlabel('$Z~[Z_\odot]$')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.savefig('%s/halo_logs/%s_profiles.png'%(args.output_dest, label), bbox_inches='tight', dpi=600)

        with open(logfile, 'w') as f:
                print('saving halofile to %s'%logfile)
                json.dump(stats, f, indent=4)
    else:
        print("%d: %s or %s don't exist!"%(rank, rdout, halofile))