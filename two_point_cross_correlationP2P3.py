'''
    At time of formation, get the two-point correlation between a P2 star and P3 stars
'''

import yt, sys, os, h5py
from halotools.mock_observables import tpcf
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser as ap
from mpi4py import MPI
from analysis_helpers import add_particle_filters
yt.set_log_level(40)
# def _sum_metallicity(field, data):
#     return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.01295
# yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

# def _p3_metallicity(field, data):
#     return (data['SN_Colour'] / data['Density']) / 0.01295
# yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')



def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    argparser = ap()

    argparser.add_argument('--sim', type=str, default=None, 
                        help="simulation name")
    argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                        help="file path to simulation directory")
    argparser.add_argument('--output_dest','-od', type=str, default='./correlation_function',
                        help='Destination for analysis logs and other output.')
    argparser.add_argument('--outputs', type=float, nargs='+', default=None,
                        help="white-space separated list of dumps to analyze")
    argparser.add_argument('--outputs_resolution','-out_res', type=float, default=1,
                        help="how many outputs to skip between analyzed ones")
    args = argparser.parse_args()

    simpath = args.sim_root
    sim = args.sim
    pltdest = args.output_dest
    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest, exist_ok = True)
    nouts = (args.outputs[1]-args.outputs[0]) // args.outputs_resolution
    os.makedirs(args.output_dest, exist_ok=True)    
    
    out_list = np.linspace(int(args.outputs[0]), int(args.outputs[1]), int(nouts), dtype=int)
    localinds = np.arange(rank, len(out_list), size)
    localouts = [out_list[i] for i in localinds]
    n_bins = 25

    final_sne = np.zeros((3,n_bins-1))
    analyzed = 0
    nrand = 10**5
    lowbin = -6.5
    highbin = -4
    rx = np.random.uniform(lowbin, highbin, size=nrand)
    ry = np.random.uniform(lowbin, highbin, size=nrand)
    rz = np.random.uniform(lowbin, highbin, size=nrand)

    for d in localouts:

        
        
        output = 'RD%04d'%d
        dataset = '%s/%s/%s/%s'%(simpath, sim, output, output)
        
        if os.path.exists(dataset) \
                and not os.path.exists('%s/%d_%s_RD%04d_correlation_fn.h5'\
                                    %(pltdest, rank, sim, d)):
            
            ds = yt.load(dataset)
            ds = add_particle_filters(ds)
            # rsds = yt.load('%s/%s/rockstar_halos/halos_%s.0.bin'%(simpath, sim, output))
            # rad = rsds.all_data()
            # halopos = [[rad['halos','particle_position_%s'%ax][idx].to('unitary') for ax in 'xyz'] for idx in range(rad['halos','virial_radius'].size)]
            # # want to plot several points for new P2 stars:
            # #   P(pisn|p2)
            # #   P(sne|p2)
            # #   p(hne|p2)
            # for i, hp in enumerate(halopos):
            
            ad = ds.all_data()
            # p2c = ad['new_p2_stars', 'particle_position'].to('Mpc/h')
            p3pos = ad['p3_stars','particle_position'].to('Mpc/h')
            
            # # if len(p2c) == 0:
            #     print("No new P2 stars found in %s #%d! D:"%(output, i))
            #     continue
            if len(p3pos) == 0:
                print("No SNRs registered in %s"%(output))
                continue
            
            # pisn_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 > 140), (ad['snr','particle_mass'].to('Msun')*1e20 < 260))
            # sne_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 < 20), (ad['snr','particle_mass'].to('Msun')*1e20 > 11))
            # hne_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 > 20), (ad['snr','particle_mass'].to('Msun')*1e20 < 40))

            # p2pos = np.vstack([ad['new_p2_stars','particle_position_%s'%ax].to('Mpc/h') for ax in 'xyz']).T
            p3pos = np.vstack([ad['p3_stars','particle_position_%s'%ax].to('Mpc/h') for ax in 'xyz']).T

            # snepos = np.vstack([ad['snr','particle_position_%s'%ax][sne_filter].to('Mpc/h') for ax in 'xyz']).T
            # hnepos = np.vstack([ad['snr','particle_position_%s'%ax][hne_filter].to('Mpc/h') for ax in 'xyz']).T
            # pisnpos = np.vstack([ad['snr','particle_position_%s'%ax][pisn_filter].to('Mpc/h') for ax in 'xyz']).T

            period = float(ds.parameters['CosmologyComovingBoxSize'])
            randlocs = np.vstack([10**rx,
                                    10**ry,
                                    10**rz]).T
            rbins = np.logspace(lowbin, highbin, n_bins)
            allcf = tpcf(p3pos, rbins, period = period, randoms=randlocs)
            final_sne += allcf[allcf != np.nan]
            analyzed += 1
            # print(final_sne.size)
            with h5py.File('%s/%d_%s_RD%04d_correlation_fn.h5'%(pltdest, rank, sim, d), 'w') as f:
                f.create_dataset('twopoint_p3', data=allcf)
                f.create_dataset('rbins', data=rbins)


    fig, ax = plt.subplots(1, 1, figsize=(4,4), sharex=True)
    ax.plot(rbins[:-1]*1e6, final_sne[0]/float(analyzed))
    ax.plot(rbins[:-1]*1e6, final_sne[1]/float(analyzed))
    ax.plot(rbins[:-1]*1e6, final_sne[2]/float(analyzed))
    ax.set_xlabel('R [pc/h]')
    ax.set_ylabel('$\\xi$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig(pltdest + '/%d_SN_p2_correlation.png'%rank, bbox_inches='tight')

    plt.close()
        
if __name__=='__main__':
    main()