'''
    At time of formation, get the two-point correlation between a P2 star and P3 stars
'''
import yt, sys, os, h5py
from halotools.mock_observables import tpcf
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser as ap
from mpi4py import MPI

def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.01295
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.01295
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')

def _all_p3(pfilter, data): # all p3 particles, past and present
    # return all present and SNr | blackhole collapses.
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0))\
        | ((data['all','particle_type'] == 1)\
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') > 1))
yt.add_particle_filter('all_p3',function=_all_p3, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _p3_stars(pfilter,data): # active Pop 3 stars
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') > 1))
yt.add_particle_filter('p3_stars',function=_p3_stars, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _snr(pfilter, data): # supernovae remnants
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') < 1))
yt.add_particle_filter('snr',function=_snr, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')


def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])

def _new_p2(pfilter, data):
    return (data['p2_stars','age'].to('Myr') < 1)
yt.add_particle_filter('new_p2_stars',function=_new_p2, requires=['age'], filtered_type='p2_stars')



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

    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest, exist_ok = True)
    nouts = (args.outputs[1]-args.outputs[0]) // args.outputs_resolution
    
    out_list = np.linspace(int(args.outputs[0]), int(args.outputs[1]), int(nouts), dtype=int)
    localinds = np.arange(rank, len(out_list), size)
    localouts = [out_list[i] for i in localinds]
    n_bins = 50

    final_sne = np.zeros((3,n_bins-1))
    final_hne = np.zeros((3,n_bins-1))
    final_pisn = np.zeros((3,n_bins-1))

    for d in localouts:
        simpath = args.sim_root
        sim = args.sim
        pltdest = args.output_dest
        
        os.makedirs(pltdest, exist_ok=True)    
        
        output = 'RD%04d'%d
        dataset = '%s/%s/%s/%s'%(simpath, sim, output, output)
        
        if os.path.exists(dataset):
            
            ds = yt.load(dataset)
            for filter in ['p3_stars','snr','p2_stars','new_p2_stars']:
                ds.add_particle_filter(filter)
            ad = ds.all_data()

            # want to plot several points for new P2 stars:
            #   P(pisn|p2)
            #   P(sne|p2)
            #   p(hne|p2)

            p2c = ad['new_p2_stars', 'particle_position'].to('Mpc/h')
            snrpos = ad['snr','particle_position'].to('Mpc/h')
            
            if len(p2c) == 0:
                print("No new P2 stars found in %s! D:"%output)
                continue
            if len(snrpos) == 0:
                print("No SNRs registered... WTF.")
                continue
            
            pisn_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 > 140), (ad['snr','particle_mass'].to('Msun')*1e20 < 260))
            sne_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 < 20), (ad['snr','particle_mass'].to('Msun')*1e20 > 11))
            hne_filter = np.logical_and((ad['snr','particle_mass'].to('Msun')*1e20 > 20), (ad['snr','particle_mass'].to('Msun')*1e20 < 40))

            p2pos = np.vstack([ad['new_p2_stars','particle_position_%s'%ax].to('Mpc/h') for ax in 'xyz']).T

            snepos = np.vstack([ad['snr','particle_position_%s'%ax][sne_filter].to('Mpc/h') for ax in 'xyz']).T
            hnepos = np.vstack([ad['snr','particle_position_%s'%ax][hne_filter].to('Mpc/h') for ax in 'xyz']).T
            pisnpos = np.vstack([ad['snr','particle_position_%s'%ax][pisn_filter].to('Mpc/h') for ax in 'xyz']).T

            period = float(ds.parameters['CosmologyComovingBoxSize'])
            rbins = np.logspace(-7,-1, n_bins)

            snecf = tpcf(p2pos, rbins, snepos, period=period)
            hnecf = tpcf(p2pos, rbins, hnepos, period=period)
            pisncf = tpcf(p2pos, rbins, pisnpos, period=period)

            if not np.any(np.isnan(final_sne)):
                final_sne += snecf
            if not np.any(np.isnan(final_hne)):
                final_hne += hnecf
            if not np.any(np.isnan(final_pisn)):
                final_pisn += pisncf
    with h5py.File('%s/%d_%s_correlation_fn.h5'%(pltdest, rank, sim), 'w') as f:
        f.create_dataset('twopoint_sne_sum', data=final_sne)
        f.create_dataset('twopoint_hne_sum', data=final_hne)
        f.create_dataset('twopoint_pisne_sum', data=final_pisn)

    fig, ax = plt.subplots(3, 1, figsize=(8,12), sharex=True)

    ax[0].plot(rbins[:-1]*1e6, final_sne[2])
    ax[1].plot(rbins[:-1]*1e6, final_hne[2])
    ax[2].plot(rbins[:-1]*1e6, final_pisn[2])
    ax[2].set_xlabel('R [pc/h]')
    ax[1].set_ylabel('\\xi')
    plt.savefig(pltdest + '/%d_SN_p2_correlation.png'%rank)


        
if __name__=='__main__':
    main()