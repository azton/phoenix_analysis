'''
    At time of formation, get the two-point correlation between a P2 star and P3 stars
'''
import yt, sys, os
from halotools.mock_observables import tpcf
import matplotlib.pyplot as plt
import numpy as np



def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.02
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.02
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun')

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
    
    sim = sys.argv[1]
    dstart = int(sys.argv[2])
    dend = int(sys.argv[3])

    final_sne = np.zeros((3,24))
    final_hne = np.zeros((3,24))
    final_pisn = np.zeros((3,24))
    
    for d in range(dstart, dend):
        simpath = '/mnt/d/starnet/simulation_data'
        dest = '/mnt/d/starnet/simulation_data/phoenix_analysis/'
        pltdest = dest+'correlation_functions'
        
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
            
            pisn_filter = (ad['snr','particle_mass'].to('Msun')*1e20 > 140) \
                            & (ad['snr','particle_mass'].to('Msun')*1e20 < 260)
            sne_filter = (ad['snr','particle_mass'].to('Msun')*1e20 < 20) \
                            & (ad['snr','particle_mass'].to('Msun')*1e20 > 11)
            hne_filter = (ad['snr','particle_mass'].to('Msun')*1e20 > 20) \
                            & (ad['snr','particle_mass'].to('Msun')*1e20 < 40)

            p2pos = np.vstack([ad['new_p2_stars','particle_position_%s'%ax].to('Mpc/h') for ax in 'xyz']).T

            snepos = np.vstack([ad['snr','particle_position_%s'%ax][sne_filter].to('Mpc/h') for ax in 'xyz']).T
            hnepos = np.vstack([ad['snr','particle_position_%s'%ax][hne_filter].to('Mpc/h') for ax in 'xyz']).T
            pisnpos = np.vstack([ad['snr','particle_position_%s'%ax][pisn_filter].to('Mpc/h') for ax in 'xyz']).T

            period = float(ds.parameters['CosmologyComovingBoxSize'])
            max_r = float(ds.quan(20, 'kpccm').to('Mpc/h'))
            rbins = np.logspace(-7,-4, 25)

            snecf = tpcf(p2pos, rbins, snepos, period=period)
            hnecf = tpcf(p2pos, rbins, hnepos, period=period)
            pisncf = tpcf(p2pos, rbins, pisnpos, period=period)

            if not np.any(np.isnan(final_sne)):
                final_sne += snecf
            if not np.any(np.isnan(final_hne)):
                final_hne += hnecf
            if not np.any(np.isnan(final_pisn)):
                final_pisn += pisncf

    fig, ax = plt.subplots(3, 1, figsize=(8,12), sharex=True)

    ax[0].plot(rbins[:-1]*1e6, final_sne[2])
    ax[1].plot(rbins[:-1]*1e6, final_hne[2])
    ax[2].plot(rbins[:-1]*1e6, final_pisn[2])
    plt.savefig(pltdest + '/SN_p2_correlation.png')


        
if __name__=='__main__':
    main()