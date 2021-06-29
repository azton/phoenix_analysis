'''
    Generate a star forming history for all P3 stars
    within phoenix simulations.  To compare with RS at
        https://ui.adsabs.harvard.edu/abs/2013ApJ...773...83X/abstract

'''

import sys, yt, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# active p3 stars
def _p3stars(pfilter, data):
    return (data['creation_time'] > 0) & (data['particle_type'] == 5) & (data['particle_mass'].to('Msun') > 1)
yt.add_particle_filter('p3_stars',function=_p3stars, requires=['creation_time', 'particle_type', 'particle_mass'])

# p3 SN remnants
def _p3remnant(pfilter, data):
    return (data['creation_time'] > 0) & (data['particle_type'] == 5) & (data['particle_mass'].to('Msun') < 1e-5)
yt.add_particle_filter('sne_remnant',function=_p3remnant, requires=['creation_time', 'particle_type', 'particle_mass'])

# p3 BH remnants
def _p3bh(pfilter, data):
    return (data['creation_time'] > 0) & (data['particle_type'] == 1) & (data['particle_mass'].to('Msun') > 1)
yt.add_particle_filter('p3_bh',function=_p3bh, requires=['creation_time', 'particle_type', 'particle_mass'])

def add_filters(ds):
    for filter in ['p3_stars','sne_remnant','p3_bh']:
        ds.add_particle_filter(filter)
    return ds

def get_redshift(ds, t):
    tstart = ds.cosmology.t_from_z(ds.parameters['CosmologyInitialRedshift'])
    znow = ds.cosmology.z_from_t(float(tstart)+t)
    return znow
'''
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                                                        Main
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
'''
def main():
    sim = sys.argv[2]
    output = int(sys.argv[3])
    sim_root = sys.argv[1]
    starfile = '%s/starfile.json'%sim

    ds = yt.load('%s/%s/RD%04d/RD%04d'%(sim_root, sim, output, output))
    ds = add_filters(ds)
    if not os.path.exists(starfile):
        # load output, get star list. compile m*(t) and dm/dt.  save txt file that has stars and thier quantities.

        ad = ds.all_data()

        stardict = {}
        stardict['sindex'] = []
        stardict['pid'] = []
        stardict['mass'] = []
        stardict['position'] = []
        stardict['birth'] = []
        stardict['metallicity'] = []
        stardict['z_birth'] = []
        stardict['lifetime'] = []
        for stype in ['p3_stars','sne_remnant','p3_bh']:
            for i, star in enumerate(ad[stype, 'particle_mass'].to('Msun')):
                if 'remnant' in stype:
                    star *= 1e20
                stardict['sindex'].append(i)
                stardict['pid'].append(int(ad[stype,'particle_index'][i]))
                stardict['mass'].append(float(star))
                stardict['position'].append([float(i) for i in ad[stype,'particle_position'][i].to('unitary')])
                stardict['birth'].append(float(ad[stype, 'creation_time'][i].to('Myr')))
                stardict['metallicity'].append(float(ad[stype, 'metallicity_fraction'][i]))
                stardict['z_birth'].append(float(get_redshift(ds, stardict['birth'][-1])))
                stardict['lifetime'].append(float(ad[stype, 'creation_time'][i].to('Myr')))

        os.makedirs(sim, exist_ok=True)
        with open(starfile, 'w') as f:
            json.dump(stardict, f, indent=4)
    else:
        with open(starfile, 'r') as f:
            stardict = json.load(f)    
    # M*(t)
  
    tend = float(max(stardict['birth']))
    tstart = float(min(stardict['birth']))
  
    # bins for time
    tbins = np.linspace(tstart, tend, 100)
    dt = (tbins[1] - tbins[0])*1e6 #yr
    mbins = np.zeros(101)
    nbins = np.zeros(101)
    for i, t in enumerate(stardict['birth']):
        if stardict['mass'][i] <= 300:
            tbin = np.digitize(t, tbins)
            mbins[tbin] += stardict['mass'][i]
            nbins[tbin] += 1
    print(np.unique(mbins))
    cnum = np.array([nbins[:i].sum() for i in range(len(nbins))])
    cmass = np.array([mbins[:i].sum() for i in range(len(mbins))])
    dmdt = np.array([(cmass[i] - cmass[i-1])/dt for i in range(1,len(cmass))]) / float(ds.parameters['CosmologyComovingBoxSize'])**3
    
    
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(5.5,8))
    tdmdt = tbins
    ax[0].plot(tdmdt, dmdt)
    ax[1].plot(tbins, cmass[1:]/ float(ds.parameters['CosmologyComovingBoxSize'])**3)
    ax[2].plot(tbins, cnum[1:]/ float(ds.parameters['CosmologyComovingBoxSize'])**3)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')

    ax[0].set_ylabel('SFR [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]')
    ax[1].set_ylabel('M* [M$_\odot$ Mpc$^{-3}$]')
    ax[2].set_ylabel('N* Mpc$^{-3}$')
    ax[2].set_xlabel('t [Myr]')


    plt.savefig('P3_SFR_%s.pdf'%sim, bbox_inches='tight')


    


    



if __name__=='__main__':
    main()