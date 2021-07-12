"""
    we have to assume there is some maximal range of influence of a P3 region
        It may depend on z and time after region starts
        describes the maximal radii to expect metals, ionization influence.
"""

import yt,sys,os,json
import numpy as np
import matplotlib.pyplot as plt
from yt.data_objects.particle_filters import add_particle_filter
from analysis_helpers import *

# yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun')
# yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun')
# yt.add_particle_filter('p3_stars',function=_p3_stars, \
#         requires=['particle_type','particle_mass','creation_time'], filtered_type='all')
# yt.add_particle_filter('snr',function=_snr, \
#         requires=['particle_type','particle_mass','creation_time'], filtered_type='all')
# yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])
# yt.add_particle_filter('new_p2_stars',function=_new_p2, requires=['age'], filtered_type='p2_stars')

d = int(sys.argv[1])
dspath = '/mnt/d/starnet/simulation_data/phx256-1'
dsfile = dspath + '/RD%04d/RD%04d'%(d,d)

# as a test, find a region and plot out the radii we would use

ds = yt.load(dsfile)
ds = add_particle_filters(ds)
ad = ds.all_data()

ctime = ad['all_p3','creation_time'].to('Myr')
pinds = ad['all_p3','particle_index']
output_list = np.array(find_correct_output(ds, ctime)) + 1

# sort so we iterate through time
srtlist = np.argsort(output_list)
pidx_srt = pinds[srtlist]
output_list_srt = output_list[srtlist]


profile_fields = ['p3_metallicity','temperature']
profile_stat = {}

 

logged_pids = []

print('First formation at d = %04d, last is at %04d'%(min(output_list), max(output_list)))
print(np.unique(output_list))
n = 0
for i, (pidx,d) in enumerate(zip(pidx_srt, output_list_srt)):
    z_profiles = []
    t_profiles = []
    labels = []
    dsfile = dspath + '/RD%04d/RD%04d'%(d,d)
    pidx = int(pidx)
    if os.path.exists(dsfile):
        ds = yt.load(dsfile)
        ds = add_particle_filters(ds)
        ad = ds.all_data()
        idx = np.where(ad['all','particle_index'] == pidx)[0][0]
        ct = ctime[i]
        c = ad['all','particle_position'][idx]
        r0 = ds.quan(200, 'kpccm').to('unitary')
        b = ds.box(c-r0/2., c+r0/2.)
        dfinal = min(595, d + 120)
        
        bf = None
        while bf == None:
            dfinfile = dspath + "/RD%04d/RD%04d"%(dfinal, dfinal)
            if os.path.exists(dfinfile):
                indx = np.where(ad['all','particle_index'] == pidx)[0][0]
                cf = ad['all','particle_position'][indx]
                rf = ds.quan(200, 'kpccm').to('unitary')
                bf = ds.box(c-r0/2., c+r0/2.)
            else:
                dfinal += 1


        for p3 in bf['all_p3','particle_index']:
            if p3 in logged_pids:
                print('Skipping %d; already did region for %d'%(pidx, p3))
                continue
        logged_pids.append(pidx)
        
        profile_stat[pidx] = {}
        profile_stat[pidx]['time'] = [] # time of measurement
        profile_stat[pidx]['p3_metallicity_radius'] = [] # calculated radius of enrichment zone
        profile_stat[pidx]['temperature_radius'] = [] # radius of hot zone

        # if b['gas','p3_metallicity'].max() > 1e-5:
        #     continue # for now, just want to analyze pristine regions.
        print('iterating %d formed in RD%04d...'%(pidx, d))
        for dd in range(d, dfinal+1):
            dsfile = dspath + "/RD%04d/RD%04d"%(dd,dd)
            print('Creating profiles for %s (%d/%d)'%(dsfile, dd, dfinal))
            if os.path.exists(dsfile):
                ds = yt.load(dsfile)
                ds = add_particle_filters(ds)
                if dd == d:
                    time = ds.current_time.to('Myr')

                ad = ds.all_data()

                # ind = np.where(output_list_srt == d)[0][0]
                # pidx = pidx_srt[ind]
                idx = np.where(ad['all','particle_index'] == pidx)[0][0]

                c = ad['all','particle_position'][idx]
                r = ds.quan(200, 'kpccm').to('unitary')

                sp = ds.sphere(c,r)

                profile_stat[pidx]['time'].append(float(ds.current_time.to('Myr') - time))
                labels.append('%0.2f Myr'%(ds.current_time.to('Myr') - time))

                for field in profile_fields:
                    prof = yt.create_profile(sp, 
                                            'radius',
                                            [('gas',field)],
                                            weight_field='cell_volume')
                    # prof.set_unit('radius','pc')
                    # take the min radius as where the profile drops to 1/100 of central value for z
                    
                    z_prof = prof.field_data[('gas',field)]
                    radius = prof.x
                    radius = radius[z_prof != 0]
                    z_prof = z_prof[z_prof != 0]
                    z_max = z_prof[:len(z_prof)//8].mean()
                    zcut = -1
                    rcut = -1
                    rfactor = 100. #if field=='sum_metallicity' else 10.
                    for i, z in enumerate(z_prof):
                            if z < z_max / rfactor:
                                zcut = z
                                rcut = radius[i].to('kpccm')
                                break
                            
                    profile_stat[pidx]['%s_radius'%field].append(float(rcut))
                    if field == 'temperature':
                        t_profiles.append(prof)
                    else:
                        z_profiles.append(prof)
                n += 1
        p1 = yt.ProfilePlot.from_profiles(z_profiles, labels=labels)
        p1.set_unit('radius','kpccm') 
        p1.save('radial_plots/%d'%pidx)
        p2 = yt.ProfilePlot.from_profiles(t_profiles, labels=labels)
        p2.set_unit('radius','kpccm')
        p2.save('radial_plots/%d'%pidx)

        with open('radial_plots/radiuslog.json', 'w') as f:
            json.dump(profile_stat, f, indent = 4)
        
with open('radial_plots/radiuslog.json', 'w') as f:
            json.dump(profile_stat, f, indent = 4)
        

        


