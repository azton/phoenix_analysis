"""
    we have to assume there is some maximal range of influence of a P3 region
        It may depend on z and time after region starts
        describes the maximal radii to expect metals, ionization influence.
"""

import yt,sys,os,json, glob
import numpy as np
import matplotlib.pyplot as plt
from yt.data_objects.particle_filters import add_particle_filter
from analysis_helpers import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


model_time = 30 # Myr

anyl_root = sys.argv[1] # prefix for the simulation directories
simname = sys.argv[2]
dspath = '%s/%s'%(anyl_root, simname)
if rank == 0:
    print('Looking for outputs in %s'%dspath)
alldspaths = glob.glob('%s/RD*/RD[01][0123456789][0123456789][0123456789]'%(dspath))
# if rank == 0:
#     print('Found: ', alldspaths)
local_inds = np.arange(rank, len(alldspaths), size, dtype=int)
localdspaths = [alldspaths[i] for i in local_inds]
# as a test, find a region and plot out the radii we would use

profile_fields = ['p3_metallicity','temperature']
profile_stat = {}


logged_pids = []

n = 0
# print('[%d] has:'%rank, localdspaths)
for i, outpath in enumerate(localdspaths):
    z_profiles = []
    t_profiles = []
    labels = []

    ds = yt.load(outpath)
    ds = add_particle_filters(ds)
    ad = ds.all_data()

    if ad['new_p3_stars','age'].size > 0:
        dnum = int(os.path.split(outpath)[-1][2:])

        for j, c in enumerate(ad['new_p3_stars','particle_position'].to('unitary')):

            r = ds.quan(2, 'kpc')
            sp = ds.sphere(c, r)
            if sp['p2_stars','age'].size > 0 \
                    or sp['snr','age'].size > 0 \
                    or sp['p3_bh', 'age'].size > 0 \
                    or sp['p3_stars','age'].size > 1:
                # theres prior star formation; skip this one
                continue


            dfinal = dnum + model_time * 5
            pidx = int(ad['new_p3_stars','particle_index'][j])
        
            profile_stat[pidx] = {}
            profile_stat[pidx]['time'] = [] # time of measurement
            profile_stat[pidx]['p3_metallicity_radius'] = [] # calculated radius of enrichment zone
            profile_stat[pidx]['temperature_radius'] = [] # radius of hot zone
            
            profile_stat[pidx]['p3_star_masses'] = []
            profile_stat[pidx]['p3_star_ctime'] = []
            profile_stat[pidx]['p2_star_masses'] = []
            profile_stat[pidx]['p2_star_ctime'] = []
            profile_stat[pidx]['p3_bh_mass'] = []
            profile_stat[pidx]['snr_mass'] = []
            # if b['gas','p3_metallicity'].max() > 1e-5:
            #     continue # for now, just want to analyze pristine regions.
            print('iterating %d formed in RD%04d...'%(pidx, dnum))
            for dd in range(dnum, dfinal+1):
                dsfile = dspath + "/RD%04d/RD%04d"%(dd,dd)
                # print('Creating profiles for %s (%d/%d)'%(dsfile, dd, dfinal))
                if os.path.exists(dsfile):
                    ds = yt.load(dsfile)
                    ds = add_particle_filters(ds)
                    if dd == dnum:
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
                    profile_stat[pidx]['p3_star_ctime'].append(np.array(sp['all_p3','creation_time'].to('Myr')).tolist())
                    profile_stat[pidx]['p2_star_ctime'].append(np.array(sp['p2_stars','creation_time'].to('Myr')).tolist())
                    profile_stat[pidx]['p2_star_masses'].append(np.array(sp['p2_stars','particle_mass'].to('Msun')).tolist())
                    profile_stat[pidx]['p3_star_masses'].append(np.array(sp['p3_stars','particle_mass'].to('Msun')).tolist())
                    for k in ['snr','p3_bh']:
                        profile_stat[pidx]['%s_mass'%k].append(np.array(sp[k, 'particle_mass'].to('Msun')).tolist())
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
            os.makedirs('radial_plots/%s/plots'%sim, exist_ok=True)
            p1.save('radial_plots/%s/plots/%d'%(simname, pidx))

            p2 = yt.ProfilePlot.from_profiles(t_profiles, labels=labels)
            p2.set_unit('radius','kpccm')
            p2.save('radial_plots/%s/plots/%d'%(simname, pidx))
            with open('radial_plots/%s/%02d_size-regionlog.json'%(simname, rank), 'w') as f:
                json.dump(profile_stat, f, indent = 4)
        
with open('radial_plots/%s/%02d_size-regionlog.json'%(simname, rank), 'w') as f:
            json.dump(profile_stat, f, indent = 4)
        

        


