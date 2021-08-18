"""
    we have to assume there is some maximal range of influence of a P3 region
        It may depend on z and time after region starts
        describes the maximal radii to expect metals, ionization influence.
"""

import yt,sys,os,json, glob, h5py
import numpy as np
import matplotlib.pyplot as plt
from yt.data_objects.particle_filters import add_particle_filter
from analysis_helpers import *
from mpi4py import MPI
from argparse import ArgumentParser as ap


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plot_exemp = True # turn off to save on written files after debugging


def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.01295
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.01295
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')

# model_time = 30 # Myr

argparser = ap()
argparser.add_argument('--sim', type=str, default=None, 
                    help="simulation name")
argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                    help="file path to simulation directory")
argparser.add_argument('--output_dest','-od', type=str, default='./size_of_feedback',
                    help='Destination for analysis logs and other output.')
argparser.add_argument('--output_skip', type=float, default = 1.0,
                    help="how many outputs to skip between region snapshots")
argparser.add_argument('--model_time', type=float, default=30.0,
                    help="duration to observe starting at first P3 formation.")
try:
    args = argparser.parse_args()
except:
    # print(e)
    args.print_help()
    exit()
if not os.path.exists(args.output_dest):
    os.makedirs(args.output_dest, exist_ok = True)
logpath = '%s/%s/%02d_size-regionlog.json'%(args.output_dest, args.sim, rank)
if not os.path.exists(os.path.split(logpath)[0]):
    os.makedirs(os.path.split(logpath)[0], exist_ok=True)
dspath = '%s/%s'%(args.sim_root, args.sim)
if rank == 0:
    print('Looking for outputs in %s'%dspath)
alldspaths = glob.glob('%s/RD*/RD[01][0123456789][0123456789][0123456789]'%(dspath))
# alldspaths = [alldspaths[i] for i in range(500,700, 10)] # reduced count for testing and validation
local_inds = np.arange(rank, len(alldspaths), size, dtype=int)
localdspaths = [alldspaths[i] for i in local_inds]
profile_fields = ['p3_metallicity','temperature']
if os.path.exists(logpath):
    with open(logfile, 'r') as f:
        profile_stat = json.load(f)
else:
    profile_stat = {}


logged_pids = []

n = 0
# print('[%d] has:'%rank, localdspaths)
for i, outpath in enumerate(localdspaths):


    ds = yt.load(outpath)
    ds = add_particle_filters(ds)

    ad0 = ds.all_data()

    if ad0['new_p3_stars','age'].size > 0:
        

        dnum = int(os.path.split(outpath)[-1][2:])

        for j, c in enumerate(ad0['new_p3_stars','particle_position'].to('unitary')):

            if plot_exemp:
                z_profiles = []
                t_profiles = []
                z_labels = []
                t_labels = []
            r = ds.quan(75, 'kpccm')
            sp = ds.sphere(c, r)
            if sp['p2_stars','age'].size > 0 \
                    or sp['snr','age'].size > 0 \
                    or sp['p3_bh', 'age'].size > 0 \
                    or np.any(sp['p3_stars','age'].to('Myr') > 0.2):
                # theres prior star formation; skip this one
                continue


            dfinal = dnum + args.model_time * 5
            pidx = int(ad0['new_p3_stars','particle_index'][j])
        
            # if weve logged the profile before
            pid_path = '%s/%s/plots/%d_1d-Profile_radius_temperature.png'%(args.output_dest, args.sim, pidx)
            if os.path.exists(pid_path):
                continue

            print('Working PID %d in RD%04d'%(int(pidx), dnum))
            profile_stat[str(int(pidx))] = {}
            profile_stat[pidx]['region_start_time'] = float(ds.current_time.to('Myr'))
            profile_stat[pidx]['time'] = [] # time of measurement
            profile_stat[pidx]['p3_metallicity_radius'] = [] # calculated radius of enrichment zone
            profile_stat[pidx]['temperature_radius'] = [] # radius of hot zone
            profile_stat[pidx]['p3_all_ctime'] = []
            profile_stat[pidx]['p3_all_position'] = []
            profile_stat[pidx]['p3_all_mass'] = []
            profile_stat[pidx]['p3_all_idx'] = []
            profile_stat[pidx]['snr_ctime'] = []
            profile_stat[pidx]['p3_bh_ctime'] = []
            profile_stat[pidx]['p3_live_masses'] = []
            profile_stat[pidx]['p3_live_ctime'] = []
            profile_stat[pidx]['p2_star_masses'] = []
            profile_stat[pidx]['p2_star_ctime'] = []
            profile_stat[pidx]['p3_bh_mass'] = []
            profile_stat[pidx]['snr_mass'] = []
            profile_stat[pidx]['radius'] = []
            # if b['gas','p3_metallicity'].max() > 1e-5:
            #     continue # for now, just want to analyze pristine regions.
            print('iterating %d formed in RD%04d...'%(pidx, dnum))
            out_dumps = np.linspace(dnum, dfinal, int(args.model_time * 5.0 / args.output_skip), dtype=int)
            for ddd, d in enumerate(out_dumps):
                # only have every 10th dump at home ><

                if d % 10 != 0 and '/mnt' in args.sim_root:
                    out_dumps[ddd] -= d % 10
            print('[%d] outputs:'%rank, out_dumps)
            for dd in out_dumps:
                dsfile = dspath + "/RD%04d/RD%04d"%(dd,dd)
                # print('Creating profiles for %s (%d/%d)'%(dsfile, dd, dfinal))
                if os.path.exists(dsfile):
                    try:
                        dsn = yt.load(dsfile)
                        dsn = add_particle_filters(dsn)
                        if dd == dnum:
                            time = dsn.current_time.to('Myr')

                        ad = dsn.all_data()
                    except:
                        print('Could not load %s'%dsfile)
                        continue
                    # ind = np.where(output_list_srt == d)[0][0]
                    # pidx = pidx_srt[ind]
                    try:
                        idx = np.where(ad['all','particle_index'] == pidx)[0][0]
                    except IndexError as ie:
                        print("IndexError raised on rank %d for %s:\n"%(rank, dsfile), ie)
                        continue
                    # find the radius where the metallicity falls off -- this may be smaller than the 
                    # temperature radius
                    zmean = 1e-5
                    r = dsn.quan(150, 'kpccm')
                    c = ad['all','particle_position'][idx]                    
                    # while zmean > 1e-6:
                    #     r += dsn.quan(25, 'kpccm')
                    #     try:
                    #         r = ds.quan(r, 'kpccm').to('unitary')
                    #         sp = dsn.sphere(c,r)
                    #         zmean = sp.quantities.weighted_average_quantity(('gas','p3_metallicity'), ('gas','cell_volume'))
                    #         if r >= 250: break
                    #     except:
                    #         continue
                    
                    # give a bit of padding to
                    # make sure we get the important business 
                    sp = dsn.sphere(c,r)

                    profile_stat[pidx]['time'].append(float(dsn.current_time.to('Myr')))
                    profile_stat[pidx]['radius'].append(float(r.to('kpccm')))
                    profile_stat[pidx]['p3_all_ctime'].append(np.array(sp['all_p3','creation_time'].to('Myr')).tolist())
                    profile_stat[pidx]['p3_all_mass'].append(np.array(sp['all_p3','particle_mass'].to('Msun')).tolist())
                    profile_stat[pidx]['p3_all_position'].append(np.array(sp['all_p3','particle_position'].to('pc')).tolist())
                    profile_stat[pidx]['p3_all_idx'].append(np.array(sp['all_p3', 'particle_index']).tolist())
                    profile_stat[pidx]['p2_star_ctime'].append(np.array(sp['p2_stars','creation_time'].to('Myr')).tolist())
                    profile_stat[pidx]['p2_star_masses'].append(np.array(sp['p2_stars','particle_mass'].to('Msun')).tolist())
                    profile_stat[pidx]['p3_live_masses'].append(np.array(sp['p3_stars','particle_mass'].to('Msun')).tolist())
                    profile_stat[pidx]['p3_live_ctime'].append(np.array(sp['p3_stars','creation_time'].to('Myr')).tolist())

                    for k in ['snr','p3_bh']:
                        profile_stat[pidx]['%s_mass'%k].append(np.array(sp[k, 'particle_mass'].to('Msun')).tolist())
                        profile_stat[pidx]['%s_ctime'%k].append(np.array(sp[k, 'creation_time'].to('Myr')).tolist())
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
                        z_max = z_prof[radius.to('kpccm') < 10].mean()
                        zcut = -1
                        rcut = -1
                        rfactor = 1000. #if field=='sum_metallicity' else 10.
                        if field == 'temperature' and z_max <= 1e4:
                            rfactor = 10. # for low temp regions, require less of a drop to find the edge of the region.
                        elif z_max >=1e4:
                            rfactor = 100.
                        for ii, z in enumerate(z_prof):
                                if z < z_max / rfactor:
                                    zcut = z
                                    rcut = radius[ii].to('kpccm')
                                    break
                                
                        profile_stat[pidx]['%s_radius'%field].append(float(rcut))
                        if plot_exemp:
                            if field == 'temperature':
                                t_profiles.append(prof)
                                t_labels.append('%0.2f Myr: %0.2f'%(dsn.current_time.to('Myr') - time, rcut))
                            else:
                                z_profiles.append(prof)
                                z_labels.append('%0.2f Myr: %0.2f'%(dsn.current_time.to('Myr') - time, rcut))
                    n += 1
            # exemplary plots of the pid, for checking
            if plot_exemp:
                p1 = yt.ProfilePlot.from_profiles(z_profiles, labels=z_labels)
                p1.set_unit('radius','kpccm') 
                os.makedirs('%s/%s/plots'%(args.output_dest, args.sim), exist_ok=True)
                p1.save('%s/%s/plots/%d'%(args.output_dest, args.sim, pidx))

                p2 = yt.ProfilePlot.from_profiles(t_profiles, labels=t_labels)
                p2.set_unit('radius','kpccm')
                p2.save('%s/%s/plots/%d'%(args.output_dest, args.sim, pidx))
            with h5py.File('%s/%s/%d_profiles.h5'%args.output_dest, args.sim, pidx, 'w') as f:
                times = []
                for i, profile in enumerate(z_profiles):
                    r = profile.x.to('kpccm')
                    z = profile.field_data[('gas','p3_metallicity')]
                    time = float(z_labels[i].split(':')[-1])
                    times.append(time)
                    f.create_dataset('radius_%0.2f'%time, data = r)
                    f.create_dataset('p3_metallicity_%0.2f'%time, data = z)
                f.attrs['times'] = times
                for i, profile in enumerate(t_profiles):
                    r = profile.x.to('kpccm')
                    z = profile.field_data[('gas','p3_metallicity')]
                    time = float(z_labels[i].split(':')[-1])
                    times.append(time)
                    f.create_dataset('radius_%0.2f'%time, data = r)
                    f.create_dataset('temperature_%0.2f'%time, data = z)
            # update logfile after each pid
            with open(logpath, 'w') as f:
                json.dump(profile_stat, f, indent = 4)
        
with open(logpath, 'w') as f:
            json.dump(profile_stat, f, indent = 4)
        

        


