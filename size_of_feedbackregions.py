"""
    we have to assume there is some maximal range of influence of a P3 region
        It may depend on z and time after region starts
        describes the maximal radii to expect metals, ionization influence.
"""
import matplotlib
matplotlib.use('Agg')
import yt,os,json, glob
import numpy as np

from analysis_helpers import *
from mpi4py import MPI
from argparse import ArgumentParser as ap
yt.set_log_level(40)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plot_exemp = True # turn off to save on written files after debugging
plot_prj = True

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
args = argparser.parse_args()

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
    with open(logpath, 'r') as f:
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
    try:
        if ad0['new_p3_stars','age'].size > 0:
            

            dnum = int(os.path.split(outpath)[-1][2:])

            for j, c in enumerate(ad0['new_p3_stars','particle_position'].to('unitary')):
                
                if plot_exemp:
                    z_profiles = []
                    t_profiles = []
                    z_labels = []
                    t_labels = []
                r = ds.quan(150, 'kpccm')
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
                pid_path = '%s/%s/profiles/%d_profiles.png'%(args.output_dest, args.sim, pidx)
                if os.path.exists(pid_path):
                    print("SKIPPING %d (ALREADY ACCOMPLISHED)"%pidx)
                    continue

                print('Working PID %d in RD%04d'%(int(pidx), dnum))
                profile_stat[pidx] = {}
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
                        except:
                            print('Could not load %s'%dsfile)
                            continue
                        # use big sphere to find particle in this output
                        # all_data() takes way too long on, i.e., phx512
                        r = dsn.quan(250, 'kpccm')
                        sp = dsn.sphere(c, r)
                        # ind = np.where(output_list_srt == d)[0][0]
                        # pidx = pidx_srt[ind]
                        try:
                            idx = np.where(sp['all','particle_index'] == pidx)[0][0]
                        except IndexError as ie:
                            print("IndexError raised on rank %d for %s:\n"%(rank, dsfile), ie)
                            continue
                    
                        r = dsn.quan(150, 'kpccm')
                        c = sp['all','particle_position'][idx]                    
                        
                        # new sphere centered on focused idx
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
                                                    weight_field=('gas','cell_volume'))
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
                        if plot_proj:
                            prjpath = '%s/%s/projections/%d'%(args.output_dest, args.sim, pidx)
                            if not os.path.exists(prjpath):
                                os.makedirs(prjpath, exist_ok=True)
                            b = ds.box(c-r, c+r)
                            prj = yt.ProjectionPlot(ds, 'z', [('gas','density'),('gas','metallicity'), ('gas','sum_metallicity')], weight_field=('gas','density'), data_source=b, center=c, width=2*r)
                            for sc in b['p3_stars','particle_position']:
                                prj.annotate_marker(sc, marker='*', plot_args={'color':'lime', 'alpha':0.5})
                            for sc in b['p2_stars', 'particle_position']:
                                prj.annotate_marker(sc, marker='P', plot_args={'color':'tab:cyan', 'alpha':0.5})
                            prj.save(prjpath)
                # exemplary plots of the pid, for checking
                if plot_exemp:
                    p1 = yt.ProfilePlot.from_profiles(z_profiles, labels=z_labels)
                    p1.set_unit('radius','kpccm') 
                    os.makedirs('%s/%s/plots'%(args.output_dest, args.sim), exist_ok=True)
                    p1.save('%s/%s/plots/%d'%(args.output_dest, args.sim, pidx))

                    p2 = yt.ProfilePlot.from_profiles(t_profiles, labels=t_labels)
                    p2.set_unit('radius','kpccm')
                    p2.save('%s/%s/plots/%d'%(args.output_dest, args.sim, pidx))
                
                profile_rec = {}
                profile_rec['pid'] = pidx
                profile_rec['times'] = []
                profile_rec['z_radii'] = []
                profile_rec['t_radii'] = []
                profile_rec['p3_metallicity'] = []
                profile_rec['temperature'] = []
                for kk, profile in enumerate(t_profiles):
                    profile_rec['times'].append(float(t_labels[kk].split(':')[-1]))
                    profile_rec['temperature'].append(np.array(profile.field_data[('gas','temperature')]).tolist())
                    profile_rec['t_radii'].append(np.array(profile.x.to('kpccm')).tolist())
                for profile in z_profiles:
                    profile_rec['p3_metallicity'].append(np.array(profile.field_data[('gas','p3_metallicity')]).tolist())
                    profile_rec['z_radii'].append(np.array(profile.x.to('kpccm')).tolist())

                if not os.path.exists('%s/%s/profiles'%(args.output_dest, args.sim)):
                    os.makedirs('%s/%s/profiles'%(args.output_dest, args.sim), exist_ok=True)
                with open('%s/%s/profiles/%d_profiles.json'%(args.output_dest, args.sim, pidx), 'w') as f:
                    json.dump(profile_rec, f, indent=4)
                
                # update logfile after each pid
                print("FINISHED %d"%pidx)

            with open(logpath, 'w') as f:
                json.dump(profile_stat, f, indent = 4)
        
    except OSError as oe:
        print("[%d] Could not load stars in %s"%(rank,outpath))
        print(oe)
    with open(logpath, 'w') as f:
            json.dump(profile_stat, f, indent = 4)
        

        


