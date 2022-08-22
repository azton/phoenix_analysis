"""
    we have to assume there is some maximal range of influence of a P3 region
        It may depend on z and time after region starts
        describes the maximal radii to expect metals, ionization influence.
"""
import matplotlib
matplotlib.use('Agg')
import yt,os,json, glob, copy
import numpy as np

from analysis_helpers import *
from mpi4py import MPI
from argparse import ArgumentParser as ap
yt.set_log_level(40)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plot_exemp = False # turn off to save on written files after debugging
plot_proj = False

def _sum_metallicity(field, data):
    return ((data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3')).to('Zsun')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']).to("Zsun")

def add_fields(ds):
    ds.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')
    ds.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units = 'Zsun', sampling_type='cell')
    return ds

def main():
    argparser = ap()
    argparser.add_argument('--sim', type=str, default=None, 
                        help="simulation name")
    argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                        help="file path to simulation directory")
    argparser.add_argument('--output_dest','-od', type=str, default='./size_of_halobase',
                        help='Destination for analysis logs and other output.')
    argparser.add_argument('--output_skip', type=float, default = 1.0,
                        help="how many outputs to skip between region snapshots")
    argparser.add_argument('--model_time', type=float, default=30.0,
                        help="duration to observe starting at first P3 formation.")
    argparser.add_argument('--final_output', type=int, default = None)
    args = argparser.parse_args()

    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest, exist_ok = True)
    logpath = '%s/%s/%03d_size-regionlog.json'%(args.output_dest, args.sim, rank)
    if not os.path.exists(os.path.split(logpath)[0]):
        os.makedirs(os.path.split(logpath)[0], exist_ok=True)
    dspath = '%s/%s'%(args.sim_root, args.sim)
    if rank == 0:
        print('Looking for outputs in %s'%dspath)
    alldnums = np.arange(200, args.final_output)
    local_inds = np.arange(rank, len(alldnums), size, dtype=int)
    localdnums = alldnums[local_inds]
    localdspaths = ['%s/%s/RD%04d/RD%04d'%(args.sim_root, args.sim, i, i) for i in alldnums[local_inds]]
    profile_fields = ['p3_metallicity','H_p1_fraction', 'sum_metallicity']
    fsmall = [3.1e-6, 0.05, 3.1e-6]

    if os.path.exists(logpath):
        with open(logpath, 'r') as f:
            profile_stat = json.load(f)
    else:
        profile_stat = {}



    n = 0
    # print('[%d] has:'%rank, localdspaths)

    # radii to use later, in kpc
    r_step = 0.1 # step to find metal radius
    r_big = 12
    r_small = 0.1
    done_p3ids = []
    for i, dnum in enumerate(localdnums):

        print('[%d] working RD%04d'%(rank, dnum))
        outpath = '%s/%s/RD%04d/RD%04d'%(args.sim_root, args.sim, dnum, dnum)
        ds = yt.load(outpath)
        ds = add_particle_filters(ds)
        ds = add_fields(ds)
        if args.final_output > 0 and dnum > args.final_output:
            continue
        try:
            rsds = yt.load('%s/%s/rockstar_halos/halos_RD%04d.0.bin'%(args.sim_root, args.sim, dnum))
            rsad = rsds.all_data()
        except:
            continue
        print('[%d] checking %d halos...'%(rank, len(rsad['halos','particle_identifier'])))
        halo_written = False # have we logged stars in this halo?
        for ii, r in enumerate(rsad['halos','virial_radius'].to('unitary')):
            halo_written = False
            hcenter = [rsad['halos','particle_position_%s'%a][ii].to('unitary') for a in 'xyz']
            # r = ds.quan(5*r, 'kpc')
            hsp = ds.sphere(hcenter, 2*r)


            if hsp['p2_stars','age'].size > 0 \
                or hsp['snr','age'].size > 0 \
                or hsp['p3_bh', 'age'].size > 0 \
                or np.any(hsp['p3_stars','age'].to('Myr') > 0.2):
                # theres prior star formation; skip this one
                # print("[%d] Theres prior stars in halo %d!"%(rank, ii))
                continue
            if hsp['new_p3_stars','age'].size > 0:
                

                print('[%d] Found stars! iterating %d new p3 stars...'%(rank,  hsp['new_p3_stars','age'].size))
                for j, c in enumerate(hsp['new_p3_stars','particle_position'].to('unitary')):
                    
                    if plot_exemp:
                        z_profiles = []
                        t_profiles = []
                        z_labels = []
                        t_labels = []
                    # r = ds.quan(r_plot, 'kpc')
                    # sp = ds.sphere(c, r)



                    dfinal = dnum + args.model_time * 5
                    pidx = int(hsp['new_p3_stars','particle_index'][j])
                    # print('[%d] working on %d'%(rank,pidx))
                    # if weve logged the profile before
                    pid_path = '%s/%s/profiles/%d_profiles.png'%(args.output_dest, args.sim, pidx)
                    # if os.path.exists(pid_path):
                    #     print("SKIPPING %d (ALREADY ACCOMPLISHED)"%pidx)
                    #     continue

                    # print('Working PID %d in RD%04d'%(int(pidx), dnum))
                    profile_stat[pidx] = {}
                    profile_stat[pidx]['region_start_time'] = float(ds.current_time.to('Myr'))
                    profile_stat[pidx]['box_size'] = float(ds.parameters['CosmologyComovingBoxSize'])
                    profile_stat[pidx]['region_center'] = []
                    profile_stat[pidx]['region_unitary'] = []
                    profile_stat[pidx]['time'] = [] # time of measurement
                    profile_stat[pidx]['redshift'] = [] # time of measurement
                    for field in profile_fields:
                        profile_stat[pidx]['%s_radius'%field] = [] # calculated radius of enrichment zone
                    for field in profile_fields:
                        profile_stat[pidx]['%s_radius_u'%field] = [] # calculated radius of enrichment zone
                    profile_stat[pidx]['p3_all_ctime'] = []
                    profile_stat[pidx]['p3_all_position'] = []
                    profile_stat[pidx]['p3_star_position'] = []
                    profile_stat[pidx]['p3_all_mass'] = []
                    profile_stat[pidx]['p3_all_idx'] = []
                    profile_stat[pidx]['snr_ctime'] = []
                    profile_stat[pidx]['p3_bh_ctime'] = []
                    profile_stat[pidx]['p3_live_masses'] = []
                    profile_stat[pidx]['p3_live_ctime'] = []
                    profile_stat[pidx]['p2_star_masses'] = []
                    profile_stat[pidx]['p2_star_ctime'] = []
                    profile_stat[pidx]['p2_star_position'] = []
                    profile_stat[pidx]['p3_bh_mass'] = []
                    profile_stat[pidx]['snr_mass'] = []

                    profile_stat[pidx]['enriched_stat'] = {}
                    profile_stat[pidx]['enriched_stat']['std'] = []
                    profile_stat[pidx]['enriched_stat']['mu'] = [] 

                    # if b['gas','p3_metallicity'].max() > 1e-5:
                    #     continue # for now, just want to analyze pristine regions.
                    print('iterating %d formed in RD%04d...'%(pidx, dnum))
                    # num = int(dnum[2:])
                    out_dumps = np.linspace(dnum, dfinal, int(args.model_time * 5.0 / args.output_skip), dtype=int)
                    for ddd, d in enumerate(out_dumps):
                        # only have every 10th dump at home ><

                        if d % 10 != 0 and '/mnt' in args.sim_root:
                            out_dumps[ddd] -= d % 10
                    print('[%d] %d iterates outputs:'%(rank, pidx), out_dumps)
                    for dcnt, dd in enumerate(out_dumps):
                        dsfile = dspath + "/RD%04d/RD%04d"%(dd,dd)
                        # print('Creating profiles for %s (%d/%d)'%(dsfile, dd, dfinal))
                        if os.path.exists(dsfile):
                            try:
                                dsn = yt.load(dsfile)
                                dsn = add_particle_filters(dsn)
                                dsn = add_fields(dsn)

                            except:
                                print('Could not load %s'%dsfile)
                                continue
                            
                            # use big sphere to find particle in this output
                            # all_data() takes way too long on, i.e., phx512
                            r = dsn.quan(1.5*r_big, 'kpc')
                            sp = dsn.sphere(c, r)

                            try:
                                idx = np.where(sp['all','particle_index'] == pidx)[0][0]
                            except IndexError as ie:
                                print("IndexError raised on rank %d for %s: pidx %d not found\n"%(rank, dsfile, pidx), ie)
                                continue

                            # use small sphere centered on the particle we found
                            # to ID the radius of metal system by falloff where
                            # mean of metal within falls to 1e-5   
                            c = sp['all','particle_position'][idx]
                            r_eval = -1
                            
                            for field, minval in zip(profile_fields, fsmall):
                            
                                r = ds.quan(r_small, 'kpc')
                                sp = dsn.sphere(c, r)

                                fval = sp.quantities.weighted_average_quantity(('gas',field), ('gas','cell_volume'))
                                fcenter = fval
                                fmax = 0
                                while fval >= minval:
                                    spl = dsn.sphere(c, r + ds.quan(r_step, 'kpc'))
                                    shell = spl - sp
                                    fldmean = shell.quantities.weighted_average_quantity(('gas',field), ('gas','cell_volume'))
                                    # if z < zmean:
                                    #     zmean = float(z)
                                    if fldmean > fmax: 
                                        fmax = float(fldmean)
                                    if r > r_eval and field == 'p3_metallicity':
                                        r_eval = r
                                    if fldmean < minval:
                                        break
                                    if r.to('kpc') > r_big:
                                        break
                                    
                                    r += dsn.quan(r_step, 'kpc')
                                    sp = dsn.sphere(c, r)
                                
                                # only log stars and other things for p3_metallicity
                                if field == 'p3_metallicity':
                                    sp = dsn.sphere(c, r)

                                    profile_stat[pidx]['time'].append(float(dsn.current_time.to('Myr')))
                                    profile_stat[pidx]['redshift'].append(float(dsn.current_redshift))
                                    profile_stat[pidx]['region_center'].append(np.array(c.to('pc')).tolist())
                                    profile_stat[pidx]['region_unitary'].append(np.array(c.to('unitary')).tolist())

                                    profile_stat[pidx]['p3_all_ctime'].append(np.array(sp['all_p3','creation_time'].to('Myr')).tolist())
                                    profile_stat[pidx]['p3_all_mass'].append(np.array(sp['all_p3','particle_mass'].to('Msun')).tolist())
                                    profile_stat[pidx]['p3_all_position'].append(np.array(sp['all_p3','particle_position'].to('pc')).tolist())
                                    profile_stat[pidx]['p3_star_position'].append(np.array(sp['p3_stars','particle_position'].to('pc')).tolist())
                                    profile_stat[pidx]['p3_all_idx'].append(np.array(sp['all_p3', 'particle_index']).tolist())
                                    profile_stat[pidx]['p3_live_masses'].append(np.array(sp['p3_stars','particle_mass'].to('Msun')).tolist())
                                    profile_stat[pidx]['p3_live_ctime'].append(np.array(sp['p3_stars','creation_time'].to('Myr')).tolist())

                                    profile_stat[pidx]['p2_star_ctime'].append(np.array(sp['p2_stars','creation_time'].to('Myr')).tolist())
                                    profile_stat[pidx]['p2_star_position'].append(np.array(sp['p2_stars','particle_position'].to('pc')).tolist())
                                    profile_stat[pidx]['p2_star_masses'].append(np.array(sp['p2_stars','particle_mass'].to('Msun')).tolist())

                                    #stats of enriched gas within volume; std and mu of (log(z) | z > z_crit)
                                    enriched = True
                                    profile_stat[pidx]['enriched_stat']['std'].append(np.std(np.log10(sp['gas','p3_metallicity'])))
                                    profile_stat[pidx]['enriched_stat']['mu'].append(float(np.log10(sp['gas','p3_metallicity']).mean() ))


                                    for k in ['snr','p3_bh']:
                                        profile_stat[pidx]['%s_mass'%k].append(np.array(sp[k, 'particle_mass'].to('Msun')).tolist())
                                        profile_stat[pidx]['%s_ctime'%k].append(np.array(sp[k, 'creation_time'].to('Myr')).tolist())
                                # log radii of others
                                profile_stat[pidx]['%s_radius'%field].append(float(r.to('kpc')))
                                profile_stat[pidx]['%s_radius_u'%field].append(float(r.to('unitary')))
                                
                                n += 1

                    halo_written = True # we only want to log the first star from the halo, as
                                                # further stars are likely near duplicate regions as the first
                # # update logfile after each pid
                # print("FINISHED %d"%pidx)

                with open(logpath, 'w') as f:
                    json.dump(profile_stat, f, indent = 4)
                

        with open(logpath, 'w') as f:
                json.dump(profile_stat, f, indent = 4)
        
if __name__=='__main__': main()
        


