'''
    The PHX simulations have unique time resolution in outputs.  
    We can use this to study the progenitors of P2 stars, since
    we know the state surrouding each P2 formation to within 200 Kyr.

    1: Find P2 stars formation times, corresponding outputs
    2: use trident to form and store a ray from each P2 formation
        to all neighboring P3 SNe Events
    3: Rays with z > z_crit along the path will count as formation
        contributors.
    4: Form dict to save that has each contributing SNe type, its distance from
        the source, metallicity for each cell along path, duration between SN and
        formation.  Will need to filter for SNe that have continuous z-path, but werent
        actually part of the event... Hmm...
    5: Projection with P2 star, all P3 contributors plotted


'''

import yt, os, sys, json
# from trident import LightRay
import numpy as np
from mpi4py import MPI


def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.012950
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.012950
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')


def find_correct_output(ds, t):
    outputlist = {} # dict correlating outputs number to time in redshift
    f = open('OutputList.txt','r')
    for l in f:
        # sample line: CosmologyOutputRedshift[0]=30.000000
        z = float(l.split('=')[-1])
        dl = l.split('[')[-1]
        d = int(dl.split(']')[0])  
        outputlist[z] = d # associate redshift to outputnumber.
    # return output number corresponding to a given redshift

    zs = ds.cosmology.z_from_t(t) # redshift at time t
    ret_outs = []
    for z in zs:
        out=None
        diff = 1e10
        for zd in outputlist:
            if z - zd > 0 and z - zd < diff:
                diff = z-zd
                out = outputlist[zd]
        ret_outs.append(out)
    return ret_outs

def _stars(pfilter,data): # only returns SNe stars, includes P3 stars that havent "formed" yet with low mass, get rid of those. 
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') < 1))
yt.add_particle_filter('stars',function=_stars, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])
def _new_p2(pfilter, data):
    return (data['p2_stars','age'].to('Myr') < 0.2)
yt.add_particle_filter('new_p2_stars',function=_new_p2, requires=['age'], filtered_type='p2_stars')
def main():

    # set up communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # paths and destinations
    sim = sys.argv[1]
    final_out = int(sys.argv[2])
    init_out = 200 if '-1' in sim else 100
    simpath = '/expanse/lustre/scratch/azton/temp_project/phoenix'
    img_dest = '/expanse/lustre/scratch/azton/temp_project/phoenix_analysis'
    data_dest = img_dest+'/%s/p2_origins/rays'%sim
    img_dest += '/%s/p2_origins/img'%sim
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)
    if rank == 0:
        print('Iterating RD%04d to RD%04d with %d ranks'%(init_out, final_out, size))
    # ds = yt.load('%s/%s/RD%04d/RD%04d'%(simpath, sim, final_out, final_out))
    # ds.add_particle_filter('p2_stars')
    # ad = ds.all_data()

    # form_times = ad['p2_stars','creation_time']
    # form_pids = ad['p2_stars', 'particle_index']
    
    # #sort chronologically
    # sort_times = np.argsort(form_times)
    # form_pids = form_pids[sort_times]
    # form_times = form_times[sort_times]

    form_outs = np.arange(init_out, final_out) # actually returns dump before formation
    nouts = len(form_outs)/float(size)
    localouts = np.arange(int(rank*nouts), int(rank*nouts)+int(nouts))
    print("HELLO! Today rank %d/%d will be iterating through RD%04d to RD%04d, %d outputs in search of the origin of P2 stars!"\
            %(rank, size, len(form_outs)/size, localouts[0], localouts[-1]-1))
    
    # logged_p2 = []
    comp_output_log = '%s_p2origin_completed_outputs.log'%sim
    if not os.path.exists(comp_output_log) and rank == 0:
        with open(comp_output_log, 'w') as f:
            f.write('#\n')
    
    enr_relations = {}

    enr_relations['enrichee_pidx']      = []
    enr_relations['enricher_pidx']      = []
    enr_relations['enricher_mass']      = {} # list of enriching masses keyed by the pid they enriched
    enr_relations['enrichee_metal']     = []
    enr_relations['formation_dt']       = []
    enr_relations['sne_dt']             = []
    enr_relations['enricher_mean_z']    = []
    enr_relations['enricher_min_z']     = []
    enr_relations['enricher_max_z']     = []
    enr_relations['distance']           = {}
    
    # log for pop3 stars: how many p2-stars enrichment events are they causally connected to?
    enr_relations['p3_stats'] = {}


    # if rank == 0:
        # print('Found %d p2 stars to log!'%(len(form_outs)))
    # localpids = form_pids[localouts]
    print(localouts)
    for i, ld in enumerate(localouts):
        d = form_outs[ld]
        # check that we dont do regions twice (in successive runs)
        comp_outs = []
        with open(comp_output_log, 'r') as f:
            for l in f:
                if l.strip() != '#':
                    comp_outs.append(int(l))
        print("Previously completed: ", sorted(comp_outs))
        if d in comp_outs:
            print('[%d] Skipping previously done RD%04d'%(rank,d))
            continue

        dspath = '%s/%s/RD%04d/RD%04d'%(simpath, sim, d, d)
        print("Checking %s"%dspath)
        if os.path.exists(dspath): # local ds's are multiples of 5
            ds = yt.load(dspath)
            ds.add_particle_filter('p2_stars')
            ds.add_particle_filter('new_p2_stars')
            ds.add_particle_filter('stars')
            ad = ds.all_data()

            for idx, p2c in enumerate(ad['new_p2_stars','particle_position'].to('unitary')):
                pid = int(ad['new_p2_stars','particle_index'][idx])
                ctime = ad['new_p2_stars','creation_time'][idx].to('Myr')
                p2r = ds.quan(500, 'pc').to('unitary')
                sp = ds.sphere(p2c, p2r) # sphere large enough to project a region with width "d"
                if (sp['enzo','Metal_Density']/sp['enzo','Density']*sp['gas','cell_volume']).sum()/(sp['gas','cell_volume'].sum()) <= 1e-5 * 0.012950: # only analyze if region seems to not have ongoing prior p2 star formation
                    print('Processing particle %d with age %f Myr in RD%04d'%(pid, ad['new_p2_stars','age'][idx].to('Myr'), d))
                    r = ds.quan(200, 'kpccm').to('unitary')
                    sp = ds.sphere(p2c, r)
                    # we dont mind loggin multiple stars from the same reigon, so long as the z is still low
                    # for p2 in sp['p2_stars','particle_index']:
                    #     if p2 in logged_p2:
                    #         continue
                    # logged_p2.append(pid)
                    p3 = np.array(sp['stars','particle_index'])
                    enr_relations['enricher_mass'][int(pid)] = []
                    enr_relations['distance'][int(pid)] = []

                    prj = yt.ProjectionPlot(ds,'z', ['density', 'p3_metallicity'],weight_field='density',
                                            center = p2c, width = r)
                    prj.set_cmap('density','cividis')
                    prj.set_cmap('p3_metallicity','cividis')
                    prj.set_zlim('p3_metallicity',1e-10, 1)
                    prj.annotate_marker(p2c, marker='*',plot_args={'color':'cyan'})
                    print('Annotating %d p3 stars in volume for %d'%(len(p3), pid))
                    r = ds.quan(1e-8, 'unitary')
                    for j, p3id in enumerate(p3):
                        p3pos = sp['stars','particle_position'][j].to('unitary')
                        p3SN = (sp['stars','creation_time'][j] + sp['stars','dynamical_time'][j]).to('Myr')
                        rp3p2 = np.sqrt(((p2c-p3pos)**2).sum())
                        
                        ray_start = p2c
                        ray_end = p3pos
                        prj.annotate_marker(p3pos, marker='*', plot_args={'color':'lime'})
                        # rayfile = "%s/%d_%d.h5"%(data_dest, pid, p3id)
                        ray =ds.r[ray_start:ray_end]   
                        # f = h5py.File(rayfile, 'r')
                        # print('Ray File keys:')
                        # for k in f['grid']:
                        #     print('\t', k)   

                        '''
                            # want continuous metal between p2 and p3, 
                            # but dont want continous pop 2 metals 
                            # (indicating the region has been enriched 
                            # and this isnt a first-formation event)
                        '''
                        if np.all(ray['gas','p3_metallicity'] > 5e-5):
                            if rp3p2 > r:
                                r = rp3p2
                            prj.annotate_ray(ray, arrow=True, plot_args={'color':'tab:orange'})
                            prj.annotate_text(p3pos, "%0.1f; %0.1f"%(ctime - sp['stars','creation_time'][j].to('Myr'), ctime - p3SN))
                            # log enrichers info
                            enr_relations['enrichee_pidx'].append(int(pid))
                            enr_relations['enricher_pidx'].append(int(p3id))
                            enr_relations['enricher_mass'][int(pid)].append(float(sp['stars','particle_mass'][j].to('Msun'))*1e20)
                            enr_relations['enrichee_metal'].append(float(ad['new_p2_stars','metallicity_fraction'][idx].to('Zsun')))
                            enr_relations['formation_dt'].append(float((ctime-sp['stars','creation_time'][j]).to("Myr")))
                            enr_relations['sne_dt'].append(float(ctime-p3SN))
                            enr_relations['enricher_mean_z'].append(float(ray['p3_metallicity'].mean()))
                            enr_relations['enricher_min_z'].append(float(ray['p3_metallicity'].min()))
                            enr_relations['enricher_max_z'].append(float(ray['p3_metallicity'].max()))
                            enr_relations['distance'][int(pid)].append(float(rp3p2.to('pc')))

                            if p3id in enr_relations['p3_stats'].keys() and r: # add to this p3 star stats
                                enr_relations['p3_stats'][p3id]['n_enriched'] += 1
                                enr_relations['p3_stats'][p3id]['r_enriched'] += [float(rp3p2.to('pc'))]
                                enr_relations['p3_stats'][p3id]['m_enriched'] += float(ad['new_p2_stars', 'particle_mass'][idx].to('Msun'))
                            elif r: # create a slot for this p3 star
                                enr_relations['p3_stats'][p3id] = {}
                                enr_relations['p3_stats'][p3id]['m_enriched'] = \
                                                float(ad['new_p2_stars', 'particle_mass'][idx].to('Msun'))
                                enr_relations['p3_stats'][p3id]['mass'] =  float(sp['stars','particle_mass'][j].to('Msun'))*1e20
                                enr_relations['p3_stats'][p3id]['n_enriched'] = 1
                                enr_relations['p3_stats'][p3id]['r_enriched'] = [float(rp3p2.to('pc'))]
                        else:
                            prj.annotate_ray(ray, arrow=True, plot_args={'color':'white'})
                    if not r:
                        continue    
                    prj.set_width(4*rp3p2)
                    prj.set_antialias(True)
                    prj.set_figure_size(16)
                    prj.annotate_title('$R_{max}$ = %0.2f pc '%(r.to('pc')))
                    prj.set_axes_unit('kpc')
                    prj.save('%s/%d'%(img_dest, pid), suffix='png')
                    # exit()
                    # write the file after each star--one file for each rank, cuz... sigh
                    with open('%s/%03d_p2_origin_qtys.json'%(sim, rank),'w') as f:
                        json.dump(enr_relations, f, indent=4)
            else:
                print('Region metallicity too high: %s'%dspath)
            with open(comp_output_log, 'a') as f:
                f.write('%d\n'%d)
        else:
            print('%s does not exist?'%dspath)

if __name__ == '__main__':
    main()