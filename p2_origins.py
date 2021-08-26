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
import matplotlib
matplotlib.use("Agg")

import yt, os, json
# from trident import LightRay
import numpy as np
from mpi4py import MPI
from argparse import ArgumentParser as ap

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
        & (data['all','particle_mass'].to('Msun') < 1)
        & (data['all','particle_mass'].to('Msun')*1e20 < 300))
yt.add_particle_filter('stars',function=_stars, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])
def _new_p2(pfilter, data):
    return (data['p2_stars','age'].to('Myr') < 0.2)
yt.add_particle_filter('new_p2_stars',function=_new_p2, requires=['age'], filtered_type='p2_stars')
def main():
    print("STARTING RUN")
    # set up communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
 
 
    argparser = ap()
    argparser.add_argument('--sim', type=str, default=None, 
                        help="simulation name")
    argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                        help="file path to simulation directory")
    argparser.add_argument('--output_dest','-od', type=str, default='./p2_origins',
                        help='Destination for analysis logs and other output.')
    argparser.add_argument('--outputs', type=int, nargs='+', default=None,
                        help="white-space separated list of dumps to analyze")
    args = argparser.parse_args()

    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest, exist_ok = True)
    # paths and destinations
    
    sim = args.sim
    final_out = args.outputs[1]
    init_out = args.outputs[0]
    simpath = args.sim_root
    img_dest = args.output_dest
    data_dest = img_dest+'/%s/rays'%sim
    img_dest += '/%s/img'%sim
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)
    if rank == 0:
        print('Iterating RD%04d to RD%04d with %d ranks'%(init_out, final_out, size))


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
    
    if os.path.exists('%s/%s/%03d_p2_origin_qtys.json'%(args.output_dest, args.sim, rank)):
        with open('%s/%s/%03d_p2_origin_qtys.json'%(args.output_dest, args.sim, rank), 'r') as f:
            enr_relations = json.load(f)
    else:
        enr_relations = {}

        enr_relations['enrichee_pidx']      = []
        enr_relations['enricher_pidx']      = []
        enr_relations['enricher_mass']      = {} # list of enriching masses keyed by the pid they enriched
        enr_relations['enrichee_metal']     = []
        enr_relations['formation_dt']       = {}
        enr_relations['sne_dt']             = {}
        enr_relations['enricher_mean_z']    = {}
        enr_relations['enricher_min_z']     = {}
        enr_relations['enricher_max_z']     = {}
        enr_relations['distance']           = {}
        
        # log for pop3 stars: how many p2-stars enrichment events are they causally connected to?
        enr_relations['p3_stats'] = {}


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
            try:
                for idx, p2c in enumerate(ad['new_p2_stars','particle_position'].to('unitary')):
                    pid = int(ad['new_p2_stars','particle_index'][idx])
                    ctime = ad['new_p2_stars','creation_time'][idx].to('Myr')
                    p2r = ds.quan(500, 'pc').to('unitary')
                    sp = ds.sphere(p2c, p2r) # sphere large enough to project a region with width "d"
                    if ((sp['enzo','Metal_Density']/sp['enzo','Density']*sp['gas','cell_volume']).sum()/(sp['gas','cell_volume'].sum())).to('Zsun') <= 5e-5: # only analyze if region seems to not have ongoing prior p2 star formation
                        print('Processing particle %d with age %f Myr in RD%04d'%(pid, ad['new_p2_stars','age'][idx].to('Myr'), d))
                        r = ds.quan(200, 'kpccm').to('unitary')
                        sp = ds.sphere(p2c, r)
                        p3 = np.array(sp['stars','particle_index'])
                        enr_relations['enricher_mass'][int(pid)] = []
                        enr_relations['distance'][int(pid)] = []
                        enr_relations['formation_dt'][int(pid)]        = []
                        enr_relations['sne_dt'][int(pid)]              = []
                        enr_relations['enricher_mean_z'][int(pid)]     = []
                        enr_relations['enricher_min_z'][int(pid)]      = []
                        enr_relations['enricher_max_z'][int(pid)]      = []

                        r = ds.quan(1e-8, 'unitary')
                        for j, p3id in enumerate(p3):
                            p3pos = sp['stars','particle_position'][j].to('unitary')
                            p3SN = (sp['stars','creation_time'][j] + sp['stars','dynamical_time'][j]).to('Myr')
                            rp3p2 = np.sqrt(((p2c-p3pos)**2).sum())
                            
                            ray_start = p2c
                            ray_end = p3pos
                            ray =ds.r[ray_start:ray_end]   

                            '''
                                # want continuous metal between p2 and p3, 
                                # but dont want continous pop 2 metals 
                                # (indicating the region has been enriched 
                                # and this isnt a first-formation event)
                            '''
                            if np.all(ray['gas','p3_metallicity'] > 5e-5) and p3SN < ctime:
                                if rp3p2 > r:
                                    r = rp3p2
                                enr_relations['enrichee_pidx'].append(int(pid))
                                enr_relations['enricher_pidx'].append(int(p3id))
                                enr_relations['enricher_mass'][int(pid)].append(float(sp['stars','particle_mass'][j].to('Msun'))*1e20)
                                enr_relations['enrichee_metal'].append(float(ad['new_p2_stars','metallicity_fraction'][idx].to('Zsun')))
                                enr_relations['formation_dt'][int(pid)].append(float((ctime-sp['stars','creation_time'][j]).to("Myr")))
                                enr_relations['sne_dt'][int(pid)].append(float(ctime-p3SN))
                                enr_relations['enricher_mean_z'][int(pid)].append(float(ray['p3_metallicity'].mean()))
                                enr_relations['enricher_min_z'][int(pid)].append(float(ray['p3_metallicity'].min()))
                                enr_relations['enricher_max_z'][int(pid)].append(float(ray['p3_metallicity'].max()))
                                enr_relations['distance'][int(pid)].append(float(rp3p2.to('pc')))

                                if p3id in enr_relations['p3_stats'].keys() and r: # add to this p3 star stats
                                    enr_relations['p3_stats'][p3id]['n_enriched'] += 1
                                    enr_relations['p3_stats'][p3id]['r_enriched'] += [float(rp3p2.to('pc'))]
                                    enr_relations['p3_stats'][p3id]['m_enriched'] += float(ad['new_p2_stars', 'particle_mass'][idx].to('Msun'))
                                    enr_relations['p3_stats'][p3id]['dt_snr'] += [float((ctime-sp['stars','creation_time'][j]).to("Myr"))]
                                elif r: # create a slot for this p3 star
                                    enr_relations['p3_stats'][p3id] = {}
                                    enr_relations['p3_stats'][p3id]['m_enriched'] = \
                                                    float(ad['new_p2_stars', 'particle_mass'][idx].to('Msun'))
                                    enr_relations['p3_stats'][p3id]['mass'] =  float(sp['stars','particle_mass'][j].to('Msun'))*1e20
                                    enr_relations['p3_stats'][p3id]['n_enriched'] = 1
                                    enr_relations['p3_stats'][p3id]['r_enriched'] = [float(rp3p2.to('pc'))]
                                    enr_relations['p3_stats'][p3id]['dt_snr'] = [float((ctime-sp['stars','creation_time'][j]).to("Myr"))]
                        if not r:
                            continue    
                        with open('%s/%s/%03d_p2_origin_qtys.json'%(args.output_dest, args.sim, rank),'w') as f:
                            json.dump(enr_relations, f, indent=4)
                else:
                    print('Region metallicity too high: %s'%dspath)
                with open(comp_output_log, 'a') as f:
                    f.write('%d\n'%d)
            except OSError as oe:
                print('[%d] Could not load/access fields "new_p2_stars" in RD%04d'%(rank,d))
                print(oe)
        else:
            print('%s does not exist?'%dspath)

if __name__ == '__main__':
    main()