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

import yt, os, sys, trident, h5py, json
import numpy as np
from mpi4py import MPI


def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.02
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.02
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun')


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

def _stars(pfilter,data): # only returns SNe stars 
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') < 1))
yt.add_particle_filter('stars',function=_stars, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])

def main():

    # set up communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # paths and destinations
    sim = sys.argv[1]
    final_out = int(sys.argv[2])
    first_out = 200 # first saved sim output
    skip = 5 # number of outputs to skip between projections

    simpath = '/mnt/d/starnet/simulation_data'
    rspath = '/mnt/d/starnet/simulation_data/phoenix_256_IC1/rockstar_halos'
    img_dest = '/mnt/d/starnet/simulation_data/phoenix_analysis/'
    data_dest = img_dest+'%s/p2_origins/rays'%sim
    img_dest += '%s/p2_origins/img'%sim
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)

    ds = yt.load('%s/%s/RD%04d/RD%04d'%(simpath, sim, final_out, final_out))
    ds.add_particle_filter('p2_stars')
    ad = ds.all_data()

    form_times = ad['p2_stars','creation_time']
    form_pids = ad['p2_stars', 'particle_index']
    
    #sort chronologically
    sort_times = np.argsort(form_times)
    form_pids = form_pids[sort_times]
    form_times = form_times[sort_times]

    form_outs = np.array(find_correct_output(ds,form_times)) + 1 # actually returns dump before formation
    logged_p2 = []

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


    if rank == 0:
        print('Found %d p2 stars to log!'%(len(form_outs)))
        print(form_outs)
    for i, d in enumerate(form_outs):
        dspath = '%s/%s/RD%04d/RD%04d'%(simpath, sim, d, d)
        if os.path.exists(dspath): # local ds's are multiples of 5
            ds = yt.load(dspath)
            ds.add_particle_filter('p2_stars')
            ds.add_particle_filter('stars')
            ad = ds.all_data()
            idx = np.where(ad['p2_stars','particle_index'] == form_pids[i])[0][0]

            p2c = ad['p2_stars','particle_position'][idx]
            ctime = ad['p2_stars','creation_time'][idx].to('Myr')
            p2r = ds.quan(200, 'pc').to('unitary')
            sp = ds.sphere(p2c, p2r) # sphere large enough to project a region with width "d"
            if (sp['gas','metallicity']*sp['cell_volume']).sum()/(sp['cell_volume'].sum()) <= 1e-5: # only analyze if region seems to not have ongoing prior p2 star formation
                print('Processing particle %d with age %f Myr in RD%04d'%(form_pids[i], ad['p2_stars','age'][idx].to('Myr'), d))
                r = ds.quan(150, 'kpccm').to('unitary')
                sp = ds.sphere(p2c, r)
                for p2 in sp['p2_stars','particle_index']:
                    if p2 in logged_p2:
                        continue
                logged_p2.append(form_pids[i])
                p3 = np.array(sp['stars','particle_index'])
                enr_relations['enricher_mass'][int(form_pids[i])] = []
                enr_relations['distance'][int(form_pids[i])] = []

                prj = yt.ProjectionPlot(ds,'z', ['density', 'p3_metallicity'],weight_field='density',
                                        center = p2c, width = r)
                prj.annotate_marker(p2c, marker='*',plot_args={'color':'tab:blue'})
                print('Annotating %d p3 stars in volume for %d'%(len(p3), form_pids[i]))
                r = 0
                for j, p3id in enumerate(p3):
                    p3pos = sp['stars','particle_position'][j].to('unitary')
                    p3SN = (sp['stars','creation_time'][j] + sp['stars','dynamical_time'][j]).to('Myr')
                    rp3p2 = np.sqrt(((p2c-p3pos)**2).sum())
                    if rp3p2 > r:
                        r = rp3p2
                    ray_start = p2c
                    ray_end = p3pos
                    prj.annotate_marker(p3pos, marker='*', plot_args={'color':'tab:green'})
                    rayfile = "%s/%d_%d.h5"%(data_dest, form_pids[i], p3id)
                    ray =trident.make_simple_ray(ds,
                                start_position=ray_start,
                                end_position=ray_end,
                                fields=['temperature','density','metallicity','sum_metallicity','p3_metallicity'],
                                data_filename=rayfile,
                                ftype='gas')   
                    f = h5py.File(rayfile, 'r')
                    # print('Ray File keys:')
                    # for k in f['grid']:
                    #     print('\t', k)   

                    '''
                        # want continuous metal between p2 and p3, 
                        # but dont want continous pop 2 metals 
                        # (indicating the region has been enriched 
                        # and this isnt a first-formation event)
                    '''
                    if np.all(f['grid']['p3_metallicity'][:] > 1e-5) :
                        prj.annotate_ray(ray, arrow=True, plot_args={'color':'tab:orange'})
                        prj.annotate_text(p3pos, "%0.1f; %0.1f"%(ctime - sp['stars','creation_time'][j].to('Myr'), ctime - p3SN))
                        # log enrichers info
                        enr_relations['enrichee_pidx'].append(int(form_pids[i]))
                        enr_relations['enricher_pidx'].append(int(p3id))
                        enr_relations['enricher_mass'][int(form_pids[i])].append(float(sp['stars','particle_mass'][j].to('Msun'))*1e20)
                        enr_relations['enrichee_metal'].append(float(ad['p2_stars','metallicity_fraction'][idx]))
                        enr_relations['formation_dt'].append(float((ctime-sp['stars','creation_time'][j]).to("Myr")))
                        enr_relations['sne_dt'].append(float(ctime-p3SN))
                        enr_relations['enricher_mean_z'].append(float(f['grid']['p3_metallicity'][:].mean()))
                        enr_relations['enricher_min_z'].append(float(f['grid']['p3_metallicity'][:].min()))
                        enr_relations['enricher_max_z'].append(float(f['grid']['p3_metallicity'][:].max()))
                        enr_relations['distance'][int(form_pids[i])].append(float(r.to('pc')))

                        if p3id in enr_relations['p3_stats'].keys(): # add to this p3 star stats
                            enr_relations['p3_stats'][p3id]['n_enriched'] += 1
                            enr_relations['p3_stats'][p3id]['r_enriched'] += [float(r.to('pc'))]
                            enr_relations['p3_stats'][p3id]['m_enriched'] += float(ad['p2_stars', 'particle_mass'][idx].to('Msun'))
                        else: # create a slot for this p3 star
                            enr_relations['p3_stats'][p3id] = {}
                            enr_relations['p3_stats'][p3id]['m_enriched'] = float(ad['p2_stars', 'particle_mass'][idx].to('Msun'))
                            enr_relations['p3_stats'][p3id]['mass'] =  float(sp['stars','particle_mass'][j].to('Msun'))*1e20
                            enr_relations['p3_stats'][p3id]['n_enriched'] = 1
                            enr_relations['p3_stats'][p3id]['r_enriched'] = [float(r.to('pc'))]
                    else:
                        prj.annotate_ray(ray, arrow=True, plot_args={'color':'white'})
                        
                prj.set_width(4*rp3p2)
                prj.set_antialias(True)
                prj.set_figure_size(16)
                prj.annotate_title('$R_{max}$ = %0.2f pc '%(r.to('pc')))
                prj.set_axes_unit('kpc')
                prj.save('%s/%d'%(img_dest, form_pids[i]), suffix='png')
                # exit()
                # write the file after each star 
                with open('%s/p2_origin_qtys.json'%sim,'w') as f:
                    json.dump(enr_relations, f, indent=4)


if __name__ == '__main__':
    main()