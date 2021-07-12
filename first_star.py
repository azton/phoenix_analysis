import yt, os, sys
import numpy as np
from mpi4py import MPI



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
    out= []
    for z in zs:
        diff = 1e10
        tout = None
        for zd in outputlist:
            if z - zd > 0 and z - zd < diff:
                diff = z-zd
                tout = outputlist[zd]
        out.append(tout)
    return out

def _stars(pfilter,data):
    return ((data['all','particle_type'] == 5) \
        & (data['all','particle_mass'].to('Msun') > 1)\
        & (data['all','creation_time'] > 0)) \
        | ((data['particle_type'] == 1)\
        & (data['creation_time'] > 0)\
        & (data['particle_mass'].to('Msun') > 1))
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
    skip = 1 # number of outputs to skip between projections

    simpath = '/mnt/d/starnet/simulation_data'
    rspath = '/mnt/d/starnet/simulation_data/phoenix_256_IC1/rockstar_halos'
    img_dest = '/mnt/d/starnet/simulation_data/phoenix_analysis'
    img_dest += '/%s/first_star'%sim
    os.makedirs(img_dest, exist_ok=True)

    # could do this for the first star in the MMH--need halo catalog though.
    #   at least at final dump...    
    # if not os.path.exists('%s/halos_RD%04d.0.bin'%(rspath, final_out)):
    #     print('halo catalog for requested RD%04d does not exist...'%final_out)
    #     exit()

    ds = yt.load('%s/%s/RD%04d/RD%04d'%(simpath, sim, final_out, final_out))
    ds.add_particle_filter('stars')
    ad = ds.all_data()

    creationTs = ad['stars','creation_time'].to('Myr')
    sortinds = np.argsort(creationTs)
    first_outs = np.array(find_correct_output(ds, creationTs))[sortinds] # returns dump BEFORE star formed.
    creationZs = ds.cosmology.z_from_t(creationTs)[sortinds]
    creationTs = creationTs[sortinds]
    pidxs = ad['stars','particle_index'][sortinds]
    print(first_outs)
    # iterate the list of potential first stars to find the first one we can use
    for i, (creationT, out, pidx) in enumerate(zip(creationTs, first_outs, pidxs)):
        opath = '%s/%s/RD%04d/RD%04d'%(simpath, sim, out+1, out+1)
        if not os.path.exists(opath):
            # skip if we dont have the first dump
            print('Dont have output RD%04d for pid %d'%(out+1, pidx))
            continue
        ds = yt.load(opath)
        ds.add_particle_filter('stars')
        ad = ds.all_data()
        ind = np.where(ad['all','particle_index'] == pidx)[0][0]
        if ind:
            
            c = ad['all','particle_position'][ind].to('unitary')
            r = ds.quan(110, 'kpccm').to('unitary')

            b = ds.box(c-r/2., c+r/2.)

            if np.any(b['SN_Colour'].to('g/cm**3') / b['density'] / 0.02 > 1e-3):
                print("Have volume, but region around star is enriched in RD%04d for pid %d"%(out, pidx))
                # skip volumes with prior enrichment
                continue
        else:
            print('index not found in RD%04d'%out)
            continue

        
        # if we got to here, we can move on to making projections
        first_out = out+1
        pidx = pidx # silly

        outs = np.arange(first_out, final_out+1, skip)
        local = np.arange(rank, len(outs), size)
        for i in local: # split work among ranks   
                d = outs[i] 
                opath = '%s/%s/RD%04d/RD%04d'%(simpath, sim, d, d)
                if os.path.exists(opath):
                    ds = yt.load(opath)
                    ds.add_particle_filter('stars')
                    ds.add_particle_filter('p2_stars')
                    ad = ds.all_data()
                    ind = np.where(ad['all','particle_index'] == pidx)[0][0]
                    if ind:
                        
                        c = ad['all','particle_position'][ind].to('unitary')
                        r = ds.quan(110, 'kpccm').to('unitary')

                        b = ds.box(c-r/2., c+r/2.)

                        fields = ['density','temperature','SN_Colour', 'metallicity','H_fraction']
                        prj = yt.ProjectionPlot(ds,'z',fields, weight_field='density',center = c, width = ds.quan(100,'kpccm').to("unitary"), axes_unit='kpc')
                        prj.set_unit('SN_Colour','g/cm**3')
                        prj.set_cmap('temperature','hot')
                        prj.set_cmap('H_fraction','cividis')
                        prj.set_cmap('SN_Colour', 'GnBu')
                        prj.set_cmap('metallicity','winter')
                        prj.hide_axes()
                        prj.annotate_scale()
                        prj.annotate_title('z = %0.2f; $dt =$ %0.2f'%(ds.current_redshift, (ds.current_time - creationT).to('Myr')))
                        for ctr in b['p2_stars','particle_position']:
                            prj.annotate_marker(ctr, marker='*', plot_args={'color':'tab:purple'})
                        for ctr in b['stars','particle_position']:
                            prj.annotate_marker(ctr, marker='*', plot_args={'color':'tab:orange'})
                        
                        os.makedirs(img_dest+'/%d'%pidx, exist_ok=True)
                        prj.save(img_dest+'/%d/'%pidx)
                        del(prj)
                        del(ds)
                        del(ad)
                        del(b)
                    else:
                        print('[%d]: PIDX not found in %s!?!?!?!'%(rank, opath))
                else: 
                    print('[%d]: %s does not exist?!?!'%(rank, opath))

if __name__ == '__main__':
    main()