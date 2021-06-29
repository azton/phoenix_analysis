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

    z = ds.cosmology.z_from_t(t) # redshift at time t
    out=None
    diff = 1e10
    for zd in outputlist:
        if z - zd > 0 and z - zd < diff:
            diff = z-zd
            out = outputlist[zd]
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

def main():

    # set up communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # paths and destinations
    sim = sys.argv[1]
    final_out = int(sys.argv[2])
    simpath = '/scratch3/06429/azton/phoenix'
    rspath = '/scratch3/06429/azton/phoenix/phoenix_256_IC1/rockstar_halos'
    img_dest = '/scratch3/06429/azton/phoenix_analysis/'
    img_dest += '%s/first_star'%sim
    os.makedirs(img_dest, exist_ok=True)

    # could do this for the first star in the MMH--need halo catalog though.
    #   at least at final dump...    
    # if not os.path.exists('%s/halos_RD%04d.0.bin'%(rspath, final_out)):
    #     print('halo catalog for requested RD%04d does not exist...'%final_out)
    #     exit()

    ds = yt.load('%s/%s/RD%04d/RD%04d'%(simpath, sim, final_out, final_out))
    ds.add_particle_filter('stars')
    ad = ds.all_data()

    first_star = np.argmin(ad['stars','creation_time'].to('Myr'))
    creationT = ad['stars','creation_time'][first_star].to('Myr')
    first_out = find_correct_output(ds, creationT) # returns dump BEFORE star formed.
    creationZ = ds.cosmology.z_from_t(creationT)
    pidx = ad['stars','particle_index'][first_star]

    local = np.arange(rank + first_out + 1, final_out + 1, size)
    for d in local: # split work among ranks    
            opath = '%s/%s/RD%04d/RD%04d'%(simpath, sim, d, d)
            if os.path.exists(opath):
                ds = yt.load(opath)
                ds.add_particle_filter('stars')
                ad = ds.all_data()
                ind = np.where(ad['stars','particle_index'] == pidx)[0][0]
                
                c = ad['stars','particle_position'][ind].to('unitary')
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
                prj.annotate_title('z = %0.2f'%ds.current_redshift)
                prj.save(img_dest)
            del(prj)
            del(ds)
            del(ad)
            del(b)
if __name__ == '__main__':
    main()