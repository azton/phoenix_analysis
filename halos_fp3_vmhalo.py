import matplotlib
matplotlib.use("Agg")
import yt, json, os, sys
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    try:
        sim = sys.argv[1]
    except:
        simpath = '/scratch3/06429/azton/phoenix'
        sims = ['phoenix_512'] #'phoenix_256_IC2','phoenix_256_IC1',
        # for sim in sims:
        sim = sims[rank]
    outputs = ['RD0985','RD0848','RD0731','RD0630', 'RD0542', 'RD0465', 'RD0397', 'RD0336']
    if '512' not in sim:
        outputs.insert(0, 'RD1100')
    if 'IC1' in sim:
        outputs.insert(0, 'RD1240')
    output = '%s/%s'%(simpath, sim)
    massbins = np.logspace(6,9,25)
    results = {}
    print("Starting iteration: ", outputs)
    for output in outputs:
        
        ds = yt.load('%s/%s/%s/%s'%(simpath,sim, output, output))
        rsds = yt.load("%s/%s/rockstar_halos/halos_%s.0.bin"%(simpath, sim,output))
        rsad = rsds.all_data()
        nall = np.zeros_like(massbins)
        nwp3 = np.zeros_like(massbins)
        sz = rsad['halos','virial_radius'].size
        for i in range(sz):
            c = rsad['halos','particle_position'][i].to('unitary')
            r = rsad['halos','virial_radius'][i].to('unitary')

            sp = ds.sphere(c,r)

            p3_filter = (sp['all','particle_type'] == 5) & (sp['all','particle_mass'].to('Msun') > 1)
            
            mass = (sp['gas','cell_mass'].sum() + sp['all','particle_mass'].sum()).to("Msun")
            bin = np.digitize(mass, massbins)
            if True in p3_filter:
                nwp3[bin] += 1
            nall[bin] += 1
            print("---%05d/%05d---"%(i, sz), end='\r')
        key = '%0.1f'%ds.current_redshift
        results[key] = {}
        results[key]['bins'] = massbins.tolist()
        results[key]['Nall'] = nall.tolist()
        results[key]['Nwp3'] = nwp3.tolist()

        fp = 'result_data/p3_fraction/%s.json'%(sim)
        if not os.path.exists(os.path.split(fp)[0]):
            os.makedirs(os.path.split(fp)[0])
        with open(fp, 'w') as f:
            json.dump(results, f, indent=4)
            
if __name__=='__main__': main()