'''
    Through time, plot the enriched volume where enriched is
    z >= z_critical (for p2 formation)

    Can also plot: 
        <z> for z > z_critical
        <z> for rho > 100 rho/<rho>
            same as function of N-remnants within volume
'''

import yt, os, json
import numpy as np
from argparse import ArgumentParser as ap
from mpi4py import MPI

from analysis_helpers import *


argparser = ap()

argparser.add_argument('--sim', type=str, default=None, 
                    help="simulation name")
argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                    help="file path to simulation directory")
argparser.add_argument('--output_dest','-od', type=str, default='./enriched_stats',
                    help='Destination for analysis logs and other output.')
argparser.add_argument('--outputs', type=int, nargs='+', default=None,
                    help="white-space separated list of dumps to analyze")
args = argparser.parse_args()

if not os.path.exists(args.output_dest):
    os.makedirs(args.output_dest, exist_ok = True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

localinds = np.arange(rank, len(args.outputs), size)

outputs = [args.outputs[i] for i in localinds]
print('\n%d processing %d outputs for enriched statistics:\n'%(rank, len(outputs)), outputs)
volstat = {}
volstat['f_enr'] = []
volstat['p3_f_enr'] = []
volstat['f_enr_od'] = {}
ods = np.linspace(10, 1000, 4)
for od in ods:
    volstat['f_enr_od'][od] = []
volstat['f_enr_high'] = {}
mets = np.logspace(np.log10(5.5e-8), -2, 10)
for zc in mets:
    volstat['f_enr_high'][zc] = []
volstat['ion_frac'] = {}
ions = np.linspace(0.1, 0.999, 10)
for i_f in ions:
    volstat['ion_frac'][i_f] = []
volstat['t'] = []
volstat['z'] = []

z_crit = 5.5e-5
z_high = 1e-3

for d in outputs:
    dsfile = '%s/%s/RD%04d/RD%04d'%(args.sim_root, args.sim, d, d)
    if os.path.exists(dsfile):
        ds = yt.load(dsfile)
        z = ds.current_redshift
        vtot = ds.quan((ds.parameters['CosmologyComovingBoxSize'] / (1.0+z))**3, 'Mpc**3').to('pc**3')
        ad = ds.all_data()
        
        # tracking qtys
        fenr = ad['gas','cell_volume'][ad['sum_metallicity'] > z_crit].sum().to('pc**3') / vtot
        p3_fenr = ad['gas','cell_volume'][ad['p3_metallicity'] > z_crit].sum().to('pc**3') / vtot
        for k in volstat['f_enr_od']:
            vhigh = ad['gas','cell_volume'][ad['gas','overdensity'] > k].sum().to('pc**3')
            fenrod = ad['gas','cell_volume'][(ad['sum_metallicity'] > z_crit) & (ad['overdensity'] > k)].sum().to('pc**3') / vhigh
            volstat['f_enr_od'][k].append(float(fenrod))
        for k in volstat['f_enr_high']:
            fenrhigh = ad['gas','cell_volume'][ad['sum_metallicity'] > 10**k].sum().to('pc**3') / vtot
            volstat['f_enr_high'][k].append(float(fenrhigh))
        for k in volstat['ion_frac']:
            ion_frac = ad['H_p1_density'] / (ad['H_p0_density'] + ad['H_p1_density'])
            ionvol = ad['gas','cell_volume'][ion_frac < k].sum().to('pc**3') / vtot
            volstat['ion_frac'][k].append(float(ionvol))
        # log
        volstat['p3_f_enr'].append(float(p3_fenr))
        volstat['f_enr'].append(float(fenr))
        volstat['t'].append(float(ds.current_time.to('Myr')))
        volstat['z'].append(float(z))

    with open('%s/%d_%s.json'%(args.output_dest, rank, args.sim), 'w') as f:
        json.dump(volstat, f, indent=4)