
=======
'''
    Through time, plot the enriched volume where enriched is
    z >= z_critical (for p2 formation)

    Can also plot: 
        <z> for z > z_critical
        <z> for rho > 100 rho/<rho>
            same as function of N-remnants within volume
'''
import matplotlib
matplotlib.use("Agg")
import yt, os, json
import numpy as np
from argparse import ArgumentParser as ap
from mpi4py import MPI

from analysis_helpers import *

def _ion_frac(pfilter, data):
    return data['gas','H_p1_number_density'] / (data['gas','H_p0_number_density'] + data['gas','H_p1_number_density'])
yt.add_field(('gas','ionized_fraction'), function=_ion_frac, units = None, sampling_type='cell')
def _sum_metallicity(field, data):
    return ((data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3')).to("Zsun")
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']).to("Zsun")
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')


argparser = ap()

argparser.add_argument('--sim', type=str, default=None, 
                    help="simulation name")
argparser.add_argument('--sim_root', '-sr', type=str, default=None,
                    help="file path to simulation directory")
argparser.add_argument('--output_dest','-od', type=str, default='./enriched_stats',
                    help='Destination for analysis logs and other output.')
argparser.add_argument('--outputs', type=int, nargs='+', default=None,
                    help="white-space separated list of dumps to analyze")
argparser.add_argument('--output_skip', type=float, default=1,
                    help='number to skip between analyzed outputs')
args = argparser.parse_args()

if not os.path.exists(args.output_dest):
    os.makedirs(args.output_dest, exist_ok = True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
args.outputs = np.arange(min(args.outputs), max(args.outputs)+1, args.output_skip)
localinds = np.arange(rank, len(args.outputs), size)

outputs = [args.outputs[i] for i in localinds]
print('\n%d processing %d outputs for enriched statistics:\n'%(rank, len(outputs)), outputs)
volstat = {}
volstat['f_enr'] = []
volstat['p3_f_enr'] = []
volstat['f_enr_od'] = {}
ods = [1, 10, 100, 500, 700, 1000]
for od in ods:
    volstat['f_enr_od'][od] = []
volstat['f_enr_od']['<1'] = []
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
        fenr = ad['gas','cell_volume'][ad['gas','sum_metallicity'] > z_crit].sum().to('pc**3') / vtot
        p3_fenr = ad['gas','cell_volume'][ad['gas','p3_metallicity'] > z_crit].sum().to('pc**3') / vtot
        for k in volstat['f_enr_od']:
            if type(k) != str:
                vhigh = ad['gas','cell_volume'][ad['gas','overdensity'] > k].sum().to('pc**3')
                fenrod = ad['gas','cell_volume'][(ad['gas','sum_metallicity'] > z_crit) & (ad['overdensity'] > k)].sum().to('pc**3') / vhigh
                volstat['f_enr_od'][k].append(float(fenrod))
            else:
                vhigh = ad['gas','cell_volume'][ad['gas','overdensity'] < 1].sum().to('pc**3')
                fenrod = ad['gas','cell_volume'][(ad['gas','sum_metallicity'] > z_crit) & (ad['overdensity'] < 1)].sum().to('pc**3') / vhigh
                volstat['f_enr_od']['<1'].append(float(fenrod))


        for k in volstat['f_enr_high']:
            fenrhigh = ad['gas','cell_volume'][ad['gas','sum_metallicity'] > k].sum().to('pc**3') / vtot
            volstat['f_enr_high'][k].append(float(fenrhigh))
        for k in volstat['ion_frac']:
            # ion_frac = ad['H_p1_density'] / (ad['H_p0_density'] + ad['H_p1_density'])
            ionvol = ad['gas','cell_volume'][ad['gas','H_p1_fraction'] > k].sum().to('pc**3') / vtot
            volstat['ion_frac'][k].append(float(ionvol))
        # log
        volstat['p3_f_enr'].append(float(p3_fenr))
        volstat['f_enr'].append(float(fenr))
        volstat['t'].append(float(ds.current_time.to('Myr')))
        volstat['z'].append(float(z))

    with open('%s/%d_%s.json'%(args.output_dest, rank, args.sim), 'w') as f:
      json.dump(volstat, f, indent=4)