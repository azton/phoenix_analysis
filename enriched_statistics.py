'''
    Through time, plot the enriched volume where enriched is
    z >= z_critical (for p2 formation)

    Can also plot: 
        <z> for z > z_critical
        <z> for rho > 100 rho/<rho>
            same as function of N-remnants within volume
'''

import yt, sys, os, json
import numpy as np

def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.02
yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.02
yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun')



sim = sys.argv[1]
dstart = int(sys.argv[2])
dend = int(sys.argv[3])
dskip = int(sys.argv[4])
simpath = '/mnt/d/starnet/simulation_data/'
datadest = '/mnt/c/Users/azton/Projects/phoenix_analysis/enriched_stats'
os.makedirs(datadest, exist_ok = True)


outputs = range(dstart, dend+1, dskip)
if outputs[-1] != dend: # make sure to include final state
    outputs.append(dend)

volstat = {}
volstat['f_enr'] = []
volstat['p3_f_enr'] = []
volstat['f_enr_od'] = {}
ods = np.linspace(10, 1000, 10)
for od in ods:
    volstat['f_enr_od'][od] = []
volstat['f_enr_high'] = {}
mets = np.logspace(np.log10(5.5e-8), -3, 10)
for zc in mets:
    volstat['f_enr_high'][zc] = []
volstat['ion_vol'] = {}
ions = np.linspace(0.1, 0.999, 10)
for i_f in ions:
    volstat['ion_vol'][i_f] = []
volstat['t'] = []
volstat['z'] = []

z_crit = 5.5e-5
z_high = 1e-3

for d in outputs:
    dsfile = '%s/%s/RD%04d/RD%04d'%(simpath, sim, d, d)
    if os.path.exists(dsfile):
        ds = yt.load(dsfile)
        z = ds.current_redshift
        vtot = ds.quan((ds.parameters['CosmologyComovingBoxSize'] / (1.0+z))**3, 'Mpc**3').to('pc**3')
        ad = ds.all_data()
        
        # tracking qtys
        fenr = ad['cell_volume'][ad['sum_metallicity'] > z_crit].sum().to('pc**3') / vtot
        p3_fenr = ad['cell_volume'][ad['p3_metallicity'] > z_crit].sum().to('pc**3') / vtot
        for k in volstat['f_enr_od']:
            vhigh = ad['cell_volume'][ad['gas','overdensity'] > k].sum().to('pc**3')
            fenrod = ad['cell_volume'][(ad['sum_metallicity'] > z_crit) & (ad['overdensity'] > k)].sum().to('pc**3') / vhigh
            volstat['f_enr_od'][k].append(float(fenrod))
        for k in volstat['f_enr_high']:
            fenrhigh = ad['cell_volume'][ad['sum_metallicity'] > 10**k].sum().to('pc**3') / vtot
            volstat['f_enr_high'][k].append(float(fenrhigh))
        for k in volstat['ion_vol']:
            ionvol = ad['cell_volume'][ad['H_fraction'] < k].sum().to('pc**3') / vtot
            volstat['ion_vol'][k].append(float(ionvol))
        # log
        volstat['p3_f_enr'].append(float(p3_fenr))
        volstat['f_enr'].append(float(fenr))
        volstat['t'].append(float(ds.current_time.to('Myr')))
        volstat['z'].append(float(z))

    with open('%s/%s_%04d-%04d.json'%(datadest, sim, dstart, d), 'w') as f:
        json.dump(volstat, f, indent=4)