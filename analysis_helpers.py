"""
    Aggregation of commonly used functions in this analysis
"""

import yt
import mysql.connector
from mysql.connector import Error


def _sum_metallicity(field, data):
    return (data['Metal_Density'] + data['SN_Colour']).to('g/cm**3')/data['Density'].to('g/cm**3') / 0.01295
# yt.add_field(('gas','sum_metallicity'), function=_sum_metallicity, units = 'Zsun', sampling_type='cell')

def _p3_metallicity(field, data):
    return (data['SN_Colour'] / data['Density']) / 0.01295
# yt.add_field(('gas','p3_metallicity'), function=_p3_metallicity, units='Zsun', sampling_type='cell')


def _p3_stars(pfilter,data): # active Pop 3 stars
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') > 1))
yt.add_particle_filter('p3_stars',function=_p3_stars, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _new_p3_stars(pfilter, data):
    return data['p3_stars','age'].to('Myr') < 0.2
yt.add_particle_filter('new_p3_stars',function=_new_p3_stars, \
        requires=['age'], filtered_type='p3_stars')



def _all_p3(pfilter, data): # all p3 particles, past and present
    # return all present and SNr | blackhole collapses.
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0))\
        | ((data['all','particle_type'] == 1)\
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') > 1))
yt.add_particle_filter('all_p3',function=_all_p3, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _snr(pfilter, data): # supernovae remnants
    return ((data['all','particle_type'] == 5) \
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') < 1)\
        & (data['all','particle_mass'].to('Msun') * 1e20 < 300))
yt.add_particle_filter('snr',function=_snr, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _bh(pfilter, data):
    return (data['all','particle_type'] == 1)\
        & (data['all','creation_time'] > 0)\
        & (data['all','particle_mass'].to('Msun') > 1)
yt.add_particle_filter('p3_bh',function=_bh, \
        requires=['particle_type','particle_mass','creation_time'], filtered_type='all')

def _p2(pfilter, data):
    return (data['all','particle_type'] == 7) & (data['all','creation_time'] > 0)
yt.add_particle_filter('p2_stars',function=_p2, requires=['particle_type', 'creation_time'])

def _new_p2(pfilter, data):
    return (data['p2_stars','age'].to('Myr') < 1)
yt.add_particle_filter('new_p2_stars',function=_new_p2, requires=['age'], filtered_type='p2_stars')


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
    if type(zs) == float:
        zs = [zs]
    for z in zs:
        out=None
        diff = 1e10
        for zd in outputlist:
            if z - zd > 0 and z - zd < diff:
                diff = z-zd
                out = outputlist[zd]
        ret_outs.append(out)
    return ret_outs

def add_particle_filters(ds):
    for filter in ['new_p2_stars',
                    'p2_stars',
                    'snr',
                    'p3_stars', 
                    'all_p3', 
                    'new_p3_stars',
                    'p3_bh']:
        ds.add_particle_filter(filter)
    return ds

def db_connect(db_name = None):
    mydb = mysql.connector.connect(
        host="localhost",
        user="Azton",
        password="DDF0rmfr33d0m",
        database=db_name
        )
    return mydb

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")