import json, glob, os, shap
import numpy as np
from argparse import ArgumentParser as ap
from sklearn.tree import DecisionTreeRegressor

from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import torch
from  torch import Tensor as tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# the right edges of the mass bins
# subdivided in the really large bins between hn and pisn
# and in the pisn mass
massbins = [1, 11, 20, 40, 100, 140, 200, 260, 300]
# tokenized birth times too with 5 myr increments
timebins = np.arange(1, 30 + 1, 5)
split_bound = 0.5
max_radii = 5000
class MLP_Regression(nn.Module):
    def __init__(self, n_input, n_output, n_layer, nh_nodes):
        """

            Simple Multi-layer perceptron
                n_input: Number of input features
                n_output: Number of predicted quantities
                n_layer: Number of hidden layers (not including input/output)
                nh_nodes: Number of hidden nodes per layer (int)
        """
        super(MLP_Regression, self).__init__()
        print(" %d layers with %d input, %d output, %d nodes per hidden"%\
            (n_layer, n_input, n_output, nh_nodes))

        l0 = [nn.Linear(n_input, nh_nodes if n_layer > 0 else n_output)]
        if n_layer > 0:
            l0 += [nn.Sequential(nn.Linear(nh_nodes, nh_nodes), nn.ReLU(), nn.Dropout(0.5)) for n in range(n_layer)]
        if n_layer > 0:
            l0 += [nn.Linear(nh_nodes, n_output)]
        self.model = nn.Sequential(*l0)
        # self.fin_act = nn.ReLU() # to make sure the output is positive semi-definite

    def forward(self, x):
        return self.model(x)



def hit_rate(y, yhat, bound):
    yt = np.sort(y)
    tr_big = y  > yt[int(len(yt) * (bound))]
    mod_big = yhat  > yt[int(len(yt) * (bound))]
    return np.logical_and(tr_big, mod_big).sum()/ (tr_big.sum())
def fa_rate(y, yhat, bound):
    yt = np.sort(y)
    tr_big = y  <= yt[int(len(yt) * (bound))]
    mod_big = yhat  > yt[int(len(yt) * (bound))]
    return np.logical_and(tr_big, mod_big).sum() / (tr_big.sum())
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_roc(y_train, yhat_train, y_val, yhat_val, path):
    cmap= mpl.cm.winter
    new_cmap = truncate_colormap(cmap, 0.0, 0.7, n=100)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,1,20), np.linspace(0,1,20), 
                    linestyle=':', 
                    label='Random', 
                    color='tab:grey')

    ys = [np.log10(y_train), np.log10(y_val)]
    phats = [np.log10(yhat_train), np.log10(yhat_val)]
    splits = ['train','val']
    for i in range(2):
        y = ys[i]
        phat = phats[i]
        split = splits[i]
        d_bounds = np.linspace(0.2, 0.8, 6)
        norm = mpl.colors.Normalize(vmin=0.1, vmax=1.5)
        tprate = []
        fprate = []
        dec = []
        yhat = np.log10(phat)
        for i, d_bound in enumerate(d_bounds):
            tprate.append(hit_rate(y, yhat, d_bound))
            fprate.append(fa_rate(y, yhat, d_bound))
        #     print("HR = %0.2f; FAR = %0.2f"%(
        #             hit_rate(Ytr, phat, dbound),
        #             fa_rate(Ytr, phat, dbound)))

        colors= new_cmap(d_bounds)
        norm = mpl.colors.Normalize(vmin=d_bounds.min(), vmax=d_bounds.max())
        print(fprate)
        print(tprate)
        ax.plot(fprate, tprate, label=split)
        ax.scatter(fprate, tprate, marker='x', c=colors, norm=norm)

    ax.set_ylabel('Hit Rate')
    ax.set_xlabel("False Alarm Rate")
    ax.legend(frameon=False)
    fig2, barax= plt.subplots(1, 1)
    map1 = barax.imshow(np.stack([d_bounds, d_bounds]),cmap=new_cmap, norm=norm)
    cb = fig.colorbar(map1, ax=ax, pad=0.0)
    cb.set_label('Bound')
    cb.cmap(colors)
    plt.savefig(path)


def get_split(position, val=True, test=True):

    if np.all(position < [0.5,0.6,0.6]) and val:
        return 'val'
    if np.all(position > [0.5,0.4,0.4]) and test:
        return 'test'
    else:
        return 'train'


def get_features(data, idx, maxt, dt, split_val=True, split_test=True, time = True, max_stars = 250):
    """
        Retrieves samples from *json archives, gets tokenized features from
        samples at time index idx

        data: dictionary containing all region data
         idx: index of the time to analyze
         maxt: maximum model time
         dt: time between data samples
         split_xx: return xx split spatially in the simulation?
                Random splitting doesnt work here, as there are multiple samples from very
                similar regions...
         time: return time features too?
    """
    timebins = np.arange(0, maxt, dt)
    mass_tok = None
    time_tok = None
    splits = []
    skipped = 0
    for i, k in enumerate(data.keys()): # iterate over pidx's
        ind = idx if len(data[k]['time']) > idx else -1
        try:
            p3pos = np.array(data[k]['p3_all_position'][idx])
        except IndexError:
            skipped += 1
            continue
        center = np.array(data[k]['region_center'][idx])
        screen = np.array([np.linalg.norm(p3-center) for p3 in p3pos])
        if np.all(screen > max_radii): continue
        ctimes = np.array(data[k]['p3_all_ctime'][ind])[screen < max_radii]
        cmasses = np.array(data[k]['p3_all_mass'][ind])[screen < max_radii]
        tnow = ctimes.min()
        if cmasses.sum() == 0: 
            skipped += 1
            continue
        try:
            position = np.array(data[k]['region_unitary'][ind])
        except:
            skipped += 1
            continue
        splits.append(get_split(position, split_val, split_test))
        if i==0:
            times = [t-tnow for t in ctimes]
            masses = [m*1e20 if m < 1e-5 else m for m in cmasses] 
            mass_tok, edges = np.histogram(masses, bins=massbins)
            time_tok, edges = np.histogram(times, bins=timebins)
        else:
            masses = [m*1e20 if m < 1e-5 else m for m in cmasses]          
            times = [t-tnow for t in ctimes]
            str_times = ', '.join(['%0.2f'%t for t in times])
            # print(str_times)
            m_tok, edges = np.histogram(masses, bins=massbins)
            t_tok, edges = np.histogram(times, bins=timebins)
            if mass_tok.sum() > 0 and time_tok.sum() > 0:
                mass_tok = np.vstack([mass_tok, m_tok])
                time_tok = np.vstack([time_tok, t_tok])
        # print(mass_tok)
        # print(time_tok)
        # exit()
        print('Loaded %04d/%04d'%(i, len(data.keys())), end='\r')
    print("Get features skipped %d samples..."%skipped, mass_tok.shape, time_tok.shape)
    splits = np.array(splits)
    if time:
        tsplit = np.append(mass_tok, time_tok, axis=1)[splits=='train']
        vsplit =np.append(mass_tok, time_tok, axis=1)[splits=='val']
        testsplit =  np.append(mass_tok, time_tok, axis=1)[splits=='test']
    else:
        tsplit = mass_tok[splits=='train'] 
        vsplit = mass_tok[splits=='val']
        testsplit = mass_tok[splits=='test']
    # # Rescaling tests
    # vsplit = (vsplit-tsplit.mean(0))/tsplit.std(0)
    # testsplit = (testsplit-tsplit.mean(0))/tsplit.std(0)
    # tsplit = (tsplit-tsplit.mean(0))/tsplit.std(0)
    return tsplit, vsplit, testsplit

def get_labels(data, idx, fields, split_val=True, split_test=True, log_radii=False):
    """
        Loads labels from *json files located at <data>
            data: dictionary containing all region data
            idx: which time index to use as label
            fields: which fields to include in labels
            split: split among training and validation?
    """
    rads = []
    splits = []
    for k in data.keys():

        ind = idx if len(data[k]['time']) > idx else -1
        try:
            p3pos = np.array(data[k]['p3_all_position'][idx])
        except IndexError:
            continue
        center = np.array(data[k]['region_center'][idx])
        screen = np.array([np.linalg.norm(p3-center) for p3 in p3pos])
        if np.all(screen > max_radii): continue
        # print('now: %0.2f'%tnow)
        # print(', '.join(['%0.2f'%t for t in data[k]['p3_all_ctime'][ind]]))
        ctimes = np.array(data[k]['p3_all_ctime'][ind])[screen < max_radii]
        cmasses = np.array(data[k]['p3_all_mass'][ind])[screen < max_radii]
        if ctimes.size == 0 or cmasses.size == 0:
            continue
        try:
            position = np.array(data[k]['region_unitary'][ind])
        except:
            continue
        # cpos = np.array(data[k]['p2_all_position'][idx])
        # center = np.array(data[k]['region_center'][idx])
        # screen = np.array([np.linalg.norm(p3-center) for p3 in p3pos])

        splits.append(get_split(position, split_val, split_test))
        local = []
        for f in fields:
            local.append(data[k][f][ind])
        rads.append(local)
    if log_radii:
        rads = np.log10(rads)
    else:
        rads = np.array(rads)
    splits = np.array(splits)
    return (rads[splits == 'train'], 
                rads[splits == 'val'],
                rads[splits == 'test'])

def split_dataset(X, Y):
    
    return xtr, ytr, xval, yval, xtest, ytest
    
def save_best_model(model, name):
    print("SAVING record for %s"%name)
    params = model.get_params(deep=True)
    for k in params.keys():
        print(k)
    if not os.path.exists('./record_models'):
        os.makedirs('./record_models', exist_ok=True)
    with open('./record_models/%s.json'%name, 'w') as f:
        json.dump(params, f)


def get_data_dict(path, sim = None):
    """
        Iterates all *json files at <path>, loads them to dictionary

        sim: restrict to only loading files from specific simulation name 
    """
    if not sim:
        logs = glob.glob('%s/*/*.json'%path)
    else:
        logs = glob.glob('%s/%s/*.json'%(path, sim))
    if len(logs) == 0:
        print("Found no logs at %s/*/*.json"%path)
    with open(logs[0], 'r') as f:
        rq = json.load(f)
    for i, log in enumerate(logs):
        if i != 0:
            with open(log, 'r') as f:
                newd = json.load(f)
            for k in newd:
                if len(newd[k]['p3_all_mass']) > 0:
                    rq[k] = newd[k]
    tlen = 0
    tmax = 0
    for k in rq:
        if len(rq[k]['time']) > tlen: tlen = len(rq[k]['time'])
        dt = max(rq[k]['time']) - min(rq[k]['time'])
        if dt > tmax: tmax = dt
    # print('Time Length: %d, Model_Time = %0.1f'%(tlen, tmax))
    # for k in rq:
    #     for kk in rq[k]:
    #         print(kk)
    #     break
    return rq, tlen, tmax


class IMF_sampler():
    """
        This class produces samples from an IMF that matches Enzo's PopIII IMF
        with a number of stars per region that matches the statistical distribution 
        of the Phoenix simulations.  Birth times, masses, and counts per region are pulled
        from CDFs that were generated from the PHX sims--either supplied as arguments or 
        on disk as text files.
    """
    def __init__(self, mchar=20, 
                mmax=300, 
                mmin=1, 
                maxtime=30, 
                dt=5, 
                num_cdf = [], 
                num_bins = None,
                mass_cdf = [],
                mass_bins = None,
                time_cdf = [],
                time_bins = None, 
                nstar_mean = 1, nstar_std = 1):
        """

        """
        if num_bins is None:
            num_cdf, num_bins = np.loadtxt('CDF_Nstar.txt', usecols=True)
        if mass_bins is None:
            mass_cdf, mass_bins = np.loadtxt('CDF_mass.txt', usecols=True)
        if time_bins is None:
            time_cdf, time_bins = np.loadtxt('CDF_creationtime.txt', usecols=True)
        self.model_time = maxtime
        self.imf_mchar = mchar
        self.imf_mmax = mmax
        self.imf_mmin = mmin
        self.mass_cdf = mass_cdf
        self.mass_cdf_bins = mass_bins
        self.ctime_cdf = time_cdf
        self.ctime_cdf_bins = time_bins
        self.nstar_cdf = num_cdf
        self.nstar_cdf_bins = num_bins
        self.nstar_mean = nstar_mean
        self.nstar_std = nstar_std
        self.massbins = [1, 11, 20, 40, 100, 140, 200, 260, 300]
        self.timebins = np.arange(0, maxtime, dt)
        # print("Timebins: ", self.timebins)
        self.dt = dt

    def generate_samples(self, nsamples, time = True):
            counts = self.get_starcounts(nsamples)
            # print("Counts: ", counts)
            masses = self.get_mstars(counts)
            times = self.get_birthtimes(counts)
            sample_tokens = self.tokenize(masses, times, time)
            return sample_tokens
            
            
    def get_starcounts(self, nsamples):
        unisamples = np.random.uniform(0,1, size=nsamples)
        indices = np.digitize(unisamples, self.nstar_cdf)
        # indices = np.array([np.argmin(np.abs(self.nstar_cdf - u)) for u in unisamples])
        counts = self.nstar_cdf_bins[indices]
        return counts
        
        
    def get_mstars(self, nstars):
        masses = []
        for nstar in nstars:
            unisamples = np.random.uniform(0,1,size=int(nstar))
            indices = np.digitize(unisamples, self.mass_cdf)
            masses.append(self.mass_cdf_bins[indices].tolist())
        return masses
    
    def get_birthtimes(self, nstars):
        times = []
        for nstar in nstars:
            unisamples = np.random.uniform(0,1,size=int(nstar))
            indices = np.digitize(unisamples, self.ctime_cdf)
            times.append(self.ctime_cdf_bins[indices].tolist())
        return times
    
    def tokenize(self, masses, times, time = True): #times are already in tokens
        # print("MASSES: ", masses)
        # print("TIMES: ", times)
        mass_tok = None
        time_tok = None
        for i in range(len(masses)):
            m_tok,_ = np.histogram(masses[i], bins=self.massbins)
            t_tok,_ = np.histogram(times[i], bins=self.timebins)
            if i == 0:
                mass_tok = m_tok
                time_tok = t_tok
            else:
                mass_tok = np.vstack([mass_tok, m_tok])
                time_tok = np.vstack([time_tok, t_tok])
        if time:
            return np.append(mass_tok, time_tok, axis=1)
        else:
            return mass_tok
            
def save_shap_trees(model, Xtr, Xv, feat_names):

                    
    explainer = shap.Explainer(model.get_booster())

    fig, ax = plt.subplots()
    explainer.feature_names = feat_names
    shap.decision_plot(explainer.expected_value, 
                    explainer.shap_values(Xv,check_additivity=False), 
                    feature_names = feat_names,
                    show=False
                    )
    ax.set_xlabel('Model Output [kpc]')
    # fig.supylabel('Model Feature')
    plt.savefig('DecisionPlot_val.pdf', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    explainer.feature_names = feat_names
    shap.decision_plot(explainer.expected_value, 
                    explainer.shap_values(Xtr,check_additivity=False), 
                    feature_names = feat_names,
                    show=False
                    )
    ax.set_xlabel('Model Output [kpc]')
    # fig.supylabel('Model Feature')
    # ax.set_xlim(0,10)
    plt.savefig('DecisionPlot_train.pdf', bbox_inches='tight', dpi=600)
    plt.close()



def get_dataloaders(model_time, tbin_dt, min_r=0.0, timefeatures=True, log_radii=False, batch_size=-1):
    logpath = '../size_of-single'
    fields = ['p3_metallicity_radius']

    data, tlen, tmax = get_data_dict(logpath)
    # print('%d keys'%len(data))
    dt = tmax // (tlen-1)
    idx = int(model_time // dt) -1# index of time you want to model-stuck at final state for now
    print('Max Time = %0.1f; T_len = %d;  DT = %0.1f; index = %d'%(tmax, tlen, dt, idx))
    # print("PIDX's:")
    # for k in data.keys():
    #     print('\t%s'%k)
    if idx >= tlen -1:
        idx = -1
    # print('Using index %d'%idx)
    Xtr, Xv, Xtest = get_features(data, idx, model_time, tbin_dt, time=timefeatures)
    Ytr, Yv, Ytest = get_labels(data, idx, fields, log_radii=log_radii)
    Ytr = Ytr.flatten()
    Yv = Yv.flatten()
    if not log_radii:
        Xtr = Xtr[Ytr > min_r]
        Ytr = Ytr[Ytr > min_r]
        Xv = Xv[Yv > min_r]
        Yv = Yv[Yv > min_r]
    nfeat = Xv[0].size
    # Xtr, Ytr, Xv, Yv, xtest, ytest = split_dataset(X,Y)
    times = ['%dMyr'%(tbin_dt*(i+1)) for i in range(tlen)]
    feat_names = ['1-11', '11-20', '20-40',
                '40-100', 
                '100-140', 
                '140-200', 
                '200-260', 
                '260+'] + times 
    print('Found train samples', Ytr.shape)
    print('With val samples', Yv.shape)
    print("Min radius = %0.2f; Max = %0.2f"%(Ytr.min(), Ytr.max()))
    print("with %d features each"%Xtr[0].size)
    print(Xtr.shape, Ytr.shape)
            
    trainset = TensorDataset(tensor(Xtr), tensor(Ytr).view( Ytr.size, 1))
    valset = TensorDataset(tensor(Xv), tensor(Yv).view(Yv.size, 1))
    testset = TensorDataset(tensor(Xtest), tensor(Ytest).view(Ytest.size,1))
    tload = DataLoader(trainset, batch_size=len(Xtr) if batch_size < 0 else batch_size, shuffle=True)
    vload = DataLoader(valset, batch_size=len(Xv) if batch_size < 0 else batch_size, shuffle=True)
    testload = DataLoader(testset, batch_size=len(Xv) if batch_size < 0 else batch_size, shuffle=True)
    # print(len(tload))

    return tload, vload, testload, Xtr[0].size

