"""
    As a baseline model,
    we can try to estimate the radii of influence of
    a P3 cluster.  This will take a series of token counts
        {lowmass, SN, HN, midmass, PISN, highmass}
    and birth times
        {>5 Myr, >10 Myr, >15 Myr, >20 Myr, >25 Myr, >30 Myr}
    to estimate the radius of (initially) metals and temperature as determined by size_of_feedbackregions.py

    Might be nice to use decision trees or something more interpretable than a neural net (+it looks good :D)
"""


import json, glob, os
import numpy as np
from argparse import ArgumentParser as ap
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, \
                            mean_absolute_error, \
                            explained_variance_score, \
                            mean_absolute_percentage_error 
from sklearn.ensemble import AdaBoostRegressor,\
                                ExtraTreesRegressor,\
                                GradientBoostingRegressor,\
                                RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# the right edges of the mass bins
# subdivided in the really large bins between hn and pisn
# and in the pisn mass
massbins = [1, 11, 20, 40, 100, 140, 200, 260, 300]
# tokenized birth times too with 5 myr increments
timebins = np.arange(1, 30 + 1, 15)

def get_features(data, idx):
    # from dict under the particle index, tokenize the mass and time features
    # index corresponds to time of model (-1 for last state guaranteed)
    mass_tok = None
    time_tok = None
    for i, k in enumerate(data.keys()): # iterate over pidx's
        ind = idx if len(data[k]['time']) > idx else -1
        tnow = data[k]['time'][ind]
        if i==0:
            times = [tnow - t for t in data[k]['p3_all_ctime'][ind]]
            masses = [m*1e20 if m < 1e-5 else m for m in data[k]['p3_all_mass'][ind]] 
            mass_tok, edges = np.histogram(masses, bins=massbins)
            time_tok, edges = np.histogram(times, bins=timebins)
        else:
            masses = [m*1e20 if m < 1e-5 else m for m in data[k]['p3_all_mass'][ind]]          
            times = [tnow - t for t in data[k]['p3_all_ctime'][ind]]
            m_tok, edges = np.histogram(masses, bins=massbins)
            t_tok, edges = np.histogram(times, bins=timebins)
            mass_tok = np.vstack([mass_tok, m_tok])
            time_tok = np.vstack([time_tok, t_tok])
    return np.append(mass_tok, time_tok, axis=1)

def get_labels(data, idx, fields):
    # index corresponds to time of model (-1 for last state guaranteed)
    rads = []
    for k in data.keys():
        ind = idx if len(data[k]['time']) > idx else -1
        local = []
        for f in fields:
            local.append(data[k][f][ind])
        rads.append(local)
    return np.array(rads)

def split_dataset(X, Y):
    xtr, xval, ytr, yval = train_test_split(X, Y, test_size=0.2, random_state=8675309)
    # xtr, xtest, ytr, ytest = train_test_split(xtr, ytr, test_size=0.125, random_state=2339479)
    
    return xtr, ytr, xval, yval

def get_data_dict(path, sim = None):
    # provide overarching path to data files
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
    print('Time Length: %d, Model_Time = %0.1f'%(tlen, tmax))
    for k in rq:
        for kk in rq[k]:
            print(kk)
        break
    return rq, tlen, tmax

def main():

    argparser = ap()
    argparser.add_argument('--logpath', type=str, default='./size_of')
    argparser.add_argument('--model_time', type=float, default=30.0)
    argparser.add_argument('--output_dir', type=str, default = './modelling_feedback')
    argparser.add_argument('--fields', type=str, nargs='+', default=None)
    args = argparser.parse_args()

    if not os.path.exists(args.logpath):
        raise(OSError("Cannot locate input logs filepath"))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_time <= 1:
        raise(ValueError('--model_time must be greater than 1 Myr'))

    # get data dict
    data, tlen, tmax = get_data_dict(args.logpath)
    dt = tmax // (tlen-1)
    idx = int(args.model_time // dt) # index of time you want to model-stuck at final state for now
    print('Max Time = %0.1f; T_len = %d;  DT = %0.1f; index = %d'%(tmax, tlen, dt, idx))
    print("PIDX's:")
    for k in data.keys():
        print('\t%s'%k)
    if idx >= tlen -1:
        idx = -1
    print('Using index %d'%idx)
    X = get_features(data, idx)
    Y = get_labels(data, idx, args.fields)
    Y = Y.flatten()
    X = X[Y > 1]
    Y = Y[Y > 1]

    Xtr, Ytr, Xv, Yv = split_dataset(X,Y)
    print("Found %d Training inputs, %d validation"\
                %(len(Xtr), len(Xv)))
    models = [
                AdaBoostRegressor,
                DecisionTreeRegressor, 
                ExtraTreesRegressor, 
                GradientBoostingRegressor,
                # RandomForestRegressor
                ]
    model_labels = [
                    'AdaBoost',
                    'DecisionTree',
                    'RandomForest+',
                    'GradBoost',
                    # 'RandomForest'
                    ]
    colors = ['tab:blue','tab:green','tab:orange', 'tab:purple']
    fig, ax = plt.subplots(2,2, sharex=True, figsize=(10,10))
    ax=ax.flatten()
    # if type(ax) != np.array: ax = [ax]
    for k in range(5): # 10 runs, to start averaging out noise
        print("<<< ITERATION %d IN PROGRESS >>>"%k)
        for i, model in enumerate(models):
            boost_dep = 3
            min_leaf = 1
            name = model_labels[i]
            est_cnt = np.linspace(10, 1000, 5, dtype=int)
            mse = []
            mae = []
            exv = []
            mape = []
            print("Processing %s"%model_labels[i])
            for j, n in enumerate(est_cnt):
                if name=='AdaBoost':
                    predictor = model(
                                        n_estimators=n, 
                                        learning_rate=.1 /(n / 10.), 
                                        loss = 'exponential',
                                        base_estimator=DecisionTreeRegressor(max_depth=boost_dep, 
                                                                                min_samples_leaf=min_leaf)
                                        # random_state=8675309
                                        )
                elif name == 'GradBoost':
                    predictor = model(
                                        n_estimators=n,
                                        learning_rate=0.1,
                                        min_samples_leaf=min_leaf,
                                        max_depth=boost_dep,
                                        # random_state=0
                                        )
                elif 'Forest' in name:
                    predictor = model(n_estimators=n, 
                                        n_jobs=6, 
                                        # min_samples_split=4, 
                                        bootstrap=True, 
                                        max_samples=0.9,
                                        max_depth=boost_dep,
                                        min_samples_leaf=min_leaf
                                        )
                else: # decision tree
                    predictor = model(
                                        # n_estimators=n, 
                                        # n_jobs=6, 
                                        # min_samples_split=4, 
                                        # bootstrap=True, 
                                        # max_samples=0.9,
                                        max_depth=7,
                                        min_samples_leaf=1
                                        )
                Ytr = Ytr.ravel()
                predictor.fit(Xtr, Ytr)
                if name == 'DecisionTreeRegressor' and k == 0:
                    treesave = predictor
                # print("R^2 = %0.5f"%(model.score(Xv, Yv)))
                phat = predictor.predict(Xv)
                if k == 4:
                    for ii in range(len(phat)):
                    
                        print("\t%0.2f: %0.2f"%(phat[ii], Yv[ii]))
                    print('\n\n')
                mse.append(mean_squared_error(phat, Yv))
                mae.append(mean_absolute_percentage_error(Yv, phat))
                exv.append(explained_variance_score(phat, Yv))
                mape.append(mean_absolute_percentage_error(phat, Yv))

            ax[0].plot(est_cnt, mse, color=colors[i], label=model_labels[i] if k == 0 else None, alpha=0.5)
            ax[1].plot(est_cnt, mae, color=colors[i], label=model_labels[i] if k == 0 else None, alpha=0.5)
            ax[2].plot(est_cnt, mape, color=colors[i], label=model_labels[i] if k == 0 else None, alpha=0.5)
            ax[3].plot(est_cnt, exv, color=colors[i], label=model_labels[i] if k == 0 else None, alpha=0.5)
    ax[0].set_title('MSE')
    ax[1].set_title('MAE')
    ax[2].set_title('MAPE')
    ax[3].set_title('Exp-Var')
    ax[0].legend(bbox_to_anchor=(1.7,1.2), ncol=4)
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[-1].set_xticks([k for k in range(len(model_labels))])
    # ax[-1].set_xticklabels(model_labels, rotation=45)
    fig.supxlabel('$N_{trees}$ ($n=$%d)'%len(Ytr))
    fig.supylabel('Err($Z_{III}$)')
    plt.savefig('%s/model_quant.pdf'%args.output_dir, bbox_inches='tight')

    if name == 'DecisionTreeRegressor':
        feat_names = ['>SN', 'SNe', 'HNe','HNe < M < 100', '100 < M < 140', 'PISN < 200', 'PISN > 200', '> 200',
                        '5Myr', '10Myr','15Myr','20Myr','25Myr','30Myr']
        fig, ax = plt.subplots(figsize=(40,40))
        tree.plot_tree(treesave, feature_names = feat_names, rounded=True, filled=True, ax=ax)
        plt.savefig('test_tree.png', bbox_inches='tight')

if __name__ == '__main__': main()