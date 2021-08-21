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
from model_feedback_radii import *

# tokenized mass bins for ez modelling
massbins = [1, 11, 20, 40, 100, 140, 200, 260, 300]
# tokenized birth times too with 5 myr increments
timebins = np.arange(1, 30 + 1, 15)

def main():
    argparser = ap()
    argparser.add_argument('--logpath', type=str, default='../size_of')
    argparser.add_argument('--model_time', type=float, default=30.0)
    argparser.add_argument('--output_dir', type=str, default = './')
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
    depths = [None, 3,5,7,9]
    leaf_samples = [1,2,3,4]
    criteria = ['friedman_mse','mae','mse']
    splitter = ['best','random']
    max_leaf_nodes = [None,3,5,9,15,30]
    savelog = "%s/hyperparameter_tuning.log"%args.output_dir
    with open(savelog, 'w') as f:
        f.write("# run\tdepth\tleaf_samples\tcriteria\tsplitter\tmax_leaf_nodes\tmse\tmae\tmape\texvar\n")
    mapemin = 1e5
    maemin = 1e5
    exvbest = -10
    for run in range(25):
        for d in depths:
            for l in leaf_samples:
                for c in criteria:
                    for split in splitter:
                        for mln in max_leaf_nodes:
                            predictor = DecisionTreeRegressor(
                                            max_depth = d,
                                            max_leaf_nodes=mln,
                                            min_samples_leaf=l,
                                            criterion=c,
                                            splitter=split
                                            )
                            predictor.fit(Xtr, Ytr)
                            phat = predictor.predict(Xv)
                            mse =  mean_squared_error(phat, Yv)
                            mae = mean_absolute_percentage_error(Yv, phat)
                            exv = explained_variance_score(phat, Yv)
                            mape = mean_absolute_percentage_error(phat, Yv)

                            with open(savelog,'a') as f:
                                f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                                    run, d, l, c, split, mln, mse, mae, mape, exv
                                ))
                            if mape < mapemin:
                                mapemin = mape
                                print("Record Run: MSE = {}; MAE = {}; MAPE = {}; exp_var = {}".format(
                                    mse, mae, mape, exv
                                    ))
                                print('\t\trun = {},d = {},l = {}, c = {}, split = {}, mln = {}\n'.format(
                                    run, d, l, c, split, mln
                                ))
                                feat_names = ['>SN', 'SNe', 'HNe','HNe < M < 100', '100 < M < 140', 'PISN < 200', 'PISN > 200', '> 200',
                                                '5Myr', '10Myr','15Myr','20Myr','25Myr','30Myr']
                                fig, ax = plt.subplots(figsize=(40,40))
                                tree.plot_tree(predictor, feature_names = feat_names, rounded=True, filled=True, ax=ax)
                                plt.savefig('Best_tree.pdf', bbox_inches='tight')

if __name__=='__main__':
    main()