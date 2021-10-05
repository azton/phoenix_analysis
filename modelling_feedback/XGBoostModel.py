import json, glob, os
import numpy as np
from argparse import ArgumentParser as ap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from model_feedback_radii import *

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
    # print("PIDX's:")
    # for k in data.keys():
    #     print('\t%s'%k)
    if idx >= tlen -1:
        idx = -1
    print('Using index %d'%idx)
    X = get_features(data, idx)
    Y = get_labels(data, idx, args.fields)
    Y = Y.flatten()
    X = X[Y > 1]
    Y = Y[Y > 1]

    Xtr, Ytr, Xv, Yv, xtest, ytest = split_dataset(X,Y)

    depths = [2,4,6]
    n_ests = [4, 12, 24]
    boost_round = [14, 20, 24, 36, 300]
    mapemin=10
    for n_est in n_ests:
        for depth in depths:
            for br in boost_round:
                for run in range(10):
                    print("<< RUN %05d: N_est = %03d, D = %02d, %d boosts >>"\
                            %(run, n_est, depth, br), end='\r')
                    lr = np.random.uniform(0.1,1.3)
                    colsamples = np.random.uniform(0.5,1, size=(3))
                    alpha = np.random.uniform(0,0.2)
                    lamb = np.random.uniform(0.8,1.2)
                    parameters = {
                        'booster':'gbtree',
                        'colsample_bytree': np.random.uniform(0.5,1),
                        'colsample_bynode': np.random.uniform(0.5,1),
                        'colsample_bylevel': np.random.uniform(0.5,1),
                        'tree_method':'auto',
                        'eta': lr,
                        'objective':'reg:squarederror',
                        'subsample':np.random.uniform(0.5,1),
                        'num_parallel_tree':n_est,
                        'max_depth':depth,
                        'eval_metric':'rmse',
                        'lambda':lamb,
                        'alpha':alpha
                    }
                    dmatrix=xgb.DMatrix(Xtr, label=Ytr)
                    predictor = xgb.train(parameters, 
                                    dmatrix, 
                                    num_boost_round=br)
                    # predictor.fit(Xtr, Ytr)
                    dtest = xgb.DMatrix(Xv, label=Yv)
                    phat = predictor.predict(dtest)
                    mse =  mean_squared_error(phat, Yv)
                    mae = mean_absolute_percentage_error(Yv, phat)
                    exv = explained_variance_score(phat, Yv)
                    mape = mean_absolute_percentage_error(phat, Yv)
                    

                    if mape < mapemin:
                        mapemin = mape
                        print("Record Run: MSE = {}; MAE = {}; MAPE = {}; exp_var = {}".format(
                            mse, mae, mape, exv
                            ))
                        for k, v in parameters.items():
                            print("\t\t{}: {}".format(k,v))
                        print("\t\tnum_boost_round: %d"%br)
                        # xgb.plot_importance(predictor)
                        # plt.savefig('importance.png')
                        # plt.close()
                        predictor.save_model('./record_models/XGB_%dMyr.model'%args.model_time)
                    run += 1
if __name__=='__main__':
    main()
