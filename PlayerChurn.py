# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:25:36 2017

@author: Dmitriy
"""

import shelve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import strftime, localtime
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier as GBC

def save_vars(fname, varlst):
    my_shlf = shelve.open(fname, flag='n')
    for key in varlst:
        try:
            my_shlf[key] = globals()[key]
        except TypeError:
                print('TypeError shelving: {0}'.format(key))
        except:
            print('Generic error shelving: {0}'.format(key))
    my_shlf.close()
    
def load_vars(fname):
    my_shlf = shelve.open(fname)
    for key in my_shlf:
        globals()[key] = my_shlf[key]
    my_shlf.close()

def cv_gb_perf(hyperpars, nfolds=5):
    cv_auc_train = []
    cv_auc_test = []
    cv_losses_train = []
    cv_losses_test = []
    cv = KFold(n_splits=nfolds, shuffle=True)
    for train_index, test_index in cv.split(dt_train_all):
        dt_train = dt_train_all.iloc[train_index,:]
        dt_test = dt_train_all.iloc[test_index,:]
        X_train = dt_train[factor_cols].values
        Y_train = dt_train[trgt_col].values
        X_test = dt_test[factor_cols].values
        Y_test = dt_test[trgt_col].values           
        pars = {'n_estimators': hyperpars['ntrees'],
                'learning_rate': hyperpars['eta'],
                'max_depth': hyperpars['max_depth'],
                'min_samples_split': int(hyperpars['min_samples_split']),
                'min_samples_leaf': int(hyperpars['min_samples_leaf']),
                'subsample': hyperpars['subsample']
        }
        clf = GBC(**pars)
        clf.fit(X=X_train, y=Y_train)
        pos_cl_ind = int(np.where(clf.classes_==1)[0])
        Y_train_fcs = clf.predict_proba(X=X_train)[:,pos_cl_ind]
        Y_test_fcs = clf.predict_proba(X=X_test)[:,pos_cl_ind]
        cv_auc_train.append(roc_auc_score(y_true=Y_train, y_score=Y_train_fcs))
        cv_auc_test.append(roc_auc_score(y_true=Y_test, y_score=Y_test_fcs))
        cv_losses_train.append(log_loss(y_true=Y_train, y_pred=Y_train_fcs))
        cv_losses_test.append(log_loss(y_true=Y_test, y_pred=Y_test_fcs))
    return {'auc_train': np.mean(cv_auc_train),
            'auc_test': np.mean(cv_auc_test),
            'loss_train': np.mean(cv_losses_train),
            'loss_test': np.mean(cv_losses_test)}

def obj_loss(hyperparams, nfolds=5):
    res_perf = cv_gb_perf(hyperpars=hyperparams, nfolds=nfolds)
    print 'Hyperparameters: ' + str(hyperparams)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['loss_test']

# Set this to use previously tuned hyperparams instead new search
sv_tunes_fname = None

Y_train_all = pd.read_csv('y_train.csv', sep=';', header=None)
dt_train_all = pd.read_csv('x_train.csv', sep=';')
X_test_all = pd.read_csv('x_test.csv', sep=';')

factor_cols = dt_train_all.columns.tolist()
trgt_col = 'IsOnline'
dt_train_all[trgt_col] = Y_train_all[0]

if sv_tunes_fname is None:
    hp_space = {
       'ntrees': hp.choice('ntrees', (200,300,500)),
       'eta': hp.qloguniform('eta', np.log(0.01), np.log(0.1), 0.001),
       'max_depth': hp.choice('max_depth', (2,3,4,5,6)),
       'min_samples_split': hp.quniform('min_samples_split', 2, 16, 1),
       'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
       'subsample': hp.quniform('subsample', 0.5, 0.9, 0.001)
    }
    trials = Trials()
    beg = timer()
    GB_tune = fmin(fn=obj_loss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=300)
    tm_spent = timer() - beg
    print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec'
    print GB_tune
    # Save tuning results
    save_vars(fname='GB_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
              varlst=['GB_tune','trials','hp_space','tm_spent'])
else:
    load_vars(fname=sv_tunes_fname)

# Estimate model with found hyperparameters
clf = GBC(learning_rate=GB_tune['eta'], n_estimators=GB_tune['ntrees'], subsample=GB_tune['subsample'],
          max_depth=GB_tune['max_depth'], min_samples_split=int(GB_tune['min_samples_split']),
          min_samples_leaf=int(GB_tune['min_samples_leaf']), random_state=0, verbose=1)

clf.fit(X=dt_train_all[factor_cols].values, y=dt_train_all[trgt_col].values)
Y_test_fcs = clf.predict_proba(X=X_test_all.values)[:,int(np.where(clf.classes_==1)[0])]
np.savetxt(fname='answer_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')



# from hyperopt.mongoexp import MongoTrials
# MongoTrials('mongo://localhost:1234/mydb/jobs', exp_key='exp1')







