# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 01:01:57 2017

@author: Dmitriy
"""

import shelve
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, Trials
from time import strftime, localtime
from sklearn.ensemble import RandomForestClassifier as RFC

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

nfolds = 5

Y_train_all = pd.read_csv('y_train.csv', sep=';', header=None)
X_train_all = pd.read_csv('x_train.csv', sep=';')
X_test_all = pd.read_csv('x_test.csv', sep=';')

factor_cols = X_train_all.columns.tolist()
nfeat = np.sqrt(len(factor_cols))
trgt_col = 'IsOnline'
dt_train_all = X_train_all.copy()
dt_train_all[trgt_col] = Y_train_all[0]

def cv_rf_perf(hyperpars):
    cv_auc_train = []
    cv_auc_test = []
    cv_losses_train = []
    cv_losses_test = []
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True)
    for train_index, test_index in cv.split(X=X_train_all.values, y=Y_train_all[0].values):
        dt_train = dt_train_all.iloc[train_index,:]
        dt_test = dt_train_all.iloc[test_index,:]
        X_train = dt_train[factor_cols].values
        Y_train = dt_train[trgt_col].values
        X_test = dt_test[factor_cols].values
        Y_test = dt_test[trgt_col].values           
        pars = {'n_estimators': int(hyperpars.get('ntrees',100)),
                'max_features': int(hyperpars.get('max_features',nfeat)),
                'max_depth': int(hyperpars.get('max_depth',3)),
                'min_samples_split': int(hyperpars.get('min_samples_split',2)),
                'min_samples_leaf': int(hyperpars.get('min_samples_leaf',1))
        }
        clf = RFC(**pars)
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
    
def obj_loss(hyperpars):
    res_perf = cv_rf_perf(hyperpars)
    print 'Hyperparameters: ' + str(hyperpars)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['loss_test']

hp_space = {
  'ntrees': hp.quniform('ntrees', 300, 500, 10),
  'max_features': hp.quniform('max_features', round(nfeat), round(2*nfeat), 1),
  'max_depth': hp.qloguniform('max_depth', np.log(2), np.log(9), 1),
  'min_samples_split': hp.quniform('min_samples_split', 10, 20, 1),
  'min_samples_leaf': hp.quniform('min_samples_leaf', 10, 30, 1)
}
trials = Trials()
beg = timer()
RF_tune = fmin(fn=obj_loss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=1500)
tm_spent = timer() - beg
print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec\nBest found hyperparameters:'
print str(RF_tune) + '\n'

save_vars(fname='RF_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
          varlst=['RF_tune','trials','hp_space','tm_spent'])

clf = RFC(n_estimators=int(RF_tune['ntrees']), max_depth=int(RF_tune['max_depth']),
          min_samples_split=int(RF_tune['min_samples_split']),
          min_samples_leaf=int(RF_tune['min_samples_leaf']),
          random_state=0)

clf.fit(X=dt_train_all[factor_cols].values, y=dt_train_all[trgt_col].values)
pos_cl_ind = int(np.where(clf.classes_==1)[0])

Y_train_fcs = clf.predict_proba(X=dt_train_all[factor_cols].values)[:,pos_cl_ind]
Y_test_fcs = clf.predict_proba(X=X_test_all.values)[:,pos_cl_ind]
np.savetxt(fname='answerRF_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')

# To use in further ensembling
pd.DataFrame({'RF_PROB': Y_train_fcs}).to_csv('RF_fcs_train.csv', index=False)
pd.DataFrame({'RF_PROB': Y_test_fcs}).to_csv('RF_fcs_test.csv', index=False)












