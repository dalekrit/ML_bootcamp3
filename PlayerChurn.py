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
    
def sigmoid(x):
    return 1/(1+np.exp(-np.asarray(x)))
    
# Set CV params
nfolds = 5 # both for hyperopt & final GB performance
nreps = 10 # only for final GB perf. (test-loss by ntrees)

# Set to existing filename to use previously tuned hyperparams instead new search
sv_tunes_fname = None

# Load and prepare data
Y_train_all = pd.read_csv('y_train.csv', sep=';', header=None)
dt_train_all = pd.read_csv('x_train.csv', sep=';')
X_test_all = pd.read_csv('x_test.csv', sep=';')

factor_cols = dt_train_all.columns.tolist()
trgt_col = 'IsOnline'
dt_train_all[trgt_col] = Y_train_all[0]

# Define objective to minimize for hyperopt search
def cv_gb_perf(hyperpars):
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
        pars = {'n_estimators': int(hyperpars.get('ntrees',100)),
                'learning_rate': hyperpars.get('eta',0.1),
                'max_depth': int(hyperpars.get('max_depth',3)),
                'min_samples_split': int(hyperpars.get('min_samples_split',2)),
                'min_samples_leaf': int(hyperpars.get('min_samples_leaf',1)),
                'subsample': hyperpars.get('subsample',1.0)
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
    
def obj_loss(hyperpars):
    res_perf = cv_gb_perf(hyperpars)
    print 'Hyperparameters: ' + str(hyperpars)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['loss_test']

if sv_tunes_fname is None:
    hp_space = {
       'ntrees': hp.quniform('ntrees', 200, 500, 100),
       'eta': hp.qloguniform('eta', np.log(0.01), np.log(0.1), 0.001),
       'max_depth': hp.quniform('max_depth', 2, 6, 1),
       'min_samples_split': hp.quniform('min_samples_split', 2, 16, 1),
       'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
       'subsample': hp.quniform('subsample', 0.5, 0.9, 0.001)
    }
    trials = Trials()
    beg = timer()
    GB_tune = fmin(fn=obj_loss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=300)
    tm_spent = timer() - beg
    print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec\nBest found hyperparameters:'
    print str(GB_tune) + '\n'
    # Save tuning results
    save_vars(fname='GB_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
              varlst=['GB_tune','trials','hp_space','tm_spent'])
else:
    load_vars(fname=sv_tunes_fname)

ntrees = int(GB_tune.get('ntrees',100))
eta = GB_tune.get('eta',0.1)
subsample = GB_tune.get('subsample',1.0)
max_depth = int(GB_tune.get('max_depth',3))
min_samples_split = int(GB_tune.get('min_samples_split',2))
min_samples_leaf = int(GB_tune.get('min_samples_leaf',1))

# Estimate GB performance with different number of trees
loss_train = np.zeros(ntrees)
loss_test = np.zeros(ntrees)
beg = timer()
print 'Performing final CV'
for i in range(nreps):
    print 'Repeat ' + str(i+1) + ' of ' + str(nreps)
    cv = KFold(n_splits=nfolds, shuffle=True)
    for train_index, test_index in cv.split(dt_train_all):
        dt_train = dt_train_all.iloc[train_index,:]
        dt_test = dt_train_all.iloc[test_index,:]
        X_train = dt_train[factor_cols].values
        Y_train = dt_train[trgt_col].values
        X_test = dt_test[factor_cols].values
        Y_test = dt_test[trgt_col].values
        clf = GBC(learning_rate=eta, n_estimators=ntrees, subsample=subsample, max_depth=max_depth,
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        clf.fit(X=X_train, y=Y_train)
        train_pred = clf.staged_decision_function(X=X_train)
        test_pred = clf.staged_decision_function(X=X_test)
        loss_train += np.asarray([log_loss(y_true=Y_train, y_pred=sigmoid(train_pred.next())) for _ in range(ntrees)])
        loss_test += np.asarray([log_loss(y_true=Y_test, y_pred=sigmoid(test_pred.next())) for _ in range(ntrees)])
        
loss_train /= nfolds*nreps
loss_test /= nfolds*nreps

print 'Time elapsed: ' + str(round(timer()-beg,2)) + ' sec'

plt.figure(figsize=(15,6))
plt.plot(loss_test, 'g', linewidth=2)
plt.plot(loss_train, 'b', linewidth=2)
plt.legend(['LogLoss Test', 'LogLoss Train'])
plt.show()

best_ntrees = loss_test.argmin() + 1
best_test_loss = loss_test[best_ntrees-1]

print 'Best number of trees is ' + str(best_ntrees) + ' of ' + str(ntrees)
print 'Minimal logarithmic loss on test sample is ' + str(best_test_loss) + '\n'

save_vars(fname='BEST_GB_PARAMS_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
          varlst=['best_ntrees','eta','subsample','max_depth','min_samples_split','min_samples_leaf'])

# Estimate model with found hyperparameters on the whole training dataset
clf = GBC(learning_rate=eta, n_estimators=best_ntrees, subsample=subsample, max_depth=max_depth,
          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=0)

clf.fit(X=dt_train_all[factor_cols].values, y=dt_train_all[trgt_col].values)
Y_test_fcs = clf.predict_proba(X=X_test_all.values)[:,int(np.where(clf.classes_==1)[0])]
np.savetxt(fname='answer_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')











