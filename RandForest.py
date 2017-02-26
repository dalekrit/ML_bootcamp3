# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 01:01:57 2017

@author: Dmitriy
"""
import shelve
import numpy as np
import pandas as pd
from time import strftime, localtime
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
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
    
def load_vars(fname):
    my_shlf = shelve.open(fname)
    for key in my_shlf:
        globals()[key] = my_shlf[key]
    my_shlf.close()

nfolds = 5
rstate = 0
njobs = -1

# Set to existing filename to use previously tuned hyperparams instead new search
sv_tunes_fname = None

# Load and prepare data
Y_train_all = pd.read_csv('y_train.csv', sep=';', header=None)[0].values
X_train_all = pd.read_csv('x_train.csv', sep=';')
X_test_all = pd.read_csv('x_test.csv', sep=';')

# Feature engineering
def add_features(df):
    df['fractionOfBonusScore'] = df['totalBonusScore']/df['totalScore']
    df['ScorePerLvl'] = df['totalScore']/df['numberOfAttemptedLevels']
    df['StarsPerLvl'] = df['totalStarsCount']/df['numberOfAttemptedLevels']
    df['AttemptsPerDay'] = df['totalNumOfAttempts']/df['numberOfDaysActuallyPlayed']
    df['RegLvl_vs_Highest'] = (df['totalNumOfAttempts'] - df['attemptsOnTheHighestLevel'])/\
           ((df['numberOfAttemptedLevels'] - 1) * df['attemptsOnTheHighestLevel'])
    df['BoosterUtilityRate'] = df['fractionOfUsefullBoosters'] * df['totalNumOfAttempts'] / df['numberOfAttemptedLevels']
    df.fillna(value=0, axis=0, inplace=True)
    return df
    
X_train_all = add_features(X_train_all)
X_test_all = add_features(X_test_all)
nfeat = np.sqrt(len(X_train_all.columns))

# Define objective to minimize for hyperopt search
def cv_rf_perf(hyperpars):
    cv_auc_train = []
    cv_auc_test = []
    cv_loss_train = []
    cv_loss_test = []
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=rstate)
    for train_index, test_index in cv.split(X=X_train_all, y=Y_train_all):
        X_train = X_train_all.iloc[train_index,:]
        X_test = X_train_all.iloc[test_index,:]
        Y_train = Y_train_all[train_index]
        Y_test = Y_train_all[test_index]
        pars = {'n_estimators': int(hyperpars.get('ntrees',100)),
                'max_features': int(hyperpars.get('max_features',nfeat)),
                'max_depth': int(hyperpars.get('max_depth',3)),
                'min_samples_split': int(hyperpars.get('min_samples_split',2)),
                'min_samples_leaf': int(hyperpars.get('min_samples_leaf',1)),
                'random_state': rstate,
                'n_jobs': njobs
        }
        clf = RFC(**pars)
        clf.fit(X=X_train, y=Y_train)
        pos_cl_ind = int(np.where(clf.classes_==1)[0])
        Y_train_fcs = clf.predict_proba(X=X_train)[:,pos_cl_ind]
        Y_test_fcs = clf.predict_proba(X=X_test)[:,pos_cl_ind]
        cv_auc_train.append(roc_auc_score(y_true=Y_train, y_score=Y_train_fcs))
        cv_auc_test.append(roc_auc_score(y_true=Y_test, y_score=Y_test_fcs))
        cv_loss_train.append(log_loss(y_true=Y_train, y_pred=Y_train_fcs))
        cv_loss_test.append(log_loss(y_true=Y_test, y_pred=Y_test_fcs))
    return {'auc_train': np.mean(cv_auc_train),
            'auc_test': np.mean(cv_auc_test),
            'logloss_train': np.mean(cv_loss_train),
            'logloss_test': np.mean(cv_loss_test)}
    
def rf_logloss(hyperpars):
    res_perf = cv_rf_perf(hyperpars)
    print 'Hyperparameters: ' + str(hyperpars)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['logloss_test']

if sv_tunes_fname is None:
    hp_space = {
      'ntrees': hp.quniform('ntrees', 400, 600, 10),
      'max_features': hp.quniform('max_features', np.floor(nfeat), np.ceil(2*nfeat), 1),
      'max_depth': hp.qloguniform('max_depth', np.log(3), np.log(9), 1),
      'min_samples_split': hp.quniform('min_samples_split', 5, 20, 1),
      'min_samples_leaf': hp.quniform('min_samples_leaf', 20, 40, 1)
    }
    trials = Trials()
    beg = timer()
    RF_tune = fmin(fn=rf_logloss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=1500)
    tm_spent = timer() - beg
    print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec\nBest found hyperparameters:'
    print str(RF_tune) + '\n'
    save_vars(fname='RF_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
              varlst=['RF_tune','trials','hp_space','tm_spent'])
else:
    load_vars(fname=sv_tunes_fname)

clf = RFC(n_estimators=int(RF_tune.get('ntrees',100)), max_depth=int(RF_tune.get('max_depth',3)),
          min_samples_split=int(RF_tune.get('min_samples_split',2)),
          min_samples_leaf=int(RF_tune.get('min_samples_leaf',1)),
          max_features=int(RF_tune.get('max_features',nfeat)),
          n_jobs=njobs, random_state=rstate)

clf.fit(X=X_train_all, y=Y_train_all)
pos_cl_ind = int(np.where(clf.classes_==1)[0])
Y_test_fcs = clf.predict_proba(X=X_test_all)[:,pos_cl_ind]
np.savetxt(fname='answerRF_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')










