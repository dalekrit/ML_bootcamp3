# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:47:53 2017

@author: Dmitriy
"""

import shelve
import numpy as np
import pandas as pd
from time import strftime, localtime
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier as SGDC

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
niter = 500
reg_type = 'elasticnet'
loss_type = 'log'

# Set to existing filename to use previously tuned hyperparams instead new search
sv_tunes_fname = None

# Load and prepare data
Y_train_all = pd.read_csv('y_train.csv', sep=';', header=None)[0].values
X_train_all = pd.read_csv('x_train.csv', sep=';')
X_test_all = pd.read_csv('x_test.csv', sep=';')

# Feature engineering & normalization
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
feature_cols = X_train_all.columns.tolist()

scaler = StandardScaler()
X_train_all = scaler.fit_transform(X=X_train_all)
X_test_all = scaler.transform(X=X_test_all)

# Define objective to minimize for hyperopt search
def cv_lm_perf(hyperpars):
    cv_auc_train = []
    cv_auc_test = []
    cv_loss_train = []
    cv_loss_test = []
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=rstate)
    for train_index, test_index in cv.split(X=X_train_all, y=Y_train_all):
        X_train = X_train_all[train_index,:]
        X_test = X_train_all[test_index,:]
        Y_train = Y_train_all[train_index]
        Y_test = Y_train_all[test_index]
        pars = {'loss': loss_type,
                'penalty': reg_type,
                'alpha': hyperpars.get('alpha',0.0001),
                'l1_ratio': hyperpars.get('l1_ratio',0.15),
                'random_state': rstate,
                'n_jobs': njobs,
                'n_iter': niter
        }
        clf = SGDC(**pars)
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
    
def lm_logloss(hyperpars):
    res_perf = cv_lm_perf(hyperpars)
    print 'Hyperparameters: ' + str(hyperpars)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['logloss_test']

if sv_tunes_fname is None:
    hp_space = {
      'alpha': hp.qloguniform('alpha', np.log(0.00001), np.log(0.1), 0.00001),
      'l1_ratio': hp.quniform('l1_ratio', 0, 1, 0.001)
    }
    trials = Trials()
    beg = timer()
    LM_tune = fmin(fn=lm_logloss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=1000)
    LM_tune['loss'] = loss_type
    LM_tune['penalty'] = reg_type
    LM_tune['n_iter'] = niter
    tm_spent = timer() - beg
    print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec\nBest found hyperparameters:'
    print str(LM_tune) + '\n'
    save_vars(fname='LM_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
              varlst=['LM_tune','trials','hp_space','tm_spent'])
else:
    load_vars(fname=sv_tunes_fname)

clf = SGDC(loss=LM_tune.get('loss','log'), penalty=LM_tune.get('penalty','elasticnet'), alpha=LM_tune.get('alpha',0.0001),
           n_iter=LM_tune.get('n_iter',5), l1_ratio=LM_tune.get('l1_ratio',0.15), n_jobs=njobs, random_state=rstate)

clf.fit(X=X_train_all, y=Y_train_all)
pos_cl_ind = int(np.where(clf.classes_==1)[0])
Y_test_fcs = clf.predict_proba(X=X_test_all)[:,pos_cl_ind]
np.savetxt(fname='answerLM_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')







