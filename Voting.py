# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:08:42 2017

@author: Dmitriy
"""

import shelve
import numpy as np
import pandas as pd
from time import strftime, localtime
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import VotingClassifier as VC
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.ensemble import RandomForestClassifier as RFC
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
    
# Set CV params
nfolds = 5 # both for hyperopt & final GB performance
rstate = 0
njobs = -1

# Set to existing filename to use previously tuned hyperparams (for voting weights) instead new search
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
feature_cols = X_train_all.columns.tolist()
nfeat = np.sqrt(len(feature_cols))

scaler = StandardScaler()
X_train_all = scaler.fit_transform(X=X_train_all)
X_test_all = scaler.transform(X=X_test_all)

# Prepare smaller grids for additional tuning
load_vars('BEST_GB_PARAMS.out')
GB_eta_rg = (np.floor(700*eta)/1000, np.ceil(1300*eta)/1000)
GB_ntree_rg = (np.floor(0.09*best_ntrees)*10, np.ceil(0.11*best_ntrees)*10)
GB_msl_rg = (np.floor(0.9*min_samples_leaf), np.ceil(1.1*min_samples_leaf))
GB_mss_rg = (np.floor(0.8*min_samples_split), np.ceil(1.2*min_samples_split))
GB_max_depth = max_depth
GB_subsample = subsample
GB_loss = loss

load_vars('RF_TUNE_RES.out')
RF_dpth_rg = (max(RF_tune.get('max_depth',3)-2, 2), RF_tune.get('max_depth',3)+2)
RF_mf_rg = (max(RF_tune.get('max_features',nfeat)-2, 2), min(RF_tune.get('max_features',nfeat)+1, len(feature_cols)))
RF_msl_rg = (np.floor(0.9*RF_tune.get('min_samples_leaf',1)), np.ceil(1.1*RF_tune.get('min_samples_leaf',1)))
RF_mss_rg = (np.floor(0.8*RF_tune.get('min_samples_split',1)), np.ceil(1.2*RF_tune.get('min_samples_split',1)))
RF_ntree_rg = (np.floor(0.095*RF_tune.get('ntrees',100))*10, np.ceil(0.105*RF_tune.get('ntrees',100))*10)

load_vars('LM_TUNE_RES.out')
LM_alpha_rg = (np.floor(70000*LM_tune.get('alpha',0.0001))/100000, np.ceil(130000*LM_tune.get('alpha',0.0001))/100000)
LM_l1r_rg = (np.floor(800*LM_tune.get('l1_ratio',0.15))/1000, np.ceil(1200*LM_tune.get('l1_ratio',0.15))/1000)
LM_loss = LM_tune.get('loss','log')
LM_reg = LM_tune.get('penalty','elasticnet')
LM_niter = LM_tune.get('n_iter',5)

# Define objective to minimize for hyperopt search
def cv_voting_perf(hyperpars):
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
        GB_pars = {'n_estimators': int(hyperpars.get('GB_ntrees',100)),
                   'learning_rate': hyperpars.get('GB_eta',0.1),
                   'max_depth': GB_max_depth,
                   'min_samples_split': int(hyperpars.get('GB_min_samples_split',2)),
                   'min_samples_leaf': int(hyperpars.get('GB_min_samples_leaf',1)),
                   'subsample': GB_subsample,
                   'loss': GB_loss,
                   'random_state': rstate
        }
        RF_pars = {'n_estimators': int(hyperpars.get('RF_ntrees',100)),
                   'max_features': int(hyperpars.get('RF_max_features',nfeat)),
                   'max_depth': int(hyperpars.get('RF_max_depth',3)),
                   'min_samples_split': int(hyperpars.get('RF_min_samples_split',2)),
                   'min_samples_leaf': int(hyperpars.get('RF_min_samples_leaf',1)),
                   'random_state': rstate,
                   'n_jobs': njobs
        }
        LM_pars = {'loss': LM_loss,
                   'penalty': LM_reg,
                   'alpha': hyperpars.get('LM_alpha',0.0001),
                   'l1_ratio': hyperpars.get('LM_l1_ratio',0.15),
                   'random_state': rstate,
                   'n_jobs': njobs,
                   'n_iter': LM_niter
        }
        clf = VC(estimators=[('GB',GBC(**GB_pars)), ('RF',RFC(**RF_pars)), ('LM',SGDC(**LM_pars))], voting='soft',
                 weights=[hyperpars.get('GBW',1), hyperpars.get('RFW',1), hyperpars.get('LMW',1)])
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
    
def voting_logloss(hyperpars):
    res_perf = cv_voting_perf(hyperpars)
    print 'Hyperparameters: ' + str(hyperpars)
    print 'Performance: ' + str({k: round(v,5) for k, v in res_perf.items()}) + '\n'
    return res_perf['logloss_test']

if sv_tunes_fname is None:
    hp_space = {
       'GB_ntrees': hp.quniform('GB_ntrees', GB_ntree_rg[0], GB_ntree_rg[1], 10),
       'GB_eta': hp.quniform('GB_eta', GB_eta_rg[0], GB_eta_rg[1], 0.001),
       'GB_min_samples_split': hp.quniform('GB_min_samples_split', GB_mss_rg[0], GB_mss_rg[1], 1),
       'GB_min_samples_leaf': hp.quniform('GB_min_samples_leaf', GB_msl_rg[0], GB_msl_rg[1], 1),
       'RF_ntrees': hp.quniform('RF_ntrees', RF_ntree_rg[0], RF_ntree_rg[1], 10),
       'RF_max_features': hp.quniform('RF_max_features', RF_mf_rg[0], RF_mf_rg[1], 1),
       'RF_max_depth': hp.quniform('RF_max_depth', RF_dpth_rg[0], RF_dpth_rg[1], 1),
       'RF_min_samples_split': hp.quniform('RF_min_samples_split', RF_mss_rg[0], RF_mss_rg[1], 1),
       'RF_min_samples_leaf': hp.quniform('RF_min_samples_leaf', RF_msl_rg[0], RF_msl_rg[1], 1),
       'LM_alpha': hp.quniform('LM_alpha', LM_alpha_rg[0], LM_alpha_rg[1], 0.00001),
       'LM_l1_ratio': hp.quniform('LM_l1_ratio', LM_l1r_rg[0], LM_l1r_rg[1], 0.0001),
       'GBW': hp.quniform('GBW', 1, 6, 1),
       'RFW': hp.quniform('RFW', 0, 6, 1),
       'LMW': hp.quniform('LMW', 0, 6, 1)
    }
    trials = Trials()
    beg = timer()
    v_tune = fmin(fn=voting_logloss, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=500)
    tm_spent = timer() - beg
    print 'Time elapsed: ' + str(round(tm_spent,2)) + ' sec\nBest found hyperparameters:'
    print str(v_tune) + '\n'
    # Save tuning results
    save_vars(fname='VOTING_TUNE_RES_'+strftime("%d%m%Y%H%M%S", localtime())+'.out',
              varlst=['v_tune','GB_max_depth','GB_loss','GB_subsample','LM_loss','LM_reg','LM_niter'])
else:
    load_vars(fname=sv_tunes_fname)

# Estimate model with found hyperparameters on the whole training dataset
GB_clf = GBC(loss=GB_loss, learning_rate=v_tune.get('GB_eta',0.1), n_estimators=int(v_tune.get('GB_ntrees',100)),
             subsample=GB_subsample, max_depth=GB_max_depth, random_state=rstate,
             min_samples_split=int(v_tune.get('GB_min_samples_split',2)),
             min_samples_leaf=int(v_tune.get('GB_min_samples_leaf',1)))

RF_clf = RFC(n_estimators=int(v_tune.get('RF_ntrees',100)), max_depth=int(v_tune.get('RF_max_depth',3)),
             min_samples_split=int(v_tune.get('RF_min_samples_split',2)),
             min_samples_leaf=int(v_tune.get('RF_min_samples_leaf',1)),
             max_features=int(v_tune.get('RF_max_features',nfeat)),
             n_jobs=njobs, random_state=rstate)

LM_clf = SGDC(loss=LM_loss, penalty=LM_reg, alpha=v_tune.get('LM_alpha',0.0001), n_iter=LM_niter,
              l1_ratio=v_tune.get('LM_l1_ratio',0.15), n_jobs=njobs, random_state=rstate)

clf = VC(estimators=[('GB',GB_clf), ('RF',RF_clf), ('LM',LM_clf)], voting='soft',
         weights=[v_tune.get('GBW',1), v_tune.get('RFW',1), v_tune.get('LMW',1)])
clf.fit(X=X_train_all, y=Y_train_all)
pos_cl_ind = int(np.where(clf.classes_==1)[0])
Y_test_fcs = clf.predict_proba(X=X_test_all)[:,pos_cl_ind]
np.savetxt(fname='answerVOTING_'+strftime("%d%m%Y%H%M%S", localtime())+'.txt', X=Y_test_fcs, newline='\n')









