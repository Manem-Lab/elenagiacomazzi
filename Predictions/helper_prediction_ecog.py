# Imports
import pandas as pd
import numpy as np
import os
from pymrmre import mrmr
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from lifelines import CoxPHFitter
from sklearn.inspection import permutation_importance
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import helper_data_load

import helper_evaluation as eval

local = False

#### Feature Selection
# through correlation
def get_most_corr_feat(X_train, y_train, nr_solutions):
    if nr_solutions == -1:
        method = "spearman"
    elif nr_solutions == -2:
        method = "pearson"
    corr_df = X_train.copy()
    corr_df["os_days"] = y_train
    corr = corr_df.corr(method=method)
    corr["os_days_abs"] = corr["os_days"].abs()
    top_feat = corr.os_days_abs.sort_values(ascending=False)[1:]
    return top_feat

# through mrmr
def get_feat_mrmr(df, y, nr_feat, solution_count):
    y = pd.DataFrame(y)
    solutions = mrmr.mrmr_ensemble(features=df, targets=y, solution_length=nr_feat, solution_count=solution_count)
    #selected_features = mrmr_regression(df, y, K = nr_feat+1)
    return solutions

# through mutual information maximazation
def fit_apply_kbest(X_train_in, y_train_in, score_func, nr_feat, X_test):
    #print("in fit apply - X_train ", X_train_in.columns)
    data_transformer = SelectKBest(score_func=score_func, k=nr_feat).fit(X_train_in, y_train_in)
    #X_train_in = X_train_in.reshape(-1, 1)
    #y_train_in = y_train_in.reshape(-1, 1)
    cols_idxs = data_transformer.get_support(indices=True)
    features_df_new = X_train_in.iloc[:,cols_idxs]
    feat_to_use = list(features_df_new.columns)
    #print(feat_to_use)
    return feat_to_use

# through permutation random forest
def perm_feat_sel(X_train, y_train, random_state):
    regr = RandomForestRegressor(random_state=random_state)
    regr.fit(X_train, y_train)
    result = permutation_importance(regr, X_train, y_train, random_state=random_state)
    importances = result.importances_mean
    perm_forest_importances = pd.Series(importances, index=X_train.columns)
    top_feat = perm_forest_importances.sort_values(ascending=False)
    return top_feat

# through importance of random forest
def randFor_feat_sel(X_train, y_train, random_state):
    max_depth_list = [int(x) for x in np.linspace(10, 110, num = 6)]
    max_depth_list.append(None)
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': max_depth_list,
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]}
    regr = RandomForestRegressor(random_state=random_state)
    model = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=random_state, n_jobs = -1)
    model.fit(X_train, y_train)
    importances = model.best_estimator_.feature_importances_
    perm_forest_importances = pd.Series(importances, index=X_train.columns)
    top_feat = perm_forest_importances.sort_values(ascending=False)
    return top_feat

# through cox regression
def cox_feat_sel(X_train, y_train, random_state):
    print("in cox_feat_sel")
    data_df = X_train.copy()
    data_df["time"] = y_train
    data_df["status"] = 1
    feat_rank_df = pd.DataFrame(columns=["feature", "c_idx"])
    cph = CoxPHFitter()
    cols_to_use = list(X_train.columns)#+[ "time", "status"]
    print(cols_to_use)
    cph.fit(data_df[cols_to_use+[ "time", "status"]], 'time', 'status',show_progress=True)
    print(cph.summary)
    
    # calculate the c index for each features individually
    for feat in X_train.columns:
        cph = CoxPHFitter()
        cph.fit(data_df[[feat, "time", "status"]], 'time', 'status')
        print(cph.summary)
        c_temp = cph.concordance_index_
        dict_temp = {"feature":feat, "c_idx":c_temp}
        df_temp = pd.DataFrame(dict_temp, index=[0])
        feat_rank_df = pd.concat([feat_rank_df, df_temp])
        break
    # sort the feature according to their prediction strength
    top_feat = feat_rank_df.sort_values(by="c_idx",ascending=False)
    top_feat = top_feat.set_index("feature")
    print(top_feat)
    #return top_feat    

def select_feats(feat_sel_method, X_train, y_train, X_test, model, nr_feat, nr_solutions=0, random_state=42):
    print("in select_feat: ", type(model))
    # remove ecog and pdl1 from X_train
    columns_X_train = list(X_train.columns)
    #print("list cols X-train before", columns_X_train)
    for feat in ["pdl1_tps", "ecog_status"]:
        columns_X_train.remove(feat)
    #print("list cols X-train after", columns_X_train)
    X_train_feat_sel = X_train[columns_X_train]
    # calculate most correlating features of that fold
    if feat_sel_method == "mrmr":
        ### mrmr
        solution_count = nr_solutions
        sol_mrmr_pred_list = []
        mrmr_solutions = get_feat_mrmr(X_train_feat_sel, y_train, nr_feat, solution_count=solution_count)
        for i in range(solution_count):
            feat_to_use = mrmr_solutions.iloc[0][i]
            #print("feat_to_use before fit and predict ", feat_to_use)
            model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_method)
            sol_mrmr_pred_list.append(y_predict)

        sol_mrmr_pred_array = np.array(sol_mrmr_pred_list)
        y_predict = np.average(sol_mrmr_pred_array, axis = 0)
            
    elif feat_sel_method == "corr":
        top_feat = get_most_corr_feat(X_train=X_train_feat_sel, y_train=y_train, nr_solutions=nr_solutions)
        feat_to_use = list(top_feat[:nr_feat].index)
        print("Feat to use ",feat_to_use)
        print("nr feat ",nr_feat)
        model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state,feat_sel_method)

    elif feat_sel_method in  ["mim", "f_reg"]:
        if feat_sel_method == "mim":
            score_func = mutual_info_regression
        if feat_sel_method == "f_reg":
            score_func = f_regression
        #print(X_train_feat_sel.columns)
        feat_to_use = fit_apply_kbest(X_train_feat_sel, y_train, score_func, nr_feat, X_test)
        #print(X_train_new.columns)
        #feat_to_use = []
        print(feat_to_use)
        model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_method)

    elif feat_sel_method == "permut_randFor":
        top_feat = perm_feat_sel(X_train=X_train_feat_sel, y_train=y_train, random_state=random_state)
        feat_to_use = top_feat[:nr_feat].index
        model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_method)
    
    elif feat_sel_method == "randFor_feat_sel":
        top_feat = randFor_feat_sel(X_train=X_train_feat_sel, y_train=y_train, random_state=random_state)
        feat_to_use = top_feat[:nr_feat].index
        model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_method)

    elif feat_sel_method == "cox_reg":
        top_feat = cox_feat_sel(X_train=X_train_feat_sel, y_train=y_train, random_state=random_state)
        feat_to_use = top_feat[:nr_feat].index
        model_str, y_predict = fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_method)

    return model_str, y_predict
    

### Model Fitting
def fit_and_predict(model, X_train, y_train, feat_to_use, X_test, random_state, feat_sel_meth):
    print(feat_to_use)
    # add ecog and pdl1 to train set
    feat_to_use.append("pdl1_tps")
    feat_to_use.append("ecog_status")
    print("feat_to_use before fit and predict ", feat_to_use)
    print("in fit_and_predict: ", type(model))
    
    X_train = X_train[feat_to_use]
    X_test = X_test[feat_to_use]
    print("X_train and test in fit and pred")
    print(X_train.columns)
    print(X_test.columns)
    print(str(model))
    model_str = str(model)
    if str(model) in ["SVR()", "RandomForestRegressor(random_state=42)", "BaggingRegressor(random_state=42)", "GradientBoostingRegressor(random_state=42)"]:
        if str(model) == "BaggingRegressor(random_state=42)":
            random_grid = {'n_estimators': [100, 300, 500, 800, 1200], 
                        'max_features': [1, 2, 5, 10, 13], 
                        'max_samples': [5, 10, 25, 50, 100]}
        elif str(model) == "GradientBoostingRegressor(random_state=42)":
            random_grid = {'learning_rate': [0.01,0.02,0.03,0.04],
                        'subsample'    : [0.9, 0.5, 0.2, 0.1],
                        'n_estimators' : [100,500,1000, 1500],
                        'max_depth'    : [4,6,8,10]
                        }
        elif str(model) == "SVR()":
            random_grid = {'kernel': ('linear', 'rbf','poly'), 
                        'C':[1.5, 10],
                        'gamma': [1e-7, 1e-4],
                        'epsilon':[0.1,0.2,0.5,0.3]}
        elif str(model) == "RandomForestRegressor(random_state=42)":
            max_depth_list = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth_list.append(None)
            random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': max_depth_list,
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'bootstrap': [True, False]}
        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=random_state, n_jobs = -1)
    print("in fit_right before error: ", type(model))
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    return model_str, y_predict
'''
def fit_and_predict_applied_feat(X_train, y_train, model, X_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model, y_predict
'''

def kFold_training(k_fold_nr, random_state, model, data_norm, nr_feat, ensemble=False, feat_sel_method="corr", nr_solutions=0, save_pred=False):
    print("in kFold_training: ", type(model))
    #### kFold for the evaluation of the best number of features and best method for the discovery cohort
    pred_list_one_fold = []
    c_idx_list = [] # to store c-index
    cv = KFold(n_splits=k_fold_nr, random_state=random_state, shuffle=True)
    y_test_list = []
    y_test_cv = []
    y_pred_cv = []
    for train_index, test_index in cv.split(data_norm):
        # get train and test sets for this fold
        X_train, X_test = data_norm.loc[train_index, data_norm.columns != 'os_days'], data_norm.loc[test_index, data_norm.columns != 'os_days']
        y_train, y_test = data_norm.loc[train_index, "os_days"], data_norm.loc[test_index, "os_days"]
        # calculate most correlating features of that fold
        model_str, y_predict = select_feats(feat_sel_method, X_train, y_train, X_test, model, nr_feat, nr_solutions, random_state)
        y_test_cv.append(y_test)
        y_pred_cv.append(y_predict)
        
    y_test_cv = [j for sub in y_test_cv for j in sub]
    y_pred_cv = [j for sub in y_pred_cv for j in sub]
    if ensemble: 
        return  y_pred_cv, y_test_cv

    else:
        if save_pred:
            df_preds = pd.DataFrame({"y_test_cv": y_test_cv, "y_pred_cv": y_pred_cv})
            file_name = "switch_cv_preds_"+str(model_str)+"_"+feat_sel_method+str(nr_solutions)+".csv"
            print(file_name)
            helper_data_load.save_csv(df_preds, file_name=file_name, local=local, folder="ecog/results/predictions")            #df_preds.to_csv("../../Data/Results/test_folder_pred/" + "cv_preds_"+str(model)+"_"+feat_sel_method+str(nr_solutions))
        else:
            c_idx = concordance_index(y_test_cv, y_pred_cv)
            #c_idx_list.append(c_idx)
            return c_idx

# Training the Models
def lin_reg_ensamble(data_norm, random_state, feat_sel_meth="corr", nr_sol=0):
    df_result_ens = pd.DataFrame(columns=["nr_feat","c_idx_ens"])
    # calculate for 1 to k_max features
    if feat_sel_meth == "mrmr":
        k_max = 10
    else:
        k_max = 50 # 99 for (IC)
    for i in range(1,k_max): 
        pred_list = []
        for nr_feat in range(1, i+1):
            lrmodel = LinearRegression()
            print("in lin_reg_ens: ", type(lrmodel))
            k_fold = 5
            # Train model with CHUM data and 5-fold and linear regression
            y_pred_cv, y_test_cv = kFold_training(k_fold, random_state, lrmodel, data_norm, nr_feat, ensemble=True, feat_sel_method=feat_sel_meth, nr_solutions=nr_sol)
            pred_list.append(y_pred_cv) # should have len of nr_feat
  
        preds_array = np.array(pred_list)
        print(preds_array.shape)
        avg_pred = np.average(preds_array, axis = 0) # THAT is the prediction
        print(avg_pred.shape)
        c_idx_one_cv = concordance_index(y_test_cv, avg_pred)

        # add errors to df
        row_dict = {"nr_feat": int(i), "c_idx_ens": c_idx_one_cv} 
        df_temp = pd.DataFrame(row_dict, index=[0])
        df_result_ens = pd.concat([df_result_ens,df_temp])

    # here run with best nr of features
    df_result_ens.reset_index(inplace=True, drop=True)
    dict_keys = df_result_ens.columns
    df_result_ens = df_result_ens.astype(float)
    print("echo {}".format(dict_keys[1]))
    print("echo {}".format(df_result_ens[dict_keys[1]]))
    idx_best_feat = df_result_ens[dict_keys[1]].idxmax()
    best_nr_feat = int(df_result_ens[df_result_ens.index == idx_best_feat]["nr_feat"])

    pred_list_best = []
    for feat in range(1, best_nr_feat+1):
        y_pred_cv, y_test_cv = kFold_training(k_fold, random_state, lrmodel, data_norm, feat, ensemble=True, feat_sel_method=feat_sel_meth, nr_solutions=nr_sol, save_pred=True)
        pred_list_best.append(y_pred_cv)

    preds_array_best = np.array(pred_list_best)
    avg_pred = np.average(preds_array_best, axis = 0) # THAT is the prediction
    ### SAVE
    df_preds = pd.DataFrame({"y_test_cv": y_test_cv, "y_pred_cv": avg_pred})
    file_name = "switch_cv_preds_ensemble_"+str(lrmodel)+"_"+feat_sel_meth+str(nr_sol)+".csv"
    helper_data_load.save_csv(df_preds, file_name=file_name, local=local, folder="ecog/results/predictions") 

    return df_result_ens

def not_ensemble_methods(method, df_result, data_norm, random_state, feat_sel_meth="corr", nr_sol=0):
    # calculate for 1 to 50 features
    if feat_sel_meth == "mrmr":
        k_max = 11
    else:
        k_max = 50 # 99 for (IC)
    for nr_feat in range(1,k_max): 
        print("NEW LOOP################################")
        print(nr_feat)
        k_fold = 5
        c_idx = kFold_training(k_fold, random_state, method, data_norm, nr_feat, ensemble=False, feat_sel_method=feat_sel_meth, nr_solutions=nr_sol)

        # add errors to df
        dict_keys = df_result.columns
        dict_values = [int(nr_feat), c_idx]
        row_dict = dict(zip(dict_keys, dict_values))
        df_temp = pd.DataFrame(row_dict, index=[0])
        df_result = pd.concat([df_result,df_temp])
    df_result = df_result.reset_index(drop=True)
    df_result = df_result.astype(float)

    ### check here for best nr of features for that model
    print("save string: "+ str(method)+feat_sel_meth+str(nr_sol))
    print("echo {}".format(dict_keys[1]))
    print("echo {}".format(df_result[dict_keys[1]]))
    idx_best_feat = df_result[dict_keys[1]].idxmax()
    best_nr_feat = int(df_result[df_result.index == idx_best_feat]["nr_feat"])
    print("Number of best feat: ", best_nr_feat)
    kFold_training(k_fold, random_state, method, data_norm, best_nr_feat, ensemble=False, feat_sel_method=feat_sel_meth, nr_solutions=nr_sol, save_pred=True)

    return df_result

# Run training for different methods
def basic_methods(data_norm, FEAT_SEL_METH, NR_SOL, random_state):
    # Ensemble Linear Regression
    df_results_ens = lin_reg_ensamble(data_norm, random_state=random_state, feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained Ensemble")
    os.system("echo Trained Ensemble")

    # Multiple Linear Regression
    df_result_multi_lr = pd.DataFrame(columns=["nr_feat","c_idx_multi"])
    lr_model = LinearRegression()
    df_result_multi_lr = not_ensemble_methods(lr_model, df_result_multi_lr, data_norm, random_state=random_state, feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained Multilinear")
    os.system("echo Trained Multilinear")
    
    # Random Forest Regressor
    df_result_random_forest = pd.DataFrame(columns=["nr_feat","c_idx_random_for"])
    rf_model = RandomForestRegressor(random_state=random_state)
    df_result_random_forest = not_ensemble_methods(rf_model, df_result_random_forest, data_norm, random_state=random_state, feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained random forest")
    os.system("echo Trained random forest")

    # SVR
    df_result_SVR= pd.DataFrame(columns=["nr_feat","c_idx_svr"])
    rf_model = SVR()
    df_result_SVR = not_ensemble_methods(rf_model, df_result_SVR, data_norm, random_state=random_state, feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained SVR")
    os.system("echo Trained SVR")

    result = pd.merge(df_results_ens, df_result_multi_lr, on='nr_feat').merge(df_result_random_forest, on='nr_feat').merge(df_result_SVR, on='nr_feat') # 
    return result

def more_methods(data_norm, FEAT_SEL_METH, NR_SOL, random_state):
    # Lasso Regression
    df_result_lasso = pd.DataFrame(columns=["nr_feat","c_idx_lasso"])
    lasso_model = linear_model.LassoCV(random_state=random_state)
    df_result_lasso = not_ensemble_methods(lasso_model, df_result_lasso, data_norm, random_state=random_state, feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained Lasso")
    os.system("echo Trained Lasso")

    # ElasticNet regression
    df_result_elastic = pd.DataFrame(columns=["nr_feat","c_idx_elastic"])
    elastic_model = ElasticNetCV(random_state=random_state)
    df_result_elastic = not_ensemble_methods(elastic_model, df_result_elastic, data_norm, random_state=random_state,  feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained ElasticNet")
    os.system("echo Trained ElasticNet")

    # Bagging regression
    df_result_bagging_reg = pd.DataFrame(columns=["nr_feat","c_idx_bagging"])
    bagging_model = BaggingRegressor(random_state=random_state)
    df_result_bagging_reg = not_ensemble_methods(bagging_model, df_result_bagging_reg, data_norm, random_state=random_state,  feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained Bagging")
    os.system("echo Trained Bagging")

    # Boosting regression
    df_result_boosting_reg = pd.DataFrame(columns=["nr_feat","c_idx_boosting"])
    boosting_model = GradientBoostingRegressor(random_state=random_state)
    df_result_boosting_reg = not_ensemble_methods(boosting_model, df_result_boosting_reg, data_norm, random_state=random_state,  feat_sel_meth=FEAT_SEL_METH, nr_sol=NR_SOL)
    print("Trained Boosting")
    os.system("echo Trained Boosting")

    result_new = pd.merge(df_result_lasso, df_result_elastic, on='nr_feat').merge(df_result_bagging_reg, on='nr_feat').merge(df_result_boosting_reg, on='nr_feat')
    return result_new

def all_methods(data_norm, random_state, FEAT_SEL_METH, NR_SOL):
    os.system("echo in all_methods")
    result_basic_methods = basic_methods(data_norm, random_state=random_state, FEAT_SEL_METH=FEAT_SEL_METH, NR_SOL=NR_SOL)
    result_new_methods = more_methods(data_norm, random_state=random_state, FEAT_SEL_METH=FEAT_SEL_METH, NR_SOL=NR_SOL)
    result_all = pd.merge(result_basic_methods, result_new_methods, on='nr_feat')
    return result_all