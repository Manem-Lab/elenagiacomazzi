from random import random
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import helper_prediction as helper
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import boto3
import helper_data_load

local = False #
if local:
    folder = "Results"
else:
    folder = "eval/preds"

# Evaluation helper functions
def rename_cols_for_plotting(result_discovery, method_list):
    rename_dict = dict(zip(result_discovery.columns, method_list))
    result_discovery.rename(columns=rename_dict, inplace=True)
    return result_discovery

def rename_for_plotting(result_nr_df):
    list_methods_naming = ["Nr. Features", "Ensemble R.", "Multivariate R.", "Random Forest R.", "SVR", "Lasso R.", "ElasticNet R.", "Bagging R.", "GradiendBoosting R."]
    result_nr_df_renamed = rename_cols_for_plotting(result_nr_df, list_methods_naming)
    return result_nr_df_renamed

def plot_result(df, title="", save=False):
    fig_size = (10,5)
    ax = df.plot(x="Nr. Features", title=title, figsize=fig_size, grid=True) #, xticks=range(1,100)
    ax.set_xlabel("Nr. Features")
    ax.set_ylabel("C-Index")
    # Save plots
    if save:
        file_name = title + "_std_scl_1-100.png"
        fig = ax.get_figure()
        if local:
            fig.savefig('../../Plots/' + file_name, dpi=150)
        else:
            s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
            ENDPOINT_URL = 'https://s3.valeria.science'
            bucket = 'oncotechdata'
            fig.savefig(f's3://{bucket}/plots/' + file_name, dpi=150, storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})

def get_nr_feat_per_method(df):
    df_best_model_nr_feat = pd.DataFrame(columns=["Method Name", "C-Index", "Nr. Features"])
    for col in df.columns:
        #print("Method: ", col)
        nr_feat_best = df[col].idxmax()
        c_idx_best = df[col].max()
        #print("For method {} the best number of features is {} with c-index {}.".format(col, nr_feat_best, c_idx_best))
        new_row = {"Method Name": col, "C-Index": c_idx_best, "Nr. Features": nr_feat_best}

        df_best_model_nr_feat = pd.concat([df_best_model_nr_feat, pd.DataFrame([new_row])], ignore_index=True)
    return df_best_model_nr_feat

#### Train Model
def train_final_model_test_on_validation_cohort(model, train_df, test_df, feat_sel_meth, nr_feat, random_state=42, ensemble=False, nr_solutions=0):
    # print("In train_final", nr_solutions)
    X_train = train_df.loc[:, train_df.columns != 'os_days']
    y_train = train_df.loc[:, "os_days"]

    X_test = test_df.loc[:, test_df.columns != 'os_days']
    y_test = test_df.loc[:, "os_days"]

    nr_feat = int(nr_feat)

    if ensemble:
        pred_list = []
        for nr_f in range(1, nr_feat+1):
            model = LinearRegression()
            model, y_pred = helper.select_feats(feat_sel_meth, X_train, y_train, X_test, model, nr_f, nr_solutions, random_state)
            pred_list.append(y_pred) # should have len of nr_feat
        
        preds_array = np.array(pred_list)
        y_predict = np.average(preds_array, axis = 0)
    else:
        model, y_predict = helper.select_feats(feat_sel_meth, X_train, y_train, X_test, model, nr_feat, nr_solutions, random_state)

    if str(model) in ["ElasticNetCV(random_state=42)", "LassoCV(random_state=42)"]:
        print(str(model))
        #print(model.alpha_)
    c_idx = concordance_index(y_test, y_predict)
    return c_idx, y_predict

def train_disc_n_validate(train_df, test_df, methods_dict, feat_sel_meth, nr_sol, val_set):
    #rint("in train_disc_n_vali", nr_sol)
    ensemble = False
    df_validation = pd.DataFrame(columns=["Method Name", "C-Index on " + val_set + " Test", "Nr. Features"])
    df_predictions_vali = pd.DataFrame(columns=["Method_Name", "Nr_feat", "Pred on " + val_set + " vali with "+feat_sel_meth+str(nr_sol)])
    for i in range(len(methods_dict)):
        keys_list = list(methods_dict.keys())
        feat_nr_temp = list(methods_dict.values())[i]
        method_temp = keys_list[i]
        if method_temp == "Ensemble Lin. R.":
            ensemble = True
        c_idx_tested_on_val_set, y_pred_vali = train_final_model_test_on_validation_cohort(method_temp, train_df, test_df, feat_sel_meth, feat_nr_temp, ensemble=ensemble, nr_solutions=nr_sol)
        
        new_row = {"Method Name": str(method_temp), "C-Index on " + val_set + " Test": c_idx_tested_on_val_set, "Nr. Features": feat_nr_temp}
        df_validation = pd.concat([df_validation, pd.DataFrame([new_row])], ignore_index=True)
        new_df = pd.DataFrame({"Method_Name": str(method_temp), "Nr_feat":feat_nr_temp, "Pred on " + val_set + " vali with "+feat_sel_meth+str(nr_sol):y_pred_vali})
        df_predictions_vali = pd.concat([df_predictions_vali, new_df])
    file_name = "preds_"+ feat_sel_meth+str(nr_sol)+ "on_vali_"+val_set+".csv"
    helper_data_load.save_csv(df_predictions_vali, file_name=file_name,local=local, folder=folder)
    return df_validation

def get_meth_feat_dict(df_dicovery_mthod_nr_feat, random_state):
    meth_feat_dict = {}
    dict_map_str_method = {"SVR": SVR(kernel="linear"), "Multivariate R.": LinearRegression(), "Random Forest R.": RandomForestRegressor(n_estimators=100, random_state=random_state), "Bagging R.": BaggingRegressor(random_state=random_state),
                            "GradiendBoosting R.": GradientBoostingRegressor(random_state=random_state), "ElasticNet R.": ElasticNetCV(random_state=random_state),
                            "Ensemble R.": "Ensemble Lin. R.", "Lasso R.": linear_model.LassoCV(random_state=random_state)}
    
    temp_dict = df_dicovery_mthod_nr_feat[["Method Name", "Nr. Features"]].to_dict('records')
    for method in range(len(temp_dict)):
        temp_meth = list(temp_dict[method].values())[0]
        feat_nr = list(temp_dict[method].values())[1]
        meth_func = dict_map_str_method[temp_meth]
        meth_feat_dict[meth_func] = feat_nr
    return meth_feat_dict

def prep_method_name(df, random_state, name_to_func):
    random_state = str(random_state)
    dict_map_str_method = {"SVR": "SVR(kernel='linear')", "Multivariate R.": "LinearRegression()", "Random Forest R.": "RandomForestRegressor(random_state="+random_state+")", "Bagging R.": "BaggingRegressor(random_state="+random_state+")",
                                "GradiendBoosting R.": "GradientBoostingRegressor(random_state="+random_state+")", "ElasticNet R.": "ElasticNetCV(random_state="+random_state+")",
                                "Ensemble R.": "Ensemble Lin. R.", "Lasso R.": "LassoCV(random_state="+random_state+")"}
    if name_to_func:
        dict_rename = dict_map_str_method
        df_temp = df.set_index("Method Name")
    else: 
        dict_rename = {v: k for k, v in dict_map_str_method.items()}
        df = df.reset_index()
        df_temp = df.set_index("Method Name")
    df_temp_plot = df_temp.rename(index=dict_rename)
    df_temp_plot.index = df_temp_plot.index.map(str)
    return df_temp_plot

def plot_vali_results(df_vali_on, discovery_methods_nr_feat, val_set, feat_sel_method, nr_sol, random_state, save_plots=False):
    if val_set=="CHUM": disc_set = "IUCPQ"
    elif val_set == "IUCPQ": disc_set = "CHUM"
    df_vali_on = prep_method_name(df_vali_on, random_state=random_state, name_to_func=False)
    discovery_methods_nr_feat.rename(columns={"C-Index":"C-Index " + disc_set + " Discovery"}, inplace=True)
    df_disc = discovery_methods_nr_feat.reset_index()
    df_vali_on = df_vali_on.reset_index()
    df_vali_on_merge = df_vali_on.merge(df_disc, on="Method Name")

    # Plotting
    colors = {"C-Index IUCPQ Discovery":"darkorange", "C-Index on CHUM Test":"tan"}
    vali_plot = df_vali_on_merge[["C-Index "+disc_set+" Discovery", "C-Index on "+val_set+" Test"]].plot.bar(rot=70, figsize=(10,3), color=colors, title=feat_sel_method+str(nr_sol), grid=True)

    #vali_plot = df_vali_on[["C-Index on "+ val_set +" Test"]].plot.bar(rot=70, figsize=(10,3), grid=True)
    vali_plot.legend(loc='lower right')
    vali_plot.set_ylabel("C-Index")
    if save_plots:
        if local:
            fig = vali_plot.get_figure()
            fig.savefig('../../Plots/os_days_validation_on_' + val_set +'_'+feat_sel_method+str(nr_sol)+'_1-100.png', dpi=150, bbox_inches = 'tight')  
        else:
            s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
            ENDPOINT_URL = 'https://s3.valeria.science'
            bucket = 'oncotechdata'
            fig = vali_plot.get_figure()
            fig.savefig(f's3://{bucket}/plots/os_days_validation_on_' + val_set +'_'+feat_sel_method+str(nr_sol)+'_1-100.png', dpi=150, bbox_inches = 'tight', storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})  

def plot_vali_results_in_one(df_vali_on_1, df_vali_on_2, df_disc_on_1, df_disc_on_2, feat_sel_method, nr_sol, save_plots=False):
    df_vali_on_1 = df_vali_on_1.reset_index()
    df_vali_on_2 = df_vali_on_2.reset_index()
    df_disc_on_1 = df_disc_on_1.reset_index()
    df_disc_on_2 = df_disc_on_2.reset_index()
    df_vali_on_merge = df_vali_on_1.merge(df_vali_on_2, on="Method Name").merge(df_disc_on_1, on="Method Name").merge(df_disc_on_2, on="Method Name")
    df_vali_on_merge = df_vali_on_merge.set_index("Method Name")
    colors = {"C-Index CHUM Discovery":"blue", "C-Index on IUCPQ Test":"tab:blue", "C-Index IUCPQ Discovery":"darkorange", "C-Index on CHUM Test":"tan"}
    vali_plot = df_vali_on_merge[["C-Index CHUM Discovery", "C-Index on IUCPQ Test", "C-Index IUCPQ Discovery", "C-Index on CHUM Test"]].plot.bar(rot=70, figsize=(10,3), color=colors, title=feat_sel_method+str(nr_sol), grid=True)
    vali_plot.set_ylabel("C-Index")
    vali_plot.legend(loc="lower right")
    
    if save_plots:
        if local:
            fig = vali_plot.get_figure()
            fig.savefig('../../Plots/os_days_validation_both_in_one_plot_'+feat_sel_method+str(nr_sol)+'_1-100.png', dpi=150, bbox_inches = 'tight')  
        else:
            s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
            ENDPOINT_URL = 'https://s3.valeria.science'
            bucket = 'oncotechdata'
            fig = vali_plot.get_figure()
            fig.savefig(f's3://{bucket}/plots/os_days_validation_both_in_one_plot_'+feat_sel_method+str(nr_sol)+'_1-100.png', dpi=150, bbox_inches = 'tight', storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})  
