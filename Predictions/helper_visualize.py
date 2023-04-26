import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score, confusion_matrix

def calc_metric(real_col, pred_col, metric="c"):
    if metric == "c":
        return concordance_index(real_col, pred_col)
    elif metric == "r2":
        return r2_score(real_col, pred_col)
    elif metric in ["sensitivity", "specificity", "acc"]:
        cm1 = confusion_matrix(real_col,pred_col)
        if metric == "sensitivity":
            sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            return sensitivity1
        elif metric == "specificity":
            specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            return specificity1
        elif metric == "acc":
            total1=sum(sum(cm1))
            accuracy1=(cm1[0,0]+cm1[1,1])/total1
            return accuracy1
    

def rename_pred_col(df, model):
    rename_pred_col = {"y_pred_cv": "y_pred_cv"+"_"+model}
    df_temp_rename = df.rename(columns=rename_pred_col)
    return df_temp_rename

def eval_disc(file_path, model_temp, feat_sel_temp, metric_df, metric):
    df_temp = pd.read_csv(file_path, index_col=0)
    df_temp_renamed = rename_pred_col(df_temp, model_temp)
    c_temp = calc_metric(df_temp_renamed.y_test_cv, df_temp_renamed["y_pred_cv_"+model_temp], metric=metric)
    metric_df.loc[model_temp, feat_sel_temp] = c_temp
    metric_df = metric_df.astype(float)
    return metric_df

def eval_vali(file_path, model_temp, feat_sel_temp, corr_meth_temp, metric_df, metric, y_vali, vali_set):
    # load saved predictions
    df_temp = pd.read_csv(file_path, index_col=0)
    # get parts of df that is important for that calculation
    if model_temp == "ensemble_LinearRegression()":
        df_temp_meth = df_temp[df_temp.Method_Name == "Ensemble Lin. R."]
    elif model_temp == "ensemble_LogisticRegressionCV(random_state=42)":
        df_temp_meth = df_temp[df_temp.Method_Name == "Ensemble Log. R."]
    elif model_temp == "LogisticRegressionCV(random_state=42)":
        df_temp_meth = df_temp[df_temp.Method_Name == "LogisticRegression()"]
    elif model_temp == "SVC()":
        df_temp_meth = df_temp[df_temp.Method_Name == "SVC(kernel='linear')"]
    elif model_temp == "SVR()":
        df_temp_meth = df_temp[df_temp.Method_Name == "SVR(kernel='linear')"]
    else:
        df_temp_meth = df_temp[df_temp.Method_Name == model_temp]
    # define column name depending on feat_sel_meth
    if feat_sel_temp in ["corr-1", "corr-2", "mrmr5"]:
        col_name = "Pred on "+vali_set+" vali with "+feat_sel_temp        
    else:
        col_name = "Pred on "+vali_set+" vali with "+feat_sel_temp + str(corr_meth_temp)
    # Get a array of predictions
    y_pred = df_temp_meth[col_name]
    # Calculate the error
    c_temp = calc_metric(y_vali, y_pred, metric=metric)
    # Write error to df
    metric_df.loc[model_temp, feat_sel_temp] = c_temp
    metric_df = metric_df.astype(float)
    return metric_df