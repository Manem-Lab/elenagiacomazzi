import pandas as pd
import helper_eval_ecog as helper_eval_ecog
import helper_data_load as helper_load
import random
import warnings
warnings.filterwarnings("ignore")
random_state= 42
random.seed(random_state)
import boto3
import os
import argparse
import helper_data_load

local = False #

def eval(discovery_chum_results_list, feat_sel_meth, nr_sol_list, save_plots=True):
    ### Plotting Discovery results
    discovery_chum_results_list_rename = [None] * len(nr_sol_list)
    discovery_chum_methods_nr_feat = [None] * len(nr_sol_list)
    dicovery_chum_meth_nr_feat_dict = [None] * len(nr_sol_list)
    df_vali_on_chum = [None] * len(nr_sol_list)
    df_vali_on_chum_rename = [None] * len(nr_sol_list)

    data_norm_chum, data_norm_iucpq = helper_load.get_norm_data(local=local, ecog=True)
    for i, nr_sol in enumerate(nr_sol_list):
        #### Discovery == CHUM, Validation == IUCPQ ####
        # rename columns for plotting
        discovery_chum_results_list_rename[i] = helper_eval_ecog.rename_for_plotting(discovery_chum_results_list[i])
        # do not need it because the results on the discovery set are not that important
        title = "os_days_Discovery_CHUM" + feat_sel_meth + str(nr_sol)
        ### helper_eval.plot_result(discovery_chum_results_list_rename[i], title, save_plots)
        file_name1 = 'os_days_disc_CHUM_renamed_1-100_'+feat_sel_meth+str(nr_sol)+'.csv'
        helper_data_load.save_csv(discovery_chum_results_list_rename[i],file_name1, local, folder="ecog/eval")
        
        # Get the number of features corresponding to the best performance for each method
        discovery_chum_results_list_rename[i].set_index("Nr. Features", inplace=True)
        discovery_chum_methods_nr_feat[i] = helper_eval_ecog.get_nr_feat_per_method(discovery_chum_results_list_rename[i])
        discovery_chum_methods_nr_feat[i] = discovery_chum_methods_nr_feat[i].sort_values("C-Index", ascending=False, ignore_index=True)
        file_name2 = "os_days_CHUM_discovery_nr_feat"+"_"+feat_sel_meth+str(nr_sol)+"_1-100.csv"
        helper_data_load.save_csv(discovery_chum_methods_nr_feat[i], file_name2, local, folder="ecog/eval")
    
        # Build a dict which maps the method to the nr of features for best performance
        dicovery_chum_meth_nr_feat_dict[i] = helper_eval_ecog.get_meth_feat_dict(discovery_chum_methods_nr_feat[i], random_state=random_state)
        
        # Train all methods with best nr_feat on the whole discovery dataset and validate on validation set
        df_vali_on_chum[i] = helper_eval_ecog.train_disc_n_validate(data_norm_chum, data_norm_iucpq, dicovery_chum_meth_nr_feat_dict[i], feat_sel_meth, nr_sol, val_set="IUCPQ")
        df_vali_on_chum[i].set_index("Method Name", inplace=True)
        # Save the validation results and plot them
        file_name3 = "os_days_results_best_model_validation_IUCPQ"+"_"+feat_sel_meth+str(nr_sol)+"_1-100.csv"
        helper_data_load.save_csv(df_vali_on_chum[i], file_name3, local, folder="ecog/eval/c_idxs")

        # Plot performance of dicovery and validation set for each method
        ### helper_eval.plot_vali_results(df_vali_on_chum[i], discovery_iucpq_methods_nr_feat[i], val_set="CHUM", feat_sel_method=feat_sel_meth, nr_sol=nr_sol, random_state=random_state, save_plots=save_plots)
        
  
def main_eval(feat_sel_method, corr_meth="spearman", random_state=42):
    print("###################### BEGIN EVALUATION ######################")
    print("echo in main with feat: {} and corr {}".format(feat_sel_method, corr_meth))
    disc = "CHUM"
    vali = "IUCPQ"
    if feat_sel_method == "corr1":
        corr_meth = "spearman"
        feat_sel_method = "corr"
    elif feat_sel_method == "corr2":
        corr_meth = "pearson"
        feat_sel_method = "corr"
    ##### Read in the earlier build results
    # Dicovery == CHUM, Validation == IUCPQ
    result_discovery_chum_list = []
    if feat_sel_method == "mrmr":
        # if mrmr select to try how many solutions
        nr_sol_list = [5] #[1, 2, 3, 4]
        # Lists for the different numbers of results list[0] = nr_sol1, list[1] = nr_sol2...
        for nr_sol in nr_sol_list:
            result_discovery_chum = helper_data_load.load_csv(file_name="os_days_results_"+ disc +"_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", folder="ecog/results", local=local)
            result_discovery_chum_list.append(result_discovery_chum)
    elif feat_sel_method == "corr":
        if corr_meth == "spearman":
            nr_sol_list = [-1]
        elif corr_meth == "pearson":
            nr_sol_list = [-2]
        nr_sol = nr_sol_list[0]
        result_discovery_chum_list.append(helper_data_load.load_csv(file_name="os_days_results_"+disc +"_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", folder="ecog/results", local=local))
    else:
        nr_sol_list = [-1]
        nr_sol = nr_sol_list[0]
        result_discovery_chum_list.append(helper_data_load.load_csv("os_days_results_"+disc+"_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", folder="ecog/results", local=local))
    eval(result_discovery_chum_list, feat_sel_method, nr_sol_list)






# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fs", "--FeatureSelection", help = "Add Featureselection method")
parser.add_argument("-cm", "--CorrMeth", help = "Add correlation method", default="None")

args = parser.parse_args()

main_eval(args.FeatureSelection, args.CorrMeth, random_state)
'''
random_state= 42
random.seed(random_state)
feat_sel_meth = ["mrmr"] #["corr", "mim", "randFor_feat_sel", "f_reg"]
corr_methods = ["spearman", "pearson"]
for feat in feat_sel_meth:
    if feat == "corr":
        for corr_meth in corr_methods:
            main(feat, corr_meth, random_state)
    else:
        main(feat, random_state)
        '''
