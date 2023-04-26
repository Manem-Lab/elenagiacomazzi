import pandas as pd
import helper_evaluation_class as helper_eval_class
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

def eval(discovery_iucpq_results_list, feat_sel_meth, nr_sol_list, save_plots=True):
    ### Plotting Discovery results
    discovery_iucpq_results_list_rename = [None] * len(nr_sol_list)
    discovery_iucpq_methods_nr_feat = [None] * len(nr_sol_list)
    dicovery_iucpq_meth_nr_feat_dict = [None] * len(nr_sol_list)
    df_vali_on_chum = [None] * len(nr_sol_list)
    df_vali_on_chum_rename = [None] * len(nr_sol_list)

    data_norm_chum, data_norm_iucpq = helper_load.get_norm_data(local=local, classification=True)
    for i, nr_sol in enumerate(nr_sol_list):
        #### Discovery == IUCPQ, Validation == CHUM ####
        # rename columns for plotting
        discovery_iucpq_results_list_rename[i] = helper_eval_class.rename_for_plotting(discovery_iucpq_results_list[i])
        # do not need it because the results on the discovery set are not that important
        title = "os_days_Discovery_IUCPQ" + feat_sel_meth + str(nr_sol)
        ### helper_eval.plot_result(discovery_iucpq_results_list_rename[i], title, save_plots)
        file_name1 = 'os_days_disc_IUCPQ_renamed_1-100_'+feat_sel_meth+str(nr_sol)+'.csv'
        helper_data_load.save_csv(discovery_iucpq_results_list_rename[i],file_name1, local, folder="eval/classification")
        
        # Get the number of features corresponding to the best performance for each method
        discovery_iucpq_results_list_rename[i].set_index("Nr. Features", inplace=True)
        discovery_iucpq_methods_nr_feat[i] = helper_eval_class.get_nr_feat_per_method(discovery_iucpq_results_list_rename[i])
        discovery_iucpq_methods_nr_feat[i] = discovery_iucpq_methods_nr_feat[i].sort_values("C-Index", ascending=False, ignore_index=True)
        file_name2 = "os_days_IUCPQ_discovery_nr_feat"+"_"+feat_sel_meth+str(nr_sol)+"_1-100.csv"
        helper_data_load.save_csv(discovery_iucpq_methods_nr_feat[i], file_name2, local, folder="eval/classification")
    
        # Build a dict which maps the method to the nr of features for best performance
        dicovery_iucpq_meth_nr_feat_dict[i] = helper_eval_class.get_meth_feat_dict(discovery_iucpq_methods_nr_feat[i], random_state=random_state)
        
        # Train all methods with best nr_feat on the whole discovery dataset and validate on validation set
        df_vali_on_chum[i] = helper_eval_class.train_disc_n_validate(data_norm_iucpq, data_norm_chum, dicovery_iucpq_meth_nr_feat_dict[i], feat_sel_meth, nr_sol, "CHUM")
        df_vali_on_chum[i].set_index("Method Name", inplace=True)
        # Save the validation results and plot them
        file_name3 = "os_days_results_best_model_validation_CHUM"+"_"+feat_sel_meth+str(nr_sol)+"_1-100.csv"
        helper_data_load.save_csv(df_vali_on_chum[i], file_name3, local, folder="eval/classification/c_idxs")

        # Plot performance of dicovery and validation set for each method
        ### helper_eval.plot_vali_results(df_vali_on_chum[i], discovery_iucpq_methods_nr_feat[i], val_set="CHUM", feat_sel_method=feat_sel_meth, nr_sol=nr_sol, random_state=random_state, save_plots=save_plots)
        
  
def main(feat_sel_method, corr_meth="spearman", random_state=42):
    print("echo in main with feat: {} and corr {}".format(feat_sel_method, corr_meth))
    ##### Read in the earlier build results
    # Dicovery == IUCPQ, Validation == CHUM
    result_discovery_iucpq_list = []
    if feat_sel_method == "mrmr":
        # if mrmr select to try how many solutions
        nr_sol_list = [5] #[1, 2, 3, 4]
        # Lists for the different numbers of results list[0] = nr_sol1, list[1] = nr_sol2...
        for nr_sol in nr_sol_list:
            result_discovery_IUCPQ = helper_data_load.load_csv(file_name="os_days_results_IUCPQ_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", local=local, folder="results/classification")
            result_discovery_iucpq_list.append(result_discovery_IUCPQ)
    else:
        nr_sol_list = [-1]
        nr_sol = nr_sol_list[0]
        result_discovery_iucpq_list.append(helper_data_load.load_csv("os_days_results_IUCPQ_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", local, folder="results/classification"))
    eval(result_discovery_iucpq_list, feat_sel_method, nr_sol_list)

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fs", "--FeatureSelection", help = "Add Featureselection method")
#parser.add_argument("-cm", "--CorrMeth", help = "Add correlation method")

args = parser.parse_args()

main(args.FeatureSelection, random_state=random_state)
