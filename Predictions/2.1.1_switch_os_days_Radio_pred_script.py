import random
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import helper_prediction as helper
import helper_data_load as helper_load
import boto3
import os
import argparse

def start_pred(data_norm, random_state, feat_sel_method, nr_sol_list):
    os.system("echo in start_pred")
    for nr_sol in nr_sol_list:
        # Dicovery == CHUM, Validation == IUCPQ
        result_discovery_CHUM_all = helper.all_methods(data_norm, random_state, FEAT_SEL_METH=feat_sel_method, NR_SOL=nr_sol)

        s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
        ENDPOINT_URL = 'https://s3.valeria.science'
        bucket = 'oncotechdata'
        result_discovery_CHUM_all.to_csv(f"s3://{bucket}/results/os_days_results_CHUM_discovery_"+feat_sel_method+str(nr_sol)+"_std_scl_1-100.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        os.system("echo saved df for feat_sel_meth {} and nr_sol {}".format(feat_sel_method, nr_sol))

def main(feat_sel_method, random_state=42, nr_sol_temp=1):
    os.system("echo in main with {}".format(feat_sel_method))
    data_norm_chum, data_norm_iucpq = helper_load.get_norm_data(local=False)

    #list_feat_sel_methods = ["corr", "mrmr", "mim", "f_reg", "cox_reg", "permut_randFor", "randFor_feat_sel"]
    if feat_sel_method == "corr2":
        list_corr_meth = ["pearson"]
    elif feat_sel_method == "corr1":
        list_corr_meth = ["spearman"]

    if feat_sel_method in ["corr1", "corr2"]:
        for corr_meth in list_corr_meth:
            if corr_meth == "spearman":
                nr_sol_list = [-1]
                feat_sel_method = "corr"
            elif corr_meth == "pearson":
                nr_sol_list = [-2]
                feat_sel_method = "corr"
            os.system("echo feat_sel_meth:{} and nr_sol_lis: {} ".format(feat_sel_method, nr_sol_list))
            start_pred(data_norm_chum, random_state, feat_sel_method, nr_sol_list)
        
    elif feat_sel_method == "mrmr":
        nr_sol_list = [5]
        #nr_sol_list = [2, 3, 4] # 1 is already calculated
    else:
        nr_sol_list = [-1]
    start_pred(data_norm_chum, random_state, feat_sel_method, nr_sol_list)
    
 
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fs", "--FeatureSelection", help = "Add Featureselection method")
parser.add_argument("-ns", "--NrSol", help = "Add number of solutions used by mrmr method", default=1)
args = parser.parse_args()

random_state= 42
random.seed(random_state)
main(args.FeatureSelection, random_state, args.NrSol)
    