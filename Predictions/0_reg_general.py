'''
Script to be strated using a SLURM command but also locally

For using SLURM I needed a complete workflow from loading the correct data, setting all the variables and executing many different feature selection and prediction algorithms.
This is the startng script which
- loads the parameters
- loads the respective data (main) using the data load helper
- sets the nr_sol_list parameter which I introduced when I tested multiple numbers of solutions for MRMR (could be removed probably but would have to be removed in the following scripts as well)
- start the training of all feature selection/ML prediction combinations (in start_pred)
- after saving the results, the evalusation (main_eval) scrit is called to test the final model configurations on the validation data
'''

import random
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import helper_prediction_refactored as helper
import helper_data_load as helper_load
import boto3
import os
import argparse
import pandas as pd

def start_pred(data_norm_train, random_state, feat_sel_method, nr_sol, local, savePath, k_max, predictor, train_set, genomics, num_genomics):
    print("echo in start_pred")
    results_train = helper.train_all_methods(data_norm_train, random_state, feat_sel_method, nr_sol, k_max, predictor, savePath, local)
    
    file_name = predictor + "_results_discovery_" + train_set + "_" + feat_sel_method + str(nr_sol) + "_1-" + str(k_max) + ".csv"
    helper_load.save_csv(results_train, file_name, local, savePath, folder="/results/")
    print("echo saved results for feat_sel_meth {} and nr_sol {}".format(feat_sel_method, nr_sol))

    ## Start Evaluation
    import eval_script_refactored as eval
    print(feat_sel_method, predictor, train_set, nr_sol, k_max, local)
    eval.main_eval(feat_sel_method, predictor, train_set, nr_sol, k_max, local, savePath, genomics, num_genomics, random_state=42)
    print("EVALUATION finished")

    
def main(feat_sel_method, local_load, savePath, k_max, predictor, train_set, genomics, num_genomics):
    random_state= 42
    random.seed(random_state)  
    print("echo in main with {}".format(feat_sel_method))

    data_norm_chum, data_norm_iucpq = helper_load.get_norm_data(local_load, predictor, genomics=genomics, num_genomics=num_genomics)
    if train_set == "CHUM": data_norm_train = data_norm_chum
    elif train_set == "IUCPQ": data_norm_train = data_norm_iucpq

    if feat_sel_method == "spearman":
        nr_sol_list = [-1]
    elif feat_sel_method == "pearson":
        nr_sol_list = [-2]
    elif feat_sel_method == "mrmr":
        nr_sol_list = [5]
    else:
        nr_sol_list = [-1]
    nr_sol = nr_sol_list[0]
    start_pred(data_norm_train, random_state, feat_sel_method, nr_sol, local_load, savePath, k_max, predictor, train_set, genomics, num_genomics)
    
 
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fs", "--FeatureSelection", type = str, help = "Add Featureselection method (one of ['pearson', 'spearman', 'mim', 'f_reg', 'mrmr'])")
parser.add_argument("-local", "--local", type = str, help = "Load data locally, 'True'/'False'")
parser.add_argument("-path", "--savePath", type = str, help = "Base path to save the data exp.: '../../exp1/")
parser.add_argument("-k_max", "--k_max", type = int, help = "Try up to k(int) features exp.: 50, 100")
parser.add_argument("-pred", "--predictor", type = str, help = "Predictor - clinical endpoint (one of ['os_days','pfs_days', "pdl1_tps']")
parser.add_argument("-trainSet", "--TrainSet", type = str, help = "Which of the two loaded dataset is used as training set, either 'CHUM' or 'IUCPQ'")
parser.add_argument("-genomics", "--Genomics", type = str, help = "Use Genomics data ? 'True'/'False'")
parser.add_argument("-num_gen", "--NumGenomics", help = "Use statset of 1000 or 5000 genes exp.: 1000")
args = parser.parse_args()

if args.local == "False": local = False
elif args.local == "True": local = True
if args.Genomics == "False": genomics = False
elif args.Genomics == "True": genomics = True

main(args.FeatureSelection, local, args.savePath, args.k_max, args.predictor, args.TrainSet, genomics, args.NumGenomics)
    
