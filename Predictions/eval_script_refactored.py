import pandas as pd
import helper_eval_refactored as helper_eval
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

#local = False #

def eval(discovery_results_list, feat_sel_method, predictor, train_set, nr_sol_list, k_max, local, savePath, genomics, num_genomics, random_state=42):
    ### Plotting Discovery results
    discovery_results_list_rename = [None] * len(nr_sol_list)
    discovery_methods_nr_feat = [None] * len(nr_sol_list)
    dicovery_meth_nr_feat_dict = [None] * len(nr_sol_list)
    df_vali = [None] * len(nr_sol_list)

    # TODO update loading function for other data
    data_norm_chum, data_norm_iucpq = helper_load.get_norm_data(local, predictor, genomics=genomics, num_genomics=num_genomics)
    if train_set == "CHUM": 
        data_norm_train = data_norm_chum
        data_norm_test = data_norm_iucpq
        val_set = "IUCPQ"
    elif train_set == "IUCPQ": 
        data_norm_train = data_norm_iucpq
        data_norm_test = data_norm_chum
        val_set = "CHUM"

    for i, nr_sol in enumerate(nr_sol_list):
        # rename columns for plotting
        print(discovery_results_list[i])
        discovery_results_list_rename[i] = helper_eval.rename_for_plotting(discovery_results_list[i]) # discovery_results_list[i]
        print(discovery_results_list_rename[i])
        file_name1 = predictor + "_disc_" + train_set + "_renamed_1-"+str(k_max) + "_" + feat_sel_method+str(nr_sol)+'.csv'
        # df, file_name, local, savePath, folder
        helper_data_load.save_csv(discovery_results_list_rename[i],file_name1, local, savePath, folder="/eval/")
        
        # Get the number of features corresponding to the best performance for each method
        discovery_results_list_rename[i].set_index("Nr. Features", inplace=True) # Nr. Features
        discovery_methods_nr_feat[i] = helper_eval.get_nr_feat_per_method(discovery_results_list_rename[i])
        print(discovery_methods_nr_feat[i])
        discovery_methods_nr_feat[i] = discovery_methods_nr_feat[i].sort_values("C-Index", ascending=False, ignore_index=True)
        file_name2 = predictor + "_disc_"+train_set + "_nr_feat_"+feat_sel_method+str(nr_sol)+"_1-"+str(k_max)+".csv"
        helper_data_load.save_csv(discovery_methods_nr_feat[i], file_name2, local, savePath, folder="/eval/")
    ####
        # Build a dict which maps the method to the nr of features for best performance
        dicovery_meth_nr_feat_dict[i] = helper_eval.get_meth_feat_dict(discovery_methods_nr_feat[i], random_state=random_state)
        print(dicovery_meth_nr_feat_dict[i])

        # Train all methods with best nr_feat on the whole discovery dataset and validate on validation set
        df_vali[i] = helper_eval.train_disc_n_validate(predictor, data_norm_train, data_norm_test, dicovery_meth_nr_feat_dict[i], feat_sel_method, nr_sol, val_set, k_max, local, savePath)
        df_vali[i].set_index("Method Name", inplace=True)
        # Save the validation results and plot them
        file_name3 = predictor + "_results_best_model_validation_" + val_set +"_"+feat_sel_method+str(nr_sol)+"_1-"+str(k_max)+".csv"
        helper_data_load.save_csv(df_vali[i], file_name3, local, savePath, folder="/eval/c_idxs/")

  
def main_eval(feat_sel_method, predictor, train_set, nr_sol, k_max, local, savePath, genomics, num_genomics, random_state=42):
    print("###################### BEGIN EVALUATION ######################")
    print("In eval main with feat: {}".format(feat_sel_method))
    
    ##### Load the earlier build results
    result_discovery_list = []
    file_name = predictor + "_results_discovery_" + train_set + "_" + feat_sel_method + str(nr_sol) + "_1-" + str(k_max) + ".csv"
    if feat_sel_method == "mrmr":
        # if mrmr select to try how many solutions
        nr_sol_list = [5] #[1, 2, 3, 4]
        # Lists for the different numbers of results list[0] = nr_sol1, list[1] = nr_sol2...
        for nr_sol in nr_sol_list:
            result_discovery = helper_data_load.load_csv(file_name, local,  savePath, folder="/results/")
            result_discovery_list.append(result_discovery)
    else:
        nr_sol_list = [nr_sol]
        result_discovery_list.append(helper_data_load.load_csv(file_name, local,  savePath, folder="/results/"))
    
    
    eval(result_discovery_list, feat_sel_method, predictor, train_set, nr_sol_list, k_max, local, savePath, genomics, num_genomics, random_state=42)




# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fs", "--FeatureSelection", help = "Add Featureselection method")
parser.add_argument("-local", "--local", type = str, help = "Load data locally")
parser.add_argument("-path", "--savePath", type = str, help = "Base path to save the data")
parser.add_argument("-k_max", "--k_max", type = int, help = "Try up to k features")
parser.add_argument("-pred", "--predictor", type = str, help = "Predictor - clinical endpoint")
parser.add_argument("-trainSet", "--TrainSet", type = str, help = "Which of the two loaded dataset is used as training set")
parser.add_argument("-genomics", "--Genomics", type = str, help = "Use Genomics data ?")
parser.add_argument("-num_gen", "--NumGenomics", help = "Use statset of 1000 or 5000 genes")
args = parser.parse_args()

if args.local == "False": local = False
elif args.local == "True": local = True
if args.Genomics == "False": genomics = False
elif args.Genomics == "True": genomics = True

if args.FeatureSelection == "spearman":
    nr_sol_list = [-1]
elif args.FeatureSelection == "pearson":
    nr_sol_list = [-2]
elif args.FeatureSelection == "mrmr":
    nr_sol_list = [5]
else:
    nr_sol_list = [-1]
nr_sol = nr_sol_list[0]

main_eval(args.FeatureSelection, args.predictor, args.TrainSet, nr_sol, args.k_max, local, args.savePath, genomics, args.NumGenomics)