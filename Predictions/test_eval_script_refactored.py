import eval_script_refactored

feat_sel_method = "spearman"
predictor = "os_days"
train_set = "CHUM"
nr_sol = -1
k_max = 2
local = True
savePath = "../../Data/test_folder"

eval_script_refactored.main_eval(feat_sel_method, predictor, train_set, nr_sol, k_max, local, savePath, random_state=42)