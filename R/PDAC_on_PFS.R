########## Required libraries
library(readr)
require(switchBox)
library(caret)
library(pROC)
library(survival)
args(SWAP.Train.KTSP)
require(data.table)

###############################
## Load Training set
gen_class_disc <- read_csv("Documents/IUCPQ/Data/CHUM_genetics_pfs_class.csv")
trainingGroup <- factor(c(gen_class_disc$class_pfs))
matTraining <- gen_class_disc[,!names(gen_class_disc) %in% c("...1", "class_pfs")]
rownames(matTraining)

trainingset_precent <- 0.8

num_patients <- length(trainingGroup) # 72
length(trainingGroup[trainingGroup==1]) # 36
length(trainingGroup[trainingGroup==0]) # 36
train_per_group <- round(num_patients*trainingset_precent/2) # 70% = 50.4

#######################################################
######################## Generating 1000 kTSP models

pred <- list()
sel_pred <- list()
count=0;
b_acc <- vector()
b_acc_ensemble <- vector()
F1 <- vector()
i=1
model <- list()
models_no=1000
count=1
selected_model=list()
set.seed(1987)
sel_b_acc=list()
num_models <- 0

for(i in 1:models_no){
  # return index or values?
  x5 <-sample(which(trainingGroup==0), train_per_group, replace=F)     # Selecting random 70%(of training set)/2 of samples from Low survival group
  y5 <-sample(which(trainingGroup==1), train_per_group, replace=F)     # Selecting random 70%(of training set)/2 samples from High survival group
  
  x1=matTraining[c(x5,y5),]                
  y_index=c(x5, y5)                                  # Selecting the classes of re-sampled samples
  y1=trainingGroup[y_index]
  
  
  ### Building k-TSP models
  zzz=paste('classifier',i,sep="")
  model[[i]]<- SWAP.KTSP.Train(t(x1), as.factor(y1) )
  
  z=setdiff(1:nrow(matTraining),c(x5,y5))                              ### Finding test samples excluded in training set
  test=matTraining[z,]   
  test_grp=trainingGroup[z]
  
  ### Testing the model on out of bag samples
  pred[[i]] <- SWAP.KTSP.Classify(t(test), model[[i]])   ### Predicting the classes of test set
  
  cc=confusionMatrix(pred[[i]], test_grp,  mode = "prec_recall")
  b_acc[i]=as.numeric(cc$byClass)[11]
  F1[i]=as.numeric(cc$byClass)[7]
  print(i)
  ## Add the balanced accuracys of the models ending up in the ensemble
  if(b_acc[i]> 0.60){
    num_models <- num_models + 1
    b_acc_ensemble[num_models] <- b_acc[i]
  }
}
# mean b_acc of all 1000 models
mean(b_acc) 
## 2x25 samples (70%): 0.5168182 
## 2x28 samples (80%): 0.5042143
# mean b_acc_ensembles of models in ensemble
mean(b_acc_ensemble) 
## 2x25 samples (70%): 0.6636364
## 2x28 samples (80%): 0.6756854

# Final ensemble model
selected_model = model[which(b_acc> 0.60)]
length(selected_model) 
## 2x25 samples (70%) uses: 145 models
## 2x28 samples (80%) uses: 198 models

# How many pairs do the models consist of (out of 2-10)
min_k <- 30
max_k <- 0
for (i in 1:length(selected_model)){
  current_k <- length(selected_model[[i]]$TSPs)/2
  if(current_k < min_k){
    min_k <- current_k
  }
  if(current_k> max_k){
    max_k <- current_k
  }
}
## 2x25 samples (70%): 4-10; 30: Models use 4 to 10 gene pairs; 20/15: 3-10
min_k
max_k

save(selected_model, file="Documents/IUCPQ/IUCPQ/R/RData/PCOSP_pfs.RData")

####################################################
######################## Final Model on Validation Set

###############################
## Load Validation set
gen_class_vali <- read_csv("Documents/IUCPQ/Data/IUCPQ_genetics_pfs_class.csv")
testGroup <- factor(c(gen_class_vali$class_pfs))
matTest <- gen_class_vali[,!names(gen_class_vali) %in% c("...1", "class_pfs")]


#################### Own validation code
# Ensemble model predict on validation cohort
val_pred <- list()
for(i in 1: length(selected_model) ){
  val_pred[[i]] <- SWAP.KTSP.Classify(t(matTest), selected_model[[i]])  
}

# Majority voting for final prediction
final_pred <- vector()
length(val_pred[[1]])
for(j in 1: length(val_pred[[1]]) ){
  temp_pred_sample <- 0
  for(i in 1: length(selected_model) ){
    temp_pred_sample <- temp_pred_sample + as.numeric(as.character(val_pred[[i]][j]))
  }
  print(temp_pred_sample)
  if (temp_pred_sample <= length(selected_model)/2) {
    final_pred[[j]] <- 0
  }
  else {
    final_pred[[j]] <- 1
  }
}
final_pred <- factor(final_pred)
length(final_pred)
length(testGroup)
c=confusionMatrix(final_pred, testGroup,  mode = "prec_recall")
# validation set confusion matrix and errors
c

## 2x25 samples (70%)
# Accuracy : 0.5593         
# 95% CI : (0.424, 0.6884)
# No Information Rate : 0.5085         
# P-Value [Acc > NIR] : 0.257828       
# Kappa : 0.1102         
# Mcnemar's Test P-Value : 0.003264       
# Precision : 0.5435         
# Recall : 0.8333         
# F1 : 0.6579         
# Prevalence : 0.5085         
# Detection Rate : 0.4237         
# Detection Prevalence : 0.7797         
# Balanced Accuracy : 0.5546         
# 'Positive' Class : 0 

## 2x28 samples (80%)
# Accuracy : 0.5593         
# 95% CI : (0.424, 0.6884)
# No Information Rate : 0.5085         
# P-Value [Acc > NIR] : 0.257828       
# Kappa : 0.1102         
# Mcnemar's Test P-Value : 0.003264       
# Precision : 0.5435         
# Recall : 0.8333         
# F1 : 0.6579         
# Prevalence : 0.5085         
# Detection Rate : 0.4237         
# Detection Prevalence : 0.7797         
# Balanced Accuracy : 0.5546         
# 'Positive' Class : 0 


b_acc_test=as.numeric(c$byClass)[11]
F1_test=as.numeric(c$byClass)[7]


######################## pcosp Probability
########
pcosp_prob = function(val_mat){
  val_pred <- list()
  
  for(i in 1: length(selected_model) ){
    val_pred[[i]] <- SWAP.KTSP.Classify(t(val_mat), selected_model[[i]])  
  }
  
  list_z1=vector()
  freq_early=vector()
  i=1
  
  for (i in 1: nrow(val_mat)){
    for (k in 1: length(selected_model) ){
      print(k)
      print(i)
      print(val_pred[[k]][i])
      list_z1=append(list_z1,as.numeric(val_pred[[k]][i]))
    }
    
    freq_early[i] =length(list_z1[list_z1==1])/length(selected_model) 
    list_z1=vector()
  }
  ret_list=list(predicted_probabilities=freq_early)
  return(ret_list)
}

## A PCOSP score is simply the proportion of models predicting good survival
# out of the total number of models in the ensemble.
ret_list <- pcosp_prob(matTest)
ret_list
