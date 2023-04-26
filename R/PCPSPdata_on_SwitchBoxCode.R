########## Required libraries
require(switchBox)
library(caret)
library(pROC)
library(survival)
args(SWAP.Train.KTSP)
require(data.table)
library(PDATK) 


###############################
## Load Training set
gen_class_disc <- read_csv("Documents/IUCPQ/Data/CHUM_genetics_pdl1_class.csv")
trainingGroup <- factor(c(gen_class_disc$class_pdl1))
matTraining <- gen_class_disc[,!names(gen_class_disc) %in% c("...1", "class_pdl1")]
rownames(matTraining)
###########
## Load their trainingset
data(sampleCohortList)
sampleCohortList

commonGenes <- findCommonGenes(sampleCohortList)
# Subsets all list items, with subset for specifying rows and select for
# specifying columns
cohortList <- subset(sampleCohortList, subset=commonGenes)

ICGCcohortList <- cohortList[grepl('ICGC', names(cohortList), ignore.case=TRUE)]

commonSamples <- findCommonSamples(ICGCcohortList)

# split into shared samples for training, the rest for testing
ICGCtrainCohorts <- subset(ICGCcohortList, select=commonSamples)
ICGCtestCohorts <- subset(ICGCcohortList, select=commonSamples, invert=TRUE)

ICGCtrainCohorts$ICGCMICRO
icgc_seq_cohort = ICGCtrainCohorts$ICGCSEQ
icgc_array_cohort = ICGCtrainCohorts$ICGCMICRO

rownames(icgc_array_cohort) == rownames(icgc_seq_cohort)

# get target survival data
icgc_array_cohort@assays@data@listData
ICGCtrainCohorts@listData[["ICGCMICRO"]]@assays@data@listData

###### Excluding samples censored before 1-yr
g1=which(as.numeric(as.character(icgc_seq_cohort$OS))<=365 &  as.numeric(as.character(icgc_seq_cohort$OS_Status))==1)
g2=which(as.numeric(as.character(icgc_seq_cohort$OS))>365)
g_ind=sort(c(g1,g2))

icgc_seq_cohort=icgc_seq_cohort[g_ind,]
icgc_array_cohort=icgc_array_cohort[g_ind,]
icgc_array_cohort
icgc_seq_cohort
icgc_array_cohort

groups_temps <- icgc_seq_cohort@colData@listData[["event_occurred"]]
length(groups_temps[groups_temps==1])
length(groups_temps[groups_temps==0])
## Assays have to be the same for rbind
icgc_array_cohort@assays <- icgc_seq_cohort@assays
merge_common <- rbind(icgc_seq_cohort,icgc_array_cohort)      ### Merged common ICGC seq and array data

trainingGroup <- merge_common
#######################################################
######################## Generating 1000 kTSP models

pred <- list()
sel_pred <- list()
count=0;
b_acc <- vector()
F1 <- vector()
i=1
model <- list()
models_no=10
count=1
selected_model=list()
set.seed(1987)
sel_b_acc=list()

for(i in 1:models_no){
  # return index or values?
  x5 <-sample(which(trainingGroup==0), 30, replace=F)     # Selecting random 40 samples from Low survival group
  y5 <-sample(which(trainingGroup==1), 30, replace=F)     # Selecting random 40 samples from High survival group
  
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
}
# Final ensembled model
selected_model = model[which(b_acc> 0.60)]
length(selected_model)
save(selected_model, file="Documents/IUCPQ/IUCPQ/R/RData/PCOSP_pdl1.RData")

####################################################
######################## Final Model on Test Set
training_pred <- list()
for(i in 1: length(selected_model) ){
  training_pred[[i]] <- SWAP.KTSP.Classify(t(matTraining), selected_model[[i]])  
}

final_pred_training <- vector()
for(j in 1: length(training_pred[[1]]) ){
  temp_pred_sample <- 0
  for(i in 1: length(selected_model) ){
    temp_pred_sample <- temp_pred_sample + as.numeric(as.character(training_pred[[i]][j]))
  }
  print(temp_pred_sample)
  if (temp_pred_sample <= length(selected_model)/2) {
    final_pred_training[[j]] <- 0
  }
  else {
    final_pred_training[[j]] <- 1
  }
}
final_pred_training <- factor(final_pred_training)
c_train=confusionMatrix(final_pred_training, trainingGroup,  mode = "prec_recall")
# test set confusin matrix and errors
c_train 
b_acc_train = as.numeric(c_train$byClass)[11]
F1_train = as.numeric(c_train$byClass)[7]



####################################################
######################## Final Model on Validation Set

###############################
## Load Validation set
gen_class_vali <- read_csv("Documents/IUCPQ/Data/IUCPQ_genetics_pdl1_class.csv")
testGroup <- factor(c(gen_class_vali$class_pdl1))
matTest <- gen_class_vali[,!names(gen_class_vali) %in% c("...1", "class_pdl1")]


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
pcosp_prob(matTest)
