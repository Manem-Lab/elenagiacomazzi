library(readr)
require(switchBox)
library(pROC)
library(survival)
library(caret)
args(SWAP.Train.KTSP)

###Loading training and test data
## Training set
gen_class_disc <- read_csv("Documents/IUCPQ/Data/CHUM_genetics_class.csv")
trainingGroup <- factor(c(gen_class_disc$class_os_days))
matTraining <- gen_class_disc[,!names(gen_class_disc) %in% c("oncotech_id", "class_os_days")]
matTraining <- t(matTraining)
## Testing set
gen_class_vali <- read_csv("Documents/IUCPQ/Data/IUCPQ_genetics_class.csv")
testingGroup <- factor(c(gen_class_vali$class_os_days))
matTesting <- gen_class_vali[,!names(gen_class_vali) %in% c("oncotech_id", "class_os_days")]
matTesting <- t(matTesting)

### Use CV
set.seed(1)
num_folds <- 5
auc_best <- 0
k_best <- 0

for (k in c(3:30)) {
  result <- SWAP.KTSP.CV(inputMat = matTraining, Groups= trainingGroup, k = num_folds, randomize = TRUE, krange=k)
  auc_temp <- result[["stats"]][["auc"]]
  if (auc_temp > auc_best){
    auc_best <- auc_temp
    k_best <- k
    print(k_best)
    
    print(cat("training: ", auc_best))
    # train classifier on complete training set with best k and predict on test set
    # most important features are selected based on Wilcoxin test
    classifier <- SWAP.KTSP.Train(inputMat = matTraining, phenoGroup = trainingGroup, krange=k)
    testPrediction <- SWAP.KTSP.Classify(matTesting, classifier)
    roc_test = roc(response=testingGroup, predictor= factor(testPrediction, ordered = TRUE), plot=TRUE)
    auc_test <- auc(roc_test)
    print(cat("testing: ", auc_test))
    
    cc=confusionMatrix(testPrediction, testingGroup,  mode = "prec_recall")
    cc$byClass
    b_acc=as.numeric(cc$byClass)[11]
    F1=as.numeric(cc$byClass)[7]
  }
}
print("Best model performance: ")
print(cat("AUC on Training Cohort (CHUM+CV): ", auc_best))
print(cat("AUC on Testing Cohort: ", auc_test))
