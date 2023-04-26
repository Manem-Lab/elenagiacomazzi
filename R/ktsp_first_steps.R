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
rownames(matTraining)
## Testing set
gen_class_vali <- read_csv("Documents/IUCPQ/Data/IUCPQ_genetics_class.csv")
testingGroup <- factor(c(gen_class_vali$class_os_days))
matTesting <- gen_class_vali[,!names(gen_class_vali) %in% c("oncotech_id", "class_os_days")]
matTesting <- t(matTesting)
rownames(matTesting)

print(length(trainingGroup))
print(ncol(matTraining))
dim(matTesting)

     
#####################
### Training kTSP and classifying new samples
classifier <- SWAP.KTSP.Train(inputMat = matTraining, phenoGroup = trainingGroup, krange=c(2:50))
classifier
ktspStatTraining <- SWAP.KTSP.Statistics(inputMat = matTraining, classifier = classifier)
ktspStatTraining
matTraining <- t(1*ktspStatTraining$comparisons)
#rownames(matTraining) <- gsub(">", "\n more express than\n", rownames(matTraining))

heatmap.2(matTraining,
          scale="none", Rowv=TRUE, Colv=TRUE,  dendrogram="none",
          trace="none", key=FALSE,
          col=c("lightsteelblue2", "pink3"),
          labCol=toupper(paste(trainingGroup, "Prognosis")),
          sepwidth=c(0.075,0.075), sepcolor="black",
          rowsep=1:ncol(ktspStatTraining$comparisons),
          colsep=1:nrow(ktspStatTraining$comparisons),
          lmat=rbind( c(0, 3), c(2, 1), c(0, 4) ), lhei=c(0.1, 5, 0.5), lwid=c(0.15, 5),
          mar=c(7.5, 12), cexRow=0.85, cexCol=0.9)

## Testing
testPrediction <- SWAP.KTSP.Classify(matTesting, classifier)
testPrediction
ktspStatTesting <- SWAP.KTSP.Statistics(inputMat = matTesting, classifier = classifier)
ktspStatTesting

matTesting <- t(1*ktspStatTesting$comparisons)
#rownames(matTesting) <- gsub(">", "\n more express than\n", rownames(matTesting))
heatmap.2(matTesting,
          scale="none", Rowv=TRUE, Colv=TRUE,  dendrogram="none",
          trace="none", key=FALSE,
          col=c("lightsteelblue2", "pink3"),
          labCol=toupper(paste(testingGroup, "Prognosis")),
          sepwidth=c(0.075,0.075), sepcolor="black",
          rowsep=1:ncol(ktspStatTesting$comparisons),
          colsep=1:nrow(ktspStatTesting$comparisons),
          lmat=rbind( c(0, 3), c(2, 1), c(0, 4) ), lhei=c(0.1, 5, 0.5), lwid=c(0.15, 5),
          mar=c(7.5, 12), cexRow=0.85, cexCol=0.9)

### Making confusion matrix
table(testPrediction, testingGroup)
testPrediction
testingGroup

roc_1 = roc(response=testingGroup, predictor= factor(testPrediction, ordered = TRUE), plot=TRUE)
plot(roc_1, col="red", lwd=3, main="ROC curve")
auc_1 <- auc(roc_1)
auc_1

### Test set performance for different k's
auc_best = 0
for (k in c(2:50)) {
  classifier <- SWAP.KTSP.Train(inputMat = matTraining, phenoGroup = trainingGroup, krange=k)
  testPrediction <- SWAP.KTSP.Classify(matTesting, classifier)
  roc_1 = roc(response=testingGroup, predictor= factor(testPrediction, ordered = TRUE), plot=TRUE)
  auc_1 <- auc(roc_1)
  if (auc_1 > auc_best) {
    print("New best k")
    print(k)
    auc_best <- auc_1
    print(auc_best)
  }
  if (k == 6) {
    print("Best training k")
    print(k)
    print(auc_1)
  }

}


### Use CV
set.seed(1)
num_folds <- 5
auc_best <- 0
k_best <- 0
for (k in c(2:15)) {
  result <- SWAP.KTSP.CV(inputMat = matTraining, Groups= trainingGroup, k = num_folds, randomize = TRUE, krange=k)
  auc_temp <- result[["stats"]][["auc"]]
  if (auc_temp > auc_best){
    auc_best <- auc_temp
    k_best <- k
    print(k_best)
    
    print(cat("training: ", auc_best))
    # train classifier on complete training set with best k and predict on test set
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

