library(PDATK)

data(sampleCohortList)
sampleCohortList

commonGenes <- findCommonGenes(sampleCohortList)
# Subsets all list items, with subset for specifying rows and select for
# specifying columns
cohortList <- subset(sampleCohortList, subset=commonGenes)

ICGCcohortList <- cohortList[grepl('ICGC', names(cohortList), ignore.case=TRUE)]
validationCohortList <- cohortList[!grepl('icgc', names(cohortList),
                                          ignore.case=TRUE)]
ICGCcohortList$ICGCMICRO

validationCohortList <- dropNotCensored(validationCohortList)
ICGCcohortList <- dropNotCensored(ICGCcohortList)

# find common samples between our training cohorts in a cohort list
commonSamples <- findCommonSamples(ICGCcohortList)

ICGCtrainCohorts <- subset(ICGCcohortList, select=commonSamples)
ICGCtestCohorts <- subset(ICGCcohortList, select=commonSamples, invert=TRUE)

validationCohortList <- c(ICGCtestCohorts, validationCohortList)

validationCohortList <- 
  validationCohortList[names(validationCohortList) != 'ICGCSEQ']

ICGCtrainCohorts$ICGCSEQ

### Model training
set.seed(1987)
PCOSPmodel <- PCOSP(ICGCtrainCohorts, minDaysSurvived=365, randomSeed=1987)

# view the model parameters; these make your model run reproducible
metadata(PCOSPmodel)$modelParams

trainedPCOSPmodel <- trainModel(PCOSPmodel, numModels=15, minAccuracy=0.6)

metadata(trainedPCOSPmodel)$modelParams




