# Structure of training and evaluating the prediction of os_days

Do for both [IUCPQ, CHUM]:
1. Data Imports
2. Merge clinical to pyrads data
3. Data Normalization (StandardScaler())

For CHUM as discovery cohort:  
Calculate Lin. R., Random Forest R., SVR, Ensembled R. & more:  
1. Fit model with nr_feat in [1-50] features:  
    - Use 5-fold cross validation  
        - Create train and test set in fold
        - Get the wanted number of features with selected feature selection method (input is train set of fold)
        - Predict test set of fold
        - Calculate concordance index and add to list of c_indecies for that nr_feat
    - Calculate mean over the c-indecies 

2. Get number of used features of best performing model for each method (largest c-idx)

3. Train model on whole Discovery cohort with number of features the best performing model showed with cross validation

4. Test model on validation cohort (IUCPQ)
    - Calculate c-index
    - Plot results