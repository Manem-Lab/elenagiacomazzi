import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import boto3
import s3fs
import os

def save_csv(df, file_name, local, savePath, folder):
    savePath = savePath + folder
    if local:
        # checking if the directory demo_folder exist or not.
        if not os.path.exists(savePath):
            # if the demo_folder directory is not present then create it.
            os.makedirs(savePath)
        df.to_csv(savePath + file_name)
    else:
        s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
        ENDPOINT_URL = 'https://s3.valeria.science'
        bucket = 'oncotechdata'
        df.to_csv(f"s3://{bucket}"+savePath+file_name, storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})

def load_csv(file_name, local, savePath, folder):
    savePath = savePath + folder
    if local:
        df = pd.read_csv(savePath + file_name, index_col=0)
    else:
        s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
        ENDPOINT_URL = 'https://s3.valeria.science'
        bucket = 'oncotechdata'
        print("echo load file: "+"s3://{bucket}"+savePath+file_name)
        df = pd.read_csv(f"s3://{bucket}"+savePath+file_name, index_col=0, storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
    return df

def data_normalization(df, scaler= StandardScaler()):
    # TRAIN
    # normalize data
    data_norm = df.copy() #Has training + test data frames combined to form single data frame
    normalizer = scaler
    df_array = normalizer.fit_transform(data_norm)
    data_norm = pd.DataFrame(data=df_array, columns=data_norm.columns)
    # rename columns because of later usage
    data_norm.columns = data_norm.columns.str.replace('.','_')
    return data_norm

def add_os_day_class(df, predictor):
    df["class_"+ predictor] = pd.qcut(df[predictor], 3, labels=False, retbins=False, precision=3, duplicates='raise')
    df = df[df["class_"+ predictor] != 1]
    df["class_"+ predictor][df["class_"+ predictor] == 2] = 1
    df["class_"+ predictor][df["class_"+ predictor] == "nan"] = np.nan
    df = df[df["class_"+ predictor].notna()]
    df = df.reset_index(drop=True)
    df = df.drop(columns=[predictor, "Unnamed: 0"])
    df["class_"+ predictor] = df["class_"+ predictor].astype('int')
    return df

def change_groupstr_to_id(df, col):
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes
    return df
    
def make_cat_to_int(df, ecog=False):
    if ecog:
        list_clin_feats_cat = ["ecog_status"]
    else:
        list_clin_feats_cat = ["sex", "smoking_habit", "ecog_status", "first_line_io", "histology_group"]
    for col in list_clin_feats_cat:
        df_cat = change_groupstr_to_id(df, col)
    return df_cat

def prep_genomics(df):
    df.set_index("Unnamed: 0", inplace=True)
    df = df.T
    df.reset_index(inplace=True)
    df = df.rename(columns={"index":"oncotech_id"})
    return df

def get_files(local, genomics=False, num_genomics=0):
    #### Accessing on Valeria
    """Accessing the S3 buckets using boto3 client"""
    s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
    ENDPOINT_URL = 'https://s3.valeria.science'
    bucket = 'oncotechdata'
    if local and (genomics==False):
        pyrads_chum = pd.read_csv("../../Data/PyRads_CHUM.csv")
        pyrads_iucpq = pd.read_csv("../../Data/PyRads_IUCPQ.csv")
        clinical_chum = pd.read_csv("../../Data/clinical_CHUM.csv")
        clinical_iucpq = pd.read_csv("../../Data/clinical_IUCPQ.csv")
    elif local and genomics:
        chum_clinical = pd.read_csv("../../Data/Discovery-ClinicalData.csv")
        chum_genomics1000 = pd.read_csv("../../Data/Discovery-GenomicsTop1000-March22023.csv")
        chum_genomics5000 = pd.read_csv("../../Data/Discovery-GenomicsTop5000-March22023.csv")
        chum_genomicsImmune = pd.read_csv("../../Data/ImmuneGenesDiscovery.csv")

        iucpq_genomics = pd.read_csv("../../Data/ValidationIUCPQ-March2 2023.csv")
        iucpq_clinical = pd.read_csv("../../Data/ValidationClinicalData.csv")
    
    elif not(local) and not(genomics):
        pyrads_chum = pd.read_csv(f"s3://{bucket}/data/PyRads_CHUM.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        pyrads_iucpq = pd.read_csv(f"s3://{bucket}/data/PyRads_IUCPQ.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        clinical_chum = pd.read_csv(f"s3://{bucket}/data/clinical_CHUM.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        clinical_iucpq = pd.read_csv(f"s3://{bucket}/data/clinical_IUCPQ.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
    #### Accessing local
    elif not(local) and genomics:
        chum_clinical = pd.read_csv(f"s3://{bucket}/data/Discovery-ClinicalData.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        chum_genomics1000 = pd.read_csv(f"s3://{bucket}/data/Discovery-GenomicsTop1000-March22023.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        chum_genomics5000 = pd.read_csv(f"s3://{bucket}/data/Discovery-GenomicsTop5000-March22023.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        chum_genomicsImmune = pd.read_csv(f"s3://{bucket}/data/ImmuneGenesDiscovery.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})

        iucpq_genomics = pd.read_csv(f"s3://{bucket}/data/ValidationIUCPQ-March2 2023.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        iucpq_clinical = pd.read_csv(f"s3://{bucket}/data/ValidationClinicalData.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})

    if not(genomics):
        pyrads_chum.drop(columns="Unnamed: 0", inplace=True)
        pyrads_iucpq.drop(columns="Unnamed: 0", inplace=True)
        clinical_chum.drop(columns="Unnamed: 0", inplace=True)
        clinical_iucpq.drop(columns="Unnamed: 0", inplace=True)

        chum = (clinical_chum, pyrads_chum)
        iucpq = (clinical_iucpq, pyrads_iucpq)
        return chum, iucpq

    elif genomics:
        iucpq_genomics_prep = prep_genomics(iucpq_genomics)
        if num_genomics == 1000:
            chum_genomics_prep = prep_genomics(chum_genomics1000)
        elif num_genomics == 5000:
            chum_genomics_prep = prep_genomics(chum_genomics5000)
        elif num_genomics == "Immune":
            chum_genomics_prep = prep_genomics(chum_genomicsImmune)

    chum = (chum_clinical, chum_genomics_prep)
    iucpq = (iucpq_clinical, iucpq_genomics_prep)
    return chum, iucpq


def get_norm_data(local, predictor, classification=False, univar=False, ecog=False, genomics=False, num_genomics=0):
'''
Returns the prepared and normalized data
local: True = loads from local path, False = loads from Valeria
predictor: clinical endpoint ("os_days", "pfs_days", "pdl1_tps")
classification: True = binarize the clinical entpoint, False = means regression task is performed and the clinical endpoint stays continuous
univar: True = loads the clinical data like age, smoking habit, ecog,..., False = does not add more clinical variables to the dataset
ecog: True = adds the ecog value of a patient to the dataset, False = does not add the ecog value
genomics: True = load Genomics data with predictor, False = load Radiomics data with predictor
num_genomics: 0 = ignore this if Radiomics are loaded, 1000 = load the 1000 genes, 5000 = load the 5000 genes
'''
    chum, iucpq = get_files(local, genomics, num_genomics)
    chum_clinical, chum_features = chum
    iucpq_clinical, iucpq_features = iucpq

    if univar:
        list_clin_feats = ["age", "smoking_habit", "ecog_status","sex", "first_line_io", "histology_group"]
        ## CHUM
        # merge clinical data to radiation data
        # left merge removes the patients with an undefined os_days
        chum_merge = chum_features.merge(chum_clinical[["oncotech_id",predictor]+list_clin_feats],how="left", on="oncotech_id")
        print("{} patients with clinical and radiation data for CHUM.".format(len(chum_merge)))
        chum_merge.drop(columns="oncotech_id", inplace=True)
        chum_merge = make_cat_to_int(chum_merge)

        ## IUCPQ
        # merge clinical data to radiation data
        iucpq_merge = iucpq_features.merge(iucpq_clinical[["oncotech_id",predictor]+list_clin_feats],how="inner", on="oncotech_id")
        print("{} patients with clinical and radiation data for IUCPQ.".format(len(iucpq_merge)))
        iucpq_merge.drop(columns="oncotech_id", inplace=True)
        iucpq_merge = make_cat_to_int(iucpq_merge)

    elif ecog:
        list_clin_feats = ["ecog_status"]
        ## CHUM
        # merge clinical data to radiation data
        # left merge removes the patients with an undefined os_days
        chum_merge = chum_features.merge(chum_clinical[["oncotech_id",predictor]+list_clin_feats],how="left", on="oncotech_id")
        print("{} patients with clinical and radiation data for CHUM.".format(len(chum_merge)))
        chum_merge.drop(columns="oncotech_id", inplace=True)
        # remove values of nan values
        chum_merge_pdl_rem = chum_merge.dropna(axis=0, subset = list_clin_feats)
        print("{} patients with predictor {} and ecog data for CHUM.".format(len(chum_merge_pdl_rem), predictor))
        chum_merge_pdl_rem = make_cat_to_int(chum_merge_pdl_rem, ecog=True)
        chum_merge = chum_merge_pdl_rem.copy()

        ## IUCPQ
        # merge clinical data to radiation data
        # left merge removes the patients with an undefined os_days
        iucpq_merge = iucpq_features.merge(iucpq_clinical[["oncotech_id",predictor]+list_clin_feats],how="left", on="oncotech_id")
        print("{} patients with clinical and radiation data for IUCPQ.".format(len(iucpq_merge)))
        # remove values of nan values
        iucpq_merge_pdl_rem = iucpq_merge.dropna(axis=0, subset=["ecog_status"])
        print("{} patients with os, radiation, ecog and pdl1 data for IUCPQ.".format(len(iucpq_merge_pdl_rem)))
        iucpq_merge_pdl_rem.drop(columns="oncotech_id", inplace=True)
        iucpq_merge_pdl_rem = make_cat_to_int(iucpq_merge_pdl_rem, ecog=True)
        iucpq_merge = iucpq_merge_pdl_rem.copy()

    else:
        ## CHUM
        # merge clinical data to radiation data
        # left merge removes the patients with an undefined os_days
        chum_merge = chum_features.merge(chum_clinical[["oncotech_id",predictor]],how="left", on="oncotech_id")
        print("{} patients with clinical and radiation data for CHUM.".format(len(chum_merge)))
        chum_merge.drop(columns="oncotech_id", inplace=True)

        ## IUCPQ
        # merge clinical data to radiation data
        iucpq_merge = iucpq_features.merge(iucpq_clinical[["oncotech_id",predictor]],how="inner", on="oncotech_id")
        print("{} patients with clinical and radiation data for IUCPQ.".format(len(iucpq_merge)))
        iucpq_merge.drop(columns="oncotech_id", inplace=True)
    

    data_norm_chum = data_normalization(chum_merge)
    data_norm_iucpq = data_normalization(iucpq_merge)

    data_norm_chum.dropna(axis=0, subset=predictor, inplace=True)
    data_norm_iucpq.dropna(axis=0, subset=predictor, inplace=True)
    print("{} patients after drop NaN - CHUM.".format(len(data_norm_chum)))
    print("{} patients after drop NaN - IUCPQ.".format(len(data_norm_iucpq)))
    data_norm_chum.reset_index(inplace=True, drop=True)
    data_norm_iucpq.reset_index(inplace=True, drop=True)

    genes_not_in_vali = list(set(data_norm_chum.columns)- set(data_norm_iucpq.columns))
    data_norm_chum = data_norm_chum.drop(columns=genes_not_in_vali)

    if classification:
        data_norm_chum_class = add_os_day_class(data_norm_chum, predictor)
        print("{} patients with clinical and radiation data for CHUM and classification.".format(len(data_norm_chum_class)))
        data_norm_iucpq_class = add_os_day_class(data_norm_iucpq, predictor)
        print("{} patients with clinical and radiation data for IUCPQ and classification.".format(len(data_norm_iucpq_class)))

        return data_norm_chum_class, data_norm_iucpq_class

    return data_norm_chum, data_norm_iucpq
