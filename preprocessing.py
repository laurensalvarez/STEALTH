import copy
import math
import sys, random
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import *

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
POSITIVE_OUTCOME = params.positive_outcome
NEGATIVE_OUTCOME = params.negative_outcome


def trim(df, dataset):
    X = df.copy()

    if dataset == "adultscensusincome":
        X = X.drop(['fnlwgt', 'education'], axis=1)

    if dataset == "compas":
        X = X.loc[(X['Days_b_screening_arrest'] <= 30) & (X['Days_b_screening_arrest'] >= -30) & (X['is_recid'] != -1) & (X['c_charge_degree'] != "O") & (X['!probability'] != "NA")]
        X['Length_of_stay'] = (pd.to_datetime(X['c_jail_out']) - pd.to_datetime(X['c_jail_in'])).dt.days
        X = X[['Age(','c_charge_degree', 'race(', 'sex(', 'Priors_count', 'Length_of_stay', "!probability", "!Probability"]]

    if dataset == "communities":
        X['!Probability'] = X['!Probability'].values.astype('float32')
        X = X.drop(['communityname', ':Fold', ':County', ':Community', 'State'], axis=1)

    if dataset == "germancredit":
        X = X.drop(["purposeOfLoan"], axis=1)

    return X


def unifyOutcomes(df, dataset):
    if dataset == "adultscensusincome":
        df['!probability'] = np.where((df['!probability'] == " >50K"), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "bankmarketing":
        df['!probability'] = np.where((df['!probability'] == "yes" ), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "compas":
        df['!probability'] = np.where((df['!probability'] == "High"), NEGATIVE_OUTCOME, POSITIVE_OUTCOME)
        # df['!Probability'] = np.where((df['!Probability'] == "1"), NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    if dataset == "defaultcredit":
        df['!Probability'] = np.where((df['!Probability'] == 0), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "diabetes":
        df['!probability'] = np.where((df['!probability'] == "negative" ), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "germancredit":
        df['!Probability'] = np.where((df['!Probability'] == 1), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "heart":
        df['!Probability'] = np.where((df['!Probability'] == 0), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "studentperformance":
        df['!Probability'] = np.where((df['!Probability'] > 14), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    return df

def makeBinary(df, dataset):
    """turning sensitive features binary."""
    if dataset == "diabetes" or "heart":
        df['Age('] = np.where((df['Age('] > 25), 0, 1)

    if dataset == "compas":
        df['race('] = np.where((df['race('] == 2), 1, 0)
        df['sex('] = np.where((df['sex('] == 1), 0, 1)
        df['Age('] = np.where((df['Age('] > 25), 0, 1)

    if dataset == "germancredit":
        df['sav('] = np.where((df['sav('] == 0) | (df['sav('] == 1) | (df['sav('] ==  4), 0, 1)
        df['Age('] = np.where((df['Age('] > 25), 0, 1)
        df['sex('] = np.where((df['sex('] == 1), 0, 1)

    if dataset == "adultscensusincome":
        df['sex('] = np.where((df['sex('] == 1), 0, 1)
        df['race('] = np.where((df['race('] == 5), 1, 0)

    if dataset == "studentperformance":
        df['sex('] = np.where((df['sex('] == 1), 0, 1)

    if dataset == "bankmarketing":
        df['Age('] = np.where((df['Age('] > 25), 0, 1)
        df['marital('] = np.where((df['marital('] == 3), 1, 0)
        df['education('] = np.where((df['education('] == 6) | (df['education('] == 7), 1, 0)

    if dataset == "defaultcredit":
        df['SEX('] = np.where((df['SEX('] == 1), 1, 0)
        df['MARRIAGE('] = np.where((df['MARRIAGE('] == 1), 1, 0)
        df['EDUCATION('] = np.where((df['EDUCATION('] == 1) | (df['EDUCATION('] == 2), 1, 0)
        df['AGE('] = np.where((df['AGE('] > 25), 0, 1)
        df['LIMIT_BAL('] = np.where((df['LIMIT_BAL('] > 25250), 1, 0)

    return df

def preprocess(path,dataset):
    raw_datadf = pd.read_csv(path, header=0, na_values=['?', ' ?'])
    raw_copy = raw_datadf.copy()
    if dataset != "compas":
        """1: remove rows with missing values."""
        # raw_copy = raw_copy.replace("?", float('NaN'), regex = True) #replace blanks with NaN
        raw_copy = raw_copy.replace(r'^s*$', float('NaN'), regex = True) #replace blanks with NaN
        raw_copy.dropna(inplace = True) #removes all rows with blank cells

    """1b: remove unwanted features or do recommended trimming."""
    trimmeddf = trim(raw_copy, dataset)
    """2: change cont. to categorical (i.e. age >= 25 = old; age < 25 is young)."""
    """3: change non-numerical to numerical (i.e. female = 0; male = 1)."""
    """4: make y's binary (i.e. 0 is neg outcome; 1 is pos outcome)."""
    prettydf = unifyOutcomes(trimmeddf, dataset)

    return prettydf



if __name__ == "__main__":
    datasets = ["adultscensusincome.csv","bankmarketing.csv", "compas.csv", "communities.csv", "defaultcredit.csv", "diabetes.csv",  "germancredit.csv", "heart.csv", "studentperformance.csv"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        filename = dataset[:-4]
        # df = pd.DataFrame(columns=columns)
        path =  "./datasets/" + filename + ".csv"
        df = preprocess(path,filename)
        df.to_csv("./datasets/processed/"  + filename + "_p.csv", index = False)
