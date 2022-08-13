import copy, math, sys, random
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
        X = X[['Age(','c_charge_degree', 'race(', 'sex(', 'Priors_count', 'Length_of_stay', "!Probability"]]

    if dataset == "communities":
        X['!Probability'] = X['!Probability'].values.astype('float32')
        X = X.drop(['communityname', ':Fold', ':County', ':Community', 'State'], axis=1)

    return X


def makeBinary(df, dataset):
    """turning sensitive features binary."""
    if dataset == "adultscensusincome":
        df['sex('] = np.where((df['sex('] == "Female"), 0, 1)
        df['race('] = np.where((df['race('] == "White"), 1, 0)
        df['!probability'] = np.where((df['!probability'] == " >50K"), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "diabetes":
        mean = df.loc[:,'Age('].mean()
        df['Age('] = np.where((df['Age('] >= mean), 0, 1)
        df['!probability'] = np.where((df['!probability'] == "negative" ), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "compas":
        df['race('] = np.where((df['race('] != 'Caucasian'), 0, 1)
        df['sex('] = np.where((df['sex('] == 'Female'), 1, 0)
        # df['!probability'] = np.where((df['!probability'] == "High"), NEGATIVE_OUTCOME, POSITIVE_OUTCOME)
        df['!Probability'] = np.where((df['!Probability'] == 0), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "communities":
        high_vcrime_threshold = 50
        majority = 53
        m = df['Racepctwhite(']
        majority_cutoff = np.percentile(m, majority)
        y = df['!Probability']
        y_cutoff = np.percentile(y, high_vcrime_threshold)
        df['!Probability'] = np.where((df['!Probability'] > y_cutoff), NEGATIVE_OUTCOME, POSITIVE_OUTCOME)
        df['Racepctwhite('] = np.where(df['Racepctwhite('] > majority_cutoff, 1, 0)

    if dataset == "germancredit":
        ## Change symbolics to numerics
        df = df.drop(['status_of_existing_account','Duration_month','purpose','Install_rate_percentage_disposalble','debtors','Present_residence','property','installment_plans','housing','Num_existng_credits','job','Num_people_liable_maintenance','telephone'],axis=1)
        df['sex('] = np.where((df['sex('] == 'A92') | (df['sex('] == 'A95'), 0, 1)
        df['sav('] = np.where((df['sav('] == 'A63') | (df['sav('] == 'A64'), 1, 0)
        df['Age('] = np.where(df['Age('] >= 25, 1, 0)
        df['f_w('] = np.where((df['f_w('] == 'A201'), 0, 1)

        df['credit_history'] = np.where(df['credit_history'] == 'A30', 1, df['credit_history'])
        df['credit_history'] = np.where(df['credit_history'] == 'A31', 1, df['credit_history'])
        df['credit_history'] = np.where(df['credit_history'] == 'A32', 1, df['credit_history'])
        df['credit_history'] = np.where(df['credit_history'] == 'A33', 2, df['credit_history'])
        df['credit_history'] = np.where(df['credit_history'] == 'A34', 3, df['credit_history'])



        df['employment'] = np.where(df['employment'] == 'A72', 1, df['employment'])
        df['employment'] = np.where(df['employment'] == 'A73', 1, df['employment'])
        df['employment'] = np.where(df['employment'] == 'A74', 2, df['employment'])
        df['employment'] = np.where(df['employment'] == 'A75', 2, df['employment'])
        df['employment'] = np.where(df['employment'] == 'A71', 3, df['employment'])

        df['!Probability'] = np.where(df['!Probability'] == 2, NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    if dataset == "heart":
        mean = df.loc[:,'Age('].mean()
        df['Age('] = np.where((df['Age('] >= mean), 0, 1)
        df['!Probability'] = np.where((df['!Probability'] == 0), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "studentperformance":
        df['sex('] = np.where(df['sex('] == 'M', 1, 0)
        df['schoolsup'] = np.where(df['schoolsup'] == 'yes', 1, 0)
        df['famsup'] = np.where(df['famsup'] == 'yes', 1, 0)
        df['paid'] = np.where(df['paid'] == 'yes', 1, 0)
        df['activities'] = np.where(df['activities'] == 'yes', 1, 0)
        df['nursery'] = np.where(df['nursery'] == 'yes', 1, 0)
        df['higher'] = np.where(df['higher'] == 'yes', 1, 0)
        df['internet'] = np.where(df['internet'] == 'yes', 1, 0)
        df['romantic'] = np.where(df['romantic'] == 'yes', 1, 0)
        df['!Probability'] = np.where((df['!Probability'] > 12), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "bankmarketing":
        df['Age('] = np.where((df['Age('] > 25), 0, 1)
        df['marital('] = np.where((df['marital('] == 3), 1, 0)
        df['education('] = np.where((df['education('] == 6) | (df['education('] == 7), 1, 0)
        df['!probability'] = np.where((df['!probability'] == "yes" ), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    if dataset == "defaultcredit":
        df['SEX('] = np.where((df['SEX('] == 2), 0, 1)
        df['MARRIAGE('] = np.where((df['MARRIAGE('] == 1), 1, 0)
        df['EDUCATION('] = np.where((df['EDUCATION('] == 1) | (df['EDUCATION('] == 2), 1, 0)
        df['LIMIT_BAL('] = np.where((df['LIMIT_BAL('] > 25250), 1, 0)
        df['!Probability'] = np.where((df['!Probability'] == 1), POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

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
    """2: change cont. to categorical (i.e. age >= 25 = old; age < 25 is young) & non-numerical to numerical (i.e. female = 0; male = 1).  make y's binary (i.e. 0 is neg outcome; 1 is pos outcome)."""
    bindf = makeBinary(trimmeddf, dataset)

    return bindf



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
