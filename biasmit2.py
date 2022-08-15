import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os,sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

# sys.path.append(os.path.abspath('..'))
from utils import *
from smote import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples

params = Params("./model_configurations/experiment_params.json")
np.random.seed(params.seed)

def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
    cols = df.columns
    smt = smote(df)
    df = smt.run()
    df.columns = cols
    return df

def getMetrics(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, samples, yname, model_num):

    recall = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall', yname)
    precision = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision', yname)
    accuracy = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', yname)
    F1 = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1', yname)
    AOD = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod', yname)
    EOD =measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod', yname)
    SPD = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD', yname)
    FA0 = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'FA0', yname)
    FA1 = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'FA1', yname)
    DI = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI', yname)
    FAR = measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far', yname)

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, FA0, FA1, DI, protected_attribute, samples, model_num]


## Load dataset
def getOGscores(path, scaler, clf, dataset):
    lilprobs = ["adultscensusincome", "diabetes", "bankmarketing"]
    dataset_og = pd.read_csv(path)
    y_s = [col for col in dataset_og.columns if "!" in col]
    yname = y_s[0]

    if dataset in lilprobs:
        y_og = dataset_og[yname].values
        dataset_og.drop([yname], axis=1, inplace=True)
    else:
        y_og = dataset_og[yname].values
        dataset_og.drop([yname], axis=1, inplace=True)
    # print(dataset_og.head())
    sensitive_features = [col for col in dataset_og.columns if "(" in col]
    sorted(sensitive_features)

    og_cat_features = []
    for col in dataset_og.columns:
        if col not in sensitive_features and col[0].islower():
            # print(col)
            og_cat_features.append(col)
            ## Drop categorical features
            dataset_og.drop([col], axis=1, inplace=True)

    # print(dataset_og.head())

    dataset_og = pd.DataFrame(scaler.fit_transform(dataset_og.values),columns = dataset_og.columns)

    X_train,X_test,y_train,y_test = train_test_split(dataset_og, y_og, test_size=0.2, shuffle = True)

    test_df = pd.DataFrame(X_test, columns = dataset_og.columns)

    # # Check Original Score from Original Training set
    colmetrics = {}
    rows = []
    for i in sensitive_features:
        colmetrics[i] = getMetrics(test_df, clf, X_train, y_train, X_test, y_test, i, len(test_df.index), yname, 0)

    for col, metric_row in colmetrics.items():
        rows.append(metric_row)
    original_scores = pd.DataFrame(rows,columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'feature', 'sample_size', 'model_num'])

    return original_scores, test_df, X_test, y_test, yname, sensitive_features
    # # Check Original Score from Original Training set
def getSurrogates(cluster_path, sensitive_features, scaler, clf, test_df, X_test, y_test, yname):
    sur = pd.read_csv(cluster_path)
    surrogate = sur.copy()
    surrogate_cat_features = []
    keep = ["samples", "predicted"]
    y_pred = "predicted"
    sample_metrics ={}

    for col in surrogate.columns:
        if col not in keep:
            if col not in sensitive_features and col[0].islower():
                surrogate_cat_features.append(col)
                surrogate.drop([col],axis=1, inplace=True)

    samples = copy.deepcopy(surrogate["samples"].tolist())
    sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
    m = 1
    for s in sortedsamples:
        # print(s)
        # print(m)
        # print(surrogate.head())
        surr_colmetrics = []
        surrogate_1 = copy.deepcopy(surrogate)
        surrogate_1.drop(surrogate.loc[surrogate['samples']!= s].index, inplace=True)
        # print(surrogate_1.head())
        surrogate_ground_truth = surrogate_1[yname].values
        surrogate_y = surrogate_1[y_pred].values
        surrogate_1.drop([yname,y_pred,'samples','Unrelated_column_one'],axis=1, inplace=True)
        scaled_surrogate = pd.DataFrame(scaler.fit_transform(surrogate_1.values),columns = surrogate_1.columns)
        # print(scaled_surrogate.head())
        for i in sensitive_features:
            # print(s)
            # print(m)
            surr_colmetrics.insert(m,getMetrics(test_df, clf, scaled_surrogate.values, surrogate_y, X_test, y_test, i, s, yname, m))
        # print(surr_colmetrics)
        sample_metrics[s] = surr_colmetrics
        m+=1

    # print(sample_metrics)
    return sample_metrics
#apply smote to OG, Surrogates

def main():
    datasets = ["compas"]#["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        og_path =  "./datasets/processed/" + dataset + "_p.csv"
        cluster_path =  "./output/cluster_preds/" + dataset + ".csv"
        scaler = MinMaxScaler()
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        original_scores, test_df, X_test, y_test, yname, sensitive_features = getOGscores(og_path, scaler, clf, dataset)
        # original_scores.append(original_scores)
        # print(original_scores.head(10))
        s_metrics = getSurrogates(cluster_path, sensitive_features, scaler, clf, test_df, X_test, y_test, yname)
        surr_rows = []
        print(s_metrics)
        for s,c in s_metrics.items():
            for metric_row in c:
                surr_rows.append(metric_row)
        # print(s_metrics)
        # print(surr_rows)
        surr_pd = pd.DataFrame(surr_rows,columns = original_scores.columns)
        # print(surr_pd.head(10))
        original_scores = original_scores.append(surr_pd)
        print(original_scores.head(10))


        original_scores.to_csv("./bias/" +  dataset + "_LR.csv", index=False)



if __name__ == "__main__":
    main()
