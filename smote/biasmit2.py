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

from smote import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples
from utils import *
params = Params("./model_configurations/experiment_params.json")
np.random.seed(params.seed)

def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
    cols = df.columns
    smt = smote(df)
    df = smt.run()
    df.columns = cols
    return df

def getMetrics(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, samples, yname, model_num, smoted):

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

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, FA0, FA1, DI, protected_attribute, samples, model_num, smoted]


## Load dataset
def getOGscores(path, scaler, dataset):
    # lilprobs = ["adultscensusincome", "diabetes", "bankmarketing"]
    dataset_og = pd.read_csv(path)
    y_s = [col for col in dataset_og.columns if "!" in col]
    yname = y_s[0]
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

    clf = RandomForestClassifier(n_estimators=100)
    colmetrics = []
    for i in sensitive_features:
        colmetrics.append(getMetrics(test_df, clf, X_train, y_train, X_test, y_test, i, len(X_train), yname, 0, 0))

    #applying apply_smote

    training_df = pd.DataFrame(copy.deepcopy(X_train), columns = dataset_og.columns)
    training_df[yname] = y_train
    training_df = apply_smote(training_df)
    y_train_smote = training_df[yname].values
    training_df.drop([yname], axis = 1, inplace = True)
    clf2 = RandomForestClassifier(n_estimators=100)

    for i in sensitive_features:
        colmetrics.append(getMetrics(test_df, clf2, training_df.values, y_train_smote, X_test, y_test, i, len(X_train), yname, 0, -1))

    original_scores = pd.DataFrame(colmetrics,columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'feature', 'sample_size', 'model_num', 'smote-ified'])
    training_df[yname] = y_train

    zero_zero = len(training_df[(training_df[yname] == 0) & (training_df[sensitive_features[0]] == 0)])
    zero_one = len(training_df[(training_df[yname] == 0) & (training_df[sensitive_features[0]] == 1)])
    one_zero = len(training_df[(training_df[yname] == 1) & (training_df[sensitive_features[0]] == 0)])
    one_one = len(training_df[(training_df[yname] == 1) & (training_df[sensitive_features[0]] == 1)])
    maximum = max(zero_zero,zero_one,one_zero,one_one)

    zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
    one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
    one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1
    df_zero_zero = training_df[(training_df[yname] == 0) & (training_df[sensitive_features[0]] == 0)]
    df_one_zero = training_df[(training_df[yname] == 1) & (training_df[sensitive_features[0]] == 0)]
    df_one_one = training_df[(training_df[yname] == 1) & (training_df[sensitive_features[0]] == 1)]

    df_zero_zero[sensitive_features[0]] = df_zero_zero[sensitive_features[0]].astype(str)
    df_zero_zero[sensitive_features[0]] = df_zero_zero[sensitive_features[0]].astype(str)
    df_one_zero[sensitive_features[0]] = df_one_zero[sensitive_features[0]].astype(str)
    df_one_one[sensitive_features[0]] = df_one_one[sensitive_features[0]].astype(str)


    df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero, dataset)
    df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero, dataset)
    df_one_one = generate_samples(one_one_to_be_incresed,df_one_one, dataset)

    df = df_zero_zero.append(df_one_zero)
    df = df.append(df_one_one)

    df[sensitive_features[0]] = df[sensitive_features[0]].astype(float)

    df_zero_one = training_df[(training_df[yname] == 0) & (training_df[sensitive_features[0]] == 1)]
    df = df.append(df_zero_one)

    X_train, y_train = df.loc[:, df.columns != yname], df[yname]

    clf3 = RandomForestClassifier(n_estimators=100)

    for i in sensitive_features:
        colmetrics.append(getMetrics(test_df, clf3, X_train, y_train, X_test, y_test, i, len(X_train), yname, 0, 1))



    return original_scores, test_df, X_test, y_test, yname, sensitive_features
    # # Check Original Score from Original Training set
def getSurrogates(cluster_path, sensitive_features, scaler, test_df, X_test, y_test, yname):
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
        clf = RandomForestClassifier(n_estimators=100)
        surr_colmetrics = []
        surrogate_1 = copy.deepcopy(surrogate)
        surrogate_1.drop(surrogate.loc[surrogate['samples']!= s].index, inplace=True)
        surrogate_ground_truth = surrogate_1[yname].values
        surrogate_y = surrogate_1[y_pred].values
        surrogate_1.drop([yname,y_pred,'samples','Unrelated_column_one'],axis=1, inplace=True)
        scaled_surrogate = pd.DataFrame(scaler.fit_transform(surrogate_1.values),columns = surrogate_1.columns)

        for i in sensitive_features:
            # print(surrogate_y)
            surr_colmetrics.insert(m,getMetrics(test_df, clf, scaled_surrogate.values, surrogate_y, X_test, y_test, i, s, yname, m, 0))

        training_df = pd.DataFrame(copy.deepcopy(scaled_surrogate.values), columns = surrogate_1.columns)
        training_df[yname] = surrogate_y
        training_df = apply_smote(training_df)
        y_train_smote = training_df[yname].values
        training_df.drop([yname], axis = 1, inplace = True)
        clf2 = RandomForestClassifier(n_estimators=100)
        for i in sensitive_features:
            surr_colmetrics.insert(m,getMetrics(test_df, clf, scaled_surrogate.values, surrogate_y, X_test, y_test, i, s, yname, m, 1))

        sample_metrics[s] = surr_colmetrics

        m+=1

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
        original_scores, test_df, X_test, y_test, yname, sensitive_features = getOGscores(og_path, scaler, dataset)

        s_metrics = getSurrogates(cluster_path, sensitive_features, scaler, test_df, X_test, y_test, yname)
        surr_rows = []
        # print(s_metrics)
        for s,c in s_metrics.items():
            for metric_row in c:
                surr_rows.append(metric_row)
        surr_pd = pd.DataFrame(surr_rows,columns = original_scores.columns)
        original_scores = original_scores.append(surr_pd)


        original_scores.to_csv("./bias/" +  dataset + "_RF.csv", index=False)



if __name__ == "__main__":
    main()
