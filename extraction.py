"""
model extraction
"""
import warnings
warnings.filterwarnings('ignore')

from utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import math, re, random, statistics, sys
import numpy as np
import pandas as pd
import collections
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2
from Measure import measure_final_score

params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)

def clusterGroups(root, features, num_points):
    if num_points != 1:
        EDT = getLeafData2(root, num_points)
        X, y = getXY2(EDT)
        df = pd.DataFrame(X, columns=features)
    else:
        MedianTable = leafmedians2(root)
        X, y = getXY2(MedianTable)
        df = pd.DataFrame(X, columns=features)

    return df, y

def getMetrics(test_df, y_test, y_pred, biased_col, samples, yname, rep):

    recall = measure_final_score(test_df, y_test, y_pred, biased_col, 'recall', yname)
    precision = measure_final_score(test_df, y_test, y_pred, biased_col, 'precision', yname)
    accuracy = measure_final_score(test_df, y_test, y_pred, biased_col, 'accuracy', yname)
    F1 = measure_final_score(test_df, y_test, y_pred, biased_col, 'F1', yname)
    AOD = measure_final_score(test_df, y_test, y_pred, biased_col, 'aod', yname)
    EOD =measure_final_score(test_df, y_test, y_pred, biased_col, 'eod', yname)
    SPD = measure_final_score(test_df, y_test, y_pred, biased_col, 'SPD', yname)
    FA0 = measure_final_score(test_df, y_test, y_pred, biased_col, 'FA0', yname)
    FA1 = measure_final_score(test_df, y_test, y_pred, biased_col, 'FA1', yname)
    DI = measure_final_score(test_df, y_test, y_pred, biased_col, 'DI', yname)

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, DI, FA0, FA1, biased_col, samples, rep]



def main():
    datasets = ["heart", "germancredit", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "adultscensusincome", "defaultcredit"]
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['SEX('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['Age('],
                'studentperformance': ['sex(']
                }
    # base = RandomForestClassifier()
    # base2 = LogisticRegression()
    # base3 = MLPClassifier()

    pbar = tqdm(datasets)
    for dataset in pbar:
        klist = keywords[dataset]
        results = []
        for keyword in klist:
            pbar.set_description("Processing %s" % dataset)
            path =  "./datasets/processed/" + dataset + "_p.csv"

            X = pd.read_csv(path)
            y_s = [col for col in X.columns if "!" in col]
            yname = y_s[0]
            y = X[yname].values
            X.drop([yname], axis=1, inplace=True)

            sensitive_features = [col for col in X.columns if "(" in col]
            sorted(sensitive_features)

            cat_features_not_encoded = []
            for col in X.columns:
                if col not in sensitive_features and col[0].islower():
                    cat_features_not_encoded.append(col)

            X = pd.get_dummies(data=X, columns=cat_features_not_encoded)
            cols = [c for c in X]

            cat_features_encoded = []
            for col in X.columns:
                if col not in sensitive_features and col[0].islower():
                    cat_features_encoded.append(col)

            sensa_indc = [cols.index(col) for col in sensitive_features]
            categorical = [cols.index(c) for c in cat_features_encoded]

            for i in range(10):
                i += 1

                xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size=0.2, random_state = i)

                ss = MinMaxScaler().fit(xtrain)
                xtrain = ss.transform(xtrain)
                xtest = ss.transform(xtest)

                testing = pd.DataFrame(deepcopy(xtest), columns = cols)
                training = pd.DataFrame(deepcopy(xtrain), columns = cols)
                training[yname] = deepcopy(ytrain)

                full_clf = RandomForestClassifier()
                full_clf.fit(xtrain, ytrain)
                fc_pred = full_clf.predict(xtest)
                results.append(getMetrics(testing, ytest, fc_pred, keyword, len(ytrain), yname, i ))

                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)

                treatment = [1,2,3,4,5]
                for num_points in treatment:
                    clustered_df, clustered_y = clusterGroups(root, cols, num_points)

                    probed_y = full_clf.predict(deepcopy(clustered_df))

                    surrogate = RandomForestClassifier().fit(clustered_df, probed_y)
                    surr_pred = surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, surr_pred, keyword, len(probed_y), yname, i ))

            #results.items() #(model, metric_row)
        metrics = pd.DataFrame(results, columns = ["recall+", "precision+", "accuracy+", "F1+", "AOD-", "EOD-", "SPD-", "DI-", "FA0-", "FA1-", "biased_col", "samples", "rep"] )
        metrics.to_csv("./output/surro_2/" +  dataset + "_" + keyword +".csv", index=False)
        # print ('-'*55)
            # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
            # print ('-'*55)






if __name__ == "__main__":
    main()
