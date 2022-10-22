"""
model extraction
"""
import warnings
warnings.filterwarnings('ignore')

from utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import math, re, random, statistics, sys, pprint
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

def getMetrics(test_df, y_test, y_pred, biased_col, samples, yname, rep, learner):

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
    MSE = round(mean_squared_error(y_test, y_pred),3)

    return [recall, precision, accuracy, F1, FA0, FA1, MSE, AOD, EOD, SPD, DI, biased_col, samples, rep, learner]



def main():
    datasets = ["communities","heart", "diabetes", "germancredit", "studentperformance", "compas", "bankmarketing", "defaultcredit", "adultscensusincome"]
    # datasets = ["diabetes", "communities", "compas", "studentperformance", "bankmarketing", "adultscensusincome", "defaultcredit"]
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
    # base3 = LinearSVC()

    pbar = tqdm(datasets)
    for dataset in pbar:
        klist = keywords[dataset]
        results = []
        feat_importance_tuple_list = []
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

                full_RF = RandomForestClassifier()
                full_RF.fit(xtrain, ytrain)
                f_RF_pred = full_RF.predict(xtest)
                results.append(getMetrics(testing, ytest, f_RF_pred, keyword, len(ytrain), yname, i, "RF" ))

                full_import = full_RF.feature_importances_
                sorted_indices = np.argsort(full_import)[::-1]
                for feat in range(xtrain.shape[1]):
                    feat_importance_tuple_list.append([keyword, i ,len(ytrain), 0, cols[sorted_indices[feat]], round(full_import[sorted_indices[feat]],3)])

                full_LSR = LogisticRegression()
                full_LSR.fit(xtrain, ytrain)
                f_LSR_pred = full_LSR.predict(xtest)
                results.append(getMetrics(testing, ytest, f_LSR_pred, keyword, len(ytrain), yname, i, "LSR" ))

                full_SVC = LinearSVC()
                full_SVC.fit(xtrain, ytrain)
                f_SVC_pred = full_SVC.predict(xtest)
                results.append(getMetrics(testing, ytest, f_SVC_pred, keyword, len(ytrain), yname, i, "SVC" ))

                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)

                # #O-CART
                # tree = DecisionTreeClassifier(min_samples_leaf = enough, random_state = i)
                # tree.fit(xtrain, ytrain)
                # leaves = tree.get_n_leaves()
                # leaf_indexes = tree.apply(xtrain)
                # new_leaves = list(set(leaf_indexes))=


                if dataset in ["compas", "bankmarketing", "defaultcredit", "adultscensusincome"]:
                    treatment = [4,5,7,8,9,10]
                else:
                    treatment = [2,3,4,5]

                for num_points in treatment:
                    subset_x, clustered_y = clusterGroups(root, cols, num_points)

                    #O-CART leaf extraction
                    # subset_x = []
                    # for l in new_leaves:
                    #     indices = [i for i, x in enumerate(leaf_indexes) if x == l] #gives indexes of leaf_num aka index of all points in leaf
                    #     x_index = np.random.choice(indices, replace = False, size = num_points)
                    #     for t in x_index:
                    #         subset_x.append(xtrain[t])

                    RF_probed_y = full_RF.predict(deepcopy(subset_x))
                    RF_surrogate = RandomForestClassifier().fit(subset_x, RF_probed_y)
                    RF_surr_pred = RF_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, RF_surr_pred, keyword, len(subset_x), yname, i, "RF_RF" ))

                    LSR_probed_y = full_LSR.predict(deepcopy(subset_x))
                    LSR_surrogate = RandomForestClassifier().fit(subset_x, LSR_probed_y)
                    LSR_surr_pred = LSR_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, LSR_surr_pred, keyword, len(subset_x), yname, i, "LSR_RF" ))

                    SVC_probed_y = full_SVC.predict(deepcopy(subset_x))
                    SVC_surrogate = RandomForestClassifier().fit(subset_x, SVC_probed_y)
                    SVC_surr_pred = SVC_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, SVC_surr_pred, keyword, len(subset_x), yname, i, "SVC_RF" ))

                    RF_surro_import = RF_surrogate.feature_importances_
                    RF_sorted_indices = np.argsort(RF_surro_import)[::-1]

                    LSR_surro_import = LSR_surrogate.feature_importances_
                    LSR_sorted_indices = np.argsort(LSR_surro_import)[::-1]

                    SVC_surro_import = SVC_surrogate.feature_importances_
                    SVC_sorted_indices = np.argsort(SVC_surro_import)[::-1]
                    for feat in range(xtrain.shape[1]):
                        feat_importance_tuple_list.append([keyword, i, len(subset_x), 1, cols[RF_sorted_indices[feat]], round(RF_surro_import[RF_sorted_indices[feat]],3)])
                        feat_importance_tuple_list.append([keyword, i, len(subset_x), 2, cols[LSR_sorted_indices[feat]], round(LSR_surro_import[LSR_sorted_indices[feat]],3)])
                        feat_importance_tuple_list.append([keyword, i, len(subset_x), 3, cols[SVC_sorted_indices[feat]], round(SVC_surro_import[SVC_sorted_indices[feat]],3)])

            #results.items() #(model, metric_row)
        metrics = pd.DataFrame(results, columns = ["recall+", "precision+", "accuracy+", "F1+", "FA0-", "FA1-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-", "biased_col", "samples", "rep", "learner"] )
        metrics.to_csv("./output/features/" +  dataset + "_FM.csv", index=False)
        feat_imp = pd.DataFrame(feat_importance_tuple_list, columns = [ "biased_col", "rep", "samples", "learner", "feature", "importance" ] )
        feat_imp.to_csv("./output/features/" +  dataset + "_feat.csv", index=False)
        # print ('-'*55)
            # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
            # print ('-'*55)






if __name__ == "__main__":
    main()
