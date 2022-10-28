"""
model extraction
"""
import warnings
warnings.filterwarnings('ignore')

from utils import *
from adversarial_models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import math, re, random, statistics, sys, pprint
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import lime
import lime.lime_tabular

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2
from Measure import measure_final_score

params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
##
## The models f and psi.  We discriminate based on sensitive for f and consider innoc feature for explanation
#
# the biased model
class biased_model_f():
    def __init__(self, sensa_indc):
        self.sensa_indc = sensa_indc
    # Decision rule: classify negative outcome if underrepresented class
    def predict(self,X):
        predictions = np.array([params.negative_outcome if x[self.sensa_indc] == 0 else params.positive_outcome for x in X])
        #### if the class balance is unbalanced add the random flips to create examples in each class distribution ####
        indices = np.random.choice(np.arange(predictions.size), replace = False, size = int(predictions.size * 0.10))
        for i in indices:
            if predictions[i] == params.negative_outcome:
                predictions[i] = params.positive_outcome
            else:
                predictions[i] = params.negative_outcome

        return predictions

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# the display model with one unrelated feature
class innocuous_model_psi:
    def __init__(self, inno_indc):
        self.inno_indc = inno_indc
    # Decision rule: classify according to innoc indc
    def predict_proba(self, X):
        return one_hot_encode(np.array([params.negative_outcome if x[self.inno_indc] > 0 else params.positive_outcome for x in X]))
##
###
def explain(xtrain, xtest, learner, categorical, features, model_num, rep):
    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=features, categorical_features= categorical, discretize_continuous=False)

    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(explainer.explain_instance(xtest[i], learner.predict_proba).as_list())

    exp_dict = experiment_summary(explanations, features)

    L = [[k, *t] for k, v in exp_dict.items() for t in v]
    for t in L:
        t.append(model_num)
        t.append(rep)
    return L

def clusterGroups(root, features, num_points):
    if num_points != 1:
        EDT = getLeafData2(root, num_points)
        X, y = getXY2(EDT)
        df = pd.DataFrame(X, columns=features)
    else:
        MedianTable = leafmedians2(root)
        X, y = getXY2(MedianTable)
        df = pd.DataFrame(X, columns=features)

    return df.to_numpy(), y

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
    datasets = ["communities", "heart", "diabetes", "studentperformance", "compas", "bankmarketing", "defaultcredit", "adultscensusincome"] #"germancredit" idk why but doesn;t work with SVC surrogate
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

    pbar = tqdm(datasets)
    for dataset in pbar:
        klist = keywords[dataset]
        results = []
        feat_importance_tuple_list = []
        lime_results = []
        for keyword in klist:
            pbar.set_description("Processing %s" % dataset)
            path =  "./datasets/processed/" + dataset + "_p.csv"

            X = pd.read_csv(path)
            y_s = [col for col in X.columns if "!" in col]
            yname = y_s[0]
            y = X[yname].values
            X.drop([yname], axis=1, inplace=True)

            # needed for the SLACK adv_model so add unrelated columns
            X['Unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])

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

            inno_indc = cols.index('Unrelated_column_one')
            sensa_indc = [cols.index(col) for col in sensitive_features]
            categorical = [cols.index(c) for c in cat_features_encoded]

            for i in range(10):
                i += 1

                xtrain,xtest,ytrain,ytest = train_test_split(X.values, y, test_size=0.2, random_state = i)

                # print(type(xtrain), type(xtest))
                # sys.exit()

                ss = MinMaxScaler().fit(xtrain)
                xtrain = ss.transform(xtrain)
                xtest = ss.transform(xtest)

                # print(type(xtrain), type(xtest))
                # sys.exit()

                testing = pd.DataFrame(xtest, columns = cols)
                training = pd.DataFrame(xtrain, columns = cols)
                training[yname] = deepcopy(ytrain)

                full_RF = RandomForestClassifier()
                full_RF.fit(xtrain, ytrain)
                f_RF_pred = full_RF.predict(xtest)
                results.append(getMetrics(testing, ytest, f_RF_pred, keyword, len(ytrain), yname, i, "RF" ))
                full_LDF = explain(xtrain, xtest, full_RF, categorical, cols, "RF", i)
                lime_results.extend(full_LDF)

                full_import = full_RF.feature_importances_
                sorted_indices = np.argsort(full_import)[::-1]
                for feat in range(xtrain.shape[1]):
                    feat_importance_tuple_list.append([i ,len(ytrain), "RF", cols[sorted_indices[feat]], round(full_import[sorted_indices[feat]],3), keyword])

                full_LSR = LogisticRegression()
                full_LSR.fit(xtrain, ytrain)
                f_LSR_pred = full_LSR.predict(xtest)
                results.append(getMetrics(testing, ytest, f_LSR_pred, keyword, len(ytrain), yname, i, "LSR" ))
                full_LDF = explain(xtrain, xtest, full_LSR, categorical, cols, "LSR", i)
                lime_results.extend(full_LDF)

                full_SVC = LinearSVC()
                full_SVC.fit(xtrain, ytrain)
                f_SVC_pred = full_SVC.predict(xtest)
                results.append(getMetrics(testing, ytest, f_SVC_pred, keyword, len(ytrain), yname, i, "SVC" ))
                #no predict.proba() not compatible

                full_Slack = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
                f_Slack_pred = full_Slack.predict(xtest)
                results.append(getMetrics(testing, ytest, f_Slack_pred, keyword, len(ytrain), yname, i, "Slack" ))
                full_LDF = explain(xtrain, xtest, full_Slack, categorical, cols, "Slack", i)
                lime_results.extend(full_LDF)

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
                    treatment = [4,5,6,7,8,9,10]
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

                    RF_probed_y = full_RF.predict(subset_x)
                    RF_surrogate = RandomForestClassifier().fit(subset_x, RF_probed_y)
                    RF_surr_pred = RF_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, RF_surr_pred, keyword, len(subset_x), yname, i, "RF_RF" ))

                    LSR_probed_y = full_LSR.predict(subset_x)
                    LSR_surrogate = RandomForestClassifier().fit(subset_x, LSR_probed_y)
                    LSR_surr_pred = LSR_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, LSR_surr_pred, keyword, len(subset_x), yname, i, "LSR_RF" ))

                    SVC_probed_y = full_SVC.predict(subset_x)
                    SVC_surrogate = RandomForestClassifier().fit(subset_x, SVC_probed_y)
                    SVC_surr_pred = SVC_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, SVC_surr_pred, keyword, len(subset_x), yname, i, "SVC_RF" ))

                    Slack_probed_y = full_Slack.predict(subset_x)
                    Slack_surrogate = RandomForestClassifier().fit(subset_x, Slack_probed_y)
                    Slack_surr_pred = Slack_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, Slack_surr_pred, keyword, len(subset_x), yname, i, "Slack_RF" ))

                    RF_surro_import = RF_surrogate.feature_importances_
                    RF_sorted_indices = np.argsort(RF_surro_import)[::-1]

                    LSR_surro_import = LSR_surrogate.feature_importances_
                    LSR_sorted_indices = np.argsort(LSR_surro_import)[::-1]

                    SVC_surro_import = SVC_surrogate.feature_importances_
                    SVC_sorted_indices = np.argsort(SVC_surro_import)[::-1]

                    Slack_surro_import = Slack_surrogate.feature_importances_
                    Slack_sorted_indices = np.argsort(Slack_surro_import)[::-1]

                    for feat in range(xtrain.shape[1]):
                        feat_importance_tuple_list.append([i, len(subset_x), "RF_RF", cols[RF_sorted_indices[feat]], round(RF_surro_import[RF_sorted_indices[feat]],3), keyword])
                        feat_importance_tuple_list.append([i, len(subset_x), "LSR_RF", cols[LSR_sorted_indices[feat]], round(LSR_surro_import[LSR_sorted_indices[feat]],3), keyword])
                        feat_importance_tuple_list.append([i, len(subset_x), "SVC_RF", cols[SVC_sorted_indices[feat]], round(SVC_surro_import[SVC_sorted_indices[feat]],3), keyword])
                        feat_importance_tuple_list.append([i, len(subset_x), "Slack_RF", cols[Slack_sorted_indices[feat]], round(Slack_surro_import[Slack_sorted_indices[feat]],3), keyword])

                    RF_surro_LDF = explain(subset_x, xtest, RF_surrogate, categorical, cols, "RF_RF", i)
                    lime_results.extend(RF_surro_LDF)

                    LSR_surro_LDF = explain(subset_x, xtest, LSR_surrogate, categorical, cols, "LSR_RF", i)
                    lime_results.extend(LSR_surro_LDF)

                    SVC_surro_LDF = explain(subset_x, xtest, SVC_surrogate, categorical, cols, "SVC_RF", i)
                    lime_results.extend(SVC_surro_LDF)

                    Slack_surro_LDF = explain(subset_x, xtest, Slack_surrogate, categorical, cols, "Slack_RF", i)
                    lime_results.extend(Slack_surro_LDF)

        metrics = pd.DataFrame(results, columns = ["recall+", "precision+", "accuracy+", "F1+", "FA0-", "FA1-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-", "biased_col", "samples", "rep", "learner"] )
        metrics.to_csv("./output/features/LIME/" +  dataset + "_FM.csv", index=False)
        feat_imp = pd.DataFrame(feat_importance_tuple_list, columns = ["rep", "samples", "learner", "feature", "importance", "biased_col"] )
        feat_imp.to_csv("./output/features/LIME/" +  dataset + "_feat.csv", index=False)
        all_L = pd.DataFrame(lime_results, columns = ["ranking","feature", "occurances_pct","model_num", "rep"])
        all_L.to_csv("./output/features/LIME/" +  dataset + "._LIME.csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)






if __name__ == "__main__":
    main()
