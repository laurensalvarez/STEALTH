"""
model extraction
"""
import warnings
warnings.filterwarnings('ignore')

from utils import *
from adversarial_models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import math, re, random, statistics, sys, pprint
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2
from Measure import measure_final_score

params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)

###
## The models f and psi.  We discriminate based on sensitive for f and consider innoc feature for explanation
#
# the biased model
class biased_model_f():
    def __init__(self, sensa_indc):
        self.sensa_indc = sensa_indc
    # Decision rule: classify negative outcome if underrepresented class
    def predict(self,X):
        predictions = np.array([params.negative_outcome if x[self.sensa_indc] == 0 else params.positive_outcome for x in X])
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

class innocuous_model_psi_two:
    def __init__(self, inno_indc):
        self.inno_indc = inno_indc

    def predict_proba(self, X):
        A = np.where(X[:,self.inno_indc] > 0, params.positive_outcome, params.negative_outcome)
        B = np.where(X[:,self.inno_indc] > 0, params.positive_outcome, params.negative_outcome)
        preds = np.logical_xor(A, B).astype(int)
        return one_hot_encode(preds)
##
###
def explain(xtrain, xtest, adv_lime, categorical, features, model):
    # print ('---------------------')
    # print ("Beginning LIME COMPAS Experiments....")
    # print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
    # print ('---------------------')

    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features= categorical, discretize_continuous=False)

    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

    exp_dict = experiment_summary(explanations, features)

    L = [(k, *t) for k, v in exp_dict.items() for t in v]
    LDF = pd.DataFrame(L, columns = ["ranking", "feature", "occurances_pct"])
    LDF["model_num"] = [model] * LDF.index.size

    return LDF
    # Display Results
    # print (str(adv_lime) + " LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
    # pprint.pprint (experiment_summary(explanations, features))
    # print ("\nFidelity:", round(adv_lime.fidelity(xtest),3))

    # # Repeat the same thing for two features
    # adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, categorical_features= categorical, feature_names=features, perturbation_multiplier=2)
    # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features= categorical, discretize_continuous=False)
    #
    # explanations = []
    # for i in range(xtest.shape[0]):
    #     explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
    #
    # print ("LIME Ranks and Pct Occurances two unrelated features:")
    # print (experiment_summary(explanations, features))
    # print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

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
    datasets = ["heart", "germancredit", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "defaultcredit", "adultscensusincome"]
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

            # add unrelated columns
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

            sensa_indc = [cols.index(col) for col in sensitive_features]
            categorical = [cols.index(c) for c in cat_features_encoded]
            inno_indc = cols.index('Unrelated_column_one')

            for i in range(10):
                i += 1

                xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size=0.2, random_state = i)

                ss = MinMaxScaler().fit(xtrain)
                xtrain = ss.transform(xtrain)
                xtest = ss.transform(xtest)

                training = pd.DataFrame(deepcopy(xtrain), columns = cols)
                training[yname] = deepcopy(ytrain)
                testing = pd.DataFrame(deepcopy(xtest), columns = cols)

                # Train the adversrial model for LIME with f and psi
                full_clf = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
                fc_pred = full_clf.predict(xtest)
                results.append(getMetrics(testing, ytest, fc_pred, keyword, len(ytrain), yname, i ))

                # full_clf = LinearSVC()
                # full_clf.fit(xtrain, ytrain)
                # fc_pred = full_clf.predict(xtest)
                # results.append(getMetrics(testing, ytest, fc_pred, keyword, len(ytrain), yname, i ))

                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)


                #O-CART
                # tree = DecisionTreeClassifier(min_samples_leaf = enough, random_state = i)
                # tree.fit(xtrain, ytrain)
                # leaves = tree.get_n_leaves()
                # leaf_indexes = tree.apply(xtrain)
                # new_leaves = list(set(leaf_indexes))=

                treatment = [2,3,4,5]
                for num_points in treatment:
                    clustered_df, clustered_y = clusterGroups(root, cols, num_points)

                    #O-CART leaf extraction
                    # subset_x = []
                    # for l in new_leaves:
                    #     indices = [i for i, x in enumerate(leaf_indexes) if x == l] #gives indexes of leaf_num aka index of all points in leaf
                    #     x_index = np.random.choice(indices, replace = False, size = num_points)
                    #     for t in x_index:
                    #         subset_x.append(xtrain[t])

                    probed_y = full_clf.predict(clustered_df.to_numpy())
                    surrogate = LinearSVC().fit(clustered_df, probed_y)
                    surr_pred = surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, surr_pred, keyword, len(probed_y), yname, i ))

        metrics = pd.DataFrame(results, columns = ["recall+", "precision+", "accuracy+", "F1+", "AOD-", "EOD-", "SPD-", "DI-", "FA0-", "FA1-", "biased_col", "samples", "rep"] )
        metrics.to_csv("./output/slack_ext/SVM/" +  dataset + ".csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)






if __name__ == "__main__":
    main()
