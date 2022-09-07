"""
Adv agent experiment
"""
import warnings
warnings.filterwarnings('ignore')

from adversarial_models import *
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

import math, re, random, statistics, sys
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap
import pprint

from sklearn.cluster import KMeans

from copy import deepcopy
from tqdm import tqdm

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2


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
        indices = np.random.choice(np.arange(predictions.size), replace = False, size = int(predictions.size * 0.25))
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

def splitUp(path, dataset):
    X = pd.read_csv(path)
    y_s = [col for col in X.columns if "!" in col]
    yname = y_s[0]
    y = X[yname].values
    X.drop([yname], axis=1, inplace=True)

    # add unrelated columns
    X['Unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
    # X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

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
    inno_indc = cols.index('Unrelated_column_one')
    categorical = [cols.index(c) for c in cat_features_encoded]

    # no_cats = pd.DataFrame(X.copy(),columns = X.columns)
    # no_cats[yname] = deepcopy(y)
    # no_cats.to_csv("./datasets/no_cats/lower/" +  dataset + ".csv", index=False)

    return X, y, yname, cols, inno_indc, sensa_indc, categorical

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

def pred(model, features, xtest, ytest, yname, fold, model_num):
    """ returing sensitive feat and predictions from ADV model (y vals) """
    pred = model.predict(deepcopy(xtest))
    sdf = pd.DataFrame(deepcopy(xtest), columns=features)

    samples_col = [sdf.index.size] * sdf.index.size
    fold_col = [fold] * sdf.index.size
    model_col = [model_num] * sdf.index.size
    sdf["predicted"] = pred
    sdf[yname] = deepcopy(ytest)
    sdf["samples"] = samples_col
    sdf["fold"] = fold_col
    sdf["model_num"] = model_col

    return sdf

def newTraining(df, yname):
    df_copy = df.copy()
    new_y = df_copy["predicted"]
    df_copy.drop(["predicted", yname ,"samples", "fold", "model_num"], axis=1, inplace=True)

    return df_copy, new_y

def transformed(df, features, yname, categorical, scaler):
    df_copy = df.copy()
    df_copy.drop(["predicted", yname ,"samples", "fold", "model_num"], axis=1, inplace=True)

    values = df_copy.values
    unscaled_values = scaler.inverse_transform(values)
    tdf = pd.DataFrame(unscaled_values, columns=features, dtype = 'int64')
    tdf["predicted"] = df["predicted"].values
    values = df[yname].values
    int_vals = []
    for v in values:
        int_vals.append(int(v))

    tdf[yname] = int_vals
    tdf["samples"] = df["samples"].values
    tdf["fold"] = df["fold"].values
    tdf["model_num"] = df["model_num"].values

    return tdf

def predTrans(model, scaler, features, categorical, xtest, ytest, yname, fold, model_num):
    df = pred(model, features, xtest, ytest, yname, fold, model_num)
    tdf = transformed(df, features, yname, categorical, scaler)
    return tdf

def main():
    # random.seed(10039)
    datasets = ["adultscensusincome","bankmarketing", "communities", "compas", "defaultcredit",  "germancredit", "heart", "studentperformance"]
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
        pbar.set_description("Processing %s" % dataset)
        path =  "./datasets/processed/" + dataset + "_p.csv"

        X, y, yname, cols, inno_indc, sensa_indc, categorical = splitUp(path, dataset)

        Xvals = X.values

        clustered_cols = deepcopy(cols)
        clustered_cols.append("predicted")
        clustered_cols.append(yname)
        clustered_cols.append("samples")
        clustered_cols.append("fold")
        clustered_cols.append("model_num")

        final_columns = deepcopy(cols)
        final_columns.append("predicted")
        final_columns.append("ytest")
        final_columns.append("samples")
        final_columns.append("fold")
        final_columns.append("model_num")

        clusters = pd.DataFrame(columns = clustered_cols)
        all_models_test = pd.DataFrame(columns = final_columns)
        all_L = pd.DataFrame(columns = ["ranking","feature","occurances_pct","model_num"])

        for i in range(10):
            i += 1

            xtrain,xtest,ytrain,ytest = train_test_split(Xvals,y,test_size=0.2, shuffle = True)

            ss = MinMaxScaler().fit(xtrain)
            xtrain = ss.transform(xtrain)
            xtest = ss.transform(xtest)

            training = pd.DataFrame(deepcopy(xtrain), columns = cols)
            training [yname] = deepcopy(ytrain)

            table = Table(11111)
            rows = deepcopy(training.values)
            header = deepcopy(list(training.columns.values))
            table + header
            for r in rows:
                table + r

            enough = int(math.sqrt(len(table.rows)))
            root = Table.clusters(table.rows, table, enough)

            # Train the adversrial model for LIME with f and psi
            adv_lime_0 = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
            L = explain(xtrain, xtest, adv_lime_0, categorical, cols, 0)
            all_L = all_L.append(L)

            treatment = [1,2,3,4,5]
            for num_points in treatment:
                clustered_df, clustered_y = clusterGroups(root, cols, num_points)
                # print(clustered_df.head(), clustered_df.index)

                probed_df = pred(adv_lime_0, cols, clustered_df.to_numpy(), clustered_y, yname, i, num_points)
                tdf = transformed(probed_df, cols, yname, categorical, ss)
                clusters = clusters.append(tdf)

                subset_training, subset_y = newTraining(probed_df, yname)

                adv_lime_clone = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(subset_training, subset_y, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

                tested_model = predTrans(adv_lime_clone, ss, cols, categorical, xtest, ytest, "ytest", i, num_points)

                all_models_test = all_models_test.append(tested_model)
                # print(all_models_test.index)

                L = explain(xtrain, xtest, adv_lime_clone, categorical, cols, num_points)

                all_L = all_L.append(L)


        clusters.to_csv("./output/cluster_preds/lower/" +  dataset + "_all.csv", index=False)
        all_models_test.to_csv("./output/clones/lower/" +  dataset + "_all.csv", index=False)
        all_L.to_csv("./output/LIME_rankings/lower/" +  dataset + ".csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)






if __name__ == "__main__":
    main()
