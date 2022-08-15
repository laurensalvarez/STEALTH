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
        indices = np.random.choice(np.arange(predictions.size), replace = False, size = int(predictions.size * 0.2))
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
    # no_cats.to_csv("./datasets/no_cats/" +  dataset + ".csv", index=False)

    return X, y, yname, cols, inno_indc, categorical, sensitive_features, sensa_indc

def clusterGroups(trainingdf, ss, features):
    table = Table(11111)
    rows = deepcopy(trainingdf.values)
    header = deepcopy(list(trainingdf.columns.values))
    table + header
    for r in rows:
        table + r

    enough = int(math.sqrt(len(table.rows)))
    root = Table.clusters(table.rows, table, enough)

    MedianTable = leafmedians2(root)
    mX, my = getXY2(MedianTable)
    mdf = pd.DataFrame(mX, columns=features)

    EDT = getLeafData2(root, 5)
    X5, y5 = getXY2(EDT)
    df5 = pd.DataFrame(X5, columns=features)

    EDT7 = getLeafData2(root, 7)
    X7, y7 = getXY2(EDT7)
    df7 = pd.DataFrame(X7, columns=features)


    return my, mdf, y5, df5, y7, df7

def pred(adv_lime,ss,features,xtest, ytest, yname, f, model_num):
    """ returing sensitive feat and predictions from ADV model (y vals) """
    pred = adv_lime.predict(deepcopy(xtest))
    sdf = pd.DataFrame(deepcopy(xtest), columns=features)

    samples_col = [sdf.index.size] * sdf.index.size
    fold_col = [f] * sdf.index.size
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

def transformed(df, cols, yname, categorical, ss):
    df_copy = df.copy()
    df_copy.drop(["predicted", yname ,"samples", "fold", "model_num"], axis=1, inplace=True)

    values = df_copy.values
    unscaled_values = ss.inverse_transform(values)
    tdf = pd.DataFrame(unscaled_values, columns=cols, dtype = 'int64')
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

def main():
    # random.seed(10039)
    datasets = ["adultscensusincome","bankmarketing", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"] #"compas"
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        path =  "./datasets/processed/" + dataset + "_p.csv"

        X, y, yname, cols, inno_indc, categorical, sensitive_features, sensa_indc = splitUp(path, dataset)
        # print(le.classes_)
        Xvals = X.values

        xtrain,xtest,ytrain,ytest = train_test_split(Xvals,y,test_size=0.2, shuffle = True)
        f = 1

        ss = MinMaxScaler().fit(xtrain)
        xtrain = ss.transform(xtrain)
        xtest = ss.transform(xtest)

        trainingdf = pd.DataFrame(deepcopy(xtrain), columns = cols)
        trainingdf[yname]= deepcopy(ytrain)

        my, mdf, y5, df5, y7, df7 = clusterGroups(trainingdf, ss, cols)

        # Train the adversarial model for LIME with f and psi
        adv_lime = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        # finalcols = cols + ["predicted", "samples", "fold", "model_num"]
        exp_df = pred(adv_lime, ss, cols, mdf.to_numpy(), my, yname, f, 0)
        t_df = transformed(exp_df, cols, yname, categorical, ss)
        clusters_df = pd.DataFrame(columns = t_df.columns)
        clusters_df = clusters_df.append(t_df)
        # t_df.to_csv("./output/cluster_preds/class_bal/" +  dataset + "_medians.csv", index=False)

        medianX, mediany = newTraining(exp_df, yname)
        adv_lime_m = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(medianX, mediany, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        exp_df5 = pred(adv_lime, ss, cols, df5.to_numpy(), y5, yname, f, 5)
        texp_df5 = transformed(exp_df5, cols, yname, categorical, ss)
        clusters_df = clusters_df.append(texp_df5)

        X5, Y5 = newTraining(exp_df5, yname)
        adv_lime_5 = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(X5, Y5, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        exp_df7 = pred(adv_lime, ss, cols, df7.to_numpy(), y7, yname, f, 7)
        texp_df7 = transformed(exp_df7, cols, yname, categorical, ss)
        clusters_df = clusters_df.append(texp_df7)
        clusters_df.to_csv("./output/cluster_preds/class_bal/" +  dataset + ".csv", index=False)

        X7, Y7 = newTraining(exp_df7, yname)
        adv_lime_7 = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(X7, Y7, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        #Test & Compare M_0, M_m, M_5
        M0 = pred(adv_lime,ss,cols,xtest, ytest, "ytest", f, 0)
        M0 = transformed(M0, cols, "ytest", categorical, ss)

        all_models_test = pd.DataFrame(columns = M0.columns)
        all_models_test = all_models_test.append(M0)
        # M0.to_csv("./output/clones/class_bal/" +  dataset + "_M0.csv", index=False)

        Mm = pred(adv_lime_m,ss,cols,xtest, ytest, "ytest", f, 1)
        Mm = transformed(Mm, cols, "ytest", categorical, ss)
        all_models_test = all_models_test.append(Mm)
        # Mm.to_csv("./output/clones/class_bal/" +  dataset + "_M1.csv", index=False)

        M5 = pred(adv_lime_5,ss,cols,xtest, ytest, "ytest", f, 5)
        M5 = transformed(M5, cols, "ytest", categorical, ss)
        all_models_test = all_models_test.append(M5)
        # M5.to_csv("./output/clones/class_bal/" +  dataset + "_M5.csv", index=False)

        M7 = pred(adv_lime_7,ss,cols,xtest, ytest, "ytest", f, 7)
        M7 = transformed(M7, cols, "ytest", categorical, ss)
        all_models_test = all_models_test.append(M7)
        # M7.to_csv("./output/clones/class_bal/" +  dataset + "_M7.csv", index=False)
        all_models_test.to_csv("./output/clones/class_bal/" +  dataset + "_all.csv", index=False)

        L = explain(xtrain, xtest, adv_lime, categorical, cols, 0)
        all_L = pd.DataFrame(columns = L.columns)
        all_L = all_L.append(L)
        L = explain(xtrain, xtest, adv_lime_m, categorical, cols, 1)
        all_L = all_L.append(L)
        L = explain(xtrain, xtest, adv_lime_5, categorical, cols, 5)
        all_L = all_L.append(L)
        L = explain(xtrain, xtest, adv_lime_7, categorical, cols, 7)
        all_L = all_L.append(L)
        all_L.to_csv("./output/LIME_rankings/class_bal/" +  dataset + ".csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)






if __name__ == "__main__":
    main()
