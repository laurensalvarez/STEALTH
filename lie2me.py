"""
Adv agent experiment
"""
import warnings
warnings.filterwarnings('ignore')

from adversarial_models import *
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import math, re, random, statistics, sys
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans

from copy import deepcopy
from tqdm import tqdm

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2


params = Params("model_configurations/experiment_params.json")
###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#
# the biased model
class biased_model_f():
    def __init__(self, sensa_indc):
        self.sensa_indc = sensa_indc
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[self.sensa_indc] == 0 else params.positive_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# the display model with one unrelated feature
class innocuous_model_psi:
    def __init__(self, inno_indc):
        self.inno_indc = inno_indc
    # Decision rule: classify according to loan rate indc
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
def explain(xtrain, xtest, adv_lime, categorical, features):
    print ('---------------------')
    print ("Beginning LIME COMPAS Experiments....")
    print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
    print ('---------------------')

    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features= categorical, discretize_continuous=False)

    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

    # Display Results
    print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
    print (experiment_summary(explanations, features))
    print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

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

def splitUp(path, dataset, labelenc):
    lilprobs = ["adultscensusincome", "diabetes", "bankmarketing"]
    X = pd.read_csv(path)
    y_s = [col for col in X.columns if "!" in col]
    yname = y_s[0]

    if dataset in lilprobs:
        y = X[yname].values
        X.drop([yname], axis=1, inplace=True)
    else:
        y = X[yname].values
        X.drop([yname], axis=1, inplace=True)

    # add unrelated columns
    X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
    # X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
    cols = [c for c in X]

    inno_indc = cols.index('unrelated_column_one')

    non_numeric_columns = list(X.select_dtypes(exclude=[np.number]).columns)

    for col in non_numeric_columns:
        X[col] = labelenc.fit_transform(X[col])

    categorical = [cols.index(c) for c in non_numeric_columns]

    sensitive_features = [col for col in X.columns if "(" in col]
    sorted(sensitive_features)
    # print("sensitive_features:", sensitive_features)
    sensa_indc = [cols.index(col) for col in sensitive_features]

    return X, y, yname, cols, inno_indc, non_numeric_columns, categorical, sensitive_features, sensa_indc

def clusterGroups(trainingdf, ss, features):
    table = Table(11111)
    rows = trainingdf.values
    header = deepcopy(list(trainingdf.columns.values))
    table + header
    for r in rows:
        table + r

    enough = int(math.sqrt(len(table.rows)))
    root = Table.clusters(table.rows, table, enough)

    MedianTable = leafmedians2(root)
    mX, my = getXY2(MedianTable)
    mX = ss.transform(mX)
    mdf = pd.DataFrame(mX, columns=features)


    EDT = getLeafData2(root, 5)
    X5, y5 = getXY2(EDT)
    X5 = ss.transform(X5)
    df5 = pd.DataFrame(X5, columns=features)

    return my, mdf, y5, df5

def pred(adv_lime,ss,features,xtest, ytest, yname):
    """ returing sensitive feat and predictions from ADV model (y vals) """

    pred = adv_lime.predict(xtest)

    # values = xtest.values
    scaled_values = ss.inverse_transform(xtest)
    sdf = pd.DataFrame(scaled_values, columns=features)

    samples_col = [sdf.index.size] * sdf.index.size
    sdf["predicted"] = pred
    sdf[yname] = ytest
    sdf["samples"] = samples_col

    return sdf

def newTraining(df, yname):
    df_copy = df.copy()
    new_y = df_copy["predicted"]
    df_copy.drop(["predicted", yname ,"samples"], axis=1, inplace=True)

    return df_copy, new_y



def main():
    random.seed(10039)
    datasets = ["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        path =  "./datasets/processed/" + dataset + "_p.csv"

        le = LabelEncoder()

        X, y, yname, cols, inno_indc, non_numeric_columns, categorical, sensitive_features, sensa_indc = splitUp(path, dataset, le)

        Xvals = X.values

        xtrain,xtest,ytrain,ytest = train_test_split(Xvals,y,test_size=0.2, shuffle = True)

        trainingdf = pd.DataFrame(xtrain, columns = cols)
        trainingdf[yname]= ytrain

        ss = MinMaxScaler().fit(xtrain)
        xtrain = ss.transform(xtrain)
        xtest = ss.transform(xtest)

        my, mdf, y5, df5 = clusterGroups(trainingdf, ss, cols)

        # Train the adversarial model for LIME with f and psi
        adv_lime = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        exp_df = pred(adv_lime, ss, cols, mdf.to_numpy(), my, yname)
        medianX, mediany = newTraining(exp_df, yname)
        adv_lime_m = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(medianX, mediany, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        exp_df.astype(int)
        exp_df.to_csv("./output/cluster_preds/" +  dataset + "_medians.csv", index=False, float_format = "%.f")

        exp_df5 = pred(adv_lime, ss, cols, df5.to_numpy(), y5, yname)
        X5, y5 = newTraining(exp_df5, yname)
        adv_lime_5 = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(X5, y5, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)

        exp_df5.astype(int)
        exp_df5.to_csv("./output/cluster_preds/" +  dataset + "_5.csv", index=False, float_format = "%.f")

        #Test & Compare M_0, M_m, M_5
        M0 = pred(adv_lime,ss,cols,xtest, ytest, "ytest")
        M0.astype(int)
        M0.to_csv("./output/clones/" +  dataset + "_M0.csv", index=False, float_format = "%.f")

        Mm = pred(adv_lime_m,ss,cols,xtest, ytest, "ytest")
        Mm.astype(int)
        Mm.to_csv("./output/clones/" +  dataset + "_Mm.csv", index=False, float_format = "%.f")

        M5 = pred(adv_lime_5,ss,cols,xtest, ytest, "ytest")
        M5.astype(int)
        M5.to_csv("./output/clones/" +  dataset + "_M5.csv", index=False, float_format = "%.f")








        # explain(xtrain, xtest, adv_lime, categorical, features)

        print ('-'*55)
        print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        print ('-'*55)






if __name__ == "__main__":
    main()
