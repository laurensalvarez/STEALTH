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

##
###

def pred(adv_lime,ss,features,xC, yC, yname):
    """
    Run through experiments for LIME.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features
    """

    pred = adv_lime.predict(xC.to_numpy())

    values = xC.values
    scaled_values = ss.inverse_transform(values)
    sdf = pd.DataFrame(scaled_values, columns=features)

    samples_col = [sdf.index.size] * sdf.index.size
    sdf["samples"] = samples_col
    sdf["predicted"] = pred
    sdf[yname] = yC

    return sdf

    # y_pred = np.append(median_pred, pred5)
    # y_true = my + y5
    # cm = confusion_matrix(y_true, y_pred)
    # print(classification_report(y_true, y_pred))
    # print(cm)

    print ('---------------------')







    # print ('---------------------')
    # print ("Beginning LIME GERMAN Experiments....")
    # print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
    # print ('---------------------')
    # """
    # LIME Explanation starts
    # """
    #
    # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=categorical)
    #
    # explanations = []
    # for i in range(xtest.shape[0]):
    #     explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
    #
    # # Display Results
    # print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
    # print (experiment_summary(explanations, features))
    # print ("Fidelity:", round(adv_lime.fidelity(xtest),2))
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

def splitUp(path, dataset):
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

    return X, y, yname


def main():
    random.seed(10039)
    datasets = ["germancredit", "compas"] #["adultscensusincome.csv","bankmarketing.csv", "compas.csv", "communities.csv", "defaultcredit.csv", "diabetes.csv",  "germancredit.csv", "heart.csv", "studentperformance.csv"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        path =  "./datasets/processed/" + dataset + "_p.csv"

        X, y, yname = splitUp(path, dataset)

        # add unrelated columns
        X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
        # X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
        cols = [c for c in X]

        inno_indc = cols.index('unrelated_column_one')

        non_numeric_columns = list(X.select_dtypes(exclude=[np.number]).columns)
        le = LabelEncoder()
        for col in non_numeric_columns:
            X[col] = le.fit_transform(X[col])

        categorical = [cols.index(c) for c in non_numeric_columns]

        sensitive_features = [col for col in X.columns if "(" in col]
        sorted(sensitive_features)
        print("sensitive_features:", sensitive_features)
        sensa_indc = [cols.index(col) for col in sensitive_features]

        Xvals = X.values

        xtrain,xtest,ytrain,ytest = train_test_split(Xvals,y,test_size=0.2, shuffle = True)

        trainingdf = pd.DataFrame(xtrain, columns = cols)
        trainingdf[yname]= ytrain

        ss = MinMaxScaler().fit(xtrain)
        xtrain = ss.transform(xtrain)
        xtest = ss.transform(xtest)
        # mean_lrpi = np.mean(xtrain[:,loan_rate_indc]) #create a general or separate for german.

        my, mdf, y5, df5 = clusterGroups(trainingdf, ss, cols)

        # print("--------sizes----------")
        # print("--------Xdf----------")
        # print(X.index, len(X.columns.values))
        # # print(Xdf.head)
        # print("--------xtraindf----------")
        # print(trainingdf.index, len(trainingdf.columns.values), "(has y col)")
        # # print(xtraindf.head)
        # print("--------mdf----------")
        # print(mdf.index, len(mdf.columns.values))
        # # print(mdf.head)
        # print("--------df5----------")
        # print(df5.index, len(df5.columns.values))
        # # print(df5.head)
        # print("--------sizes----------")

        # Train the adversarial model for LIME with f and psi
        adv_lime = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)


        full_df = pd.DataFrame(columns=cols)

        exp_df = pred(adv_lime, ss, cols, mdf, my, yname)

        full_df = full_df.append(exp_df)

        exp_df = pred(adv_lime, ss, cols, df5, y5, yname)

        full_df = full_df.append(exp_df)


        final_columns = sensitive_features.copy()
        final_columns.append("predicted")
        final_columns.append(yname)
        final_columns.append("samples")
        output_df = full_df[final_columns]
        output_df = output_df.astype(int)
        # print(output_df.index, len(output_df.columns.values))
        output_df.to_csv("./output/" +  dataset + "exp_test.csv", index=False, float_format = "%.f")







if __name__ == "__main__":
    main()
