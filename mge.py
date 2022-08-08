"""
The experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore')

from adversarial_models import *
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math, re, random, statistics, sys
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans

from copy import deepcopy

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
Xdf, y, cols = get_and_preprocess_german(params)

features = [c for c in Xdf]

gender_indc = features.index('gender(')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = Xdf.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

xtraindf = pd.DataFrame(xtrain, columns = features)
xtraindf["GoodCustomer!"]= ytrain

# print(xtraindf.head)

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])


categorical = ['gender(', 'ForeignWorker', 'Single', 'HasTelephone','CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200','SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500','MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere','OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor','OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1','YearsAtCurrentJob_geq_4','JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X):
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###

def experiment_main():
    """
    Run through experiments for LIME on GERMAN.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features
    """

    # Train the adversarial model for LIME with f and psi
    adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=2, categorical_features=categorical)

    table = Table(11111)
    xrows = xtraindf.values
    header = deepcopy(list(xtraindf.columns.values))
    print(header)
    table + header
    for r in xrows:
        table + r

    enough = int(math.sqrt(len(table.rows)))
    root = Table.clusters(table.rows, table, enough)

    tcols = deepcopy(table.header)
    tcols.append("predicted")
    tcols.append("samples")
    tcols.append("total_pts")
    full_df = pd.DataFrame(columns=tcols)
    mdf = pd.DataFrame(columns=tcols)
    df5 = pd.DataFrame(columns=tcols)
    df7 = pd.DataFrame(columns=tcols)

    treatments = [1,5,7]
    for samples in treatments:
        if samples == 1:
            MedianTable = leafmedians2(root)
            total_pts = len(MedianTable.rows)
            print(total_pts)
            full = []
            xm, ym = getXY(MedianTable)
            xm = np.array(xm)

            y_pred = adv_lime.predict(xm)

            y_pred_list = y_pred.tolist()
            print(y_pred_list)
            for x in xm:
                full.append(deepcopy(x))
            for j in range(len(ym)):
                full[j] = np.append(full[j],ym[j])
                full[j] = np.append(full[j],y_pred_list[j])
                full[j] = np.append(full[j], samples)
                full[j] = np.append(full[j], total_pts)
            for row in full:
                a_series = pd.Series(row, index= mdf.columns)
                mdf = mdf.append(a_series, ignore_index=True)

        else:
            EDT = getLeafData2(root, samples)

            total_pts = len(EDT.rows)
            print(total_pts)
            full = []
            x, y = getXY(EDT)
            x = np.array(x)

            y_pred = adv_lime.predict(x)

            y_pred_list = y_pred.tolist()
            print(y_pred_list)
            for x in x:
                full.append(deepcopy(x))
            for j in range(len(y)):
                full[j] = np.append(full[j],y[j])
                full[j] = np.append(full[j],y_pred_list[j])
                full[j] = np.append(full[j], samples)
                full[j] = np.append(full[j], total_pts)
            for row in full:
                a_series = pd.Series(row, index= mdf.columns)
                df5 = df5.append(a_series, ignore_index=True)


        full_df = full_df.append(mdf)
        full_df = full_df.append(df5)
        # full_df = full_df.append(df7)

    final_columns = []
    final_columns.append("gender(")
    final_columns.append("predicted")
    final_columns.append("samples")
    final_columns.append("total_pts")
    output_df = full_df[final_columns]
    output_df.to_csv("./output/" + "german_test2.csv", index=False)

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


if __name__ == "__main__":
    experiment_main()
