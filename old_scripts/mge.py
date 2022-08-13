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
from sklearn.metrics import confusion_matrix, classification_report

import math, re, random, statistics, sys
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans

from copy import deepcopy

from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
Xdf, y, cols = get_and_preprocess_german(params)

features = [c for c in Xdf]

gender_indc = features.index('gender(')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = Xdf.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)

xtraindf = pd.DataFrame(xtrain, columns = features)
xtraindf["GoodCustomer!"]= ytrain

ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

table = Table(11111)
xrows = xtraindf.values
header = deepcopy(list(xtraindf.columns.values))
table + header
for r in xrows:
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


print("--------sizes----------")
print("--------Xdf----------")
print(Xdf.index, len(Xdf.columns.values))
# print(Xdf.head)
print("--------xtraindf----------")
print(xtraindf.index, len(xtraindf.columns.values), "(has y col)")
# print(xtraindf.head)
print("--------mdf----------")
print(mdf.index, len(mdf.columns.values))
# print(mdf.head)
print("--------df5----------")
print(df5.index, len(df5.columns.values))
# print(df5.head)
print("--------sizes----------")

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
    full_df = pd.DataFrame(columns=features)
    # Train the adversarial model for LIME with f and psi
    adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=2, categorical_features=categorical)
    median_pred = adv_lime.predict(mdf.to_numpy())

    values = mdf.values
    svalues = ss.inverse_transform(values)
    sdf = pd.DataFrame(svalues, columns=features)

    col = [sdf.index.size] * sdf.index.size
    sdf["samples"] = col
    sdf["predicted"] = median_pred
    sdf["!Probability"] = my

    full_df = full_df.append(sdf)

    pred5 = adv_lime.predict(df5.to_numpy())

    values = df5.values
    svalues = ss.inverse_transform(values)
    sdf5 = pd.DataFrame(svalues, columns=features)

    col = [sdf5.index.size] * sdf5.index.size
    sdf5["samples"] = col
    sdf5["predicted"] = pred5
    sdf5["!Probability"] = y5

    full_df = full_df.append(sdf5)


    final_columns = []
    final_columns.append("gender(")
    final_columns.append("predicted")
    final_columns.append("!Probability")
    final_columns.append("samples")
    output_df = full_df[final_columns]
    print(output_df.index, len(output_df.columns.values))
    output_df.to_csv("./output/" + "german_mdf2.csv", index=False, float_format = "%g")

    y_pred = np.append(median_pred, pred5)
    y_true = my + y5
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
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


if __name__ == "__main__":
    random.seed(10039)
    experiment_main()
