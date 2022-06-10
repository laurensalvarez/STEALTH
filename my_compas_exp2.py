"""
The experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
         1: [('length_of_stay', 0.002592352559948153), ('unrelated_column', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values
"""
import warnings

from adversarial_models import *
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy

import pprint

###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X):
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
    def predict_proba(self, X):
        A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
        B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
        preds = np.logical_xor(A, B).astype(int)
        return one_hot_encode(preds)
#
##
###

def experiment_main(X, y, p, s, categorical_feature_indcs, features):
    # """
    # Run through experiments for LIME/SHAP on compas using both one and two unrelated features.
    # * This may take some time given that we iterate through every point in the test set
    # * We print out the rate at which features occur in the top three features
    # """

    xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
    ss = StandardScaler().fit(xtrain)
    xtrain = ss.transform(xtrain)
    xtest = ss.transform(xtest)

    # print ('---------------------')
    # print ("Beginning LIME COMPAS Experiments....")
    # print ("(These take some time to run because we have to generate explanations for every point in the test set) ") # 'two_year_recid','c_charge_degree'
    # print ('---------------------')

    # Train the adversarial model for LIME with f and psi
    adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=categorical_feature_indcs, feature_names=features, perturbation_multiplier=p)

    # To get a baseline, we'll look at LIME applied to the biased model f
    biased_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                              discretize_continuous=False,
                                                              categorical_features=categorical_feature_indcs)
    biased_explanations = []
    for i in range(xtest.shape[0]):
        biased_explanations.append(biased_explainer.explain_instance(xtest[i], racist_model_f().predict_proba, num_samples = s).as_list())

    # print ("Explanation on biased f:\n",biased_explanations[:3],"\n\n")

    print ("BIASED: LIME Ranks and Pct Occurances:")
    pprint.pprint  (experiment_summary(biased_explanations, features))

    # Now, lets look at the explanations on the adversarial model
    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features=categorical_feature_indcs, discretize_continuous=False)

    adv_explanations = []
    for i in range(xtest.shape[0]):
        adv_explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba, num_samples = s).as_list())

    # print ("\nExplanation on adversarial model:\n",adv_explanations[:3],"\n")

    # Display Results

    print ("ADV: LIME Ranks and Pct Occurances:")
    pprint.pprint (experiment_summary(adv_explanations, features))
    print ("\nFidelity:", round(adv_lime.fidelity(xtest),2))

    # print ('---------------------')
    # print ("Beginning 2 Feature Attack Experiments....")
    # print ('---------------------')
    # # Repeat the same thing for two features
    # adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, categorical_features= categorical_feature_indcs, feature_names=features, perturbation_multiplier=p)
    # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), categorical_features= categorical_feature_indcs, discretize_continuous=False)
    #
    # explanations = []
    # for i in range(xtest.shape[0]):
    #     explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
    #
    # print ("LIME Ranks and Pct Occurances two unrelated features:")
    # pprint.pprint (experiment_summary(explanations, features))
    # print ("Fidelity:", round(adv_lime.fidelity(xtest),2))


if __name__ == "__main__":

    # Set up experiment parameters
    params = Params("model_configurations/experiment_params.json")
    np.random.seed(params.seed)
    X, y, cols = get_and_preprocess_compas_data(params)

    # add unrelated columns, setup
    X2 = deepcopy(X)
    X['unrelated_column'] = np.random.choice([0,1],size=X.shape[0])

    X2['unrelated_column'] = np.random.choice([0,1],size=X.shape[0])
    X2['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

    features = [c for c in X]
    features2 = [c for c in X2]

    categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',\
                                'sex_Female', 'sex_Male', 'race', 'unrelated_column']

    categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

    race_indc = features.index('race')
    unrelated_indcs = features.index('unrelated_column')

    race_indc = features2.index('race')
    unrelated_indcs = features2.index('unrelated_column')
    unrelated_indcs1 = features2.index('unrelated_column_two')

    X = X.values
    X2 = X2.values

    per_mul = [1, 2, 5, 10, 30]
    num_samples = [2, 3, 4, 5, 10, 25,50, 625, 1250, 2500, 3750, 5000]

    for p in per_mul:
        for s in num_samples:
            print("---------------------------------------------------------------------------- \n\n")
            print("Testing with perturbation_multiplier = " , p , " with", s ,"LIME neighborhood samples \n\n")
            print("---------------------------------------------------------------------------- \n\n")
            print ('---------------------')
            print ("1 Feature Attack Experiments....")
            print ('---------------------')
            experiment_main(X, y, p, s, categorical_feature_indcs, features)

            print ('---------------------')
            print ("2 Feature Attack Experiments....")
            print ('---------------------')

            experiment_main(X2, y, p, s, categorical_feature_indcs, features2)
