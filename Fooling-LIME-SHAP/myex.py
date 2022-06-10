import warnings
warnings.filterwarnings('ignore')

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

prt_mul = [1,2,3,4,5,10,15,20,25,30,35]

for p in prt_mul:
    print ("---------------------------------------------------------------------------- \n\n")
    print ("Testing with perturbation_multiplier = " , p , "\n\n")
    print ("---------------------------------------------------------------------------- \n\n")
    # Get the data set and do some preprocessing
    params = Params("model_configurations/experiment_params.json")
    np.random.seed(params.seed)
    X, y, cols = get_and_preprocess_compas_data(params)

    # Add a random column -- this is what we'll have LIME/SHAP explain.
    X['unrelated_column'] = np.random.choice([0,1],size=X.shape[0])
    features = [c for c in X]

    categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',\
                                'sex_Female', 'sex_Male', 'race', 'unrelated_column']

    categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

    race_indc = features.index('race')
    unrelated_indcs = features.index('unrelated_column')
    X = X.values

    class racist_model_f:
        # Decision rule: classify negatively if race is black
        def predict(self,X):
            return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

        def predict_proba(self, X):
            return one_hot_encode(self.predict(X))

        def score(self, X,y):
            return np.sum(self.predict(X)==y) / len(X)

    class innocuous_model_psi:
        # Decision rule: classify according to randomly drawn column 'unrelated column'
        def predict(self,X):
            return np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X])

        def predict_proba(self, X):
            return one_hot_encode(self.predict(X))

        def score(self, X,y):
            return np.sum(self.predict(X)==y) / len(X)


    # Split the data and normalize
    xtrain,xtest,ytrain,ytest = train_test_split(X,y)
    xtest_not_normalized = deepcopy(xtest)
    ss = StandardScaler().fit(xtrain)
    xtrain = ss.transform(xtrain)
    xtest = ss.transform(xtest)


    # Train the adversarial model for LIME with f and psi
    adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).\
                train(xtrain, ytrain, feature_names=features, perturbation_multiplier = p, categorical_features=categorical_feature_indcs)


    # Let's just look at a the first example in the test set
    ex_indc = np.random.choice(xtest.shape[0])

    # To get a baseline, we'll look at LIME applied to the biased model f
    normal_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                              discretize_continuous=False,
                                                              categorical_features=categorical_feature_indcs)

    normal_exp = normal_explainer.explain_instance(xtest[ex_indc], racist_model_f().predict_proba).as_list()

    print ("Explanation on biased f:\n",normal_exp[:3],"\n\n")

    # Now, lets look at the explanations on the adversarial model
    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                           discretize_continuous=False,
                                                           categorical_features=categorical_feature_indcs)

    adv_exp = adv_explainer.explain_instance(xtest[ex_indc], adv_lime.predict_proba).as_list()

    print ("Explanation on adversarial model:\n",adv_exp[:3],"\n")

    print("Prediction fidelity: {0:3.2}".format(adv_lime.fidelity(xtest[ex_indc:ex_indc+1])))
