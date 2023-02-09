import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from scipy.optimize import differential_evolution

import math, time, sys
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import lime
import lime.lime_tabular

from cols import Table,leafmedians2, getLeafData2, getXY2
from slack.utils import *
from slack.adversarial_models import *
from metrics.Measure import getMetrics
from xFair import *
from mitigation import Fair_Smote, maat

params = Params("./model_configurations/experiment_params.json")
np.random.seed(params.seed)

## SLACK 
## The models f and psi.  We discriminate based on sensitive for f and consider innoc feature for explanation
#
# the biased model
class biased_model_f():
    def __init__(self, sensa_indc):
        self.sensa_indc = sensa_indc
    # Decision rule: classify negative outcome if underrepresented class
    def predict(self,X):
        return np.array([params.negative_outcome if x[self.sensa_indc] == 0 else params.positive_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

##SLACK
# the display model with one unrelated feature
class innocuous_model_psi:
    def __init__(self, inno_indc):
        self.inno_indc = inno_indc
    # Decision rule: classify according to innoc indc
    def predict_proba(self, X):
        return one_hot_encode(np.array([params.negative_outcome if x[self.inno_indc] > 0 else params.positive_outcome for x in X]))


def explain(xtrain, xtest, learner, categorical, features, model, keyword, treatment, samples, rep, rt, dataset):
    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=features, categorical_features= categorical, discretize_continuous=False, random_state=rep)

    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(explainer.explain_instance(xtest[i], learner.predict_proba).as_list())
        
    exp_dict = experiment_summary(explanations, features)

    L = [[k, *t] for k, v in exp_dict.items() for t in v]
    for t in L:
        t.append(model)
        t.append(keyword)
        t.append(time.time() - rt)
        t.append(treatment)
        t.append(samples)
        t.append(rep)

    print(dataset + " " + model + " " + str(treatment) + "\nLIME Round", (rep), "finished.", round(time.time() - rt, 2) )
    return L


def clusterGroups(root, features, num_points):
    if num_points != 0:
        EDT = getLeafData2(root, num_points)
        X, y = getXY2(EDT)
        df = pd.DataFrame(X, columns=features)
    else:
        MedianTable = leafmedians2(root)
        X, y = getXY2(MedianTable)
        df = pd.DataFrame(X, columns=features)

    return df.to_numpy(), y

#Differential EVO
# Define the function to optimize
def optimization_function(pars, x, y):
    # Unpack the parameters
    n_estimators, max_depth, min_samples_split = pars
    # Set the hyperparameters of the model
    model = RandomForestClassifier(n_estimators= int(n_estimators),
                                    max_depth= int(max_depth),
                                    min_samples_split= int(min_samples_split))
    # model = RandomForestClassifier()
    # Calculate the cross-validated score using 3 folds
    score = -1 * cross_val_score(model, x, y, cv=3, scoring="accuracy").mean()
    return score


def main():
    datasets =  ['communities', 'heart','diabetes',  'german', 'student', 'meps', 'compas', 'bank', 'default', 'adult'] #'communities', 'heart',
    keywords = {'adult': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bank': ['Age('],
                'communities': ['Racepctwhite('],
                'default': ['SEX('],
                'diabetes': ['Age('],
                'german': ['sex('],
                'heart': ['Age('],
                'student': ['sex('],
                'meps': ['race(']
                }

    pbar = tqdm(datasets)
    i = int(sys.argv[-1])
    
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

            # needed for the SLACK adv_model; add at least one unrelated column
            X['Unrelated'] = np.random.choice([0,1],size=X.shape[0])

            sensitive_features = [col for col in X.columns if "(" in col]
            sensitive_features.sort()

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

            inno_indc = cols.index('Unrelated')
            categorical = [cols.index(c) for c in cat_features_encoded]
            key_indc = cols.index(keyword)
            
            start = time.time()

            xtrain,xrem,ytrain,yrem = train_test_split(X.values, y, train_size=0.4, random_state = i)
            
            xvalid,xtest,yvalid,ytest = train_test_split(xrem, yrem, test_size=0.2, random_state = i)

            ss = MinMaxScaler().fit(xtrain)
            xtrain = ss.transform(xtrain)
            xtest = ss.transform(xtest)

            testing = pd.DataFrame(xtest, columns = cols)
            testing[yname] = deepcopy(ytest)

            training = pd.DataFrame(xtrain, columns = cols)
            training[yname] = deepcopy(ytrain)

            validating = pd.DataFrame(xvalid, columns = cols)
            validating[yname] = deepcopy(yvalid)

            # Define the bounds for the hyperparameters
            # bounds = [(50,150),(1,50),(2,20)] #training.apply(lambda x: pd.Series((x.min(), x.max()))).T.values.tolist()

            # Perform the optimization
            # DE = differential_evolution(optimization_function, bounds, args = (xtrain, ytrain), seed = i)

            # Print the optimal hyperparameters
            # print("Optimal hyperparameters:", DE.x)

            # Use the optimal hyperparameters to train a new model
            full_RF = RandomForestClassifier()

            # full_RF = RandomForestClassifier() 
            full_RF.fit(xtrain, ytrain)
            f_RF_pred = full_RF.predict(xtest)

            results.append(getMetrics(testing, f_RF_pred, keyword, 100, len(ytrain), yname, i, "RF", start, clf = full_RF)) #original/raw/baseline performance
            
            #Baseline Fairness Tools for full set
            results.append(Fair_Smote(training, testing, full_RF, keyword, 100, len(ytrain), yname, i, "RFs"))
            results.append(maat(training, testing, full_RF, ss, keyword, 100, len(ytrain), yname, i, "RFm", dataset))
            results.append(xFAIR(training, testing, full_RF, DecisionTreeRegressor(), keyword, yname, 100, "RFx", rep = i))

            # LIME Explanations
            # full_LDF = explain(xtrain, xtest, full_RF, categorical, cols, "RF", keyword, 100, len(xtrain), i, start, dataset)
            # lime_results.extend(full_LDF)

            #Slack
            full_Slack = Adversarial_Lime_Model(biased_model_f(key_indc), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
            f_Slack_pred = full_Slack.predict(xtest)
            results.append(getMetrics(testing, f_Slack_pred, keyword, 100, len(ytrain), yname, i, "Slack", start))

            # results.append(Fair_Smote(training, testing, full_Slack, keyword, 100, len(ytrain), yname, i, "Slacks")) #SLACK doesn't have a fit function 
            # results.append(maat(training, testing, full_Slack, ss, keyword, 100, len(ytrain), yname, i, "Slackm", dataset)) #After doing WAE and all the MAAT it loses the proper indicatior indicies for SLACK & can't be used 
            # results.append(xFAIR(training, testing, full_Slack, DecisionTreeRegressor(), keyword, yname, 100, "Slackx", rep = i)) #SLACK doesn't have a fit function 
            
            # LIME Explanations
            # full_LDF = explain(xtrain, xtest, full_Slack, categorical, cols, "Slack", keyword, 100, len(xtrain), i, start, dataset)
            # lime_results.extend(full_LDF)

            start2 = time.time()
            #Clustering
            table = Table(i)
            rows = deepcopy(validating.values)
            header = deepcopy(list(validating.columns.values))
            table + header
            for r in rows:
                table + r

            enough = int(math.sqrt(len(table.rows)))
            root = Table.clusters(table.rows, table, enough)

            treatment = [1]
            #Surrogate Experiments 
            for num_points in treatment:
                subset_x, _ = clusterGroups(root, cols, num_points)
                subset_df = pd.DataFrame(subset_x, columns = cols)

                RF_probed_y = full_RF.predict(subset_x)

                # Perform the optimization optimization_function, bounds, args = (xtrain, ytrain), seed = i
                # DE = differential_evolution(optimization_function, bounds, args=(subset_x, RF_probed_y), seed = i)

                # Use the optimal hyperparameters to train a new model
                RF_surrogate = RandomForestClassifier()

                RF_surrogate.fit(subset_x, RF_probed_y) 
                RF_surr_pred = RF_surrogate.predict(xtest)
                results.append(getMetrics(testing, RF_surr_pred, keyword, num_points, len(RF_probed_y), yname, i, "RF", start2, clf = RF_surrogate )) #surro performance
                
                #Baseline Fairness Tools for Surrogate set
                # subset_df[yname] = RF_probed_y
                # results.append(Fair_Smote(subset_df, testing, RF_surrogate, keyword, num_points, len(RF_probed_y), yname, i, "RFs"))
                # results.append(maat(subset_df, testing, RF_surrogate, ss, keyword, num_points, len(RF_probed_y), yname, i, "RFm", dataset))
                # results.append(xFAIR(subset_df, testing, RF_surrogate, DecisionTreeRegressor(), keyword, yname, num_points, "RFx", rep = i))
                # subset_df.drop([yname], axis=1, inplace=True)

                # SLACK Surrogate
                Slack_probed_y = full_Slack.predict(subset_x)

                # Perform the optimization
                # DE = differential_evolution(optimization_function, bounds, args=(subset_x, Slack_probed_y), seed = i)

                # Use the optimal hyperparameters to train a new model
                Slack_surrogate = RandomForestClassifier()

                Slack_surrogate.fit(subset_x, Slack_probed_y) 
                Slack_surr_pred = Slack_surrogate.predict(xtest)
                results.append(getMetrics(testing, Slack_surr_pred, keyword, num_points, len(Slack_probed_y), yname, i, "Slack", start2))
                
                # subset_df[yname] = Slack_probed_y
                # results.append(Fair_Smote(subset_df, testing, Slack_surrogate, keyword, num_points, len(ytrain), yname, i, "Slacks"))
                # results.append(maat(subset_df, testing, Slack_surrogate, ss, keyword, num_points, len(ytrain), yname, i, "Slackm", dataset))
                # results.append(xFAIR(subset_df, testing, Slack_surrogate, DecisionTreeRegressor(), keyword, yname, num_points, "Slackx", rep = i))
                # subset_df.drop([yname], axis=1, inplace=True)


                # Surrogate LIME Explanations; LIME doesn't work with only one class (KNN/2 class dependent)
                # print(RF_surrogate.classes_ , Slack_surrogate.classes_)
                if len(RF_surrogate.classes_ ) <= 1 :
                    # RF_surro_LDF = explain(subset_x, xtest, RF_surrogate, categorical, cols, "RF", keyword, num_points, len(subset_x), i, start2, dataset)
                    # lime_results.extend(RF_surro_LDF)
                # else: 
                    print("Error with " + dataset + " RF surrogate classes: ", RF_surrogate.classes_ )
                
                if len(Slack_surrogate.classes_) <= 1 :
                    # Slack_surro_LDF = explain(subset_x, xtest, Slack_surrogate, categorical, cols, "Slack", keyword, num_points, len(subset_x), i, start2, dataset)
                    # lime_results.extend(Slack_surro_LDF)
                # else: 
                    print("Error with "+ dataset + " Slack surrogate classes: ", Slack_surrogate.classes_ )


        mets = pd.DataFrame(results, columns = ["rep", "learner", "biased_col","treatment", "samples", "runtime", "rec+", "prec+", "acc+", "F1+", "FA0-", "FA1-","MCC-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-", "Flip"]) 
        mets.to_csv("./paper/" +  dataset + "_metrics" + str(i) + ".csv", index=False)

        all_L = pd.DataFrame(lime_results, columns = ["ranking","feature", "occurances_pct", "learner", "biased_col", "runtime", "treatment", "samples", "rep"])
        all_L.to_csv("./paper/" +  dataset + "_LIME" + str(i) + ".csv", index=False)


if __name__ == "__main__":
    main()
