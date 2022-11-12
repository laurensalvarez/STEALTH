import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier

import math, time
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import lime
import lime.lime_tabular

from slack.utils import *
from slack.adversarial_models import *
from cols import Table,leafmedians2, getLeafData2, getXY2
from metrics.Measure import getMetrics
from xFair import *
from smoting import Fair_Smote, maat

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
##SLACK
#
def explain(xtrain, xtest, learner, categorical, features, model, keyword, treatment, samples, rep, rt):
    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=features, categorical_features= categorical, discretize_continuous=False)

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


def main():
    datasets = [ "bankmarketing", "adultscensusincome"] #"communities","heart", "diabetes", "germancredit", "studentperformance", "meps", "compas", "defaultcredit",
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['SEX('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['Age('],
                'studentperformance': ['sex('],
                'meps': ['race(']
                }

    pbar = tqdm(datasets)
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

            # needed for the SLACK adv_model so add unrelated columns
            X['Unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])

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

            inno_indc = cols.index('Unrelated_column_one')
            sensa_indc = [cols.index(col) for col in sensitive_features]
            categorical = [cols.index(c) for c in cat_features_encoded]

            for i in range(20):
                i += 1
                start = time.time()

                xtrain,xtest,ytrain,ytest = train_test_split(X.values, y, test_size=0.2, random_state = i)

                ss = MinMaxScaler().fit(xtrain)
                xtrain = ss.transform(xtrain)
                xtest = ss.transform(xtest)

                testing = pd.DataFrame(xtest, columns = cols)
                testing[yname] = deepcopy(ytest)

                training = pd.DataFrame(xtrain, columns = cols)
                training[yname] = deepcopy(ytrain)

                full_RF = RandomForestClassifier()
                full_RF.fit(xtrain, ytrain)
                # f_RF_pred = full_RF.predict(xtest)

                # results.append(getMetrics(testing, f_RF_pred, keyword, 100, len(ytrain), yname, i, "RF", start)) #original/raw/baseline performance
                
                # #Baseline Fairness Tools for full set
                # results.append(Fair_Smote(training, testing, full_RF, keyword, 100, len(ytrain), yname, i, "RF_s"))
                # results.append(maat(training, testing, full_RF, ss, keyword, 100, len(ytrain), yname, i, "RF_m", dataset))
                # results.append(xFAIR(training, testing, full_RF, DecisionTreeRegressor(), keyword, yname, 100, "RF_x", rep = i))

                Explanations
                full_LDF = explain(xtrain, xtest, full_RF, categorical, cols, "RF", keyword, 100, len(xtrain), i, start)
                lime_results.extend(full_LDF)

                RF Explanations
                full_import = full_RF.feature_importances_
                sorted_indices = np.argsort(full_import)[::-1]
                for feat in range(xtrain.shape[1]):
                    timer = time.time() - start
                    feat_importance_tuple_list.append([i ,"RF", keyword, len(ytrain), timer , cols[sorted_indices[feat]], round(full_import[sorted_indices[feat]],3)])

                #Slack
                full_Slack = Adversarial_Lime_Model(biased_model_f(cols.index(keyword)), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
                # f_Slack_pred = full_Slack.predict(xtest)
                # results.append(getMetrics(testing, f_Slack_pred, keyword, 100, len(ytrain), yname, i, "Slack", start))

                # results.append(Fair_Smote(training, testing, full_Slack, keyword, 100, len(ytrain), yname, i, "Slack_s"))
                # results.append(maat(training, testing, full_Slack, ss, keyword, 100, len(ytrain), yname, i, "Slack_m", dataset))
                # results.append(xFAIR(training, testing, full_Slack, DecisionTreeRegressor(), keyword, yname, 100, "Slack_x", rep = i))
                

                full_LDF = explain(xtrain, xtest, full_Slack, categorical, cols, "Slack", keyword, 100, len(xtrain), i, start)
                lime_results.extend(full_LDF)

                
                start2 = time.time()

                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)

                treatment = [2]

                for num_points in treatment:
                    subset_x, clustered_y = clusterGroups(root, cols, num_points)
                    # subset_df = pd.DataFrame(subset_x, columns = cols)

                    RF_probed_y = full_RF.predict(subset_x)
                    RF_surrogate = RandomForestClassifier().fit(subset_x, RF_probed_y)
                    # RF_surr_pred = RF_surrogate.predict(xtest)
                    # results.append(getMetrics(testing, RF_surr_pred, keyword, num_points, len(RF_probed_y), yname, i, "RF", start2 )) #surro performance
                    
                    #Baseline Fairness Tools for distilled set
                    # subset_df[yname] = RF_probed_y
                    # results.append(Fair_Smote(subset_df, testing, full_RF, keyword, num_points, len(RF_probed_y), yname, i, "RF_s"))
                    # results.append(maat(subset_df, testing, full_RF, ss, keyword, num_points, len(RF_probed_y), yname, i, "RF_m", dataset))
                    # results.append(xFAIR(subset_df, testing, RF_surrogate, DecisionTreeRegressor(), keyword, yname, num_points, "RF_x", rep = i))
                    # subset_df.drop([yname], axis=1, inplace=True)


                    Slack_probed_y = full_Slack.predict(subset_x)
                    Slack_surrogate = RandomForestClassifier().fit(subset_x, Slack_probed_y)
                    # Slack_surr_pred = Slack_surrogate.predict(xtest)
                    # results.append(getMetrics(testing, Slack_surr_pred, keyword, num_points, len(Slack_probed_y), yname, i, "Slack", start2 ))
                    
                    # SLACK SURRO
                    # results.append(getMetrics(testing, f_RF_pred, keyword, 100, len(Slack_probed_y), yname, i, "Slack", start2)) 
                    
                    # subset_df[yname] = Slack_probed_y
                    # results.append(Fair_Smote(subset_df, testing, full_RF, keyword, num_points, len(ytrain), yname, i, "Slack_s"))
                    # results.append(maat(subset_df, testing, full_RF, ss, keyword, num_points, len(ytrain), yname, i, "Slack_m", dataset))
                    # results.append(xFAIR(subset_df, testing, Slack_surrogate, DecisionTreeRegressor(), keyword, yname, num_points, "Slack_x", rep = i))
                    # subset_df.drop([yname], axis=1, inplace=True)

                    Explanations 
                    RF_surro_import = RF_surrogate.feature_importances_
                    RF_sorted_indices = np.argsort(RF_surro_import)[::-1]

                    Slack_surro_import = Slack_surrogate.feature_importances_
                    Slack_sorted_indices = np.argsort(Slack_surro_import)[::-1]

                    RF 
                    print(RF_surrogate.classes_ , Slack_surrogate.classes_)
                    full_LDF = explain(xtrain, xtest, full_RF, categorical, cols, "RF", keyword, 100, len(xtrain), i, start)
                    RF_surro_LDF = explain(subset_x, xtest, RF_surrogate, categorical, cols, "RF", keyword, num_points, len(subset_x), i, start2)
                    lime_results.extend(RF_surro_LDF)

                    Slack_surro_LDF = explain(subset_x, xtest, Slack_surrogate, categorical, cols, "Slack", keyword, num_points, len(subset_x), i, start2)
                    lime_results.extend(Slack_surro_LDF)

                    for feat in range(xtrain.shape[1]):
                        timer2 = time.time() - start
                        feat_importance_tuple_list.append([i, "RF", keyword, num_points,timer2, len(subset_x), cols[RF_sorted_indices[feat]], round(RF_surro_import[RF_sorted_indices[feat]],3)])
                        feat_importance_tuple_list.append([i, "Slack", keyword, num_points, timer2, len(subset_x),cols[Slack_sorted_indices[feat]], round(Slack_surro_import[Slack_sorted_indices[feat]],3)])



        # mets = pd.DataFrame(results, columns = ["rep", "learner", "biased_col","treatment", "samples", "runtime", "rec+", "prec+", "acc+", "F1+", "FA0-", "FA1-","MCC-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-", "Flip"]) 
        # mets.to_csv("./final/" +  dataset +"_metrics.csv", index=False)
        feat_imp = pd.DataFrame(feat_importance_tuple_list, columns = ["rep", "learner", "biased_col","runtime", "treatment", "samples", "feature", "importance"])
        feat_imp.to_csv("./final/" +  dataset + "_FI.csv", index=False)
        all_L = pd.DataFrame(lime_results, columns = ["ranking","feature", "occurances_pct", "learner", "biased_col", "runtime", "treatment", "samples", "rep"])
        all_L.to_csv("./final/" +  dataset + "._LIME.csv", index=False)


if __name__ == "__main__":
    main()
