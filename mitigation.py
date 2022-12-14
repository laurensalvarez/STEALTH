import warnings
warnings.filterwarnings('ignore')

import random,time,csv,math,copy,os,sys
import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from metrics.Measure import situation, getMetrics
from smote.Generate_Samples import generate_samples
from slack.utils import *
from slack.adversarial_models import *
from datasets.preprocessing import classBal


from maat.WAE import data_dis


##
# MAAT
def maat(X_train, X_test, clf, ss, keyword, num_points, samples, yname, rep, learner, dataset):
    start2 = time.time()
    X_train = deepcopy(X_train)
    X_test = deepcopy(X_test)
    y_train = X_train[yname].values
    X_train.drop([yname], axis=1, inplace=True)

    inverse = ss.inverse_transform(X_train.values)
    raw_xtrain = pd.DataFrame(inverse, columns = X_train.columns)
    raw_xtrain[yname] = y_train 

    zero_zero, zero_one, one_zero, one_one = classBal(raw_xtrain, yname, keyword)
    # print("MAAT (rep, samples):", (rep, samples ), "a:", str(zero_one+one_one), "\n class distribution:", (zero_zero, zero_one, one_zero, one_one))

    if (zero_one+one_one == 0) or (zero_one < 0) or (zero_zero < 0) or (one_one < 0) or (one_zero <0) :
        print("MAAT CANCELLED (rep, samples):", (rep, samples ), "a:", str(zero_one+one_one), "\n class distribution:", (zero_zero, zero_one, one_zero, one_one))
        timer = round(time.time() - start2, 2)
        return [rep, learner, keyword, num_points, samples, timer, None, None, None, None, None, None, None, None, None, None, None, None,None]
   
    X_train_WAE = data_dis(raw_xtrain,keyword,dataset, yname)
    y_train_WAE = X_train_WAE[yname].values
    X_train_WAE.drop([yname], axis=1, inplace=True)

    sc = MinMaxScaler().fit(X_train_WAE)
    X_train_WAE = pd.DataFrame(sc.transform(X_train_WAE), columns = X_train.columns)
    clf_WAE = RandomForestClassifier().fit(X_train_WAE,y_train_WAE)
 
    y_test = X_test[yname].values
    X_test.drop([yname], axis=1, inplace=True)
    X_test_WAE = pd.DataFrame(sc.transform(X_test.values), columns = X_test.columns)
    
    pred_de1 = clf.predict_proba(X_test)
    pred_de2 = clf_WAE.predict_proba(X_test_WAE)

    # print(len(pred_de1), pred_de1[0])
    # print(len(pred_de2), pred_de2[0])

    if (len(clf_WAE.classes_) != 2) or (len(clf.classes_) != 2) :
        print("MAAT CANCELLED for inbalanced class preds (rep, samples):", (rep, samples ), "\nclf_WAE classes:", clf_WAE.classes_, "clf classes:", clf.classes_, "\n class distribution:", (zero_zero, zero_one, one_zero, one_one))
        timer = round(time.time() - start2, 2)
        return [rep, learner, keyword, num_points, samples, timer, None, None, None, None, None, None, None, None, None, None, None, None,None]

    pred = []
    for i in range(len(pred_de1)):
        prob_t = (pred_de1[i][1]+pred_de2[i][1])/2
        if prob_t >= 0.5:
            pred.append(1)
        else:
            pred.append(0)

    y_pred = np.array(pred)

    X_test[yname] = y_test
    res = getMetrics(X_test, y_pred, keyword, num_points, samples, yname, rep, learner, start2)
    print("MAAT Round", (rep), "finished.")

    return res



## Fair-SMOTE
#
def classBal(ds, yname, protected_attribute):
    zero_zero_zero = len(ds[(ds[yname] == 0) & (ds[protected_attribute] == 0)])
    zero_one_zero = len(ds[(ds[yname] == 0) & (ds[protected_attribute] == 1)])
    one_zero_zero = len(ds[(ds[yname] == 1) & (ds[protected_attribute] == 0)])
    one_one_zero = len(ds[(ds[yname] == 1) & (ds[protected_attribute] == 1)])

    # print("Protected_attribute", protected_attribute, "\n class distribution: \n0 0: ", zero_zero_zero, "\n0 1: ",zero_one_zero, "\n1 0: ", one_zero_zero,"\n1 1: ",one_one_zero)
    return zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero



def Fair_Smote(training_df, testing_df, base_clf, keyword, num_points, samples, yname, rep, learner):
    start2 = time.time()
    train_df = deepcopy(training_df)
    test_df = deepcopy(testing_df)
    acc, pre, recall, f1 = [], [], [], []
    aod1, eod1, spd1, di1 = [], [], [], []
    fr=[]

    zero_zero_zero = len(train_df[(train_df[yname] == 0) & (train_df[keyword] == 0)])

    zero_one_zero = len(train_df[(train_df[yname] == 0) & (train_df[keyword] == 1)])

    one_zero_zero = len(train_df[(train_df[yname] == 1) & (train_df[keyword] == 0)])

    one_one_zero = len(train_df[(train_df[yname] == 1) & (train_df[keyword] == 1)])

    # print("class distribution:", (zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero))
    if (zero_zero_zero < 3) or (zero_one_zero < 3) or (one_zero_zero < 3) or (one_one_zero < 3):
        print("SMOTE CANCELLED (rep, samples):", (rep, samples ), "\n class distribution:", (zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero))
        timer = round(time.time() - start2, 2)
        return [rep, learner, keyword, num_points, samples, timer, None, None, None, None, None, None, None, None, None, None, None, None, None]

    maximum = max(zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero)
    zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
    zero_one_zero_to_be_incresed = maximum - zero_one_zero
    one_zero_zero_to_be_incresed = maximum - one_zero_zero
    one_one_zero_to_be_incresed = maximum - one_one_zero

    df_zero_zero_zero = train_df[(train_df[yname] == 0) & (train_df[keyword] == 0)]

    df_zero_one_zero = train_df[(train_df[yname] == 0) & (train_df[keyword] == 1)]

    df_one_zero_zero = train_df[(train_df[yname] == 1) & (train_df[keyword] == 0)]

    df_one_one_zero = train_df[(train_df[yname] == 1) & (train_df[keyword] == 1)]

    df_zero_zero_zero[keyword] = df_zero_zero_zero[keyword].astype(str)
    df_zero_one_zero[keyword] = df_zero_one_zero[keyword].astype(str)
    df_one_zero_zero[keyword] = df_one_zero_zero[keyword].astype(str)
    df_one_one_zero[keyword] = df_one_one_zero[keyword].astype(str)

    # print("Start generating samples...")
    df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed, df_zero_zero_zero, '')
    df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed, df_zero_one_zero, '')
    df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed, df_one_zero_zero, '')
    df_one_one_zero = generate_samples(one_one_zero_to_be_incresed, df_one_one_zero, '')
    samples_df = pd.concat([df_zero_zero_zero, df_zero_one_zero,df_one_zero_zero, df_one_one_zero])

    samples_df.columns = train_df.columns
    clf = base_clf
    clf.fit(train_df.loc[:, train_df.columns != yname], train_df[yname])
    X_train, y_train = samples_df.loc[:, samples_df.columns != yname], samples_df[yname]
    # print("Situational testing...")
    X_train, y_train = situation(clf, X_train, y_train, keyword)
    X_test, y_test = test_df.loc[:, test_df.columns != yname], test_df[yname]

    clf2 = base_clf
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)

    res = getMetrics(test_df, y_pred, keyword, num_points, samples, yname, rep, learner, start2)
    print("Fair-SMOTE Round", (rep), "finished.")
    return res

## to add: xFAIR?? maybe
#

## MAIN Experiment
#

def main():
    params = Params("./model_configurations/experiment_params.json")
    np.random.seed(params.seed)
    datasets = ["communities","heart", "diabetes", "germancredit", "studentperformance","compas", "bankmarketing", "defaultcredit", "adultscensusincome"] # , 'meps'
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

            inno_indc = cols.index('Unrelated_column_one')
            sensa_indc = [cols.index(col) for col in sensitive_features]
            categorical = [cols.index(c) for c in cat_features_encoded]


            for i in range(10):
                i += 1
                start = time.time()

                xtrain,xtest,ytrain,ytest = train_test_split(X.values, y, test_size=0.2, random_state = i)
                
                # xtrain_copy = pd.DataFrame(copy.deepcopy(xtrain), columns = cols)
                # xtrain_copy[yname] = copy.deepcopy(ytrain)
                
                ss = MinMaxScaler().fit(xtrain)
                training = pd.DataFrame(ss.transform(xtrain), columns = cols)
                training[yname] = deepcopy(ytrain)
                # print(training.head())
         

                testing = pd.DataFrame(ss.transform(xtest), columns = cols)
                testing[yname] = deepcopy(ytest)

                full_RF = RandomForestClassifier()
                full_RF.fit(xtrain, ytrain)
                # f_RF_pred = full_RF.predict(xtest)
                # results.append(getMetrics(testing, f_RF_pred, keyword, len(ytrain), yname, i, "RF"))
                results.append(Fair_Smote(training, testing, full_RF, keyword, 100, i, len(ytrain), yname, "RF_s", start))
                results.append(maat(training, testing, full_RF, ss, keyword, 100, len(ytrain), i, "RF_m", dataset, yname, start))


                # full_Slack = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(cols.index(keyword)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
                # f_Slack_pred = full_Slack.predict(xtest)
                # results.append(getMetrics(testing, f_Slack_pred, keyword, len(ytrain), yname, i, "Slack"))
                # results.append(Fair_Smote(training, testing, LinearSVC(), keyword, i, len(ytrain), yname, "Slack_s"))
                
                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)


                treatment = [2,3,4,5]

                for num_points in treatment:
                    subset_x, clustered_y = clusterGroups(root, cols, num_points)
                    subset_df = pd.DataFrame(subset_x, columns = cols)


                    RF_probed_y = full_RF.predict(subset_x)
                    RF_surrogate = RandomForestClassifier().fit(subset_x, RF_probed_y)
                    # RF_surr_pred = RF_surrogate.predict(xtest)
                    # results.append(getMetrics(testing, RF_surr_pred, keyword, len(subset_x), yname, i, "RF"))

                    subset_df[yname] = RF_probed_y
                    results.append(Fair_Smote(subset_df, testing, RandomForestClassifier(), keyword, num_points, i, len(subset_x), yname, "RF_s", start))
                    results.append(maat(subset_df, testing, RF_surrogate, ss, keyword, num_points, len(subset_x), i, "RF_m", dataset, yname, start))
                    # subset_df.drop([yname], axis=1, inplace=True)

                    # Slack_probed_y = full_Slack.predict(subset_x)
                    # Slack_surrogate = RandomForestClassifier().fit(subset_x, Slack_probed_y)
                    # Slack_surr_pred = Slack_surrogate.predict(xtest)
                    # results.append(getMetrics(testing, Slack_surr_pred, keyword, len(subset_x), yname, i, "Slack_RF", 0 ))

                    # subset_df[yname] = Slack_probed_y
                    # results.append(Fair_Smote(subset_df, testing, RandomForestClassifier(), keyword, i, len(subset_x), yname, "RF_RF"))
                    # subset_df.drop([yname], axis=1, inplace=True) 


        metrics = pd.DataFrame(results, columns = ["rep", "learner","biased_col", "treatment", "runtime", "samples", "rec+", "prec+", "acc+", "F1+", "FA0-", "FA1-", "MCC-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-"] )
       
        metrics.to_csv("./final/maat/" +  dataset + ".csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)


if __name__ == "__main__":
    main()
