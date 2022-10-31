import warnings
warnings.filterwarnings('ignore')

import random,time,csv,math,copy,os,sys
import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

# sys.path.append(os.path.abspath('..'))

from smote import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples
from utils import *
from cols import Table, Col, Sym, Num, leafmedians2, getLeafData2, getXY2
from extraction import clusterGroups
params = Params("./model_configurations/experiment_params.json")
np.random.seed(params.seed)

def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
    cols = df.columns
    smt = smote(df)
    df = smt.run()
    df.columns = cols
    return df



def getMetrics(test_df, y_test, y_pred, biased_col, samples, yname, rep, learner):
#extraction run
    recall = measure_final_score(test_df, y_test, y_pred, biased_col, 'recall', yname)
    precision = measure_final_score(test_df, y_test, y_pred, biased_col, 'precision', yname)
    accuracy = measure_final_score(test_df, y_test, y_pred, biased_col, 'accuracy', yname)
    F1 = measure_final_score(test_df, y_test, y_pred, biased_col, 'F1', yname)
    AOD = measure_final_score(test_df, y_test, y_pred, biased_col, 'aod', yname)
    EOD =measure_final_score(test_df, y_test, y_pred, biased_col, 'eod', yname)
    SPD = measure_final_score(test_df, y_test, y_pred, biased_col, 'SPD', yname)
    FA0 = measure_final_score(test_df, y_test, y_pred, biased_col, 'FA0', yname)
    FA1 = measure_final_score(test_df, y_test, y_pred, biased_col, 'FA1', yname)
    DI = measure_final_score(test_df, y_test, y_pred, biased_col, 'DI', yname)
    MSE = round(mean_squared_error(y_test, y_pred),3)


    return [recall, precision, accuracy, F1, FA0, FA1, MSE, AOD, EOD, SPD, DI, biased_col, samples, rep, learner]

def flip(X_test,keyword):
    X_flip = X_test.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    return X_flip

def calculate_flip(clf,X_test,keyword):
    X_flip = flip(X_test,keyword)
    a = np.array(clf.predict(X_test))
    b = np.array(clf.predict(X_flip))
    total = X_test.shape[0]
    same = np.count_nonzero(a==b)
    return (total-same)/total

def situation(clf,X_train,y_train,keyword):
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    same = [1 if each else 0 for each in same]
    X_train['same'] = same
    X_train['y'] = y_train
    X_rest = X_train[X_train['same']==1]
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','y'])
    return X_rest,y_rest


def Fair_Smote(training_df, testing_df, base_clf, keyword, rep, samples, yname, learner):
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
    if zero_zero_zero or zero_one_zero or one_zero_zero or one_one_zero <= 3:
        return [None, None, None, None, None, None, None, None, None, None, None, biased_col, samples, rep, learner]

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
    cm = confusion_matrix(y_test, y_pred)

    res = getMetrics(test_df, y_test, y_pred, keyword, samples, yname, rep, learner)
    flip_rate = calculate_flip(clf2,X_test,keyword)
    res.append(round(flip_rate,3))

    return res

#apply smote to OG, Surrogates

def main():
    datasets = ["communities", "heart", "diabetes", "studentperformance", "compas", "bankmarketing", "defaultcredit", "adultscensusincome"] #"germancredit" idk why but doesn;t work with SVC surrogate
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['SEX('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['Age('],
                'studentperformance': ['sex(']
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


            # scaler = MinMaxScaler()
            # data_df = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
            #
            # train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=i)
            # X_train, y_train = train_df.loc[:, train_df.columns != yname], train_df[yname]
            # X_test, y_test = test_df.loc[:, test_df.columns != yname], test_df[yname]

            for i in range(10):
                i += 1

                xtrain,xtest,ytrain,ytest = train_test_split(X.values, y, test_size=0.2, random_state = i)

                ss = MinMaxScaler().fit(xtrain)
                xtrain = ss.transform(xtrain)
                xtest = ss.transform(xtest)


                testing = pd.DataFrame(xtest, columns = cols)
                training = pd.DataFrame(xtrain, columns = cols)
                training[yname] = deepcopy(ytrain)
                testing[yname] = deepcopy(ytest)

                full_RF = RandomForestClassifier()
                full_RF.fit(xtrain, ytrain)
                f_RF_pred = full_RF.predict(xtest)
                results.append(getMetrics(testing, ytest, f_RF_pred, keyword, len(ytrain), yname, i, "RF" ))
                results.append(Fair_Smote(training, testing, RandomForestClassifier(), keyword, i, len(ytrain), yname, "RF"))

                full_LSR = LogisticRegression()
                full_LSR.fit(xtrain, ytrain)
                f_LSR_pred = full_LSR.predict(xtest)
                results.append(getMetrics(testing, ytest, f_LSR_pred, keyword, len(ytrain), yname, i, "LSR" ))
                results.append(Fair_Smote(training, testing, LogisticRegression(), keyword, i, len(ytrain), yname, "LSR"))

                full_SVC = LinearSVC()
                full_SVC.fit(xtrain, ytrain)
                f_SVC_pred = full_SVC.predict(xtest)
                results.append(getMetrics(testing, ytest, f_SVC_pred, keyword, len(ytrain), yname, i, "SVC" ))
                #no predict.proba() not compatible
                results.append(Fair_Smote(training, testing, LinearSVC(), keyword, i, len(ytrain), yname, "SVC"))

                # full_Slack = Adversarial_Lime_Model(biased_model_f(sensa_indc[0]), innocuous_model_psi(inno_indc)).train(xtrain, ytrain, feature_names=cols, perturbation_multiplier=2, categorical_features=categorical)
                # f_Slack_pred = full_Slack.predict(xtest)
                # results.append(getMetrics(testing, ytest, f_Slack_pred, keyword, len(ytrain), yname, i, "Slack" ))
                # results.append(Fair_Smote(training, LinearSVC(), keyword, i, len(ytrain), yname, "Slack"))

                table = Table(i)
                rows = deepcopy(training.values)
                header = deepcopy(list(training.columns.values))
                table + header
                for r in rows:
                    table + r

                enough = int(math.sqrt(len(table.rows)))
                root = Table.clusters(table.rows, table, enough)

                # #O-CART
                # tree = DecisionTreeClassifier(min_samples_leaf = enough, random_state = i)
                # tree.fit(xtrain, ytrain)
                # leaves = tree.get_n_leaves()
                # leaf_indexes = tree.apply(xtrain)
                # new_leaves = list(set(leaf_indexes))=

                if dataset in ["compas", "bankmarketing", "defaultcredit", "adultscensusincome"]:
                    treatment = [4,5,6,7,8,9,10]
                else:
                    treatment = [2,3,4,5]

                for num_points in treatment:
                    subset_x, clustered_y = clusterGroups(root, cols, num_points)
                    subset_df = pd.DataFrame(subset_x, columns = cols)

                    #O-CART leaf extraction
                    # subset_x = []
                    # for l in new_leaves:
                    #     indices = [i for i, x in enumerate(leaf_indexes) if x == l] #gives indexes of leaf_num aka index of all points in leaf
                    #     x_index = np.random.choice(indices, replace = False, size = num_points)
                    #     for t in x_index:
                    #         subset_x.append(xtrain[t])

                    RF_probed_y = full_RF.predict(subset_x)
                    RF_surrogate = RandomForestClassifier().fit(subset_x, RF_probed_y)
                    RF_surr_pred = RF_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, RF_surr_pred, keyword, len(subset_x), yname, i, "RF_RF" ))

                    subset_df[yname] = RF_probed_y
                    results.append(Fair_Smote(subset_df, testing, RandomForestClassifier(), keyword, i, len(subset_x), yname, "RF_RF"))
                    subset_df.drop([yname], axis=1, inplace=True)

                    LSR_probed_y = full_LSR.predict(subset_x)
                    LSR_surrogate = RandomForestClassifier().fit(subset_x, LSR_probed_y)
                    LSR_surr_pred = LSR_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, LSR_surr_pred, keyword, len(subset_x), yname, i, "LSR_RF" ))

                    subset_df[yname] = LSR_probed_y
                    results.append(Fair_Smote(subset_df, testing, RandomForestClassifier(), keyword, i, len(subset_x), yname, "LSR_RF"))
                    subset_df.drop([yname], axis=1, inplace=True)


                    SVC_probed_y = full_SVC.predict(subset_x)
                    SVC_surrogate = RandomForestClassifier().fit(subset_x, SVC_probed_y)
                    SVC_surr_pred = SVC_surrogate.predict(xtest)
                    results.append(getMetrics(testing, ytest, SVC_surr_pred, keyword, len(subset_x), yname, i, "SVC_RF" ))

                    subset_df[yname] = SVC_probed_y
                    results.append(Fair_Smote(subset_df, testing, RandomForestClassifier(), keyword, i, len(subset_x), yname, "SVC_RF"))
                    subset_df.drop([yname], axis=1, inplace=True)

                    # Slack_probed_y = full_Slack.predict(subset_x)
                    # Slack_surrogate = RandomForestClassifier().fit(subset_x, Slack_probed_y)
                    # Slack_surr_pred = Slack_surrogate.predict(xtest)
                    # results.append(getMetrics(testing, ytest, Slack_surr_pred, keyword, len(subset_x), yname, i, "Slack_RF" ))

        metrics = pd.DataFrame(results, columns = ["recall+", "precision+", "accuracy+", "F1+", "FA0-", "FA1-", "MSE-", "AOD-", "EOD-", "SPD-", "DI-", "biased_col", "samples", "rep", "learner", "Flip"] )
        metrics.to_csv("./output/features/SMOTE/" +  dataset + "_FM.csv", index=False)
        # print ('-'*55)
        # print("Finished " + dataset + " ; biased_model's FEATURE: ", str(sensitive_features[0]))
        # print ('-'*55)


if __name__ == "__main__":
    main()
