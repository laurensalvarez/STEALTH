import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
from copy import deepcopy

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
def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out


def xFAIR(traindf, testdf, base_clf, base2, keyword, yname, treatment, learner, rep, ratio=.2, smote1=True, verbose=False, thresh=.5):

    start = time.time()
    X_train = copy.deepcopy(traindf.loc[:, traindf.columns != yname])
    y_train = copy.deepcopy(traindf[yname])
    X_test = copy.deepcopy(testdf.loc[:, testdf.columns != yname])
    y_test = copy.deepcopy(testdf[yname])

    reduced = list(X_train.columns)
    reduced.remove(keyword)
    X_reduced, y_reduced = X_train.loc[:, reduced], X_train[keyword]
    # Build model to predict the protect attribute
    clf1 = copy.deepcopy(base2)
    if smote1:
        sm = SMOTE()
        X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
        clf = copy.deepcopy(base_clf)
        clf.fit(X_trains, y_trains)
        y_proba = clf.predict_proba(X_trains)
        y_proba = [each[1] for each in y_proba]
        if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
            clf1.fit(X_trains, y_trains)
        else:
            clf1.fit(X_trains, y_proba)
    else:
        clf = copy.deepcopy(base_clf)
        clf.fit(X_reduced, y_reduced)
        y_proba = clf.predict_proba(X_reduced)
        y_proba = [each[1] for each in y_proba]
        if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
            clf1.fit(X_reduced, y_reduced)
        else:
            clf1.fit(X_reduced, y_proba)

    if verbose:
        if isinstance(clf1, LinearRegression):
            importances = (clf1.coef_)
        elif isinstance(clf1, LogisticRegression):
            importances = (clf1.coef_[0])
            print("coef:", clf1.coef_[0], "intercept:", clf1.intercept_)
        else:
            importances = clf1.feature_importances_
        indices = np.argsort(importances)
        features = X_reduced.columns

        plt.rcParams.update({'font.size': 14})
        plt.title('Feature Importances on sensitive attribute')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    X_test_reduced = X_test.loc[:, X_test.columns != keyword]
    protected_pred = clf1.predict(X_test_reduced)
    if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
        protected_pred = reg2clf(protected_pred, threshold=thresh)
    # Build model to predict the taget attribute Y
    clf2 = copy.deepcopy(base_clf)

    X_test.loc[:, keyword] = protected_pred
    y_pred = clf2.predict(X_test)

    X_test[yname] = y_test
    res = getMetrics(X_test, y_pred, keyword, treatment, len(y_train), yname, rep, learner, start)
   
    print("XFAIR Round", (rep), "finished.")
    # print('Time', time.time() - start)
    return res



if __name__ == "__main__":
    main()
