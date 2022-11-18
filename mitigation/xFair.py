import pandas as pd
import time,copy
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from metrics.Measure import *


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
    datasets =  ['communities', 'heart', 'diabetes',  'german', 'student', 'meps', 'compas', 'bank', 'default', 'adult']
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
    base = RandomForestClassifier()
    base2 = DecisionTreeRegressor()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            df1 = pd.read_csv("./Data/"+fname + "_processed.csv")
            result1 = xFAIR(df1, base,base2, keyword=keyword, rep=20,verbose=True)
            a, p, r, f, ao, eo, spd, di = result1
            print("**"*50)
            print(fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))