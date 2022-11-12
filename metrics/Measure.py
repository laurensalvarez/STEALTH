import numpy as np
import copy,math, time
from sklearn.metrics import mean_squared_error, matthews_corrcoef


def get_counts(df, y_pred, biased_col, metric, yname):

    # TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    test_df_copy = copy.deepcopy(df)
    test_df_copy['y_pred' + biased_col] = y_pred

    # print("test_df_copy:", test_df_copy.head())

    test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy[yname] == 1) &
                                           (test_df_copy['y_pred' + biased_col] == 1) &
                                           (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy[yname] == 0) &
                                                  (test_df_copy['y_pred' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy[yname] == 1) &
                                                  (test_df_copy['y_pred' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy[yname] == 0) &
                                                  (test_df_copy['y_pred' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy[yname] == 1) &
                                                  (test_df_copy['y_pred' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy[yname] == 0) &
                                                  (test_df_copy['y_pred' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy[yname] == 1) &
                                                  (test_df_copy['y_pred' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy[yname] == 0) &
                                                  (test_df_copy['y_pred' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    a = test_df_copy['TP_' + biased_col + "_1"].sum()
    b = test_df_copy['TN_' + biased_col + "_1"].sum()
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()


    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    elif metric=='recall':
        return calculate_recall(e,h,g,f)
    elif metric=='far':
        return calculate_far(e,h,g,f)
    elif metric=='precision':
        return calculate_precision(e,h,g,f)
    elif metric=='accuracy':
        return calculate_accuracy(e,h,g,f)
    elif metric=='F1':
        return calculate_F1(e,h,g,f)
    elif metric=='TPR':
        return calculate_TPR_difference(a, b, c, d, e, f, g, h)
    elif metric=='FPR':
        return calculate_FPR_difference(a, b, c, d, e, f, g, h)
    elif metric == "DI":
    	return calculate_Disparate_Impact(a, b, c, d, e, f, g, h)
    elif metric == "SPD":
    	return calculate_SPD(a, b, c, d, e, f, g, h)
    elif metric == "FA1":
        return calculate_false_alarm(a,d,c,b)
    elif metric == "FA0":
        return calculate_false_alarm(e,h,g,f)



def calculate_average_odds_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    FPR_diff = calculate_FPR_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0)
    TPR_diff = calculate_TPR_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0)
    average_odds_difference = (FPR_diff + TPR_diff)/2
    return abs(round(average_odds_difference,2))

def calculate_Disparate_Impact(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    P_1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P_0 = (TP_0 + FP_0)/(TP_0 + TN_0 + FN_0 +  FP_0)
    DI = (P_0/P_1)
    return round(abs(1 - DI),2)

def calculate_SPD(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    P_1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P_0 = (TP_0 + FP_0) /(TP_0 + TN_0 + FN_0 +  FP_0)
    SPD = (P_0 - P_1)
    return abs(round(SPD,2))


def calculate_equal_opportunity_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    score = calculate_TPR_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0)
    return abs(score)

def calculate_TPR_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    if (TP_0+FN_0) != 0:
        TPR_0 = TP_0/(TP_0+FN_0)
    else:
        TPR_0 = 0

    if (TP_1+FN_1) != 0:
        TPR_1 = TP_1/(TP_1+FN_1)
    else:
        TPR_1 = 0

    diff = (TPR_0 - TPR_1)
    return round(diff,2)

def calculate_FPR_difference(TP_1 , TN_1, FN_1,FP_1, TP_0 , TN_0 , FN_0,  FP_0):
    if (FP_0+TN_0) != 0:
        FPR_0 = FP_0/(FP_0+TN_0)
    else:
        FPR_0 = 0
    if (FP_1+TN_1) != 0:
        FPR_1 = FP_1/(FP_1+TN_1)
    else:
        FPR_1 = 0
    diff = (FPR_0 - FPR_1)
    return round(diff,2)

def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return round(recall,2)

def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return round(far,2)

def calculate_false_alarm(TP,FP,FN,TN):
    if (TP + FN) != 0:
        alarm = FN / (FN + TP)
    else:
        alarm = 0
    return round(alarm,2)

def calculate_precision(TP,FP,FN,TN):
    if (TP + FP) != 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return round(prec,2)

def calculate_F1(TP,FP,FN,TN):
    precision = calculate_precision(TP,FP,FN,TN)
    recall = calculate_recall(TP,FP,FN,TN)
    if (precision + recall) != 0:
        F1 = (2 * precision * recall)/(precision + recall)
    else:
        F1 = 0
    return round(F1,2)

def calculate_accuracy(TP,FP,FN,TN):
    return round((TP + TN)/(TP + TN + FP + FN),2)

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


def measure_final_score(test_df, y_pred, biased_col, metric, yname):
    df = copy.deepcopy(test_df)
    return get_counts(df, y_pred, biased_col, metric, yname)

def getMetrics(test_df, y_pred, biased_col, treatment, samples, yname, rep, learner, start, clf = None):

    recall = measure_final_score(test_df, y_pred, biased_col, 'recall', yname)
    precision = measure_final_score(test_df, y_pred, biased_col, 'precision', yname)
    accuracy = measure_final_score(test_df, y_pred, biased_col, 'accuracy', yname)
    F1 = measure_final_score(test_df, y_pred, biased_col, 'F1', yname)
    AOD = measure_final_score(test_df, y_pred, biased_col, 'aod', yname)
    EOD =measure_final_score(test_df, y_pred, biased_col, 'eod', yname)
    SPD = measure_final_score(test_df, y_pred, biased_col, 'SPD', yname)
    FA0 = measure_final_score(test_df, y_pred, biased_col, 'FA0', yname)
    FA1 = measure_final_score(test_df, y_pred, biased_col, 'FA1', yname)
    DI = measure_final_score(test_df, y_pred, biased_col, 'DI', yname)
    MSE = round(mean_squared_error(test_df[yname], y_pred),2)
    MCC = round(matthews_corrcoef(test_df[yname], y_pred), 2)
    timer = round(time.time() - start, 2)
    if clf :
        Flip = round(calculate_flip(clf, test_df[:, test_df.columns != yname],biased_col),2)
    else:
        Flip = None

    return [rep, learner, biased_col, treatment, samples, timer, recall, precision, accuracy, F1, FA0, FA1, MCC, MSE, AOD, EOD, SPD, DI, Flip]

