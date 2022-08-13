import numpy as np
import copy,math,sys
from sklearn.metrics import confusion_matrix, classification_report


def get_counts(test_df, y_pred, y_true, biased_col, metric):

    # TN, FP, FN, TP = confusion_matrix(y_true,y_pred,labels=[0, 1]).ravel()


    test_df_copy = copy.deepcopy(test_df)
    # print(test_df_copy)
    # test_df_copy['predicted'] = test_df_copy['predicted']
    # test_df_copy['!Probability'] = test_df_copy['!Probability']
    # test_df_copy[biased_col] = test_df_copy[biased_col]

    if '!probability' in test_df_copy.columns:
        test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['!probability']  == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['!probability'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['!probability'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['!probability'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['!probability'] == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['!probability'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['!probability'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['!probability'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

    elif '!Probability' in test_df_copy.columns:

        test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['!Probability']  == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['!Probability'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['!Probability'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['!Probability'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['!Probability'] == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['!Probability'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['!Probability'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['!Probability'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

    else:
        test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['ytest']  == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['ytest'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['ytest'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['ytest'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 1), 1, 0)

        test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['ytest'] == 1) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['ytest'] == 0) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['ytest'] == 1) & (test_df_copy['predicted'] == 0) & (test_df_copy[biased_col] == 0), 1, 0)

        test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['ytest'] == 0) & (test_df_copy['predicted'] == 1) & (test_df_copy[biased_col] == 0), 1, 0)

    e = test_df_copy['TP_' + biased_col + "_1"].sum()
    f = test_df_copy['TN_' + biased_col + "_1"].sum()
    g = test_df_copy['FN_' + biased_col + "_1"].sum()
    h = test_df_copy['FP_' + biased_col + "_1"].sum()
    a = test_df_copy['TP_' + biased_col + "_0"].sum()
    b = test_df_copy['TN_' + biased_col + "_0"].sum()
    c = test_df_copy['FN_' + biased_col + "_0"].sum()
    d = test_df_copy['FP_' + biased_col + "_0"].sum()

    # print("a:", a)
    # print("b:", b)
    # print("c:", c)
    # print("d:", d)
    # print("e:", e)
    # print("f:", f)
    # print("g:", g)


    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    elif metric=='recall':
        return calculate_recall(a,d,c,b)
    elif metric=='precision':
        return calculate_precision(a,d,c,b)
    elif metric=='accuracy':
        return calculate_accuracy(a,d,c,b)
    elif metric=='F1':
        return calculate_F1(a,d,c,b)
    elif metric=='TPR':
        return calculate_TPR_difference(a, b, c, d, e, f, g, h)
    elif metric=='FPR':
        return calculate_FPR_difference(a, b, c, d, e, f, g, h)
    elif metric == "SPD":
    	return calculate_SPD(a, b, c, d, e, f, g, h)
    elif metric == "FA0":
        return calculate_false_alarm(a,d,c,b)
    elif metric == "FA1":
        return calculate_false_alarm(e,h,g,f)





def calculate_average_odds_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1):
    FPR_diff = calculate_FPR_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1)
    TPR_diff = calculate_TPR_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1)
    average_odds_difference = (FPR_diff + TPR_diff)/2
    return round(average_odds_difference,2)


def calculate_SPD(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1):
    P_0 = (TP_0 + FP_0)/(TP_0 + TN_0 + FN_0 + FP_0)
    P_1 = (TP_1 + FP_1) /(TP_1 + TN_1 + FN_1 +  FP_1)
    SPD = (P_0 - P_1)
    return round(abs(SPD),2)

def calculate_equal_opportunity_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1):
    return calculate_TPR_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1)

def calculate_TPR_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1):
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

def calculate_FPR_difference(TP_0 , TN_0, FN_0,FP_0, TP_1 , TN_1 , FN_1,  FP_1):
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

def calculate_false_alarm(TP,FP,FN,TN):
    if (TP + FN) != 0:
        alarm = FN / (FN + TP)
    else:
        alarm = 0
    return round(alarm,2)

def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return round(recall,2)

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

def measure_final_score(test_df, y_train, y_true, biased_col, metric):
    df = copy.deepcopy(test_df)
    return get_counts(df, y_train, y_true, biased_col, metric=metric)
