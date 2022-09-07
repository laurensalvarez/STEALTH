import numpy as np
import copy,math
import math
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from Measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy
from cols import Table, Col, Sym, Num



###############################################
### confusion matrix
###############################################

# normal matrix
def getMetrics(test_df, y_test, y_pred, biased_col, samples, model_num, fold, yname):

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

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, DI, FA0, FA1, biased_col, samples, model_num, fold]

def getBiasCols(dataset):
    bias_cols = []
    if dataset == "adultscensusincome":
        bias_cols = ["race(","sex("]
    if dataset == "bankmarketing":
        bias_cols = ["Age("]
    if dataset == "compas":
        bias_cols = ["race(","sex("]
    if dataset == "communities":
        bias_cols = ["Racepctwhite("] #Racepctblack(,Racepctwhite(,Racepctasian(,Racepcthisp(
    if dataset == "defaultcredit":
        bias_cols = ["SEX("]
    if dataset == "diabetes":
        bias_cols = ["Age("]
    if dataset == "germancredit":
        bias_cols = ["sex("]
    if dataset == "heart":
        bias_cols = ["Age("]
    if dataset == "studentperformance":
        bias_cols = ["sex("]
    if dataset == "MEPS":
        bias_cols = ["race("]
    return bias_cols

# def getBiasCols2(dataset):
#     bias_cols = []
#     if dataset == "CleanCOMPAS53":
#         bias_cols = ["sex(", "Age(","race("]
#
#     if dataset == "GermanCredit":
#         bias_cols = ["C_a(","sav(", "sex(" , "Age(", "f_w("]
#
#     if dataset == "diabetes":
#         bias_cols = ["Age("]
#
#     if dataset == "adultscensusincome":
#         bias_cols = ["sex(", "race("]
#
#     if dataset == "bankmarketing":
#         bias_cols = ["Age(", "marital(", "education("]
#
#     if dataset == "defaultcredit":
#         bias_cols = ["LIMIT_BAL(", "SEX(","EDUCATION(","MARRIAGE(","AGE("]
#     return bias_cols


def sampleMetrics(test_df, y_test, y_pred, biased_cols, samples, model_num, f, yname):
    colmetrics = {}
    for i in biased_cols:
        colmetrics[i] = getMetrics(test_df, y_test, y_pred, i, samples, model_num, f, yname)
    return colmetrics

###############################################
###
###############################################
def main():
    datasets = ["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        filepath = r'./output/clones/lower/' + dataset + "_all.csv"

        preddf = pd.read_csv(filepath)

        biased_cols = getBiasCols(dataset)

        samples = copy.deepcopy(preddf["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        models = copy.deepcopy(preddf["model_num"].tolist())
        sortedmodels = sorted(set(models), key = lambda ele: models.count(ele))
        all_metrics = {}
        rows = []
        yname = "ytest"

        for m in sortedmodels:
            print("Metrics for model num:", m, "\n")
            dfs = copy.deepcopy(preddf)
            dfs.drop(dfs.loc[dfs['model_num']!= m].index, inplace=True)
            list = []
            for s in sortedsamples:
                dfr = copy.deepcopy(dfs)
                dfr.drop(dfs.loc[dfs['samples']!= s].index, inplace=True)

                for f in range(10):
                    f+=1
                    dfr = copy.deepcopy(dfs)
                    dfr.drop(dfs.loc[dfs['fold']!= f].index, inplace=True)

                    y_test = dfr[yname]

                    y_pred = dfr["predicted"]
                    list.insert(f, sampleMetrics(dfr, y_test, y_pred, biased_cols, s, m, f, yname))
            all_metrics[m] = list
            # print("all_metrics:", all_metrics)

        for model, fold_list in all_metrics.items():
            # print("model_num: ", model, "fold_list:", fold_list)
            for i in range(len(fold_list)):
                for biased_col, metric_row in fold_list[i].items():
                    tmp = metric_row
                    rows.append(tmp)

        # sys.exit()

        fulldf = pd.DataFrame(rows, columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'feature', 'sample_size', 'model_num', 'fold'])

        fulldf.to_csv("./metrics/all_models/lower/" + dataset + ".csv", index=False)



# self = options(__doc__)
if __name__ == '__main__':
    main()
