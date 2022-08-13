import numpy as np
import copy,math
import math
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from measure import measure_final_score,calculate_recall,calculate_precision,calculate_accuracy
from cols import Table, Col, Sym, Num



###############################################
### confusion matrix
###############################################

# normal matrix
def getMetrics(test_df, y_true, y_pred, biased_col, samples, fold):

    recall = measure_final_score(test_df, y_true, y_pred, biased_col, 'recall')
    precision = measure_final_score(test_df, y_true, y_pred, biased_col, 'precision')
    accuracy = measure_final_score(test_df, y_true, y_pred, biased_col, 'accuracy')
    F1 = measure_final_score(test_df, y_true, y_pred, biased_col, 'F1')
    AOD = measure_final_score(test_df, y_true, y_pred, biased_col, 'aod')
    EOD =measure_final_score(test_df, y_true, y_pred, biased_col, 'eod')
    SPD = measure_final_score(test_df, y_true, y_pred, biased_col, 'SPD')
    FA0 = measure_final_score(test_df, y_true, y_pred, biased_col, 'FA0')
    FA1 = measure_final_score(test_df, y_true, y_pred, biased_col, 'FA1')

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, FA0, FA1, biased_col, samples, fold]

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

def getBiasCols2(dataset):
    bias_cols = []
    if dataset == "CleanCOMPAS53":
        bias_cols = ["sex(", "Age(","race("]

    if dataset == "GermanCredit":
        bias_cols = ["C_a(","sav(", "sex(" , "Age(", "f_w("]

    if dataset == "diabetes":
        bias_cols = ["Age("]

    if dataset == "adultscensusincome":
        bias_cols = ["sex(", "race("]

    if dataset == "bankmarketing":
        bias_cols = ["Age(", "marital(", "education("]

    if dataset == "defaultcredit":
        bias_cols = ["LIMIT_BAL(", "SEX(","EDUCATION(","MARRIAGE(","AGE("]


    return bias_cols


def sampleMetrics(test_df, y_true, y_pred, biased_cols, samples, f):
    colmetrics = {}
    for i in biased_cols:
        colmetrics[i] = getMetrics(test_df, y_true, y_pred, i, samples, f)
    return colmetrics

###############################################
###
###############################################
def main():
    datasets = ["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        filepath = r'./output/clones/' + filename + "_M0.csv"

        preddf = pd.read_csv(filepath)

        biased_cols = getBiasCols(filename)

        samples = copy.deepcopy(bintdf["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
        all_metrics = {}
        rows = []

        for s in sortedsamples:
            print("Metrics for", s, "samples: \n")
            dfs = copy.deepcopy(bintdf)
            dfs.drop(dfs.loc[dfs['samples']!= s].index, inplace=True)
            list = []
            for i in range(1,21):

                dfr = copy.deepcopy(dfs)
                dfr.drop(dfs.loc[dfs['run_num']!= i].index, inplace=True)
                if '!probability' in dfr.columns:
                    y_true = dfr["!probability"]
                else:
                    y_true = dfr["!Probability"]
                y_pred = dfr["predicted"]
                # print(dfr)
                # rnum = dfs["run_num"]
                list.insert(i, sampleMetrics(dfr, y_true, y_pred, biased_cols, s, i))
            all_metrics[s] = list
            # print("all_metrics:", all_metrics)

        for key, v in all_metrics.items():
            # print("data dict: " , key, "v:", v)
            for i in range(len(v)):
                for key2, v2 in v[i].items():
                    tmp = v2
                    # print(tmp)
                    rows.append(tmp)

        fulldf = pd.DataFrame(rows, columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'feature', 'sample_size', 'fold'])

        fulldf.to_csv("./metrics/M0/" + filename + "_metrics.csv", index=False)



# self = options(__doc__)
if __name__ == '__main__':
    main()
