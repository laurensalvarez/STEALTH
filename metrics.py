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
def getMetrics(test_df, y_true, y_pred, biased_col, samples, total_pts, fold):
    # print("samples", samples, "run_num", run_num)
    # print("y_true:", y_true,"\n", "y_pred:", y_pred, "\n")
    # cm = confusion_matrix(y_true,y_pred,labels=[0, 1])
    # # print(cm)
    # TN, FP, FN, TP = cm.ravel()

    recall = measure_final_score(test_df, y_true, y_pred, biased_col, 'recall')
    precision = measure_final_score(test_df, y_true, y_pred, biased_col, 'precision')
    accuracy = measure_final_score(test_df, y_true, y_pred, biased_col, 'accuracy')
    F1 = measure_final_score(test_df, y_true, y_pred, biased_col, 'F1')
    AOD = measure_final_score(test_df, y_true, y_pred, biased_col, 'aod')
    EOD =measure_final_score(test_df, y_true, y_pred, biased_col, 'eod')
    SPD = measure_final_score(test_df, y_true, y_pred, biased_col, 'SPD')
    FA0 = measure_final_score(test_df, y_true, y_pred, biased_col, 'FA0')
    FA1 = measure_final_score(test_df, y_true, y_pred, biased_col, 'FA1')

    # print("recall :", recall)
    # print("precision :", precision)
    # print("accuracy :", accuracy)
    # print("F1 Score :", F1)
    # print("AOD :" AOD)
    # print("EOD :" + biased_col , EOD)
    # print("SPD:", SPD)

    return [recall, precision, accuracy, F1, AOD, EOD, SPD, FA0, FA1, biased_col, samples, total_pts, fold]


def makeBinary(preddf, dataset):
    if dataset == "diabetes":
        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        # preddf.replace({'Age(':{'Malaysia' : 56, 'Paris' : 778 }})
        # preddf['Age('] = np.where((preddf['Age('] <= 25), 1, preddf['Age('])

    if dataset == "CleanCOMPAS53":
        # preddf['race('] = np.where(preddf['race('] == 0 or 1 or 3 or 4 or 5, 0, preddf['race('])
        preddf['race('] = np.where((preddf['race('] == 2), 1, 0)

        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)

    if dataset == "GermanCredit":
        preddf['sav('] = np.where((preddf['sav('] == 0) | (preddf['sav('] == 1) | (preddf['sav('] ==  4), 0, 1)
        # preddf['savings('] = np.where(preddf['savings('] == 2 or 3, 1, preddf['savings('])

        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        # preddf['Age('] = np.where(preddf['Age('] > 25, 0, preddf['Age('])
        # preddf['Age('] = np.where(preddf['Age('] <= 25, 1, preddf['Age('])

        preddf['sex('] = np.where((preddf['sex('] == 1), 0, 1)
        # preddf['sex('] = np.where(preddf['sex('] == 0 or 2 or 3, 1, preddf['sex('])

    if dataset == "adultscensusincome":
        # preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        preddf['sex('] = np.where((preddf['sex('] == 1), 0, 1)
        preddf['race('] = np.where((preddf['race('] == 5), 1, 0)

    if dataset == "bankmarketing":
        preddf['Age('] = np.where((preddf['Age('] > 25), 0, 1)
        preddf['marital('] = np.where((preddf['marital('] == 3), 1, 0)
        preddf['education('] = np.where((preddf['education('] == 6) | (preddf['education('] == 7), 1, 0)


    if dataset == "defaultcredit":
        preddf['SEX('] = np.where((preddf['SEX('] == 1), 1, 0)
        preddf['MARRIAGE('] = np.where((preddf['MARRIAGE('] == 1), 1, 0)
        preddf['EDUCATION('] = np.where((preddf['EDUCATION('] == 1) | (preddf['EDUCATION('] == 2), 1, 0)
        preddf['AGE('] = np.where((preddf['AGE('] > 25), 0, 1)
        preddf['LIMIT_BAL('] = np.where((preddf['LIMIT_BAL('] > 25250), 1, 0)

    return preddf

def getColNames(table):
    colNames = []
    for col in table.cols:
        colNames.append(col.name)
    return colNames

def getBiasCols(dataset):
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


def sampleMetrics(test_df, y_true, y_pred, biased_cols, samples, f, run_num):
    colmetrics = {}

    for i in biased_cols:
        # print(test_df, y_true, y_pred, i, samples, run_num)
        colmetrics[i] = getMetrics(test_df, y_true, y_pred, i, samples, f , run_num)
    return colmetrics

###############################################
###
###############################################
def main():
    datasets = ["adultscensusincome.csv", "bankmarketing.csv", "defaultcredit.csv", "diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv"]
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        filename = dataset[:-4]
        filepath = r'./output/RAND/' + filename + "_RF.csv"
        # print(filepath)
        # predlines = Table.readfile(r'./output/fold/' + filename + "_folded_RF.csv")

        # predtable = Table(2)
        # predtable + predlines[0]
        # for l in predlines[1:]:
        #     predtable + l
        #
        # predColNames = getColNames(predtable)

        preddf = pd.read_csv(filepath)

        bintdf = makeBinary(preddf, filename)
        # print(bintdf)
        # print(bintdf["Age("])
        #
        # sys.exit()

        biased_cols = getBiasCols(filename)

        # using sorted() + set() + count()
        # sorting and removal of duplicates
        samples = copy.deepcopy(bintdf["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
        sortedsamples = sorted(sortedsamples)


        all_metrics = {}
        rows = []
        # print("sorted samples:", sortedsamples)
        treatments = sortedsamples[:4]
        full_set = sortedsamples[4:]

        for s in treatments:
            # print("Metrics for", s, "samples: \n")
            dfs = copy.deepcopy(bintdf)
            dfs.drop(dfs.loc[dfs['samples']!= s].index, inplace=True)
            list = []
            for f in range(1,26):

                dfr = copy.deepcopy(dfs)
                dfr.drop(dfs.loc[dfs['fold']!= f].index, inplace=True)
                if '!probability' in dfr.columns:
                    y_true = dfr["!probability"]
                else:
                    y_true = dfr["!Probability"]
                y_pred = dfr["predicted"]
                # print(dfr)
                tp = dfs["total_pts"].tolist()
                list.insert(f, sampleMetrics(dfr, y_true, y_pred, biased_cols, s, tp[0], f))
            all_metrics[s] = list
            # print("all_metrics:", all_metrics)
        smaller = full_set[0]
        bigger = full_set[1]
        # print("fullset:", full_set)

        # print("Metrics for", smaller, "and", bigger, "samples: \n")
        dfs2 = copy.deepcopy(bintdf)
        dfs2.drop(dfs2.loc[dfs2['samples'] < smaller].index, inplace=True)
        list2 = []
        for f in range(1,26):
            dfr2 = copy.deepcopy(dfs2)
            dfr2.drop(dfs2.loc[dfs2['fold']!= f].index, inplace=True)
            if '!probability' in dfr2.columns:
                y_true = dfr2["!probability"]
            else:
                y_true = dfr2["!Probability"]
            y_pred = dfr2["predicted"]
            # print(dfr)
            tp = dfs2["total_pts"].tolist()
            list2.insert(f, sampleMetrics(dfr2, y_true, y_pred, biased_cols, smaller, tp[0], f))
        all_metrics[smaller] = list2

        for key, v in all_metrics.items():
            # print("data dict: " , key, "v:", v)
            for i in range(len(v)):
                for key2, v2 in v[i].items():
                    tmp = v2
                    # print(tmp)
                    rows.append(tmp)

        fulldf = pd.DataFrame(rows, columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'feature', 'sample_size', 'total_pts', 'fold'])

        fulldf.to_csv("./metrics/RAND_C2/" + filename + "_RF_metrics.csv", index=False)



# self = options(__doc__)
if __name__ == '__main__':
    main()
