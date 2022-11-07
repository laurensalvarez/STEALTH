import sys, random, statistics, math, copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import count, groupby, chain
from collections import defaultdict


def rearrange(path):
    df = pd.read_csv(path)

    df1 = copy.deepcopy(df)

    all = []

    model_num = copy.deepcopy(df1["model_num"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['model_num']!= m].index, inplace=True)

        for r in [1,2,3]:
            dfRF3 = copy.deepcopy(dfRF2)
            dfRF3.drop(dfRF3.loc[dfRF3['ranking']!= r].index, inplace=True)

            features = copy.deepcopy(dfRF3["feature"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                dfRF4 = copy.deepcopy(dfRF3)
                dfRF4.drop(dfRF4.loc[dfRF4['feature']!= f].index, inplace=True)
                vals = dfRF4['occurances_pct'].values

                median = round(statistics.median(vals),2)
                # print(m)

                newRow = [m, r, f, median]
                all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["model_num", "ranking", "feature", "median_occurances_pct"])
    return prettydf

def addModel(path):
    df = pd.read_csv(path)
    df2 = copy.deepcopy(df)
    all = []

    treats = copy.deepcopy(df2["samples"].tolist())
    sortedtreats = sorted(set(treats), key = lambda ele: treats.count(ele))
    sortedtreats = sorted(sortedtreats)

    for m in sortedtreats:
        print(sortedtreats)
        dfRF2 = copy.deepcopy(df2)
        dfRF2.drop(dfRF2.loc[dfRF2['samples']!= m].index, inplace=True)

        learners = copy.deepcopy(dfRF2["learner"].tolist())
        sortedlearners= sorted(set(learners), key = lambda ele: learners.count(ele))
        sortedlearners = sorted(sortedlearners)
        for l in sortedlearners:
            dfRF5 = copy.deepcopy(dfRF2)
            dfRF5.drop(dfRF5.loc[dfRF5['learner']!= l].index, inplace=True)

            for rank in [1,2,3]:
                dfRF3 = copy.deepcopy(dfRF5)
                dfRF3.drop(dfRF3.loc[dfRF3['ranking']!= rank].index, inplace=True)

                features = copy.deepcopy(dfRF3["feature"].tolist())
                sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

                for f in sortedfeatures:
                    dfRF4 = copy.deepcopy(dfRF3)
                    dfRF4.drop(dfRF4.loc[dfRF4['feature']!= f].index, inplace=True)
                    vals = dfRF4['occurances_pct'].values

                    median = round(statistics.median(vals),2)
                    # print(m)

                    newRow = [m, l, rank, f, median]
                    all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["samples", "learner", "ranking", "feature", "median_occurances_pct"])
    return prettydf



if __name__ == "__main__":
    datasets = ["heart", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "defaultcredit"] #, "adultscensusincome"]
# "germancredit"
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'FA0-', 'FA1-','MSE-', 'AOD-', 'EOD-', 'SPD-',  'DI-']
    #LIME COls  ["ranking","feature", "occurances_pct","model_num", "rep"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        path =  "./output/final/" + dataset + "._LIME.csv"
        prettydf = addModel(path)

        # print(prettydf.head())
        prettydf.to_csv("./LIME_rankings/final/" + dataset + ".csv", index = False)
