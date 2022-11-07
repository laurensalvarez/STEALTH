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

    model_num = copy.deepcopy(df1["samples"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))
    sortedmodels = sorted(sortedmodels)

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['samples']!= m].index, inplace=True)

        learner = copy.deepcopy(dfRF2["learner"].tolist())
        sortedlearner = sorted(set(learner), key = lambda ele: learner.count(ele))
        sortedlearner = sorted(sortedlearner)

        for r in sortedlearner:
            dfRF3 = copy.deepcopy(dfRF2)
            dfRF3.drop(dfRF3.loc[dfRF3['learner']!= r].index, inplace=True)

            features = copy.deepcopy(dfRF3["feature"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                dfRF4 = copy.deepcopy(dfRF3)
                dfRF4.drop(dfRF3.loc[dfRF4['feature']!= f].index, inplace=True)
                vals = dfRF4['importance'].values

                median = round(statistics.median(vals),2)
                # print(m)

                newRow = [m, r, f, median]
                all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["samples", "learner", "feature", "median_importance"])
    return prettydf


if __name__ == "__main__":
    datasets = ["heart", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "defaultcredit"]#, "adultscensusincome"]
# "germancredit"
    # metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'FA0-', 'FA1-','MSE-', 'AOD-', 'EOD-', 'SPD-',  'DI-']
    #LIME COls  rep,samples,learner,feature,importance,biased_col
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        path =  "./output/final/" + dataset + "_feat.csv"
        prettydf = rearrange(path)

        # print(prettydf.head())
        prettydf.to_csv("./Feat/final/" + dataset + ".csv", index = False)
