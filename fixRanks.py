import copy
import math
import sys, random, statistics
from tqdm import tqdm
import pandas as pd
import numpy as np

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
                dfRF4.drop(dfRF3.loc[dfRF4['feature']!= f].index, inplace=True)
                vals = dfRF4['occurances_pct'].values

                median = round(statistics.median(vals),2)
                # print(m)

                newRow = [m, r, f, median]
                all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["model_num", "ranking", "feature", "median_occurances_pct"])
    return prettydf



if __name__ == "__main__":
    datasets = ["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-'] #feature,sample_size,model_num,smoted
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        path =  "./output/LIME_rankings/lower/" + dataset + ".csv"
        prettydf = rearrange(path)

        print(prettydf.head())
        prettydf.to_csv("./output/LIME_rankings/lower/pretty/" + dataset + ".csv", index = False)
