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
                dfRF4.drop(dfRF3.loc[dfRF4['feature']!= f].index, inplace=True)
                vals = dfRF4['occurances_pct'].values

                median = round(statistics.median(vals),2)
                # print(m)

                newRow = [m, r, f, median]
                all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["model_num", "ranking", "feature", "median_occurances_pct"])
    return prettydf

def addModel(path):
    df = pd.read_csv(path)
    df1 = copy.deepcopy(df)
    all = []

    for r in range(10):
        r+=1

        nums = []
        mods = []
        alts = []
        df2 = copy.deepcopy(df1)
        df2.drop(df2.loc[df2['rep']!= r].index, inplace=True)

        df3 = copy.deepcopy(df2)
        # df3.drop(df3.loc[df3['ranking']!= ranking].index, inplace=True)

        models = df3['model_num'].values
        counts = list(chain.from_iterable([(k, len(list(v)))] for k, v in groupby(models)))

        c = defaultdict(count)
        mods = [tuple[0] for tuple in counts]
        multiples = [tuple[1] for tuple in counts]

        nums = [next(c[n]) for n in mods]

        for i in range(len(nums)):
            alts.extend([nums[i]] * int(multiples[i]))


        # print("alts",alts)


    df3["treatment"] = alts

    # dfRF2 = dfRF2.groupby([dfRF2['model_num'].ne(dfRF2['model_num'].shift()).cumsum(), 'model_num']).size().to_frame('size').reset_index()
    # df3['Counts'] = df3.groupby(['model_num'])['ranking'].transform('count')

    # print(df3.head(10))
    # print(df3["model_num"].values)
    # print(df3["ounts"].values)

    #     ranking +=1
    #     df3 = copy.deepcopy(dfRF2)
    #     df3.drop(df3.loc[df3['ranking']!= ranking].index, inplace=True)
    #     c = defaultdict(count)
    #     model_nums = [next(c[n]) for n in df3['model_num'].values]
    #
    #     print(df3['model_num'].values)
    #     print(model_nums)
    # for learners in ['RF_RF', 'LSR_RF', 'SVC_RF', 'Slack_RF']:

    treats = copy.deepcopy(df3["treatment"].tolist())
    sortedtreats = sorted(set(treats), key = lambda ele: treats.count(ele))
    sortedtreats = sorted(sortedtreats)


    for m in sortedtreats:
        print(sortedtreats)
        dfRF2 = copy.deepcopy(df3)
        dfRF2.drop(dfRF2.loc[dfRF2['treatment']!= m].index, inplace=True)

        learners = copy.deepcopy(dfRF2["model_num"].tolist())
        sortedlearners= sorted(set(learners), key = lambda ele: learners.count(ele))
        sortedlearners = sorted(sortedlearners)
        for l in sortedlearners:
            dfRF5 = copy.deepcopy(dfRF2)
            dfRF5.drop(dfRF5.loc[dfRF5['model_num']!= l].index, inplace=True)

            for rank in [1,2,3]:
                dfRF3 = copy.deepcopy(dfRF5)
                dfRF3.drop(dfRF3.loc[dfRF3['ranking']!= rank].index, inplace=True)

                features = copy.deepcopy(dfRF3["feature"].tolist())
                sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

                for f in sortedfeatures:
                    dfRF4 = copy.deepcopy(dfRF3)
                    dfRF4.drop(dfRF3.loc[dfRF4['feature']!= f].index, inplace=True)
                    vals = dfRF4['occurances_pct'].values

                    median = round(statistics.median(vals),2)
                    # print(m)

                    newRow = [m, l, rank, f, median]
                    all.append(newRow)

    prettydf = pd.DataFrame(all, columns = ["treatment", "model_num", "ranking", "feature", "median_occurances_pct"])
    return prettydf



if __name__ == "__main__":
    datasets = ["heart", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "defaultcredit", "adultscensusincome"]
# "germancredit"
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'FA0-', 'FA1-','MSE-', 'AOD-', 'EOD-', 'SPD-',  'DI-']
    #LIME COls  ["ranking","feature", "occurances_pct","model_num", "rep"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        path =  "./output/features/LIME/" + dataset + "._LIME.csv"
        prettydf = addModel(path)

        # print(prettydf.head())
        prettydf.to_csv("./LIME_rankings/features/" + dataset + ".csv", index = False)
