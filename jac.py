import warnings
warnings.filterwarnings('ignore')
import sys, random, statistics, math, copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import count, groupby, chain
from collections import defaultdict
from cliffs_delta import cliffs_delta #https://pypi.org/project/cliffs-delta/

from sk import jaccard_similarity, cliffsDelta, bootstrap


def compareLIMERanks(path, dataset):
    df = pd.read_csv(path)
    df1 = copy.deepcopy(df)

    rows = []
    reslist, cDlist, blist, jlist, = [], [], [], []
    full_names = ['RF', "Slack"] #"LSR", ]
    surro_names = ['RF_RF', 'Slack_RF'] #'LSR_RF',

    # if dataset in ["compas", "bankmarketing", "defaultcredit", "adultscensusincome"]:
    #     treatment = range(7)
    # else:
    #     treatment = range(4)
    treatment = [2,3,4,5]

    samples = copy.deepcopy(df1["samples"].tolist())
    sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
    sortedsamples = sorted(sortedsamples, reverse = True)

    for n in range(2):

        df2 = copy.deepcopy(df1)
        df2.drop(df2.loc[df2['learner']!= full_names[n]].index, inplace=True)
        df2.drop(df2.loc[df2['samples']!= sortedsamples[0]].index, inplace=True)
        df2.drop(df2.loc[df2['ranking']!= 1].index, inplace=True)

        # print(df2.head(10))

        full_rank_1 = df2['feature'].values

        df3 = copy.deepcopy(df1)
        df3.drop(df3.loc[df3['learner']!= surro_names[n]].index, inplace=True)
        for t in sortedsamples[1:]:
            r = []
            df4 = copy.deepcopy(df3)
            df4.drop(df4.loc[df4['samples']!= t].index, inplace=True)
            df4.drop(df4.loc[df4['ranking']!= 1].index, inplace=True)

            # print(df4.head(10))

            surro_rank_1 = df4['feature'].values

            print("full_rank_1", full_rank_1, "\n \n surro_rank_1", surro_rank_1)

            # cdelta, res = cliffs_delta(full_rank_1,surro_rank_1)

            r.append(t)
            r.append(surro_names[n])
            r.append(jaccard_similarity(full_rank_1,surro_rank_1))
            r.append(res)
            r.append(round(cdelta,2))
            # r.append("same" if bootstrap(full_rank_1,surro_rank_1) else 'different')
            # r.append(cliffsDelta(full_rank_1,surro_rank_1))

            rows.append(r)
    prettydf = pd.DataFrame(rows, columns = ["samples", "learner", "jacc", "CD_res", "CD"])
    print(prettydf.head(10))
    return prettydf

def compareRFRanks(path, dataset):
    df = pd.read_csv(path)
    df1 = copy.deepcopy(df)

    rows = []
    reslist, cDlist, blist, jlist, = [], [], [], []
    full_names = ['RF']
    surro_names = ['RF_RF', 'Slack_RF']#, 'LSR_RF', 'SVC_RF']

    # learners = copy.deepcopy(df1["learners"].tolist())
    # sortedlearners = sorted(set(learners), key = lambda ele: learners.count(ele))
    # sortedsamples = sorted(sortedlearners, reverse = True)

    samples = copy.deepcopy(df1["samples"].tolist())
    sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
    sortedsamples = sorted(sortedsamples, reverse = True)

    # imp_perc = copy.deepcopy(df1["median_importance"].tolist())
    # sortedperc = sorted(set(imp_perc), key = lambda ele: imp_perc.count(ele))
    # sortedperc = sorted(sortedperc, reverse = True)
    # cut_off = round(statistics.median(sortedperc),2)
    # print("cutoff", cut_off, "\n")
    for s in sortedsamples[1:]:
        df2 = copy.deepcopy(df1)
        df2.drop(df2.loc[df2['samples']!= sortedsamples[0]].index, inplace=True)
        df2.drop(df2.loc[df2['learner']!= full_names[0]].index, inplace=True)

        full_perc = copy.deepcopy(df2["median_importance"].tolist())
        sorted_fullperc = sorted(set(full_perc), key = lambda ele: full_perc.count(ele))
        sorted_fullperc = sorted(sorted_fullperc, reverse = True)

        df2.drop(df2.loc[df2['median_importance'] <= sorted_fullperc[2]].index, inplace=True)

        full_rank_1 = df2['feature'].values

        for n in range(len(surro_names)):
            r = []
            df3 = copy.deepcopy(df1)
            df3.drop(df3.loc[df3['learner']!= surro_names[n]].index, inplace=True)
            df3.drop(df3.loc[df3['samples']!= s].index, inplace=True)

            surro_perc = copy.deepcopy(df3["median_importance"].tolist())
            sorted_surroperc = sorted(set(surro_perc), key = lambda ele: surro_perc.count(ele))
            sorted_surroperc = sorted(sorted_surroperc, reverse = True)

            df3.drop(df3.loc[df3['median_importance'] <= sorted_surroperc[3]].index, inplace=True)
            surro_rank_1 = df3['feature'].values

            # print("full_rank_1", full_rank_1, "\n \n surro_rank_1", surro_rank_1, "\n \n ______________________")

            cdelta, res = cliffs_delta(full_rank_1,surro_rank_1)
            r.append(s)
            r.append(surro_names[n])
            r.append(jaccard_similarity(full_rank_1,surro_rank_1))
            r.append(res)
            r.append(round(cdelta,2))
            # r.append("same" if bootstrap(full_rank_1,surro_rank_1) else 'different')
            # r.append(cliffsDelta(full_rank_1,surro_rank_1))

            rows.append(r)
    prettydf = pd.DataFrame(rows, columns = ["samples", "learner", "jacc", "CD_res", "CD"])
    print(prettydf.head(10))
    return prettydf


if __name__ == "__main__":
    datasets = ["communities","heart", "diabetes", "studentperformance", "compas", "bankmarketing", "defaultcredit"]#, "adultscensusincome"]
# "germancredit"
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'FA0-', 'FA1-','MSE-', 'AOD-', 'EOD-', 'SPD-',  'DI-']
    #LIME COls  ["ranking","feature", "occurances_pct","model_num", "rep"]
    pbar = tqdm(datasets)
    columns = ["order", "dataset", "samples", "learner", "jacc", "CD_res", "CD"]
    datasetdf = pd.DataFrame(columns=columns)
    order = 0

    for dataset in pbar:
        order += 1
        pbar.set_description("Processing %s" % dataset)

        path =  "./LIME_rankings/final/" + dataset + ".csv"
        jacdf = compareLIMERanks(path, dataset)

        jacdf['dataset'] = dataset
        jacdf['order'] = order
        datasetdf = pd.concat([datasetdf, jacdf], ignore_index=True)

        # print(prettydf.head())
    datasetdf.to_csv("./LIME_rankings/final/all.csv", index = False)
