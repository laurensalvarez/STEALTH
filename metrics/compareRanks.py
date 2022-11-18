import warnings
warnings.filterwarnings('ignore')
import sys, random, statistics, math, copy, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import count, groupby, chain
from collections import defaultdict
from cliffs_delta import cliffs_delta #https://pypi.org/project/cliffs-delta/

from sk import jaccard_similarity, cliffsDelta, bootstrap
# sys.path.append(os.path.abspath('.'))

def compareLIMERanks(path, dataset, k):
    df = pd.read_csv(path)
    df1 = copy.deepcopy(df)

    rows = []
    reslist, cDlist, blist, jlist, = [], [], [], []
    names = ['RF', "Slack"] 

    # if dataset in ["compas", "bankmarketing", "defaultcredit", "adultscensusincome"]:
    #     treatment = range(7)
    # else:
    #     treatment = range(4)

    # df1 = df1[~df1['learner'].isin(['Slack', 'Slack_RF'])]
    # df1["learner"].replace(["RF_s", "RF_m","RF_x" ],["RFs", "RFm", "RFx"], inplace = True)

    samples = copy.deepcopy(df1["samples"].tolist())
    sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
    sortedsamples.sort(reverse = True)


    for n in names:
        df2 = copy.deepcopy(df1)
        df2.drop(df2.loc[df2['treatment']!= 100].index, inplace=True)
        df2.drop(df2.loc[df2['learner']!= n].index, inplace=True)
        df2.drop(df2.loc[df2['ranking']!= 1].index, inplace=True)
        df2.drop(df2.loc[df2['biased_col']!= k].index, inplace=True)

        # print(df2.head(10))

        full_rank_1 = df2['feature'].values

        df3 = copy.deepcopy(df1)
        df3.drop(df3.loc[df3['learner']!= n].index, inplace=True)
        for t in sortedsamples[1:]:
            r = []
            df4 = copy.deepcopy(df3)
            df4.drop(df4.loc[df4['samples']!= t].index, inplace=True)
            df4.drop(df4.loc[df4['ranking']!= 1].index, inplace=True)

            # print(df4.head(10))

            surro_rank_1 = df4['feature'].values

            # print("full_rank_1", full_rank_1, "\n \n surro_rank_1", surro_rank_1)

            # cdelta, res = cliffs_delta(full_rank_1,surro_rank_1)

            r.append(t)
            r.append(n)
            r.append(k)
            r.append(jaccard_similarity(full_rank_1,surro_rank_1))
            # r.append(res)
            # r.append(round(cdelta,2))
            # r.append("same" if bootstrap(full_rank_1,surro_rank_1) else 'different')
            # r.append(cliffsDelta(full_rank_1,surro_rank_1))

            rows.append(r)
    prettydf = pd.DataFrame(rows, columns = ["samples", "learner", "biased_col", "jacc"])
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

def compareFair(distilledDF, baselineDF, metrics,keyword):
    distilleddf1 = copy.deepcopy(distilledDF)
    basedf1 = copy.deepcopy(baselineDF)
    rows =[]
    distilleddf1.drop(distilleddf1.loc[~distilleddf1['biased_col'].isin([keyword]) ].index, inplace=True)
    basedf1.drop(basedf1.loc[~basedf1['biased_col'].isin([keyword]) ].index, inplace=True)
    # print("distilleddf1", distilleddf1.head())

    #compare the samples to each other 
    # the comparison is always to surrogates scores 
    # then the second is the same samples but from Fair-SMOTE & maat'
    distilleddf1.drop(distilleddf1.loc[distilleddf1['learner'].isin(["Slack", "Slack_RF"]) ].index, inplace=True)
    # print("distilleddf1", distilleddf1.head())

    samples = copy.deepcopy(distilleddf1["samples"].tolist())
    sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
    sortedsamples = sorted(sortedsamples, reverse = True)

    base_learners = copy.deepcopy(basedf1["learner"].tolist())
    sortedlearners = sorted(set(base_learners), key = lambda ele: base_learners.count(ele))
 
    for s in sortedsamples:
        distilleddf2 = copy.deepcopy(distilleddf1)
        basedf2 = copy.deepcopy(basedf1)
        distilleddf2.drop(distilleddf2.loc[distilleddf2['samples']!= s].index, inplace=True)
        basedf2.drop(basedf2.loc[basedf2['samples']!= s].index, inplace=True)
        # print("distilleddf2", distilleddf2.head())
        # print("basedf2", basedf2.head())

        for l in sortedlearners:
            
            basedf3 = copy.deepcopy(basedf2)
            basedf3.drop(basedf3.loc[basedf3['learner']!= l].index, inplace=True)
            # print("basedf3", basedf3.head())

            for m in metrics:
                r = []
                # print("distilleddf2", distilleddf2.head())
                distilled_vals = distilleddf2[m].values
                base_vals = basedf3[m].values

                # print("learner", l ,"distilled_vals", distilled_vals, len(distilled_vals), "\n \n base_vals", base_vals, len(base_vals))
            
                cdelta, res = cliffs_delta(distilled_vals,base_vals)
                r.append(l)
                r.append(m)
                r.append(s)
                r.append(jaccard_similarity(distilled_vals,base_vals))
                r.append(res)
                r.append(round(cdelta,2))
                r.append("same" if bootstrap(distilled_vals,base_vals) else 'different')
                # r.append(cliffsDelta(distilled_vals,base_vals))

                rows.append(r)
    prettydf = pd.DataFrame(rows, columns = ["learner", "metric", "samples", "jacc", "CD_res", "CD", "BS"])
    # print(prettydf.head(10))

    return prettydf


if __name__ == "__main__":
    datasets =  ['communities', 'heart', 'diabetes',  'german', 'student', 'meps', 'compas', 'bank', 'default', 'adult']
    keywords = {'adult': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bank': ['Age('],
                'communities': ['Racepctwhite('],
                'default': ['SEX('],
                'diabetes': ['Age('],
                'german': ['sex('],
                'heart': ['Age('],
                'student': ['sex('],
                'meps': ['race(']
                }
    metrics = ['rec+','prec+','acc+', 'F1+', 'FA0-', 'FA1-', 'MCC-', 'MSE-', 'AOD-', 'EOD-', 'SPD-',  'DI-']
    pbar = tqdm(datasets)
    columns = ["order", "dataset","learner","biased_col", "samples", "jacc"]
    datasetdf = pd.DataFrame(columns=columns)
    order = 0

    for dataset in pbar:
        klist = keywords[dataset]
        order += 1
        pbar.set_description("Processing %s" % dataset)
        for k in klist:
            jacdf = compareLIMERanks("./LIME_rankings/final/" + dataset + "_LIME.csv", dataset, k)
            # distilledpath = "./output/final/" + dataset + "_metrics.csv"
            # distilledDF = pd.read_csv(distilledpath)
            # basepath = "./final/maat/" + dataset + ".csv"
            # baseDF = pd.read_csv(basepath)
            # jacdf = compareFair(distilledDF,baseDF, metrics,k)

            jacdf['dataset'] = dataset
            jacdf['order'] = order
            datasetdf = pd.concat([datasetdf, jacdf], ignore_index=True)

        # print(prettydf.head())
    datasetdf.to_csv("./LIME_rankings/final/all_jac.csv", index = False)
