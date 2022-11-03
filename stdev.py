import copy
import math
import sys, random, statistics
from tqdm import tqdm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def getSmotedMedians(path,metrics):
    df = pd.read_csv(path)
    row = []

    df1 = copy.deepcopy(df)
    # print(df1.head)

    flip_vals = df1['smoted'].values
    smote_vals = df1['Flip'].values

    df1.drop(['Flip'], axis=1, inplace=True)
    df1.drop(['smoted'], axis=1, inplace=True)

    df1['smoted'] = smote_vals
    df1['Flip'] = flip_vals
    # print(df1.head)

    df1.drop(df1.loc[df1['smoted']!= 1].index, inplace=True)
    # print(df1.head)

    model_num = copy.deepcopy(df1["learner"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        df2.drop(df2.loc[df2['learner']!= m].index, inplace=True)

        samples = copy.deepcopy(df2["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        for s in sortedsamples:
            df3 = copy.deepcopy(df2)
            df3.drop(df3.loc[df3['samples']!= s].index, inplace=True)

            features = copy.deepcopy(df3["biased_col"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                df4 = copy.deepcopy(df3)
                df4.drop(df4.loc[df4['biased_col']!= f].index, inplace=True)
                r = [round(statistics.median(df4[col].values),2) for col in metrics]
                r.append(f)
                r.append(s)
                r.append(m)
                row.append(r)

    cols = metrics + ["biased_col", "samples", "learner"]
    mediandf = pd.DataFrame(row, columns = cols)
    # print(mediandf)

    return mediandf

def getMedians(path,metrics):
    mdf = pd.read_csv(path)
    row = []

    mdf1 = copy.deepcopy(mdf)
    # print(df1.head)

    model_num = copy.deepcopy(mdf1["learner"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        mdf2 = copy.deepcopy(mdf1)
        mdf2.drop(mdf2.loc[mdf2['learner']!= m].index, inplace=True)

        samples = copy.deepcopy(mdf2["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        for s in sortedsamples:
            mdf3 = copy.deepcopy(mdf2)
            mdf3.drop(mdf3.loc[mdf3['samples']!= s].index, inplace=True)

            features = copy.deepcopy(mdf3["biased_col"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                mdf4 = copy.deepcopy(mdf3)
                mdf4.drop(mdf4.loc[mdf4['biased_col']!= f].index, inplace=True)
                r = [round(statistics.median(mdf4[col].values),2) for col in metrics]
                r.append(f)
                r.append(s)
                r.append(m)
                row.append(r)

    cols = metrics + ["biased_col", "samples", "learner"]
    mediandf = pd.DataFrame(row, columns = cols)
    # print(mediandf)

    return mediandf

def twinsies(maxdict, stdvdict, val, col):
    high = maxdict[col] + stdvdict[col]
    low = maxdict[col] - stdvdict[col]

    if low <= val and val <= high:
        newval = str(val) + "Y"
    else:
        newval = str(val) + "N"
    return newval


def printStdv(path, metrics, mediandf):
    df = pd.read_csv(path)
    row = []
    valss = []

    df1 = copy.deepcopy(df)
    # print(df1.head)
    mediandf = mediandf[~mediandf['learner'].isin(['Slack', 'Slack_RF'])]
    mediandf["learner"].replace(["RF_RF", "SVC_RF","LSR_RF" ],["RF", "SVC", "LSR"], inplace = True)

    df1 = df1[~df1['learner'].isin(['Slack', 'Slack_RF'])]

    features = copy.deepcopy(df1["biased_col"].tolist())
    sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))
    sortedfeatures = sorted(sortedfeatures)

    for f in sortedfeatures:
        df4 = copy.deepcopy(df1)
        df4.drop(df4.loc[df4['biased_col']!= f].index, inplace=True)
        stddf = round(df4.std()*0.35,2)
        stdd = stddf.to_dict()
        del stdd['rep']
        del stdd['samples']
        print("stdev dict:", stdd)

        model_num = copy.deepcopy(mediandf["learner"].tolist())
        sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))
        sortedmodels = sorted(sortedmodels)

        for m in sortedmodels:
            fulldict = {}
            df2 = copy.deepcopy(mediandf)
            df2.drop(df2.loc[df2['learner']!= m].index, inplace=True)

            samples = copy.deepcopy(df2["samples"].tolist())
            sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
            sortedsamples = sorted(sortedsamples, reverse = True)
            # print("sortedsamples", sortedsamples)
            # fulllearner = sortedsamples[0]



            df3 = copy.deepcopy(df2)
            df3.drop(df3.loc[df3['samples'] != sortedsamples[0]].index, inplace=True)

            for col in metrics:
                fulldict[col] = df3[col].values[0]
            # print("sortedsamples", sortedsamples)
            # print("fulldict:", fulldict)
            for s in sortedsamples:
                r = []
                df5 = copy.deepcopy(df2)
                df5.drop(df5.loc[df5['samples']!= s].index, inplace=True)

                for col in metrics:
                    val = df5[col].values[0]
                    newval = twinsies(fulldict, stdd, val, col)
                    r.append(newval)
                r.append(s)
                r.append(m)
                r.append(f)
                row.append(r)

    cols = metrics + ["samples", "learner", "biased_col"]
    stdvdf = pd.DataFrame(row, columns = cols)
    return stdvdf

if __name__ == "__main__":
    datasets = ["communities", "heart", "diabetes", "compas", "studentperformance", "bankmarketing", "defaultcredit", "adultscensusincome"]
    # , "germancredit"
    metrics = ['recall+', 'precision+', 'accuracy+', 'F1+','FA0-', 'FA1-', 'MSE-', 'AOD-', 'EOD-', 'SPD-', 'DI-']
    pbar = tqdm(datasets)

    columns = ['order','dataset', 'biased_col','learner', "samples", 'recall+', 'precision+', 'accuracy+', 'F1+', 'FA0-', 'FA1-','MSE-', 'AOD-', 'EOD-', 'SPD-', 'DI-']
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)
    order = 0

    for dataset in pbar:
        order += 1

        pbar.set_description("Processing %s" % dataset)
        # mediandf = pd.DataFrame(columns=columns)

        path =  "./output/features/LIME/" + dataset + "_FM.csv"
        mediandf = getMedians(path, metrics)
        stdvdf = printStdv(path, metrics, mediandf)

        # mediandf.to_csv("./medians/surro/" + dataset + "_medians.csv", index = False)
        stdvdf['dataset'] = dataset
        stdvdf['order'] = order
        datasetdf = pd.concat([datasetdf, stdvdf], ignore_index=True)
    # print(datasetdf)

    fulldf = datasetdf[columns]
    fulldf.to_csv("./stdv/allmedians.csv", index = False)
