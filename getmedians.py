import copy
import math
import sys, random, statistics
from tqdm import tqdm
import pandas as pd
import numpy as np





def getDiscritizedMedians(path,metrics):
    df = pd.read_csv(path)
    row = []

    df1 = copy.deepcopy(df)
    # to_bin = pd.Series(df1["samples"].tolist())
    # a = pd.qcut(to_bin.rank(method = 'first'), 5, labels = False, retbins = True)
    # # print(a)
    # df1["d_samples"] = a[0]

    model_num = copy.deepcopy(df1["d_samples"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['d_samples']!= m].index, inplace=True)

        features = copy.deepcopy(dfRF2["biased_col"].tolist())
        sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

        for f in sortedfeatures:
            dfRF3 = copy.deepcopy(dfRF2)
            dfRF3.drop(dfRF3.loc[dfRF3['biased_col']!= f].index, inplace=True)
            r = [round(statistics.median(dfRF3[col].values),2) for col in metrics]
            r.append(f)
            r.append(m)
            row.append(r)
            # print(m)
    cols = metrics + ["biased_col", "d_samples"]
    mediandf = pd.DataFrame(row, columns = cols)
    # print(mediandf)

    return mediandf

def getMedians(path,metrics):
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
    print(df1.head)

    model_num = copy.deepcopy(df1["learner"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['learner']!= m].index, inplace=True)

        samples = copy.deepcopy(dfRF2["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))

        for s in sortedsamples:
            dfRF3 = copy.deepcopy(dfRF2)
            dfRF3.drop(dfRF3.loc[dfRF3['samples']!= s].index, inplace=True)

            features = copy.deepcopy(dfRF3["biased_col"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                dfRF4 = copy.deepcopy(dfRF3)
                dfRF4.drop(dfRF4.loc[dfRF4['biased_col']!= f].index, inplace=True)
                r = [round(statistics.median(dfRF4[col].values),2) for col in metrics]
                r.append(f)
                r.append(s)
                r.append(m)
                row.append(r)

    cols = metrics + ["biased_col", "samples", "learner"]
    mediandf = pd.DataFrame(row, columns = cols)
    print(mediandf)

    return mediandf


if __name__ == "__main__":
    datasets = ["communities", "heart" , "diabetes", "compas", "studentperformance", "bankmarketing", "defaultcredit", "adultscensusincome"]
    # , "germancredit"
    metrics = ['recall+', 'precision+', 'accuracy+', 'F1+','MSE-', 'FA0-', 'FA1-', 'AOD-', 'EOD-', 'SPD-', 'DI-', 'Flip'] #feature,sample_size,model_num,smoted
    pbar = tqdm(datasets)

    columns = ['dataset','recall+', 'precision+', 'accuracy+', 'F1+', 'MSE-', 'FA0-', 'FA1-','AOD-', 'EOD-', 'SPD-', 'DI-', 'Flip','biased_col','samples', "learner"]
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        # mediandf = pd.DataFrame(columns=columns)

        path =  "./output/features/SMOTE/" + dataset + "_FM.csv"
        mediandf = getMedians(path, metrics)
        # print(mediandf)

        # mediandf.to_csv("./medians/surro/" + dataset + "_medians.csv", index = False)
        mediandf['dataset'] = dataset
        datasetdf = pd.concat([datasetdf, mediandf], ignore_index=True)
    # print(datasetdf)

    fulldf = datasetdf[columns]
    fulldf.to_csv("./medians/features/smoted/allmedians.csv", index = False)
