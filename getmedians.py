import copy
import math
import sys, random, statistics
from tqdm import tqdm
import pandas as pd
import numpy as np





def getMedians(path,metrics):
    df = pd.read_csv(path)
    row = []

    df1 = copy.deepcopy(df)


    # df1.drop(df1.loc[df1['smoted']!= 1].index, inplace=True)

    model_num = copy.deepcopy(df1["samples"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['samples']!= m].index, inplace=True)

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
    cols = metrics + ["biased_col", "samples"]
    mediandf = pd.DataFrame(row, columns = cols)
    # print(mediandf)

    return mediandf





if __name__ == "__main__":
    datasets = ["heart",  "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "adultscensusincome", "defaultcredit"]
    # "germancredit",
    metrics = ['recall+', 'precision+', 'accuracy+', 'F1+', 'AOD-', 'EOD-', 'SPD-', 'DI-', 'FA0-', 'FA1-'] #feature,sample_size,model_num,smoted
    pbar = tqdm(datasets)

    columns = ['dataset','recall+', 'precision+', 'accuracy+', 'F1+','AOD-', 'EOD-', 'SPD-', 'DI-', 'FA0-', 'FA1-', 'biased_col','samples']
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        # mediandf = pd.DataFrame(columns=columns)

        path =  "./output/surro_2/SVM/" + dataset + ".csv"
        mediandf = getMedians(path, metrics)
        print(mediandf)

        # mediandf.to_csv("./medians/surro/" + dataset + "_medians.csv", index = False)
        mediandf['dataset'] = dataset
        datasetdf = pd.concat([datasetdf, mediandf], ignore_index=True)
    # print(datasetdf)

    fulldf = datasetdf[columns]
    fulldf.to_csv("./medians/surro/allmedians_SVM.csv", index = False)
