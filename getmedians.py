import copy
import math
import sys, random, statistics
from tqdm import tqdm
import pandas as pd
import numpy as np





def getMedians(path, mediandf):
    df = pd.read_csv(path)

    df1 = copy.deepcopy(df)

    df1.drop(df1.loc[df1['smoted']!= 1].index, inplace=True)

    model_num = copy.deepcopy(df1["model_num"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        dfRF2 = copy.deepcopy(df1)
        dfRF2.drop(dfRF2.loc[dfRF2['model_num']!= m].index, inplace=True)

        features = copy.deepcopy(dfRF2["feature"].tolist())
        sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

        for f in sortedfeatures:
            dfRF3 = copy.deepcopy(dfRF2)
            dfRF3.drop(dfRF3.loc[dfRF3['feature']!= f].index, inplace=True)
            m = [statistics.median(dfRF3[col].values) for col in dfRF3.columns]
            print(m)


    return mediandf





if __name__ == "__main__":
    datasets = ["bankmarketing"] #, "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'flip_rate'] #feature,sample_size,model_num,smoted
    pbar = tqdm(datasets)

    columns = ['dataset','recall+', 'prec+', 'acc+', 'F1+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'flip_rate', 'feature','sample_size','model_num','smoted']
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        mediandf = pd.DataFrame(columns=columns)

        path =  "./bias/" + dataset + "_RF.csv"
        mediandf = getMedians(path, mediandf)

        mediandf.to_csv("./medians/OG/" + dataset + "_medians.csv", index = False)
        mediandf['dataset'] = dataset
        datasetdf = pd.concat([datasetdf, mediandf], ignore_index=True)
    print(datasetdf)

    fulldf = datasetdf[columns]
    fulldf.to_csv("./medians/OG/allmedians.csv", index = False)
