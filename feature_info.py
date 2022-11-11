import copy,sys, math,statistics
from tqdm import tqdm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def featureInfo(df):
    rows = []
    
    df1 = copy.deepcopy(df)

    num_cols = df1.select_dtypes(include=np.number).columns.tolist()
    
    for col in num_cols:
        row = []
        row.append(col)
        row.append(round(df1[col].min(),2))
        row.append(round(df1[col].max(),2))
        row.append(round(df1[col].mean(),2))
        row.append(round(df1[col].std(),2))
        rows.append(row)
   
    cols = ['numeric_col', 'min', 'max', 'mean', 'std']
    featuredf = pd.DataFrame(rows, columns = cols)
    # print(mediandf)
    return featuredf


if __name__ == "__main__":
    datasets = ["communities","heart", "diabetes", "germancredit", "studentperformance", "meps", "compas", "defaultcredit","bankmarketing", "adultscensusincome"] 
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['SEX('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['Age('],
                'studentperformance': ['sex('],
                'meps': ['race(']
                }

    pbar = tqdm(datasets)

    columns = ['dataset', 'num_rows','numeric_col', 'min', 'max', 'mean', 'std']
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)

        path = "./datasets/processed/" + dataset + "_p.csv"
        df = pd.read_csv(path)
        num_rows = df.shape[0]
        
        featuredf = featureInfo(df)
        featuredf['num_rows'] = num_rows
        featuredf['dataset'] = dataset

        datasetdf = pd.concat([datasetdf, featuredf], ignore_index=True)
    
    fulldf = datasetdf[columns]
    fulldf.to_csv("./stdv/feature_info.csv", index = False)
