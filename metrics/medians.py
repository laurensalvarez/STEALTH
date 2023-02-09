import copy,statistics
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def getMedians(mdf,allmetrics):
    # mdf = pd.read_csv(path)
    row = []

    mdf1 = copy.deepcopy(mdf)

    model_num = copy.deepcopy(mdf1["learner"].tolist())
    sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

    for m in sortedmodels:
        mdf2 = copy.deepcopy(mdf1)
        mdf2.drop(mdf2.loc[mdf2['learner']!= m].index, inplace=True)

        samples = copy.deepcopy(mdf2["samples"].tolist())
        sortedsamples = sorted(set(samples), key = lambda ele: samples.count(ele))
        
        for t in [0,1,2,5,10,100]:
            mdf32 = copy.deepcopy(mdf2)
            mdf32.drop(mdf32.loc[mdf32["treatment"]!= t].index, inplace=True)

            for s in sortedsamples:
                mdf3 = copy.deepcopy(mdf32)
                mdf3.drop(mdf3.loc[mdf3["samples"]!= s].index, inplace=True)

                features = copy.deepcopy(mdf3["biased_col"].tolist())
                sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

                for f in sortedfeatures:
                    mdf4 = copy.deepcopy(mdf3)
                    mdf4.drop(mdf4.loc[mdf4['biased_col']!= f].index, inplace=True)
                    r = [round(statistics.median(mdf4[col].values),2) for col in allmetrics]
                    r.append(f)
                    r.append(t)
                    r.append(int(s))
                    r.append(m)
                    row.append(r)

    cols = allmetrics + ["biased_col", "treatment", "samples", "learner"]
    mediandf = pd.DataFrame(row, columns = cols)

    return mediandf

def twinsies(maxdict, stdvdict, val, col, hmetrics):
    high = maxdict[col] + stdvdict[col]
    low = maxdict[col] - stdvdict[col]
    newval = None

    if col in hmetrics:
        if (low <= val and val <=high) or val >= high:
            newval = str(val) + "Y"
        else:
            newval = str(val) + "N"
    else:
        if low >= val or (low <=val and val <= high):
            newval = str(val) + "Y"
        else:
            newval = str(val) + "N"

    return newval


def printStdv(df, hmetrics, lmetrics, allmetrics, mediandf):
    # df = pd.read_csv(path)
    row = []
    valss = []

    df1 = copy.deepcopy(df)
    mediandf = mediandf[~mediandf['learner'].isin(['Slack'])]
    df1 = df1[~df1['learner'].isin(['Slack'])]

    features = copy.deepcopy(df1["biased_col"].tolist())
    sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))
    sortedfeatures = sorted(sortedfeatures)

    for f in sortedfeatures:
        df4 = copy.deepcopy(df1)
        df4.drop(df4.loc[df4['biased_col']!= f].index, inplace=True)

        stddf = round(df4.std()*0.35,2)
        stdd = stddf.to_dict()
        del stdd['rep']
        del stdd["treatment"]
        # del stdd['treatment']
        # del stdd['runtime']
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
            sortedsamples.sort(reverse = True)
            df3 = copy.deepcopy(df2)
            df3.drop(df3.loc[df3["samples"] != sortedsamples[0]].index, inplace=True)

            for col in allmetrics:
                fulldict[col] = df3[col].values[0]
            for s in sortedsamples[1:]:
                r = []
                df5 = copy.deepcopy(df2)
                df5.drop(df5.loc[df5["samples"]!= s].index, inplace=True)

                for col in allmetrics:
                    val = df5[col].values[0]
                    newval= twinsies(fulldict, stdd, val, col, hmetrics)
                    r.append(newval)
                r.append(f)
                r.append(int(s))
                r.append(m)
                row.append(r)

    cols = allmetrics + ["biased_col","samples", "learner"]
    stdvdf = pd.DataFrame(row, columns = cols)
    return stdvdf, mediandf

if __name__ == "__main__":
    datasets =  [ "diabetes",  "german", "student", 'meps', "compas", "bank", "default", "adult"] #
    allmetrics = ['runtime', 'rec+', 'prec+', 'acc+', 'F1+','FA0-', 'FA1-', 'MCC-', 'MSE-', 'AOD-', 'EOD-', 'SPD-', 'DI-', 'Flip'] #
    hmetrics = ['rec+', 'prec+', 'acc+', 'F1+', 'Flip']
    lmetrics = ['FA0-', 'FA1-', 'MCC-', 'MSE-', 'AOD-', 'EOD-', 'SPD-', 'DI-'] #'MCC-',
    pbar = tqdm(datasets)

    columns = ['order','dataset',"learner","biased_col","treatment", 'samples', 'runtime', 'rec+', 'prec+', 'acc+', 'F1+', 'FA0-', 'FA1-','MCC-','MSE-', 'AOD-', 'EOD-', 'SPD-', 'DI-', 'Flip'] #,'MCC-'
    fulldf = pd.DataFrame(columns=columns)
    datasetdf = pd.DataFrame(columns=columns)
    meddatasetdf = pd.DataFrame(columns=columns)
    medfulldf = pd.DataFrame(columns=columns)
    order = 0

    for dataset in pbar:
        order += 1
        pbar.set_description("Processing %s" % dataset)
        path = "./paper/" + dataset + "_metrics1.csv"
        df = pd.read_csv(path)
        
        for v in range(20):
            v += 1
            path = "./paper/" + dataset + "_metrics" + str(v) + ".csv"
            df2 = pd.read_csv(path)
            df = pd.concat([df, df2], ignore_index=True)
 
        mediandf = getMedians(df, allmetrics)
        
 
        mediandf['dataset'] = dataset
        mediandf['order'] = order
        meddatasetdf = pd.concat([meddatasetdf, mediandf], ignore_index=True)

    medfulldf = meddatasetdf[columns]
    medfulldf.to_csv("./output/medians.csv", index = False)
    