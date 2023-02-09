import warnings
warnings.filterwarnings('ignore')
import copy
import pandas as pd
from tqdm import tqdm

from metrics.sk import jaccard_similarity

def compareLIMERanks(path, dataset, k):
    df = pd.read_csv(path)
    df1 = copy.deepcopy(df)

    rows = []
    names = ['RF', "Slack"] 

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



if __name__ == "__main__":
    datasets =  ['communities', 'heart', 'diabetes',  'german', 'student', 'meps', 'compas', 'bank', 'default','adult']
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
    metrics = ['rec+','prec+','acc+', 'F1+', 'FA0-', 'FA1-', 'AOD-', 'EOD-', 'SPD-',  'DI-'] #'MCC-', 'MSE-',
    pbar = tqdm(datasets)
    columns = ["order", "dataset","learner","biased_col", "samples", "jacc"]
    datasetdf = pd.DataFrame(columns=columns)
    order = 0

    for dataset in pbar:
        klist = keywords[dataset]
        order += 1
        pbar.set_description("Processing %s" % dataset)
        for k in klist:
            jacdf = compareLIMERanks("./output/" + dataset + "_LIME.csv", dataset, k)
            jacdf['dataset'] = dataset
            jacdf['order'] = order
            datasetdf = pd.concat([datasetdf, jacdf], ignore_index=True)

        # print(prettydf.head())
    datasetdf.to_csv("./output/LIME_jaccard.csv", index = False)
