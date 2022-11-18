import copy
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def main():
    datasets =  ["communities", "heart", "diabetes",  "german", "student", 'meps',"compas",   "bank", "default", "adult"]
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
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        df = pd.read_csv('./final/' + dataset + "_metrics.csv")
        klist = keywords[dataset]

        df1 = copy.deepcopy(df)
        # df1 = df1[~df1['learner'].isin(['Slack'])]

        model_num = copy.deepcopy(df1["samples"].tolist())
        sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))
        sortedmodels.sort(reverse = True)  
        max_samples =  sortedmodels[0]

        recalldict = defaultdict(dict)
        precisiondict = defaultdict(dict)
        accdict = defaultdict(dict)
        F1dict = defaultdict(dict)
        FA0dict = defaultdict(dict)
        FA1dict = defaultdict(dict)
        MSEdict = defaultdict(dict)
        MCCdict = defaultdict(dict)
        AODdict = defaultdict(dict)
        EODdict = defaultdict(dict)
        SPDdict = defaultdict(dict)
        DIdict = defaultdict(dict)

        for m in sortedmodels:
            dfRF2 = copy.deepcopy(df1)
            dfRF2.drop(dfRF2.loc[dfRF2['samples']!= m].index, inplace=True)

            learner = copy.deepcopy(dfRF2["learner"].tolist())
            sortedlearner = sorted(set(learner), key = lambda ele: learner.count(ele))


            for l in sortedlearner:
                dfRF3 = copy.deepcopy(dfRF2)
                dfRF3.drop(dfRF2.loc[dfRF3['learner']!= l].index, inplace=True)

                features = copy.deepcopy(dfRF3["biased_col"].tolist())
                sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

                for f in sortedfeatures:
                    # print("Grouping DF-RF by feature", f, " \n")
                    dfRF4 = copy.deepcopy(dfRF3)
                    dfRF4.drop(dfRF3.loc[dfRF4['biased_col']!= f].index, inplace=True)

                    recall = dfRF4 ['rec+']
                    precision = dfRF4 ['prec+']
                    accuracy = dfRF4 ['acc+']
                    F1_Score = dfRF4 ['F1+']
                    FA0 = dfRF4 ['FA0-']
                    FA1 = dfRF4 ['FA1-']
                    MSE = dfRF4 ['MSE-']
                    MCC = dfRF4['MCC-']
                    AOD = dfRF4 ['AOD-']
                    EOD = dfRF4 ['EOD-']
                    SPD = dfRF4 ['SPD-']
                    DI = dfRF4 ['DI-']
                    FLIP = dfRF3 ['flip_rate']

                    recalldict[str(m) + "_" + l + "_" + f] = recall.values
                    precisiondict[str(m) + "_" + l + "_" + f] = precision.values
                    accdict[str(m) + "_" + l + "_" + f] = accuracy.values
                    F1dict[str(m) + "_" + l + "_" + f] = F1_Score.values
                    FA0dict[str(m) + "_" + l + "_" + f] = FA0.values
                    FA1dict[str(m) + "_" + l + "_" + f] = FA1.values
                    MSEdict[str(m) + "_" + l + "_" + f] = MSE.values
                    MCCdict[str(m) + "_" + l + "_" + f] = MCC.values
                    AODdict[str(m) + "_" + l + "_" + f] = AOD.values
                    EODdict[str(m) + "_" + l + "_" + f] = EOD.values
                    SPDdict[str(m) + "_" + l + "_" + f] = SPD.values
                    DIdict[str(m) + "_" + l + "_" + f] = DI.values
                    FLIPdict[str(m) + "_" + l + "_" + f] = FLIP.values

        dictionaries = [recalldict,precisiondict, accdict,F1dict, FA0dict, FA1dict, MSEdict, MCCdict, AODdict, EODdict, SPDdict, DIdict]
        for k in klist:
            removeThese = [str(max_samples) + "_RFx_" + k, str(max_samples) + "_RFm_" + k, str(max_samples) + "_RFs_" + k]
            print(removeThese)

            for d in dictionaries:
                for r in removeThese:
                    del d[r] 


        recall_df = pd.DataFrame.from_dict(recalldict, orient = 'index')
        recall_df.to_csv("./sk_data/final/" + dataset + "_recall+_.csv", header = None, index=True, sep=' ')
        
        prec_df = pd.DataFrame.from_dict(precisiondict, orient = 'index')
        prec_df.to_csv("./sk_data/final/" + dataset + "_prec+_.csv", header = None, index=True, sep=' ')

        acc_df = pd.DataFrame.from_dict(accdict, orient = 'index')
        acc_df.to_csv("./sk_data/final/" + dataset + "_acc+_.csv", header = None, index=True, sep=' ')

        F1_df = pd.DataFrame.from_dict(F1dict, orient = 'index')
        F1_df.to_csv("./sk_data/final/" + dataset + "_F1+_.csv", header = None, index=True, sep=' ')

        FA0_df = pd.DataFrame.from_dict(FA0dict, orient = 'index')
        FA0_df.to_csv("./sk_data/final/" + dataset + "_FA0-_.csv", header = None, index=True, sep=' ')

        FA1_df = pd.DataFrame.from_dict(FA1dict, orient = 'index')
        FA1_df.to_csv("./sk_data/final/" + dataset + "_FA1-_.csv", header = None, index=True, sep=' ')

        MSE_df = pd.DataFrame.from_dict(MSEdict, orient = 'index')
        MSE_df.to_csv("./sk_data/final/" + dataset + "_MSE-_.csv", header = None, index=True, sep=' ')

        MCC_df = pd.DataFrame.from_dict(MCCdict, orient = 'index')
        MCC_df.to_csv("./sk_data/final/" + dataset + "_MCC-_.csv", header = None, index=True, sep=' ')

        AOD_df = pd.DataFrame.from_dict(AODdict, orient = 'index')
        AOD_df.to_csv("./sk_data/final/" + dataset + "_AOD-_.csv", header = None, index=True, sep=' ')

        EOD_df = pd.DataFrame.from_dict(EODdict, orient = 'index')
        EOD_df.to_csv("./sk_data/final/" + dataset + "_EOD-_.csv", header = None, index=True, sep=' ')

        SPD_df = pd.DataFrame.from_dict(SPDdict, orient = 'index')
        SPD_df.to_csv("./sk_data/final/" + dataset + "_SPD-_.csv", header = None, index=True, sep=' ')

        DI_df = pd.DataFrame.from_dict(DIdict, orient = 'index')
        DI_df.to_csv("./sk_data/final/" + dataset + "_DI-_.csv", header = None, index=True, sep=' ')

        FLIP_df = pd.DataFrame.from_dict(FLIPdict, orient = 'index')
        FLIP_df.to_csv("./sk_data/final/" + dataset + "_Flip_.csv", header = None, index=True, sep=' ')

if __name__ == '__main__':
    main()
