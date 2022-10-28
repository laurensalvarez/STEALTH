import copy,math, sys, statistics, pprint
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def main():
    datasets = ["heart" , "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "defaultcredit", "adultscensusincome"]
# "germancredit",
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['SEX('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['Age('],
                'studentperformance': ['sex(']
                }
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        df = pd.read_csv(r'./output/features/LIME/' + dataset + "_FM.csv")

        df1 = copy.deepcopy(df)
        # to_bin = pd.Series(df1["samples"].tolist())
        # a = pd.qcut(to_bin.rank(method = 'first'), 5, labels = False, retbins = True)
        # df1["d_samples"] = a[0]
        #


        # df1.drop(df1.loc[df1['smoted']!= 0].index, inplace=True)

        model_num = copy.deepcopy(df1["samples"].tolist())
        sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

        recalldict = defaultdict(dict)
        precisiondict = defaultdict(dict)
        accdict = defaultdict(dict)
        F1dict = defaultdict(dict)
        FA0dict = defaultdict(dict)
        FA1dict = defaultdict(dict)
        MSEdict = defaultdict(dict)
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

            # features = copy.deepcopy(dfRF2["biased_col"].tolist())
            # sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

                for f in sortedfeatures:
                    # print("Grouping DF-RF by feature", f, " \n")
                    dfRF4 = copy.deepcopy(dfRF3)
                    dfRF4.drop(dfRF3.loc[dfRF4['biased_col']!= f].index, inplace=True)

                    recall = dfRF4 ['recall+']
                    precision = dfRF4 ['precision+']
                    accuracy = dfRF4 ['accuracy+']
                    F1_Score = dfRF4 ['F1+']
                    FA0 = dfRF4 ['FA0-']
                    FA1 = dfRF4 ['FA1-']
                    MSE = dfRF4 ['MSE-']
                    AOD = dfRF4 ['AOD-']
                    EOD = dfRF4 ['EOD-']
                    SPD = dfRF4 ['SPD-']
                    DI = dfRF4 ['DI-']
                    # FLIP = dfRF3 ['flip_rate']

                    l = l.split('_')[0]

                    recalldict[str(m) + "_" + l][f] = recall.values
                    precisiondict[str(m) + "_" + l][f] = precision.values
                    accdict[str(m) + "_" + l][f] = accuracy.values
                    F1dict[str(m) + "_" + l][f] = F1_Score.values
                    FA0dict[str(m) + "_" + l][f] = FA0.values
                    FA1dict[str(m) + "_" + l][f] = FA1.values
                    MSEdict[str(m) + "_" + l][f] = MSE.values
                    AODdict[str(m) + "_" + l][f] = AOD.values
                    EODdict[str(m) + "_" + l][f] = EOD.values
                    SPDdict[str(m) + "_" + l][f] = SPD.values
                    DIdict[str(m) + "_" + l][f] = DI.values
                    # FLIPdict[m][f] = FLIP.values

        reformed_recalldict = {}
        for outerKey, innerDict in recalldict.items():
            for innerKey, values in innerDict.items():
                reformed_recalldict[(outerKey,innerKey)] = values

        reformed_predict = {}
        for outerKey, innerDict in precisiondict.items():
            for innerKey, values in innerDict.items():
                reformed_predict[(outerKey,innerKey)] = values

        reformed_accdict = {}
        for outerKey, innerDict in accdict.items():
            for innerKey, values in innerDict.items():
                reformed_accdict[(outerKey,innerKey)] = values

        reformed_F1dict = {}
        for outerKey, innerDict in F1dict.items():
            for innerKey, values in innerDict.items():
                reformed_F1dict[(outerKey,innerKey)] = values

        reformed_FA0dict = {}
        for outerKey, innerDict in FA0dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA0dict[(outerKey,innerKey)] = values

        reformed_FA1dict = {}
        for outerKey, innerDict in FA1dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA1dict[(outerKey,innerKey)] = values

        reformed_MSEdict = {}
        for outerKey, innerDict in MSEdict.items():
            for innerKey, values in innerDict.items():
                reformed_MSEdict[(outerKey,innerKey)] = values

        reformed_AODdict = {}
        for outerKey, innerDict in AODdict.items():
            for innerKey, values in innerDict.items():
                reformed_AODdict[(outerKey,innerKey)] = values

        reformed_EODdict = {}
        for outerKey, innerDict in EODdict.items():
            for innerKey, values in innerDict.items():
                reformed_EODdict[(outerKey,innerKey)] = values

        reformed_SPDdict = {}
        for outerKey, innerDict in SPDdict.items():
            for innerKey, values in innerDict.items():
                reformed_SPDdict[(outerKey,innerKey)] = values

        reformed_DIdict = {}
        for outerKey, innerDict in DIdict.items():
            for innerKey, values in innerDict.items():
                reformed_DIdict[(outerKey,innerKey)] = values

        recall_df.columns = ['_'.join(map(str, x)) for x in recall_df.columns]
        recall_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_recall+_.csv", header = None, index=True, sep=' ')

        prec_df = pd.DataFrame(reformed_predict)
        prec_df.columns = ['_'.join(map(str, x)) for x in prec_df.columns]
        prec_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_prec+_.csv", header = None, index=True, sep=' ')

        acc_df = pd.DataFrame(reformed_accdict)
        acc_df.columns = ['_'.join(map(str, x)) for x in acc_df.columns]
        acc_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_acc+_.csv", header = None, index=True, sep=' ')

        F1_df = pd.DataFrame(reformed_F1dict)
        F1_df.columns = ['_'.join(map(str, x)) for x in F1_df.columns]
        F1_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_F1+_.csv", header = None, index=True, sep=' ')

        FA0_df = pd.DataFrame(reformed_FA0dict)
        FA0_df.columns = ['_'.join(map(str, x)) for x in FA0_df.columns]
        FA0_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_FA0-_.csv", header = None, index=True, sep=' ')

        FA1_df = pd.DataFrame(reformed_FA1dict)
        FA1_df.columns = ['_'.join(map(str, x)) for x in FA1_df.columns]
        FA1_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_FA1-_.csv", header = None, index=True, sep=' ')

        MSE_df = pd.DataFrame(reformed_MSEdict)
        MSE_df.columns = ['_'.join(map(str, x)) for x in MSE_df.columns]
        MSE_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_MSE-_.csv", header = None, index=True, sep=' ')

        AOD_df = pd.DataFrame(reformed_AODdict)
        AOD_df.columns = ['_'.join(map(str, x)) for x in AOD_df.columns]
        AOD_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_AOD-_.csv", header = None, index=True, sep=' ')

        EOD_df = pd.DataFrame(reformed_EODdict)
        EOD_df.columns = ['_'.join(map(str, x)) for x in EOD_df.columns]
        EOD_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_EOD-_.csv", header = None, index=True, sep=' ')

        SPD_df = pd.DataFrame(reformed_SPDdict)
        SPD_df.columns = ['_'.join(map(str, x)) for x in SPD_df.columns]
        SPD_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_SPD-_.csv", header = None, index=True, sep=' ')

        DI_df = pd.DataFrame(reformed_DIdict)
        DI_df.columns = ['_'.join(map(str, x)) for x in DI_df.columns]
        DI_df.transpose().to_csv("./sk_data/features/" + dataset + model + "_DI-_.csv", header = None, index=True, sep=' ')



if __name__ == '__main__':
    main()
