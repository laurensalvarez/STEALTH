import copy,math, sys, statistics
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def main():
    datasets = ["heart", "diabetes", "communities", "compas", "studentperformance", "bankmarketing", "adultscensusincome", "defaultcredit"]
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
        df = pd.read_csv(r'./output/surro_2/SVM/' + dataset + ".csv")

        df1 = copy.deepcopy(df)

        # df1.drop(df1.loc[df1['smoted']!= 0].index, inplace=True)

        model_num = copy.deepcopy(df1["samples"].tolist())
        sortedmodels = sorted(set(model_num), key = lambda ele: model_num.count(ele))

        recalldict = defaultdict(dict)
        precisiondict = defaultdict(dict)
        accdict = defaultdict(dict)
        F1dict = defaultdict(dict)
        AODdict = defaultdict(dict)
        EODdict = defaultdict(dict)
        SPDdict = defaultdict(dict)
        FA0dict = defaultdict(dict)
        FA1dict = defaultdict(dict)
        DIdict = defaultdict(dict)
        FLIPdict = defaultdict(dict)

        for m in sortedmodels:
            dfRF2 = copy.deepcopy(df1)
            dfRF2.drop(dfRF2.loc[dfRF2['samples']!= m].index, inplace=True)

            features = copy.deepcopy(dfRF2["biased_col"].tolist())
            sortedfeatures = sorted(set(features), key = lambda ele: features.count(ele))

            for f in sortedfeatures:
                # print("Grouping DF-RF by feature", f, " \n")
                dfRF3 = copy.deepcopy(dfRF2)
                dfRF3.drop(dfRF3.loc[dfRF3['biased_col']!= f].index, inplace=True)

                recall = dfRF3 ['recall+']
                precision = dfRF3 ['precision+']
                accuracy = dfRF3 ['accuracy+']
                F1_Score = dfRF3 ['F1+']
                AOD = dfRF3 ['AOD-']
                EOD = dfRF3 ['EOD-']
                SPD = dfRF3 ['SPD-']
                FA0 = dfRF3 ['FA0-']
                FA1 = dfRF3 ['FA1-']
                DI = dfRF3 ['DI-']
                # FLIP = dfRF3 ['flip_rate']

                recalldict[m][f] = recall.values
                precisiondict[m][f] = precision.values
                accdict[m][f] = accuracy.values
                F1dict[m][f] = F1_Score.values
                AODdict[m][f] = AOD.values
                EODdict[m][f] = EOD.values
                SPDdict[m][f] = SPD.values
                FA0dict[m][f] = FA0.values
                FA1dict[m][f] = FA1.values
                DIdict[m][f] = DI.values
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

        reformed_FA0dict = {}
        for outerKey, innerDict in FA0dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA0dict[(outerKey,innerKey)] = values

        reformed_FA1dict = {}
        for outerKey, innerDict in FA1dict.items():
            for innerKey, values in innerDict.items():
                reformed_FA1dict[(outerKey,innerKey)] = values

        reformed_DIdict = {}
        for outerKey, innerDict in DIdict.items():
            for innerKey, values in innerDict.items():
                reformed_DIdict[(outerKey,innerKey)] = values

        # reformed_FLIPdict = {}
        # for outerKey, innerDict in FLIPdict.items():
        #     for innerKey, values in innerDict.items():
        #         reformed_FLIPdict[(outerKey,innerKey)] = values


        recall_df = pd.DataFrame(reformed_recalldict)
        recall_df.columns = ['_'.join(map(str, x)) for x in recall_df.columns]
        recall_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_recall+_.csv", header = None, index=True, sep=' ')

        prec_df = pd.DataFrame(reformed_predict)
        prec_df.columns = ['_'.join(map(str, x)) for x in prec_df.columns]
        prec_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_prec+_.csv", header = None, index=True, sep=' ')

        acc_df = pd.DataFrame(reformed_accdict)
        acc_df.columns = ['_'.join(map(str, x)) for x in acc_df.columns]
        acc_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_acc+_.csv", header = None, index=True, sep=' ')

        F1_df = pd.DataFrame(reformed_F1dict)
        F1_df.columns = ['_'.join(map(str, x)) for x in F1_df.columns]
        F1_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_F1+_.csv", header = None, index=True, sep=' ')

        AOD_df = pd.DataFrame(reformed_AODdict)
        AOD_df.columns = ['_'.join(map(str, x)) for x in AOD_df.columns]
        AOD_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_AOD-_.csv", header = None, index=True, sep=' ')

        EOD_df = pd.DataFrame(reformed_EODdict)
        EOD_df.columns = ['_'.join(map(str, x)) for x in EOD_df.columns]
        EOD_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_EOD-_.csv", header = None, index=True, sep=' ')

        SPD_df = pd.DataFrame(reformed_SPDdict)
        SPD_df.columns = ['_'.join(map(str, x)) for x in SPD_df.columns]
        SPD_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_SPD-_.csv", header = None, index=True, sep=' ')

        FA0_df = pd.DataFrame(reformed_FA0dict)
        FA0_df.columns = ['_'.join(map(str, x)) for x in FA0_df.columns]
        FA0_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_FA0-_.csv", header = None, index=True, sep=' ')

        FA1_df = pd.DataFrame(reformed_FA1dict)
        FA1_df.columns = ['_'.join(map(str, x)) for x in FA1_df.columns]
        FA1_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_FA1-_.csv", header = None, index=True, sep=' ')

        DI_df = pd.DataFrame(reformed_DIdict)
        DI_df.columns = ['_'.join(map(str, x)) for x in DI_df.columns]
        DI_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_DI-_.csv", header = None, index=True, sep=' ')

        # flip_df = pd.DataFrame(reformed_FLIPdict)
        # flip_df.columns = ['_'.join(map(str, x)) for x in flip_df.columns]
        # flip_df.transpose().to_csv("./sk_data/surro_2/SVM/" + dataset + "_flip_rate_.csv", header = None, index=True, sep=' ')


if __name__ == '__main__':
    main()
