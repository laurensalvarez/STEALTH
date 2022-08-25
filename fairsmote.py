import warnings
warnings.filterwarnings('ignore')
import random,time,csv, math,copy,os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
# sys.path.append(os.path.abspath('..'))
from Measure import measure_final_score
from Generate_Samples import generate_samples

from utils import *
params = Params("./model_configurations/experiment_params.json")
seed = np.random.seed(params.seed)


def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out

def flip(X_test,keyword):
    X_flip = X_test.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    return X_flip

def calculate_flip(clf,X_test,keyword):
    X_flip = flip(X_test,keyword)
    a = np.array(clf.predict(X_test))
    b = np.array(clf.predict(X_flip))
    total = X_test.shape[0]
    same = np.count_nonzero(a==b)
    return round((total-same)/total,2)

def situation(clf,X_train,y_train,keyword):
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    same = [1 if each else 0 for each in same]
    X_train['same'] = same
    X_train['y'] = y_train
    X_rest = X_train[X_train['same']==1]
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','y'])
    return X_rest,y_rest

def classBal(ds_train, yname, protected_attribute):
    zero_zero_zero = len(ds_train[(ds_train[yname] == 0) & (ds_train[protected_attribute] == 0)])
    zero_one_zero = len(ds_train[(ds_train[yname] == 0) & (ds_train[protected_attribute] == 1)])
    one_zero_zero = len(ds_train[(ds_train[yname] == 1) & (ds_train[protected_attribute] == 0)])
    one_one_zero = len(ds_train[(ds_train[yname] == 1) & (ds_train[protected_attribute] == 1)])

    print("class distribution:","\n0 0: ", zero_zero_zero, "\n0 1: ",zero_one_zero, "\n1 0: ", one_zero_zero,"\n1 1: ",one_one_zero)
    maximum = max(zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero)
    zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
    zero_one_zero_to_be_incresed = maximum - zero_one_zero
    one_zero_zero_to_be_incresed = maximum - one_zero_zero
    one_one_zero_to_be_incresed = maximum - one_one_zero

    df_zero_zero_zero = ds_train[(ds_train[yname] == 0) & (ds_train[protected_attribute] == 0)]

    df_zero_one_zero = ds_train[(ds_train[yname] == 0) & (ds_train[protected_attribute] == 1)]

    df_one_zero_zero = ds_train[(ds_train[yname] == 1) & (ds_train[protected_attribute] == 0)]

    df_one_one_zero = ds_train[(ds_train[yname] == 1) & (ds_train[protected_attribute] == 1)]

    df_zero_zero_zero[protected_attribute] = df_zero_zero_zero[protected_attribute].astype(str)
    df_zero_one_zero[protected_attribute] = df_zero_one_zero[protected_attribute].astype(str)
    df_one_zero_zero[protected_attribute] = df_one_zero_zero[protected_attribute].astype(str)
    df_one_one_zero[protected_attribute] = df_one_one_zero[protected_attribute].astype(str)

    print("Start generating samples...")
    df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed, df_zero_zero_zero, '')
    df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed, df_zero_one_zero, '')
    df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed, df_one_zero_zero, '')
    df_one_one_zero = generate_samples(one_one_zero_to_be_incresed, df_one_one_zero, '')
    df = pd.concat([df_zero_zero_zero, df_zero_one_zero,df_one_zero_zero, df_one_one_zero])

    return df

def Fair_Smote(df1, base_clf, scaler, keyword, rep, yname, m, size, X_test1, y_test1): #remove rep
    ds1 = df1.dropna()
    ds = pd.DataFrame(scaler.fit_transform(ds1), columns=ds1.columns)  #dtype = 'int64'
    og_res = []
    res1 = []
    protected_attribute = keyword
    # X_test, y_test = X_test, y_test


        # start = time.time()
    for i in range(rep):
        print("Round", (i + 1), "Model", str(m), "Feature", keyword, "started.")

        if X_test1.empty and y_test1.empty:
            print("EMPTY\n \n")
            ds_train, ds_test1 = train_test_split(ds, test_size=0.2)
            X_train, y_train = ds_train.loc[:, ds_train.columns != yname], ds_train[yname]
            X_test, y_test = ds_test1.loc[:, ds_test1.columns != yname], ds_test1[yname]
        else:
            print("NOT empty \n \n")
            ds_train = ds.copy()
            # print("PRE TRAINING ds_train:", ds_train.shape, ds_train.columns )
            X_train, y_train = ds_train.loc[:, ds_train.columns != yname], ds_train[yname]
            # ds_train.drop([yname], axis=1, inplace=True)
            X_test, y_test = X_test1, y_test1

        cols = [col for col in ds_train.columns]
        if 'model_num' in cols:
            ds_train.drop(['model_num'], axis=1, inplace=True)
            X_train.drop(['model_num'], axis=1, inplace=True)

        # print("PRE TRAINING TRAIN:", X_train.shape, X_train.columns )
        # # # # print(y_train.values, len(y_train.values))
        # print("PRE TRAINING TEST:", X_test.shape, X_test.columns )
        # print("PRE TRAINING ds_train:", ds_train.shape, ds_train.columns )
        # print(y_test.values, len(y_test.values))
        # print(y_train.shape)
        # print(y_test.head())

        clf4 = base_clf
        clf4.fit(X_train, y_train)
        y_pred4 = clf4.predict(X_test)
        # print("y_pred", len(y_pred4))
        ds_test = pd.DataFrame(copy.deepcopy(X_test.values), columns = X_test.columns)
        # print("PRE TRAINING ds_test:", ds_test.shape, ds_test.columns )
        # ds_test[yname] = y_test


        fr = 0
        print("Round", (i + 1), "finished.")
        acc = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', yname)
        pre = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'precision', yname)
        recall = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'recall', yname)
        f1 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'F1', yname)
        aod1 =measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'aod', yname)
        eod1 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'eod', yname)
        spd1 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'SPD',yname)
        FA0 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'FA0',yname)
        FA1 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'FA1',yname)
        di1 = measure_final_score(ds_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'DI',yname)
        smoted = 0
        og_res = ([recall, acc, pre, f1, aod1, eod1, spd1, di1, FA0, FA1, fr, protected_attribute, size, m, smoted])
        res1.append(og_res)

        df = classBal(ds_train, yname, protected_attribute)
        # print(df.head())

        df.columns = ds_train.columns
        clf2 = base_clf
        clf2.fit(X_train, y_train)
        X_train2, y_train2 = df.loc[:, df.columns != yname], df[yname]
        print("Situational testing...")
        X_train_sitch, y_train_sitch = situation(clf2, X_train2, y_train2, protected_attribute)
        # X_test, y_test = ds_test.loc[:, ds_test.columns != yname], ds_test[yname]

        clf3 = base_clf
        clf3.fit(X_train_sitch, y_train_sitch)
        y_pred3 = clf3.predict(X_test)

        fr = calculate_flip(clf3, X_test, protected_attribute)
        print("Round", (i + 1), "finished.")
        acc = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'accuracy', yname)
        pre = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'precision', yname)
        recall = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'recall', yname)
        f1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'F1', yname)
        aod1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'aod', yname)
        eod1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'eod', yname)
        spd1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'SPD',yname)
        FA0 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'FA0',yname)
        FA1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'FA1',yname)
        di1 = measure_final_score(ds_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'DI',yname)
        smoted = 1
        # print('Time', time.time() - start)

        res1.append([recall, acc, pre, f1, aod1, eod1, spd1, di1, FA0, FA1, fr, protected_attribute, size, m, smoted])
    return res1, X_test, y_test

if __name__ == "__main__":
    datasets = ["defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    # datasets = ["adultscensusincome","bankmarketing", "communities", "compas", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
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
    base = RandomForestClassifier()
    scaler = MinMaxScaler()
    results_dict={}
    rows = []
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        klist = keywords[dataset]
        # k = 0
        for keyword in klist:
            df1 = pd.read_csv("./datasets/no_cats/"+ dataset + ".csv")
            y_s = [col for col in df1.columns if "!" in col]
            yname = y_s[0]
            df1.drop(['Unrelated_column_one'], axis=1, inplace = True)
            result0, X_test, y_test = Fair_Smote(df1, base, scaler, keyword, 10, yname, 0, len(df1.index), pd.DataFrame(), pd.DataFrame())
            # print (X_test, y_test)
            results_dict[0] = result0

            surrogatedf = pd.read_csv("./output/cluster_preds/class_bal/"+ dataset + ".csv")
            surrogate_cat_features = []
            y_pred = "predicted"
            y_pred_vals = surrogatedf[y_pred].values
            # for i in y_pred_vals:
            #     print(i)
            # print("ORIGINAL:", surrogatedf.head())

            # sys.exit()

            surrogatedf.drop([yname,y_pred,'samples','fold','Unrelated_column_one'],axis=1, inplace=True)
            # print("ORIGINAL dropped:", surrogatedf.head())
            surrogatedf[yname] = y_pred_vals

            surrogate_1 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_5 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_7 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)

            models = copy.deepcopy(surrogatedf["model_num"].tolist())
            sortedmodels = sorted(set(models), key = lambda ele: models.count(ele))
            # print(surrogate_1.head())

            surrogate_1.drop(surrogatedf.loc[surrogatedf['model_num'] != 1].index, inplace=True)
            # print("ONE:", surrogate_1.head())
            surrogate_5.drop(surrogatedf.loc[surrogatedf['model_num'] != 5].index, inplace=True)
            surrogate_7.drop(surrogatedf.loc[surrogatedf['model_num'] != 7].index, inplace=True)

            print(len(surrogate_1.index))
            # print(surrogate_5.head())
            # print(surrogate_7.head())

            # # result1, _, _ = Fair_Smote(surrogate_1, base, scaler, keyword, 10, yname, 1, len(surrogate_1.index), X_test, y_test)
            # results_dict[1] = result1
            result5, _, _ = Fair_Smote(surrogate_5, base, scaler, keyword, 10, yname, 5, len(surrogate_5.index), X_test, y_test)
            results_dict[5] = result5
            result7, _, _ = Fair_Smote(surrogate_7, base, scaler, keyword, 10, yname, 7, len(surrogate_7.index), X_test, y_test)
            results_dict[7] = result7

            # pprint.pprint(results_dict)

            for s,c in results_dict.items():
                for metric_row in c:
                    # print(metric_row)
                    rows.append(metric_row)

        final_df = pd.DataFrame(rows,columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'flip_rate', 'feature', 'sample_size', 'model_num', 'smoted'])
        final_df.to_csv("./bias/" +  dataset + "_RF.csv", index=False)
