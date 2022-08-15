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
np.random.seed(params.seed)


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

def Fair_Smote(df, base_clf, scaler, keyword, rep, yname, m, s):
    dataset_orig = df.dropna()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

    # acc, pre, recall, f1 = [], [], [], []
    # aod1, eod1, spd1, di1 = [], [], [], []
    # fr=[]
    # model =[]
    # protected_attribute_list =[]
    # samples = []
    # smoted = []
    og_res = []
    res1 = []
    protected_attribute = keyword


    for i in range(rep):
        print("Round", (i + 1), "Model", str(m), "Feature", keyword, "started.")
        start = time.time()
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=i)
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != yname], dataset_orig_train[yname]
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != yname], dataset_orig_test[yname]

        clf4 = base_clf
        clf4.fit(X_train, y_train)
        y_pred4 = clf4.predict(X_test)

        fr = 0
        print("Round", (i + 1), "finished.")
        acc = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', yname)
        pre = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'precision', yname)
        recall = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'recall', yname)
        f1 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'F1', yname)
        aod1 =measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'aod', yname)
        eod1 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'eod', yname)
        spd1 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'SPD',yname)
        FA0 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'FA0',yname)
        FA1 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'FA1',yname)
        di1 = measure_final_score(dataset_orig_test, y_pred4, X_train, y_train, X_test, y_test, protected_attribute, 'DI',yname)
        smoted = 0
        og_res = ([recall, acc, pre, f1, aod1, eod1, spd1, di1, FA0, FA1, fr, protected_attribute, s, m, smoted])

        zero_zero_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        # print("class distribution:","\n0 0: ", zero_zero_zero, "\n0 1: ",zero_one_zero, "\n1 0: ", one_zero_zero,"\n1 1: ",one_one_zero)
        maximum = max(zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero)
        zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
        zero_one_zero_to_be_incresed = maximum - zero_one_zero
        one_zero_zero_to_be_incresed = maximum - one_zero_zero
        one_one_zero_to_be_incresed = maximum - one_one_zero
        df_zero_zero_zero = dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 0)]

        df_zero_one_zero = dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 1)]

        df_one_zero_zero = dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 0)]

        df_one_one_zero = dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 1)]

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

        df.columns = dataset_orig.columns
        clf2 = base_clf
        clf2.fit(X_train, y_train)
        X_train2, y_train2 = df.loc[:, df.columns != yname], df[yname]
        print("Situational testing...")
        X_train_sitch, y_train_sitch = situation(clf2, X_train2, y_train2, protected_attribute)
        # X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != yname], dataset_orig_test[yname]

        clf3 = base_clf
        clf3.fit(X_train_sitch, y_train_sitch)
        y_pred3 = clf3.predict(X_test)

        fr = calculate_flip(clf3, X_test, protected_attribute)
        print("Round", (i + 1), "finished.")
        acc = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'accuracy', yname)
        pre = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'precision', yname)
        recall = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'recall', yname)
        f1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'F1', yname)
        aod1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'aod', yname)
        eod1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'eod', yname)
        spd1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'SPD',yname)
        FA0 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'FA0',yname)
        FA1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'FA1',yname)
        di1 = measure_final_score(dataset_orig_test, y_pred3, X_train_sitch, y_train_sitch, X_test, y_test, protected_attribute, 'DI',yname)
        smoted = 1
        print('Time', time.time() - start)
    res1.append(og_res)
    res1.append([recall, acc, pre, f1, aod1, eod1, spd1, di1, FA0, FA1, fr, protected_attribute, s, m, smoted])
    return res1

if __name__ == "__main__":
    datasets = ["compas"]#["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    keywords = {'adultscensusincome': ['race(', 'sex('],
                'compas': ['race(','sex('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['sex('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['age('],
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
            result0 = Fair_Smote(df1, base, scaler, keyword, 2, yname, 0, len(df1.index))
            # feature_list[k] = results0
            results_dict[0] = result0

            surrogatedf = pd.read_csv("./output/cluster_preds/class_bal/"+ dataset + ".csv")
            surrogate_cat_features = []
            y_pred = "predicted"
            y_pred_vals = surrogatedf["predicted"].values

            surrogatedf.drop([yname,y_pred,'samples','Unrelated_column_one'],axis=1, inplace=True)
            surrogatedf[yname] = y_pred_vals

            models = copy.deepcopy(surrogatedf["model_num"].tolist())
            sortedmodels = sorted(set(models), key = lambda ele: models.count(ele))

            surrogate_1 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_5 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_7 = pd.DataFrame(copy.deepcopy(surrogatedf.values),columns = surrogatedf.columns)

            surrogate_1.drop(surrogatedf.loc[surrogatedf['model_num']!= sortedmodels[0]].index, inplace=True)
            surrogate_5.drop(surrogatedf.loc[surrogatedf['model_num']!= sortedmodels[1]].index, inplace=True)
            surrogate_7.drop(surrogatedf.loc[surrogatedf['model_num']!= sortedmodels[2]].index, inplace=True)

            # print(surrogate_1.head())
            # print(surrogate_5.head())
            # print(surrogate_7.head())

            result1 = Fair_Smote(surrogate_1, base, scaler, keyword, 2, yname, 1, len(surrogate_1.index))
            results_dict[1] = result1
            result5 = Fair_Smote(surrogate_5, base, scaler, keyword, 2, yname, 5, len(surrogate_5.index))
            results_dict[5] = result5
            result7 = Fair_Smote(surrogate_7, base, scaler, keyword, 2, yname, 7, len(surrogate_7.index))
            results_dict[7] = result7

            pprint.pprint(results_dict)

            for s,c in results_dict.items():
                for metric_row in c:
                    # print(metric_row)
                    rows.append(metric_row)

        final_df = pd.DataFrame(rows,columns = ['recall+', 'precision+', 'accuracy+', 'F1_Score+', 'AOD-', 'EOD-', 'SPD-', 'FA0-', 'FA1-', 'DI-', 'flip_rate', 'feature', 'sample_size', 'model_num', 'smoted'])
        final_df.to_csv("./bias/" +  dataset + "_RF.csv", index=False)




            # k+=1








            # a, p, r, f, ao, eo, spd, di,fr = result1
            # print("**"*50)
            # print(fname, keyword)
            # print("+Accuracy", np.mean(a))
            # print("+Precision", np.mean(p))
            # print("+Recall", np.mean(r))
            # print("+F1", np.mean(f))
            # print("-AOD", np.mean(ao))
            # print("-EOD", np.mean(eo))
            # print("-SPD", np.mean(spd))
            # print("-DI", np.mean(di))
            # print("-FR", np.mean(fr))
