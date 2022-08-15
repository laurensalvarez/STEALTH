import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath('..'))
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
    return (total-same)/total

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

def Fair_Smote(df, base_clf, keyword, rep, yname,m):
    dataset_orig = df.dropna()
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

    acc, pre, recall, f1 = [], [], [], []
    aod1, eod1, spd1, di1 = [], [], [], []
    fr=[]
    model =[]
    protected_attribute_list =[]
    protected_attribute = keyword


    for i in range(rep):
        start = time.time()
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=i)
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != yname], dataset_orig_train[yname]
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != yname], dataset_orig_test[yname]

        clf3 = base_clf



        zero_zero_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 0)])

        zero_one_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 0) & (dataset_orig_train[protected_attribute] == 1)])

        one_zero_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 0)])

        one_one_zero = len(dataset_orig_train[(dataset_orig_train[yname] == 1) & (dataset_orig_train[protected_attribute] == 1)])

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
        X_train, y_train = df.loc[:, df.columns != yname], df[yname]
        print("Situational testing...")
        X_train, y_train = situation(clf2, X_train, y_train, protected_attribute)
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != yname], dataset_orig_test[yname]

        clf = base_clf
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # cm = confusion_matrix(y_test, y_pred)

        flip_rate = calculate_flip(clf, X_test, protected_attribute)
        # print("FLIP rate1:", flip_rate)
        # print("# of flips1:", flip_rate * X_test.shape[0])
        fr.append(flip_rate)
        print("Round", (i + 1), "finished.")
        acc.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', yname)))
        pre.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute 'precision', yname)))
        recall.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall', yname)))
        f1.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1', yname)))
        aod1.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod', yname))
        eod1.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod', yname))
        spd1.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD',yname))
        di1.append(measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI',yname))
        protected_attribute_list.append(protected_attribute)
        model.append(m)
        print('Time', time.time() - start)
    res1 = [acc, pre, recall, f1, aod1, eod1, spd1, di1,fr, protected_attribute_list, model]
    return res1

if __name__ == "__main__":
    filenames = ["compas"]#["adultscensusincome","bankmarketing", "compas", "communities", "defaultcredit", "diabetes",  "germancredit", "heart", "studentperformance"]
    keywords = {'adultscensusincome': ['sex(', 'race('],
                'compas': ['sex(', 'race('],
                'bankmarketing': ['Age('],
                'communities': ['Racepctwhite('],
                'defaultcredit': ['sex('],
                'diabetes': ['Age('],
                'germancredit': ['sex('],
                'heart': ['age('],
                'studentperformance': ['sex(']
                }
    base = RandomForestClassifier()
    results_dict={}
    feature_dict = {}
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            df1 = pd.read_csv("./datasets/no_cats/"+ fname + ".csv")
            y_s = [col for col in df1.columns if "!" in col]
            yname = y_s[0]
            result0 = Fair_Smote(df1, base, keyword, 10, yname,0)
            feature_dict{keyword} = results0
            results_dict{"original"} = feature_dict

            surrogatedf = pd.read_csv("./output/cluster_preds/"+ fname + ".csv")
            surrogate_cat_features = []
            keep = ["samples", "predicted"]
            y_pred = "predicted"
            sample_metrics ={}

            models = copy.deepcopy(surrogatedf["model_num"].tolist())
            sortedmodels = sorted(set(models), key = lambda ele: models.count(ele))

            surrogate_1 = pd.DataFrame(deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_5 = pd.DataFrame(deepcopy(surrogatedf.values),columns = surrogatedf.columns)
            surrogate_7 = pd.DataFrame(deepcopy(surrogatedf.values),columns = surrogatedf.columns)

            surrogate_1.drop(surrogatedf.loc[surrogatedf['sortedmodels']!= m[0]].index, inplace=True)
            surrogate_5.drop(surrogatedf.loc[surrogatedf['sortedmodels']!= m[1]].index, inplace=True)
            surrogate_7.drop(surrogatedf.loc[surrogatedf['sortedmodels']!= m[2]].index, inplace=True)

            result1 = Fair_Smote(surrogate_1, base, keyword, 10, yname,1)
            result5 = Fair_Smote(surrogate_1, base, keyword, 10, yname,5)
            result7 = Fair_Smote(surrogate_1, base, keyword, 10, yname,7)









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
