import math, re, random, statistics, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cProfile
import collections
from collections import defaultdict
from copy import deepcopy
from itertools import count
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

big = sys.maxsize
tiny = 1 / big

# ------------------------------------------------------------------------------
# Column Class
# ------------------------------------------------------------------------------
class Col:
    def __init__(self,
                 name):  # The __init__ method lets the class initialize the object's attributes and serves no other purpose
        self._id = 0  # underscore means hidden var
        self.name = name  # self is an instance of the class with all of the attributes & methods

    def __add__(self, v):
        return v

    def diversity(self):
        return 0

    def mid(self):
        return 0

    def dist(self, x, y):
        return 0

    @staticmethod
    def id(self):
        self._id += 1
        return self._id

# ------------------------------------------------------------------------------
# Symbolic Column Class
# ------------------------------------------------------------------------------
class Sym(Col):
    def __init__(self, name, uid, data=None):  # will override Col inheritance (could use super() to inherit)
        Col.__init__(self,
                     name)  # invoking the __init__ of the parent class; If you forget to invoke the __init__() of the parent class then its instance variables would not be available to the child class.
        self.n = 0
        self.most = 0
        self.mode = ""
        self.uid = uid  # uid --> it allows for permanence and recalling necessary subtables
        self.count = defaultdict(int)  # will never throw a key error bc it will guve default value as missing key
        # self.encoder = preprocessing.LabelEncoder()
        self.coltype = 0
        self.vals = []
        if data != None:  # initializes the empty col with val
            for val in data:
                self + val  # calls __add__

    # def __str__(self):
    # print to a sym; overrides print(); could replace dump() TBD

    def __add__(self, v):
        return self.add(v, 1)  # need to adds? i forgot why

    def add(self, v, inc=1):  # want to be able to control the increments
        self.n += inc  # add value to the count
        self.vals.append(v)
        self.count[v] += inc  # add value to the dictionary with count +1
        tmp = self.count[v]
        if tmp > self.most:  # check which is the most seen; if it's the most then assign and update mode
            self.most, self.mode = tmp, v  # a,b = b,a
        return v

    def diversity(self):  # entropy of all of n
        # is a measure of the randomness in the information being processed.
        # The higher the entropy, the harder it is to draw any conclusions from
        # that information.
        e = 0
        for k, v in self.count.items():  # iterate through a list of tuples (key, value)
            p = v / self.n
            e -= p * math.log(p) / math.log(2)
        return e

    def mid(self):  # midpoint as mode (which sym appears the most)
        return self.mode

    def dist(self, x, y):  # Aha's distance between two syms
        if (x == "?" or x == "") or (y == "?" or y == ""):  # check if the empty is just a bug
            return 1
        return 0 if x == y else 1

# ------------------------------------------------------------------------------
# Numeric Column Class
# ------------------------------------------------------------------------------
class Num(Col):
    def __init__(self, name, uid, data=None):
        Col.__init__(self, name)
        self.n = 0
        self.mu = 0  #
        self.m2 = 0  # for moving std dev
        self.sd = 0
        self.lo = big  # float('inf')
        self.hi = -big  # -float('inf')
        self.vals = []
        self.uid = uid
        self.count = defaultdict(int)
        self.most = 0
        self.median = 0
        self.coltype = 1
        if data != None:
            for val in data:
                self + val  # calls __add__

    def __add__(self, v):
        # add the column; calculate the new lo/hi, get the sd using the 'chasing the mean'
        self.n += 1
        self.vals.append(v)  # add value to a list
        self.count[v] += 1  # add value to the dictionary with count +1
        tmp = self.count[v]
        if tmp > self.most:  # check which is the most seen; if it's the most then assign and update mode
            self.most, self.mode = tmp, v  # a,b = b,a
        try:
            if v < self.lo:  # if the val is < the lowest; reassign
                self.lo = v
            if v > self.hi:  # if the val is > the highest; reassign
                self.hi = v
            d = v - self.mu  # distance to the mean
            self.mu += d / self.n  # normalize it
            self.m2 += d * (v - self.mu)  # calculate momentumn of the series in realtionship to the distance
            self.sd = self._numSd()  #
            self.median = self.mid()
        except:
            print("failed col name:", self.name, "id:", self.uid)
        return v

    def _numSd(self):
        # Standard deviation is a number that describes how
        # spread out the values are. A low standard deviation
        # means that most of the numbers are close to the mean (average) value.
        # A high standard deviation means that the values are spread out over a wider range.
        if self.m2 < 0:  # means there's is no momentum to the series
            return 0
        if self.n < 2:  # if there's two items
            return 0
        return math.sqrt(self.m2 / (self.n - 1))  # calculate std dev

    def diversity(self):  # return standard dev
        return self.sd

    def mid(self):  # get midpoint for nums (median)
        # NO statistics.median(self.vals)
        listLen = len(self.vals)
        self.vals.sort()

        if listLen == 0:
            self.median = 0
            return self.median

        if listLen % 2 == 0:
            median1 = self.vals[listLen // 2]
            median2 = self.vals[listLen // 2 - 1]
            median = (median1 + median2) / 2
        else:
            median = self.vals[listLen // 2]

        self.median = median

        return self.median
        # returns median

    def dist(self, x, y):  # Aha's distance bw two nums
        if (x == "?" or x == "") and (y == "?" or y == ""):
            return 1
        if (x == "?" or x == "") or (y == "?" or y == ""):
            x = x if (y == "?" or y == "") else y
            x = self._numNorm(x)
            y = 0 if x > 0.5 else 1
            return y - x
        return abs(self._numNorm(x) - self._numNorm(y))

    def _numNorm(self, x):
        "normalize the column."  # Min-Max Normalization + a super tiny num so you never divide by 0
        return (x - self.lo) / (self.hi - self.lo + tiny)

# ------------------------------------------------------------------------------
# Table Class: Reads in CSV and Produces Table of Nums/Syms
# ------------------------------------------------------------------------------
class Table:
    def __init__(self, uid):
        self.uid = uid
        self.count = 0
        self.cols = []
        self.rows = []
        self.encodedrows = []
        self.fileline = 0
        self.linesize = 0
        self.skip = []
        self.y = []
        self.nums = []
        self.syms = []
        self.goals = []
        self.klass = []
        self.protected = []
        self.w = defaultdict(int)
        self.x = []
        self.xnums = []  # num x points (not including goals/klass)
        self.xsyms = []  # sym x points
        self.header = ""
        self.clabels = []

    # ------------------------------------------------------------------------------
    # Table Class: Helper Functions
    # ------------------------------------------------------------------------------
    @staticmethod  # helper function; not to create or instantiate; don't use with instantiated obj
    def compiler(x):  # checks & compiles data type; a python thing
        try:
            int(x)
            return int(x)
        except:
            try:
                float(x)
                return float(x)
            except ValueError:
                return str(x)

    @staticmethod
    def readfile(file, sep=",", doomed=r'([\n\t\r"\' ]|#.*)'):  # reads in file
        datalines = []
        finallines = []

        with open(file) as f:  # ensures that the file will be closed when control leaves the block
            curline = ""
            for line in f:
                line = line.strip()  # get rid of all the white space
                if line[len(line) - 1] == ",":
                    curline += line
                else:
                    curline += line
                    datalines.append(Table.compiler(curline))
                    curline = ""

        for l in datalines:
            line = l.strip()
            line = re.sub(doomed, '', line)  # uses regular exp package to replace substrings in strings
            if line:
                finallines.append([Table.compiler(x) for x in line.split(sep)])
        return finallines  # returns all the pretty readable lines

    # ------------------------------------------------------------------------------
    # Table Class: Class Methods
    # ------------------------------------------------------------------------------
    def __add__(self, line):
        if len(self.header) > 0:  # if line has a header
            self.insert_row(line)  # insert the row
        else:
            self.create_cols(line)  # if not; create a col

    def create_cols(self, line):
        self.header = line  # since add recognized no header, assign first line as a header.
        index = 0

        for val in line:
            val = self.compiler(val)  # compile the val datatype

            if val[0] == ":" or val[
                0] == "?":  # do we skip? if we skip then it doesn't matter what we do? bc it'll never be populated?
                if val[0].isupper():
                    self.skip.append(Num(''.join(c for c in val),
                                         index))  # take all the items in val as long as it's not ?/: ;join()takes all items in an iterable and joins them as a string
                else:
                    self.skip.append(Sym(''.join(c for c in val), index))

            col = None

            if val[0].isupper():  # is it a num?
                col = Num(''.join(c for c in val if not c in ['?', ':']), index)
                self.nums.append(col)
                self.cols.append(col)

            else:  # no, it's a sym
                col = Sym(''.join(c for c in val if not c in ['?', ':']), index)
                self.syms.append(col)
                self.cols.append(col)

            if "!" in val or "-" in val or "+" in val:  # is it a klass, or goal (goals are y)
                self.y.append(col)
                self.goals.append(col)
                if "-" in val:
                    self.w[index] = -1
                if "+" in val:
                    self.w[index] = 1
                if "!" in val:
                    self.klass.append(col)

            if "-" not in val and "+" not in val and "!" not in val:  # then it's an x
                self.x.append(col)
                if val[0].isupper():
                    self.xnums.append(col)
                else:
                    self.xsyms.append(col)

            if "(" in val:  # is it a protected?
                self.protected.append(col)

            index += 1  # increase by one
            self.linesize = index
            self.fileline += 1

    def insert_row(self, line):
        self.fileline += 1
        if len(line) != self.linesize:
            print("len(line)", len(line), "self.linesize", self.linesize)
            print("Line", self.fileline, "has an error")
            return

        if isValid(self, line):
            realline = []
            index = 0
            for val in line:
                if index not in self.skip:  # check if it needs to be skipped
                    if val == "?" or val == "":
                        realline.append(val)  # add to realline
                        index += 1
                        continue
                    self.cols[index] + self.compiler(val)
                    realline.append(val)
                index += 1

            self.rows.append(realline)
            self.count += 1
        # else:
        # print("Line", self.fileline, "has missing values")

    # def encode_lines(self):
    #     # for all Syms
    #     # initialize the LabelEncoder
    #     # fit from all dictionary keys
    #     encodedrows = []
    #     for col in self.cols:
    #         if col.coltype == 0:
    #             keys = list(col.count.keys())
    #             col.encoder.fit(keys)
    #     for line in self.rows:
    #         newline = []
    #         for i, val in enumerate(line):
    #             newval = val
    #             if self.cols[i].coltype == 0:
    #                 newval = self.cols[i].encoder.transform([val])[-1]
    #             else:
    #                 newval = self.compiler(val)
    #             newline.append(newval)
    #         encodedrows.append(newline)
    #     self.encodedrows = encodedrows
        # for all lines, if col of line is Sym encode with le.transform([val])
        # store all encoded lines

    # ------------------------------------------------------------------------------
    # Clustering Fastmap;still in table class (change to it's own class???)
    # ------------------------------------------------------------------------------
    def split(self, top=None):  # Implements continous space Fastmap for bin chop on data
        if top == None:
            top = self
        pivot = random.choice(self.rows)  # pick a random row
        left = top.mostDistant(pivot, self.rows)  # get most distant point from the pivot
        right = top.mostDistant(left, self.rows)  # get most distant point from the leftTable
        c = top.distance(left, right)  # get distance between two points
        items = [[row, 0] for row in self.rows]  # make an array for the row & distance but initialize to 0 to start

        for x in items:
            a = top.distance(x[0], right)  # for each row get the distance between that and the farthest point right
            b = top.distance(x[0], left)  # for each row get the distance between that and the farthest point left
            x[1] = (a ** 2 + c ** 2 - b ** 2) / (
                        2 * c + 10e-32)  # cosine rule for the distance assign to dist in (row, dist)

        items.sort(key=lambda x: x[
            1])  # sort by distance (method sorts the list ascending by default; can have sorting criteria)
        splitpoint = len(items) // 2  # integral divison
        leftItems = [x[0] for x in items[: splitpoint]]  # left are the rows to the splitpoint
        rightItems = [x[0] for x in items[splitpoint:]]  # right are the rows from the splitpoint onward

        return [top, left, right, leftItems, rightItems]

    def distance(self, rowA, rowB):  # distance between two points
        distance = 0
        if len(rowA) != len(rowB):  # catch if they can't be compared?? why??
            return -big
        # for i, (a,b) in enumerate(zip(rowA, rowB)):#to iterate through an interable: an get the index with enumerate(), and get the elements of multiple iterables with zip()
        for col in self.x:  # to include y self.cols ; for just x vals self.x
            i = col.uid
            d = self.cols[i].dist(self.compiler(rowA[i]), self.compiler(
                rowB[i]))  # distance of both rows in each of the columns; compile the a & b bc it's in a text format
            distance += d  # add the distances together
        return distance

    def mostDistant(self, rowA, localRows):  # find the furthest point from row A
        distance = -big
        farthestRow = None  # assign to null; python uses None datatype

        for row in self.rows:
            d = self.distance(rowA, row)  # for each of the rows find the distance to row A
            if d > distance:  # if it's bigger than the distance
                distance = d  # assign the new distance to be d
                farthestRow = row  # make point the far row
        # print("most distant = ", distance, "away and is ", farthestRow[-1])
        return farthestRow  # return the far point/row

    def closestPoint(self, rowA):
        distance = big
        closestRow = None  # assign to null; python uses None datatype
        secondClosest = None

        for row in self.rows:
            d = self.distance(rowA, row)  # for each of the rows find the distance to row A
            if d < distance:  # if it's smaller than the distance
                distance = d  # assign the new distance to be d
                closestRow = row  # make point the close row
        return closestRow  # return the close point/row

    @staticmethod
    def clusters(items, table, enough, top=None, depth=0):
        # print("|.. " * depth,len(table.rows))
        # print("top cluster:", top)
        if len(items) < enough:  # if/while the length of the less than the stopping criteria #should be changable from command line
            leftTable = Table(0)  # make a table w/ uid = 0
            leftTable + table.header  # able the table header to the table ; leftTable.header = table.header?
            for item in items:  # add all the items to the table
                leftTable + item
            return TreeNode(None, None, leftTable, None, table, None, None, True,
                            table.header)  # make a leaf treenode when the cluster have enough rows in them
        # if you don't enough items
        if top != None:
            _, left, right, leftItems, rightItems = table.split(top)
        else:
            top, left, right, leftItems, rightItems = table.split(top)

        leftTable = Table(0)
        leftTable + table.header
        for item in leftItems:
            leftTable + item

        rightTable = Table(0)
        rightTable + table.header
        for item in rightItems:
            rightTable + item
        # print(rightTable.rows)
        leftNode = Table.clusters(leftItems, leftTable, enough, top, depth=depth + 1)
        rightNode = Table.clusters(rightItems, rightTable, enough, top, depth=depth + 1)
        root = TreeNode(left, right, leftTable, rightTable, table, leftNode, rightNode, False, table.header)
        return root

# ------------------------------------------------------------------------------
# Tree class
# ------------------------------------------------------------------------------
class TreeNode:
    _ids = count(0)

    def __init__(self, left, right, leftTable, rightTable, currentTable, leftNode, rightNode, leaf, header):
        self.uid = next(self._ids)
        self.left = left
        self.right = right
        self.leftTable = leftTable
        self.rightTable = rightTable
        self.currentTable = currentTable
        self.leaf = leaf
        self.header = header
        self.leftNode = leftNode
        self.rightNode = rightNode

# ------------------------------------------------------------------------------
# TreeNode Class Helper Fuctions: Functional Tree Traversal
# ------------------------------------------------------------------------------
def nodes(root):  # gets all the leaf nodes
    if root:
        for node in nodes(root.leftNode): yield node  # yield returns from a function without destroying it
        if root.leaf:  yield root
        for node in nodes(root.rightNode): yield node


def names(root: TreeNode):  # gets all the col names of the node
    for node in nodes(root):
        for i in range(len(node.leftTable.cols) - 1):
            print(node.leftTable.cols[i].name)


def rowSize(t): return len(t.leftTable.rows)  # gets the size of the rows


def leafmedians(root, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    MedianTable = Table(222)
    header = root.header
    MedianTable + header
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid = [col.mid() for col in t.cols]
        MedianTable + mid
        # print(len(t.rows), [col.mid() for col in t.cols], t.cols[-1].count)
    # MedianTable.encode_lines()
    return MedianTable

def leafmedians2(root, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    MedianTable = Table(222)
    header = root.header
    MedianTable + header
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid = [float(col.mid()) for col in t.cols]
        MedianTable + mid
        # print(len(t.rows), [col.mid() for col in t.cols], t.cols[-1].count)
    # MedianTable.encode_lines()
    return MedianTable

def getLeafData2(root, samples_per_leaf, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    EDT = Table(samples_per_leaf)
    header = root.header
    EDT + header
    counter = 0
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        for i in range(samples_per_leaf):
            randomrow = random.choice(t.rows)
            EDT + randomrow
            counter += 1
    # EDT.encode_lines()
    return EDT

def getLeafData(root, samples_per_leaf, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    EDT = Table(samples_per_leaf)
    header = root.header
    EDT + header
    counter = 0
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        for i in range(samples_per_leaf):
            randomrow = random.choice(t.rows)
            EDT + randomrow
            counter += 1
    # EDT.encode_lines()
    return EDT

def getLeafMedClass(root, samples_per_leaf,how=None):  # for all of the leaves from smallest to largest get x samples per leaf with median class label
    EDT = Table(samples_per_leaf)
    header = root.header
    EDT + header
    counter = 0
    newy = []
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid =  t.y[-1].mid()
        leafys = [mid for _ in range(samples_per_leaf)]
        newy.extend(leafys)
        for i in range(samples_per_leaf):
            randomrow = random.choice(t.rows)
            EDT + randomrow

    numrows = len(EDT.rows)
    newy = [mid for r in numrows]
    EDT.y = newy
    # EDT.encode_lines()
    return EDT

def getLeafModes(root, samples_per_leaf,how=None):  # for all of the leaves from smallest to largest get x samples per leaf with median class label
    EDT = Table(samples_per_leaf)
    header = root.header
    EDT + header
    counter = 0
    newy = []
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mode =  t.y[-1].mode
        leafys = [mode for _ in range(samples_per_leaf)]
        newy.extend(leafys)
        for i in range(samples_per_leaf):
            randomrow = random.choice(t.rows)
            EDT + randomrow
    EDT.y[-1].vals = newy
    # EDT.encode_lines()
    return EDT

    def dump(self, f):
        # DFS
        if self.leaf:
            f.write("Dump Leaf Node: " + str(self.uid) + "\n")
            f.write("Dump Leaf Table: " + "\n")
            self.leftTable.dump(f)
            return

        if self.leftNode is not None:
            self.leftNode.dump(f)
        if self.rightNode is not None:
            self.rightNode.dump(f)

    def csvDump(self, f):
        # DFS
        if self.leaf:
            self.leftTable.csvDump(f)
            return

        if self.leftNode is not None:
            self.leftNode.csvDump(f)

        if self.rightNode is not None:
            self.rightNode.csvDump(f)


def isValid(self, row):
    for val in row:
        if val == '?':
            return False
    return True


# def getXY(table):
#     X = []
#     y = []
#     y_index = table.y[-1].uid
#
#     for row in table.encodedrows:
#         X_row = []
#         y_row = -1
#         for i, val in enumerate(row):
#             if i == y_index:  # for multiple y if i in y_indexes:
#                 y_row = val
#             else:
#                 X_row.append(val)
#         X.append(X_row)
#         y.append(y_row)
#     return X,y

def getXY2(table):
    X = []
    y = []
    y_index = table.y[-1].uid

    for row in table.rows:
        X_row = []
        y_row = -1
        for i, val in enumerate(row):
            if i == y_index:  # for multiple y if i in y_indexes:
                y_row = val
            else:
                X_row.append(val)
        X.append(X_row)
        y.append(y_row)
    return X,y


def classify(table, df, X_test, y_test, samples, total_pts, f, enough_multiplier):
    # i = 1
    full = []
    X_train, y_train = getXY(table)

    #LR RF SVC
    # clf = LogisticRegression(random_state=0)
    clf = RandomForestClassifier(random_state=0)
    # clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred_list = y_pred.tolist()
    y_test_list = y_test.tolist()
    for x in list(X_test.values):
        full.append(deepcopy(x))
    for j in range(len(y_test_list)):
        full[j] = np.append(full[j],y_test_list[j])
        full[j] = np.append(full[j],y_pred_list[j])
        full[j] = np.append(full[j], samples)
        full[j] = np.append(full[j], total_pts)
        full[j] = np.append(full[j], f)
        full[j] = np.append(full[j], enough_multiplier)
        # full[j] = np.append(full[j], i)
    for row in full:
        a_series = pd.Series(row, index=df.columns)
        df = df.append(a_series, ignore_index=True)
    return df

def fullclassify(df, X_train, y_train, X_test, y_test, samples, total_pts, f, enough_multiplier):
    # i = 1
    full = []
    # X_train, y_train = getXY(table)

    #LR RF SVC
    clf = LogisticRegression(random_state=0)
    # clf = RandomForestClassifier(random_state=0)
    # clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    y_pred_list = y_pred.tolist()
    y_test_list = y_test.tolist()
    for x in list(X_test.values):
        full.append(deepcopy(x))
    for j in range(len(y_test_list)):
        full[j] = np.append(full[j],y_test_list[j])
        full[j] = np.append(full[j],y_pred_list[j])
        full[j] = np.append(full[j], samples)
        full[j] = np.append(full[j], total_pts)
        full[j] = np.append(full[j], f)
        full[j] = np.append(full[j], enough_multiplier)
        # full[j] = np.append(full[j], i)
    for row in full:
        a_series = pd.Series(row, index=df.columns)
        df = df.append(a_series, ignore_index=True)
    return df

# -----------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def getTable(csv, limiter = None):
    dataset = csv
    filename = dataset[:-4]  # cut off the .csv
    lines = Table.readfile(r'./datasets/' + dataset)
    table = Table(1)
    table + lines[0]

    lines.pop(0)
    random.shuffle(lines)

    if limiter != None:
        for l in lines[1:limiter]:
            table + l
    else:
        for l in lines[1:]:
            table + l

    return table, filename

def clusterandclassify(table, filename):
    # table.encode_lines()
    y_index = table.y[-1].name
    tcolumns = deepcopy(table.header)
    # trows = deepcopy(table.encodedrows)
    dsdf = pd.DataFrame(data= trows, columns=tcolumns)

    tcols = deepcopy(table.header)
    tcols.append("predicted")
    tcols.append("samples")
    tcols.append("total_pts")
    tcols.append("fold")
    tcols.append("enough_multiplier")
    # tcols.append("run_num")
    sampledf = pd.DataFrame(columns=tcols)
    full_df = pd.DataFrame(columns=tcols)
    onedf = pd.DataFrame(columns=tcols)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=222)

    f = 1 #do we want to count the folds??
    for train_index, test_index in rkf.split(dsdf):
        X_train = dsdf.iloc[train_index]
        X_test = dsdf.iloc[test_index].drop(columns = [y_index])
        y_test = dsdf.iloc[test_index][y_index]

        X_train_for_all_pts = dsdf.iloc[train_index].drop(columns = [y_index])
        y_train_for_all_pts = dsdf.iloc[train_index][y_index]

        num_rows = len(X_train_for_all_pts.values)
        onedf = fullclassify(onedf, X_train_for_all_pts, y_train_for_all_pts, X_test, y_test, num_rows, num_rows, f, 100)
        full_df = full_df.append(onedf)

        table2 = Table(10)
        nprows = X_train.values
        header = list(X_train.columns.values)
        table2 + header
        for l in nprows:
            table2 + l

        mtreatments = [1,2,3,5]
        for m in mtreatments:
            enough = int(m * math.sqrt(len(table2.rows)))
            root = Table.clusters(table2.rows, table2, enough)

            treatments = [1,2,3,5]
            for samples in treatments:
                if samples == 1:
                    MedianTable = leafmedians(root)
                    sampledf = classify(MedianTable, sampledf, X_test, y_test, samples, len(MedianTable.rows), f, m)
                else:
                    # EDT = getLeafModes(root, samples)
                    EDT = getLeafData(root,samples) #gets rand points with real label
                    sampledf = classify(EDT, sampledf, X_test, y_test, samples, len(EDT.rows), f, m)
                full_df = full_df.append(sampledf)
        print("f:", f)
        f += 1

    final_columns = []
    for col in table.protected:
        final_columns.append(col.name)
    for col in table.klass:
        final_columns.append(col.name)
    final_columns.append("predicted")
    final_columns.append("samples")
    final_columns.append("total_pts")
    final_columns.append("fold")
    final_columns.append("enough_multiplier")
    # final_columns.append("run_num")
    output_df = full_df[final_columns]
    output_df.to_csv("./output/RAND/" + filename + "_RF.csv", index=False)

def main():
    random.seed(10039)
    datasets = ["adultscensusincome.csv", "bankmarketing.csv", "diabetes.csv", "CleanCOMPAS53.csv", "GermanCredit.csv", "defaultcredit.csv"]
    pbar = tqdm(datasets)

    for dataset in pbar:
        pbar.set_description("Processing %s" % dataset)
        table, filename = getTable(dataset, limiter=1000)
        clusterandclassify(table, filename)


    # clusterandclassify("diabetes.csv") #clusters
    # clusterandclassify("adultscensusincome.csv") #clusters
    # clusterandclassify("bankmarketing.csv") #clusters
    # clusterandclassify("CleanCOMPAS53.csv") #problem with empty cols?
    # clusterandclassify("GermanCredit.csv") #clusters
    # clusterandclassify("processed.clevelandhearthealth.csv") #clusters
    # clusterandclassify("defaultcredit.csv") #clusters
    # clusterandclassify("homecreditapplication_train.csv") # loaded 266113 rows after 2 hours; error on compiling sym/num cols


# self = options(__doc__)
if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.print_stats(sort='time')
