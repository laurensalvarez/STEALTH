import math, re, random, sys
from collections import defaultdict
from itertools import count

import warnings
warnings.filterwarnings("ignore")

big = sys.maxsize
tiny = 1 / big

# ------------------------------------------------------------------------------
# Column Class
# ------------------------------------------------------------------------------
class Col:
    def __init__(self,
                 name):
        self._id = 0 
        self.name = name  
        
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
    def __init__(self, name, uid, data=None):
        Col.__init__(self,
                     name)
        self.n = 0
        self.most = 0
        self.mode = ""
        self.uid = uid 
        self.count = defaultdict(int)
        # self.encoder = preprocessing.LabelEncoder()
        self.coltype = 0
        self.vals = []
        if data != None:
            for val in data:
                self + val

    # def __str__(self):
    # print to a sym; overrides print(); could replace dump() TBD

    def __add__(self, v):
        return self.add(v, 1)

    def add(self, v, inc=1):
        self.n += inc
        self.vals.append(v)
        self.count[v] += inc
        tmp = self.count[v]
        if tmp > self.most: 
            self.most, self.mode = tmp, v
        return v

    def diversity(self):  # entropy of all of n
        e = 0
        for k, v in self.count.items():
            p = v / self.n
            e -= p * math.log(p) / math.log(2)
        return e

    def mid(self):  # midpoint as mode (which sym appears the most)
        return self.mode

    def dist(self, x, y):  # Aha's distance between two syms
        if (x == "?" or x == "") or (y == "?" or y == ""):
            return 1
        return 0 if x == y else 1

# ------------------------------------------------------------------------------
# Numeric Column Class
# ------------------------------------------------------------------------------
class Num(Col):
    def __init__(self, name, uid, data=None):
        Col.__init__(self, name)
        self.n = 0
        self.mu = 0
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
                self + val

    def __add__(self, v):
        # add the column; calculate the new lo/hi, get the sd using the 'chasing the mean'
        self.n += 1
        self.vals.append(v)
        self.count[v] += 1 
        tmp = self.count[v]
        if tmp > self.most:  # check which is the most seen; if it's the most then assign and update mode
            self.most, self.mode = tmp, v
        try:
            if v < self.lo: 
                self.lo = v
            if v > self.hi:
                self.hi = v
            d = v - self.mu  # distance to the mean
            self.mu += d / self.n  # normalize it
            self.m2 += d * (v - self.mu)  # calculate momentumn of the series in realtionship to the distance
            self.sd = self._numSd()
            self.median = self.mid()
        except:
            print("failed col name:", self.name, "id:", self.uid)
        return v

    def _numSd(self):        
        if self.m2 < 0:  # means there's is no momentum to the series
            return 0
        if self.n < 2:  # if there's two items
            return 0
        return math.sqrt(self.m2 / (self.n - 1))  # calculate std dev

    def diversity(self):  # return standard dev
        return self.sd

    def mid(self):  # get midpoint for nums (median)
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
        self.xnums = [] 
        self.xsyms = []
        self.header = ""
        self.clabels = []

    # ------------------------------------------------------------------------------
    # Table Class: Helper Functions
    # ------------------------------------------------------------------------------
    @staticmethod 
    def compiler(x): 
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

        with open(file) as f:
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
        return finallines

    # ------------------------------------------------------------------------------
    # Table Class: Class Methods
    # ------------------------------------------------------------------------------
    def __add__(self, line):
        if len(self.header) > 0: 
            self.insert_row(line)
        else:
            self.create_cols(line) 
            
    def create_cols(self, line):
        self.header = line  # since add recognized no header, assign first line as a header.
        index = 0

        for val in line:
            val = self.compiler(val)  # compile the val datatype

            if val[0] == ":" or val[
                0] == "?":
                if val[0].isupper():
                    self.skip.append(Num(''.join(c for c in val),
                                         index))
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
                if index not in self.skip:
                    if val == "?" or val == "":
                        realline.append(val)
                        index += 1
                        continue
                    self.cols[index] + self.compiler(val)
                    realline.append(val)
                index += 1

            self.rows.append(realline)
            self.count += 1

    # ------------------------------------------------------------------------------
    # Clustering Fastmap;still in table class
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
        if len(rowA) != len(rowB):
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
        for node in nodes(root.leftNode): yield node
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
    return MedianTable

def leafmedians2(root, how=None):  # for all of the leaves from smallest to largest print len of rows & median
    MedianTable = Table(222)
    header = root.header
    MedianTable + header
    for leaf in sorted(nodes(root), key=how or rowSize):
        t = leaf.leftTable
        mid = [float(col.mid()) for col in t.cols]
        MedianTable + mid
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



