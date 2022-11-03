#!/usr/bin/env python3
# vim: sta:et:sw=2:ts=2:sts=2 :

from copy import deepcopy as kopy
import sys,random, statistics, pprint
import pandas as pd
from cliffs_delta import cliffs_delta #https://pypi.org/project/cliffs-delta/
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

"""
Scott-Knot test + non parametric effect size + significance tests.
Tim Menzies, 2019. Share and enjoy. No warranty. Caveat Emptor.

Accepts data as per the following exmaple (you can ignore the "*n"
stuff, that is just there for the purposes of demos on larger
and larger data)

Ouputs treatments, clustered such that things that have similar
results get the same ranks.

For a demo of this code, just run

    python3 sk.py

"""

#-----------------------------------------------------
# Examples

def skDemo(n=5) :
  #Rx.data is one way to run the code
  return Rx.data( x1 =[ 0.34, 0.49 ,0.51, 0.6]*n,
                  x2  =[0.6  ,0.7 , 0.8 , 0.89]*n,
                  x3  =[0.13 ,0.23, 0.38 , 0.38]*n,
                  x4  =[0.6  ,0.7,  0.8 , 0.9]*n,
                  x5  =[0.1  ,0.2,  0.3 , 0.4]*n)

"""
Another is to make a file

x1  0.34  0.49  0.51  0.6
x2  0.6   0.7   0.8   0.9
x3  0.15  0.25  0.4   0.35
x4  0.6   0.7   0.8   0.9
x5  0.1   0.2   0.3   0.4

Then call

   Rx.fileIn( fileName )

"""

#-----------------------------------------------------
# Config

class o:
  def __init__(i,**d) : i.__dict__.update(**d)

class THE:
  cliffs = o(dull= [0.147, # small
                    0.33,  # medium
                    0.474 # large
                    ][0])
  bs=     o( conf=0.05,
             b=500)
  mine =  o( private="_")
  char =  o( skip="?")
  rx   =  o( show="%5s %20s %10s")
  tile =  o( width=50,
             chops=[0.1 ,0.3,0.5,0.7,0.9],
             marks=[" " ,"-","-","-"," "],
             bar="|",
             star="*",
             show=" %5.3f")
#-----------------------------------------------------
def cliffsDeltaSlow(lst1,lst2, dull = THE.cliffs.dull):
  """Returns true if there are more than 'dull' difference.
     Warning: O(N)^2."""
  n= gt = lt = 0.0
  for x in lst1:
    for y in lst2:
      n += 1
      if x > y:  gt += 1
      if x < y:  lt += 1
  return abs(lt - gt)/n <= dull

def cliffsDelta(lst1, lst2,  dull=THE.cliffs.dull):
  "By pre-soring the lists, this cliffsDelta runs in NlogN time"
  def runs(lst):
    for j,two in enumerate(lst):
      if j == 0: one,i = two,0
      if one!=two:
        yield j - i,one
        i = j
      one=two
    yield j - i + 1,two
  #---------------------
  m, n = len(lst1), len(lst2)
  lst2 = sorted(lst2)
  j = more = less = 0
  for repeats,x in runs(sorted(lst1)):
    while j <= (n - 1) and lst2[j] <  x: j += 1
    more += j*repeats
    while j <= (n - 1) and lst2[j] == x: j += 1
    less += (n - j)*repeats
  d= (more - less) / (m*n)
  return abs(d)  <= dull




def bootstrap(y0,z0,conf=THE.bs.conf,b=THE.bs.b):
  """
  two  lists y0,z0 are the same if the same patterns can be seen in all of them, as well
  as in 100s to 1000s  sub-samples from each.
  From p220 to 223 of the Efron text  'introduction to the boostrap'.
  Typically, conf=0.05 and b is 100s to 1000s.
  """
  class Sum():
    def __init__(i,some=[]):
      i.sum = i.n = i.mu = 0 ; i.all=[]
      for one in some: i.put(one)
    def put(i,x):
      i.all.append(x);
      i.sum +=x; i.n += 1; i.mu = float(i.sum)/i.n
    def __add__(i1,i2): return Sum(i1.all + i2.all)
  def testStatistic(y,z):
     tmp1 = tmp2 = 0
     for y1 in y.all: tmp1 += (y1 - y.mu)**2
     for z1 in z.all: tmp2 += (z1 - z.mu)**2
     s1    = float(tmp1)/(y.n - 1)
     s2    = float(tmp2)/(z.n - 1)
     delta = z.mu - y.mu
     if s1+s2:
       delta =  delta/((s1/y.n + s2/z.n)**0.5)
     return delta
  def one(lst): return lst[ int(any(len(lst))) ]
  def any(n)  : return random.uniform(0,n)
  y,z  = Sum(y0), Sum(z0)
  x    = y + z
  baseline = testStatistic(y,z)
  yhat = [y1 - y.mu + x.mu for y1 in y.all]
  zhat = [z1 - z.mu + x.mu for z1 in z.all]
  bigger = 0
  for i in range(b):
    if testStatistic(Sum([one(yhat) for _ in yhat]),
                     Sum([one(zhat) for _ in zhat])) > baseline:
      bigger += 1
  return bigger / b >= conf

#-------------------------------------------------------
# misc functions
def same(x): return x

class Mine:
  "class that, amongst other times, pretty prints objects"
  oid = 0
  def identify(i):
    Mine.oid += 1
    i.oid = Mine.oid
    return i.oid
  def __repr__(i):
    pairs = sorted([(k, v) for k, v in i.__dict__.items()
                    if k[0] != THE.mine.private])
    pre = i.__class__.__name__ + '{'
    def q(z):
     if isinstance(z,str): return "'%s'" % z
     if callable(z): return "fun(%s)" % z.__name__
     return str(z)
    return pre + ", ".join(['%s=%s' % (k, q(v))])

#-------------------------------------------------------
class Rx(Mine):
  "place to manage pairs of (TreatmentName,ListofResults)"
  def __init__(i, rx="",vals=[], key=same):
    i.rx   = rx
    i.vals = sorted([x for x in vals if x != THE.char.skip])
    i.n    = len(i.vals)
    i.med  = i.vals[int(i.n/2)]
    i.mu   = sum(i.vals)/i.n
    i.rank = 1
    i.sd = statistics.stdev(i.vals)
    i.cohen = i.sd* 0.35
  def tiles(i,lo=0,hi=1): return  xtile(i.vals,lo,hi)
  def __lt__(i,j):        return i.med < j.med
  def __eq__(i,j):
    return cliffsDelta(i.vals,j.vals) and bootstrap(i.vals,j.vals)
  def __repr__(i):
    return '%10s %20s %10s' % (i.rank, i.rx, i.tiles())
  def xpect(i,j,b4):
    "Expected value of difference in means before and after a split"
    n = i.n + j.n
    return i.n/n * (b4.med- i.med)**2 + j.n/n * (j.med-b4.med)**2

  #-- end instance methods --------------------------

  @staticmethod
  def data(**d):
    "convert dictionary to list of treatments"
    return [Rx(k,v) for k,v in d.items()]

  def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

  @staticmethod
  def fileIn(f, df, klist, lo=0,hi=1):
    d={}
    what=None
    for word in words(f):
       x = thing(word)
       if isinstance(x,str):
          what=x
          d[what] = d.get(what,[])
       else:
          d[what] += [x]

    cDlist = []
    blist = []
    dlist = []
    modellist = []
    reslist = []
    jlist = []

    statdf = pd.DataFrame(columns = ["cliffsDelta"])
    for keyword in klist:
      newd = {i:d[i] for i in d if keyword in i}

      df1 = Rx.show(Rx.sk(Rx.data(**newd)),lo=lo,hi=hi)
      df = pd.concat([df, df1], ignore_index = True)
    # print(df.head(), df.index.size)

      slist = sorted(newd.keys(), key = lambda x: int(x.split('_')[0]))
      max_key = slist[-1]

      for key2 in slist[:-1]:
        modellist.append(key2)
        cdelta, res = cliffs_delta(newd.get(max_key),newd.get(key2))
        dlist.append(cdelta)
        reslist.append(res)
        cDlist.append(cliffsDelta(newd.get(max_key),newd.get(key2)))
        blist.append("same" if bootstrap(newd.get(max_key),newd.get(key2)) else 'different')
        jlist.append(jaccard_similarity(newd.get(max_key),newd.get(key2)))

    statdf["model"] = modellist
    statdf["cliffsDelta"] = dlist
    statdf["CD_res"] = reslist
    statdf["tm_cliffsDelta"] = cDlist
    # statdf["scip_bootstrap"] = bslist
    statdf["tm_bootstrap"] = blist
    statdf["jacc"] = jlist


    return df, statdf

  @staticmethod
  def sum(rxs):
    "make a new rx from all the rxs' vals"
    all = []
    for rx in rxs:
        for val in rx.vals:
            all += [val]
    return Rx(vals=all)

  @staticmethod
  def show(rxs,lo=0,hi=1):
    "pretty print set of treatments"
    tmp=Rx.sum(rxs)
    mlist = []
    ranklist =[]
    sdlist =[]
    medlist = []
    avglist = []
    # clist = []
    skdf = pd.DataFrame(columns=["dataset", "model", "metric", "median", "StandDev", "mean", "sk_rank"])
    # lo,hi=tmp.vals[0], tmp.vals[-1]
    for rx in sorted(rxs):
        mlist.append(rx.rx)
        ranklist.append(rx.rank)
        sdlist.append(round(rx.sd,3))
        medlist.append(rx.med)
        avglist.append(round(rx.mu,3))
        # clist.append(round(rx.cohen,3))
        print(THE.rx.show % (rx.rank, rx.rx,
              rx.tiles(lo=lo,hi=hi)))

    skdf["model"] = mlist
    skdf["sk_rank"] = ranklist
    skdf["median"] = medlist
    skdf["mean"] = avglist
    skdf["StandDev"] = sdlist
    # skdf["cohens"] = clist
    return skdf

  @staticmethod
  def sk(rxs):
    "sort treatments and rank them"
    def divide(lo,hi,b4,rank):
      cut = left=right=None
      best = 0
      for j in range(lo+1,hi):
          left0  = Rx.sum( rxs[lo:j] )
          right0 = Rx.sum( rxs[j:hi] )
          now    = left0.xpect(right0, b4)
          if now > best:
              if left0 != right0:
                  best, cut,left,right = now,j,kopy(left0),kopy(right0)
      if cut:
        rank = divide(lo, cut, left, rank) + 1
        rank = divide(cut ,hi, right,rank)
      else:
        for rx in rxs[lo:hi]:
          rx.rank = rank
      return rank
    #-- sk main
    rxs=sorted(rxs)
    divide(0, len(rxs),Rx.sum(rxs),1)
    return rxs

#-------------------------------------------------------
def pairs(lst):
    "Return all pairs of items i,i+1 from a list."
    last=lst[0]
    for i in lst[1:]:
         yield last,i
         last = i

def words(f):
  with open(f) as fp:
    for line in fp:
       for word in line.split():
          yield word

def xtile(lst,lo,hi,
             width= THE.tile.width,
             chops= THE.tile.chops,
             marks= THE.tile.marks,
             bar=   THE.tile.bar,
             star=  THE.tile.star,
             show=  THE.tile.show):
  """The function _xtile_ takes a list of (possibly)
  unsorted numbers and presents them as a horizontal
  xtile chart (in ascii format). The default is a
  contracted _quintile_ that shows the
  10,30,50,70,90 breaks in the data (but this can be
  changed- see the optional flags of the function).
  """
  def pos(p)   : return ordered[int(len(lst)*p)]
  def place(x) :
    return int(width*float((x - lo))/(hi - lo+0.00001))
  def pretty(lst) :
    return ', '.join([show % x for x in lst])
  ordered = sorted(lst)
  lo      = min(lo,ordered[0])
  hi      = max(hi,ordered[-1])
  what    = [pos(p)   for p in chops]
  where   = [place(n) for n in  what]
  out     = [" "] * width
  for one,two in pairs(where):
    for i in range(one,two):
      out[i] = marks[0]
    marks = marks[1:]
  out[int(width/2)]    = bar
  out[place(pos(0.5))] = star
  return '('+''.join(out) +  ")," +  pretty(what)

def thing(x):
  "Numbers become numbers; every other x is a symbol."
  try: return int(x)
  except ValueError:
    try: return float(x)
    except ValueError:
      return x

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return round(float(intersection) / union,2)

#-------------------------------------------------------
# def _cliffsDelta():
#   "demo function"
#   lst1=[1,2,3,4,5,6,7]*100
#   n=1
#   for _ in range(10):
#       lst2=[x*n for x in lst1]
#       print("_cliffsDelta",
#            cliffsDelta(lst1,lst2),n) # should return False
#       n*=1.03
#
# def bsTest(n=1000,mu1=10,sigma1=1,mu2=10.2,sigma2=1):
#    def g(mu,sigma) : return random.gauss(mu,sigma)
#    x = [g(mu1,sigma1) for i in range(n)]
#    y = [g(mu2,sigma2) for i in range(n)]
#    return n,mu1,sigma1,mu2,sigma2,\
#           'same' if bootstrap(x,y) else 'different'

#-------------------------------------------------------
#
# if __name__ == "__main__":
#   random.seed(1)
#   Rx.fileIn("sk1.csv")
#   print("-"*50)
#   Rx.fileIn("sk2.csv",lo=0,hi=100)
#   print("-"*50)
#   _cliffsDelta()
#   print("-"*50)
#   print("bootstrap",  bsTest(100, 10, .5, 10, .5) )
#   print("bootstrap",  bsTest(100, 10, 1, 20, 1) )
#   print("bootstrap",  bsTest(100, 10, 10, 10.5, 10) )
#   print("-"*50)
#   n=1
#   l1= [1,2,3,4,5,6,7,8,9,10]*10
#   for _ in range(10):
#     l2=[n*x for x in l1]
#     print('same' if bootstrap(l1,l2) else 'different',n)
#     n*=1.02
#   n=1
#   print("-"*50)
#   for _ in range(4):
#     print()
#     print(n*5)
#     Rx.show(Rx.sk(skDemo(n)))
#     n*=5
from tqdm import tqdm
from utils import *

if __name__ == "__main__":
  params = Params("model_configurations/experiment_params.json")
  np.random.seed(params.seed)
  datasets = [ "communities" ,"heart","diabetes" ,"studentperformance", "compas", "bankmarketing", "defaultcredit", "adultscensusincome"]
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
  metrics = ['recall+', 'prec+', 'acc+', 'F1+', 'MSE-', 'FA0-', 'FA1-', 'AOD-', 'EOD-', 'SPD-', 'DI-']
  learners = ['RF', 'LSR', 'SVC']
  pbar = tqdm(datasets)

  datasetsdf = pd.DataFrame(columns=["dataset", "order", "model", "metric", "median", "mean", "StandDev", "sk_rank"])
  statsdf = pd.DataFrame(columns = ["dataset","order","model", "metric", "median", "mean", "StandDev", "sk_rank", "CD_res", "tm_bootstrap", "jacc", "tm_cliffsDelta", "cliffsDelta"])
  order = 0
  for dataset in pbar:
    order += 1
    pbar.set_description("Processing %s" % dataset)
    klist = keywords[dataset]
    metricdf = pd.DataFrame(columns=["dataset","model", "metric", "median", "mean", "StandDev", "sk_rank"])
    # statdf = pd.DataFrame(columns = ["model", "cliffsDelta", "bootstrap"])
    for m in metrics:
      df = pd.read_csv(r'./sk_data/features/full/' + dataset + "_" + m +"_.csv", sep=' ', header = None)
      df = df.transpose()
      df.columns = df.iloc[0]
      df = df[1:]
      for l in learners:
        learner_cols = [col for col in df.columns if l in col]
        output = df[learner_cols]
        output.transpose().to_csv("./sk_data/features/learners/" + l + "/" + dataset + "_" +  l +"_" + m +"_.csv", header = None, index=True, sep=' ')

    for l in learners:
      for m in metrics:
        print("\n" +"-" + dataset +"-" + l +"-"+ m + "\n"  )
        metric2df, statdf = Rx.fileIn("./sk_data/features/learners/" + l + "/" + dataset + "_" +  l + "_" + m +"_.csv", metricdf, klist)
        metric2df["metric"] = [m] * metric2df.index.size
        metric2df["dataset"] = [dataset] * metric2df.index.size
        metric2df["order"] = [order] * metric2df.index.size
        statdf["metric"] = [m] * statdf.index.size
        df_merged = pd.merge(metric2df, statdf, on = ["model", "metric"], how = "left")
        # print("df_merged",df_merged.index.size )
        statsdf = pd.concat([statsdf, df_merged], ignore_index=True)
        # print("statsdf",statsdf.index.size )

  datasetsdf = pd.concat([datasetsdf, statsdf], ignore_index=True)
  # print("datasetdf",datasetsdf.index.size )
  datasetsdf.to_csv("./sk_graphs/features/FM_RF.csv", index = False)
  print("-"*100)
