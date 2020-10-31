
from collections import defaultdict  
from queue import Queue 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import pandas as pd
import numpy as np
import sklearn
import math
import time
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
import string


# ## Helper Functions and variables


def categorize(lst):
  lst = np.array(lst)
  uniq = np.unique(lst)
  indx =  np.arange(len( uniq)) 
  for i in indx:
    lst[ lst == uniq[i] ] = str(i)

  return [str(i) for i in lst]

def binning(lst, quantile = None):
  # labels= [str(i) for i in np.arange(quantile)]
  if quantile is None:
    quantile = 4
  try:
    tmpLst = pd.qcut(lst, q = quantile)
  except:
    tmpLst = pd.cut(lst, quantile)
  return tmpLst


def castToType(x):
    try:
        return x.astype('float').astype('Int64') #pd 0.24+ 
    except:
        try:
            return x.astype('float')
        except:
            return x


def measure_performance(y_test, y_pred):
    labels = list(set(y_test))
    print('\t\t\t', labels)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN) * 100
    print('True positive rate ', TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) * 100
    print('True negative rate ', TNR)

    # Precision or positive predictive value
    PPV = TP/(TP+FP) * 100
    print('Positive predictive value ', PPV)

    # Negative predictive value
    NPV = TN/(TN+FN) * 100
    # print('True positive rate ', TPR)

    # Fall out or false positive rate
    FPR = FP/(FP+TN) * 100
    # print('True positive rate ', TPR)

    # False negative rate
    FNR = FN/(TP+FN) * 100


    # False discovery rate
    FDR = FP/(TP+FP) * 100
    print('False discovery rate ', FDR)

    F1_score = 2 * PPV * TPR / (PPV + TPR) 
    print('F1 score ', F1_score)


    print(metrics.accuracy_score(y_test,y_pred) * 100)


def isNumeric(num):
    return isinstance(num, numbers.Number)

missing_values = ["n/a", "na", "--","NA","N/A","?"]


# ## Decision Tree 

import numpy as np
import math
import numbers
import random
MX_CONT_SIZE = 8000
PER_ROW_MEDIAN_ENTRY = 2

def targetEntropy( lst):
        lst = np.array(lst)
        values, counts = np.unique(lst, return_counts=True)
        total = np.sum(counts)
        entr = counts/total * np.log2(counts/total) 
        return -np.sum(entr)


class Question:

    def __init__(self, colForQuest, isNum, valLst=None, splitNum=None):
        self.colForQuest = colForQuest
        self.isNum = isNum
        self.valLst = valLst
        self.splitNum = splitNum

    def __str__(self):
        return str(self.colForQuest) + " " + str(self.isNum)  + " "+ str(self.valLst) + " " + str(self.splitNum) 
        
    def giveMatchingChild(self, data):
        if self.isNum:
            if data[self.colForQuest] <= self.splitNum:
                return 0
            else:
                return 1
        else:
            
            for i in range(len(self.valLst)):
                if self.valLst[i] == data[self.colForQuest]:
                    return i

            return 0




class Node:

    def __init__(self, dataset, depth): 
        self.depth = depth 
        colList = dataset[0,:]
        tmpDataset = dataset[1:,:]
        self.entropy = targetEntropy(tmpDataset[:,-1])
        mxInfoGain = -10000
        splitIndx = 0
        self.children = []
        self.question = None
        info = 0
        self.splitNum = -1
        tmpSplitVal = -1

        self.isLeaf = ( depth == 0 ) or (self.entropy == 0) or (len(tmpDataset[0]) == 1 )

        
        if not self.isLeaf:
            for i in range(len(tmpDataset[0])-1):

                if isNumeric(tmpDataset[0][i]):
                    info, tmpSplitVal = self.nodeEntropy(tmpDataset[:,i], tmpDataset[:, -1])
                else:
                    info = self.nodeEntropy(tmpDataset[:,i], tmpDataset[:, -1])
                infoGain = self.entropy - info
                if mxInfoGain < infoGain:
                    if isNumeric(tmpDataset[0][i]):
                        self.splitNum = tmpSplitVal
                    mxInfoGain, splitIndx = infoGain, i

            if mxInfoGain <= 0:
                self.isLeaf = True
                
            else:
                isNum = isNumeric( tmpDataset[0][splitIndx])
                self.valLst = []

                idxs = list( range(len(tmpDataset[0])))
                idxs.pop(splitIndx) #this removes elements from the list
            

                if isNum:

                    if not self.isLeaf:
                        
                        self.children.append(Node( np.vstack((colList, tmpDataset[ tmpDataset[:,splitIndx] <= self.splitNum]))[:,idxs], depth - 1 )) 
                        self.children.append(Node( np.vstack( (colList,tmpDataset[ tmpDataset[:,splitIndx] > self.splitNum]) )[:, idxs], depth - 1 )) 

                else:
                    self.valLst = np.unique(tmpDataset[:, splitIndx])

                    if not self.isLeaf:
                        for i in self.valLst:
                            self.children.append(Node( np.vstack((colList,tmpDataset[ tmpDataset[:,splitIndx]  == i]) )[:, idxs], depth - 1 )) 

                self.question = Question(colList[splitIndx] , isNum, valLst=self.valLst, splitNum= self.splitNum ) 

        if self.isLeaf:
            targetCol = np.array(tmpDataset[:,-1], dtype='O') 
            val, cnt = np.unique( targetCol, return_counts=True)
            idx = random.choice(np.argwhere(cnt == max(cnt)))[0]
            self.predictedVal =  val[ idx] 
    
    def nodeEntropy(self, splitByCol, targetCol):
        isNum = isNumeric(splitByCol[0])

        combindedLst = np.array( list(zip(splitByCol, targetCol)), dtype='O')
        weightLst = []
        total = len(targetCol)

        if isNum:
            combindedLst = combindedLst[ combindedLst[:,0].argsort()]
            vals = np.unique(combindedLst[:,0]) 
            vals = vals[:-1] + np.diff(vals)/2.0 
            
            # for time optimization, build the median list of every consecutive 4/5 elements of sorted array  
            # print(vals.shape[0])
            # if vals.shape[0] > MX_CONT_SIZE:
            #     vals.resize(vals.shape[0]//PER_ROW_MEDIAN_ENTRY + 1 , PER_ROW_MEDIAN_ENTRY)
            #     vals = np.median(vals, axis=1)  
            #     print(vals.shape)

            leastEntropy,splitVal = 1000000, -10

            for i in vals:
                entrLst = []
                weightLst = []

                for j in range(2):

                    tmpList = []
                    indx = np.searchsorted( combindedLst[:,0], j, side='right') 
                    if j == 0:
                        tmpList = combindedLst[:indx, :]
                    else:
                        tmpList = combindedLst[indx:, :]

                    val, cnt = np.unique(tmpList[:,1], return_counts=True)
                    tmpCnt = np.sum(cnt)
                    weightLst.append(tmpCnt)
                    entrLst.append(-np.sum( cnt/tmpCnt * np.log2(cnt/tmpCnt) ) )  
                    
                entropy = np.sum(  np.array(entrLst) * np.array(weightLst)/total )
                if entropy < leastEntropy:
                    leastEntropy, splitVal = entropy, i
            return leastEntropy, splitVal
        else:
            values, counts = np.unique(splitByCol, return_counts=True)
            entrLst = []
            weightLst = []

            for i in values:
                tmpList = combindedLst[combindedLst[:,0] == i]
                val, cnt = np.unique(tmpList[:,1], return_counts=True)
                tmpCnt = np.sum(cnt)
                weightLst.append(tmpCnt)
                entrLst.append(-np.sum( cnt/tmpCnt * np.log2(cnt/tmpCnt) ) )  
            
            return np.sum( np.array(entrLst) * np.array(weightLst)/total)

    def predict(self, data):

        if self.isLeaf:
            return self.predictedVal # highest count target column value 
        else:
            childIndx = self.question.giveMatchingChild(data)
            return self.children[childIndx].predict(data) 
            
    def print(self):
      if self.isLeaf:
        print('(leaf) Entropy=', self.entropy,' Highest Count=', self.predictedVal, ' Height=',self.depth,'\t',end=' ')
      else:
        print('Entropy=', self.entropy, ' Split on=', self.question.colForQuest,' Height=',self.depth,'\t',end=' ')
    
class DecisionTree:

    def __init__(self, depth = 1000):
        self.entropy = 0
        self.depth = depth 

    def fit(self, x_train, y_train):
        a = np.array(x_train, dtype='O')
        b = np.array(y_train, dtype='O')
        self.dataset = np.column_stack((a,b))
        colNo = np.array(list(range( self.dataset.shape[1] )), dtype='O')
        self.dataset = np.vstack((colNo, self.dataset))
        self.entropy = targetEntropy(y_train)
        self.rootNode = Node(self.dataset, self.depth)

    def predict(self, x_test):
        y_pred = []
        for data in  x_test:
            y_pred.append(self.rootNode.predict(data))
        return y_pred

    def print(self):
      q = Queue()
      q.put(self.rootNode)
      depth = self.rootNode.depth
      while not q.empty()  :
        node = q.get()     
        print(node.depth*'\t',end=' ')
        node.print()
        print('')


        for child in node.children:
          q.put(child)
          depth = child.depth


# ## Adaboost implementation




from collections import defaultdict  
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import random
import numpy as np
import math

class Adaboost:
    
    def __init__(self, numOfHypothesis):
        self.hypoNum = numOfHypothesis
        self.hypoList = [] 
        self.hypoWeight = [1] * numOfHypothesis

    def normalize(self, lst):
        lst = np.array(lst)
        sum = np.sum(lst)
        return lst/sum

    def getResampleData(self):
        chosenIdxs = random.choices( population= list( range( len(self.dataset) ) ), weights=self.sampleWeight, k=len(self.dataset) )
        return self.dataset[chosenIdxs]

    def fit(self, x_train, y_train):
        x_train = np.array(x_train, dtype="O")
        y_train = np.array(y_train,  dtype="O")
        self.dataset = np.column_stack((x_train, y_train))
        self.sampleWeight = [1/len(self.dataset)] * len(self.dataset)

        k = 0
        while k < self.hypoNum:
            dtc = DecisionTree(1)
            data = self.getResampleData()

            # dtc = DecisionTreeClassifier(random_state=0, max_depth=1)
            # data = self.dataset
            tmpHypoResult = self.hypothesisResult( dtc, data )
            error = self.getError(tmpHypoResult, y_train)

            if error > 0.5:
                continue
            for j in range(len(y_train)):
                if tmpHypoResult[j] == y_train[j]:
                    self.sampleWeight[j] = self.sampleWeight[j] * error/(1 - error)
            
            self.sampleWeight = self.normalize(self.sampleWeight)
            self.hypoList.append(dtc)
            self.hypoWeight[k] = math.log2( (1-error)/ error )
            k = k + 1




    def getError(self, y_pred, y_test):
        error = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_test[i]:
                error = error + self.sampleWeight[i]
        return error

    def hypothesisResult(self, dtc, resampleData ):
        dtc.fit(resampleData[:,:-1], resampleData[:,-1] )

        return dtc.predict(self.dataset[:,:-1])
 

    def predict(self, x_test):
        y_predLst = []
        y_test = []
        for hypo in self.hypoList:
            y_test.append( hypo.predict(x_test) ) 
            
        y_test = np.array(y_test)
        for i in range(len(x_test)):
            y_pred = y_test[:,i]
            voting = defaultdict(int)
            for k in range( self.hypoNum ):
                voting[ y_pred[k] ] += self.hypoWeight[k]
            y_predLst.append( max(voting, key=voting.get) ) 
        
        return y_predLst


# ## Hyper-Parameter Tuning 
# Only tree-depth is being optimized here.
# Things like max_split/number of leaves are also subjet to optimization


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return metrics.accuracy_score(y_test,model.predict(X_test))

#Now only finds out the optimal tree depth 
def hyperParaTuning():
  from sklearn.model_selection import StratifiedKFold
  folds = StratifiedKFold(n_splits=3)

  scores_tree = []
  best_height, best_score = 5, 0
  
  for h in random.sample( list(range(3,13)), k = 5):
    for train_index, test_index in folds.split(x_train, y_train):
        _X_train, _X_test, _Y_train, _Y_test = [x_train[i] for i in train_index],                                                 [x_train[i] for i in test_index],                                                   [y_train[i] for i in train_index],                                                  [y_train[i] for i in test_index] 
        scores_tree.append(get_score(DecisionTree(5), _X_train, _X_test, _Y_train, _Y_test))  
    if np.average(scores_tree) > best_score:
      best_score, best_height = np.average(scores_tree) , h

  return best_height


# ## Telco Dataset 

df = pd.read_csv('telco', na_values = missing_values)
df = df.drop(columns=['gender'])

colToCategorise =  ['SeniorCitizen','Partner','Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod' ]
colToBinning = ['tenure',	'MonthlyCharges', 'TotalCharges']


# ## Adult Dataset


# df = pd.read_csv('adult.data', header=None, na_values = missing_values, skip_blank_lines=True)
# df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
#               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

# colToCategorise = ['workclass', 'education',  'marital-status', 'occupation',
#               'relationship', 'race', 'sex', 'native-country']
# colToBinning = ['age' , 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']


# # ## Credit-Card Dataset


# df = pd.read_csv('creditcard', na_values = missing_values, skip_blank_lines=True)
# pos_sample = df[df.iloc[:,-1] == 1]
# neg_sample = df[df.iloc[:,-1] == 0].sample(n=20000, random_state=0)
# df = shuffle(pd.concat( (pos_sample,neg_sample), axis=0), random_state=0) 
# df.reset_index(drop=True)

# colToCategorise = []
# colToBinning = df.columns[:-1]


# ## Data preprocessing + Cleaning


start_time = time.time()
print('Data processing start')
# -----------------------------------Data cleaning--------------------------------
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

df.replace(r'^[\s?]*$', np.nan, regex=True, inplace = True)

for col in df.columns:
    if len(df[col].unique()) == 1:
        print(col)
        df.drop(col,inplace=True,axis=1)

for col in df.columns:
    if len(df[col].unique()) == len(df[col]):
        print(col)
        df.drop(col,inplace=True,axis=1)

df = df.apply(castToType)

for col in df.columns:

    if df[col].isnull().any() :
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].mean())
        # else:
        #   #Max fill function for categorical columns
        #   data['column_name'].fillna(data['column_name'].value_counts()
        #   .idxmax(), inplace=True)

#drop all columns with 60% or more null values 
nullThreshold = 0.7

df.dropna(thresh=df.shape[0]*nullThreshold,how='all',axis=1, inplace= True)
df.dropna(axis = 0, inplace = True)
# -----------------------------------Data cleaning--------------------------------


# -------------------------------------------Categorization/Binning------------------------------

for col in colToCategorise:
  df[col] = categorize(df[col])

for col in colToBinning:
  df[col] = binning(df[col])
# -------------------------------------------Categorization/Binning------------------------------

from sklearn.model_selection import train_test_split
train, test = train_test_split(df,test_size=0.2, random_state=0)

x_train, y_train = train.drop(train.columns[-1], axis=1).values.tolist(),  train[train.columns[-1]].values.tolist()
x_test, y_test = test.drop(test.columns[-1], axis=1).values.tolist(),  test[test.columns[-1]].values.tolist()

print('Data processing Done')
print('Time taken ----', time.time() - start_time)


# ## Decision Tree Testing


best_depth = hyperParaTuning()
# best_depth = 7 #for telco, adult
print(best_depth)

dtc = DecisionTree(best_depth)



start_time = time.time()
print('-- Data fitting start')

dtc.fit(x_train, y_train)

print('Data fitting finished')
print('Time taken ----', time.time() - start_time)
# dtc.print()
print('Data prediction start')
start_time = time.time()

# y_pred = boost.predict(x_test)
y_pred = dtc.predict(x_test)
y_train_pred = dtc.predict(x_train)
print('Data fitting finished')
print('Time taken ----', time.time() - start_time)

# print(y_pred)
# print(y_test)
print('Training set performance')
print(metrics.classification_report(y_train,y_train_pred))
print(metrics.accuracy_score(y_train,y_train_pred) * 100)
# print(metrics.confusion_matrix(y_train, y_train_pred))
measure_performance(y_train, y_train_pred)

print('Testing set performance')
print(metrics.classification_report(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred) * 100)
# print(metrics.confusion_matrix(y_test, y_pred))
measure_performance(y_test, y_pred)
# print(calc_perfrom_BinClass(y_test, y_pred))

print('Data prediction finished')
print('Time taken ----', time.time() - start_time)


# ## Boosting Test 


for stump in [5,10,15,20]:
  boost = Adaboost(stump)

  start_time = time.time()
  print('-- Data fitting start')

  boost.fit(x_train, y_train)

  print('Data fitting finished')
  print('Time taken ----', time.time() - start_time)

  print('Data prediction start')
  start_time = time.time()

  y_pred = boost.predict(x_test)
  y_train_pred = boost.predict(x_train)

  print('Time taken ----', time.time() - start_time)

  print('No of stumps ',stump)
  print('Training set performance')
  print(metrics.accuracy_score(y_train,y_train_pred) * 100)

  print('Testing set performance')
  print(metrics.accuracy_score(y_test,y_pred) * 100)
  print('Data prediction finished')
  print('Time taken ----', time.time() - start_time)
  print('------------------------------------------')



