'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import operator
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings,  uSimMat, iSimMat,DiDrAMat, K, num_thread,train):

    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _uSim
    global _iSim
    global _DiDrAMat
    global _train
    _model = model
    _testRatings = testRatings
    _K = K
    _uSim = uSimMat
    _iSim = iSimMat
    _DiDrAMat=DiDrAMat
    _train=train
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    hit=getHitRation()
    auc,fpr, tpr, thre,area=evl_rating()
    return hit,auc,fpr, tpr, thre,area

def evl_rating():
    users, items,label,scores = [], [],[],[]
    users_c,items_c=[],[]

    for idx in _testRatings:
        u=int(idx[0])
        i=int(idx[1])
        la=int(idx[2])
        users.append(_uSim[u])
        items.append(_iSim[i])
        items_c.append(_DiDrAMat[i])
        users_c.append(_DiDrAMat[:,u])
        label.append(la)
    predictions = _model.predict([np.array(users), np.array(users_c),np.array(items),np.array(items_c)],
                                 batch_size=100, verbose=0)
    for index in range(len(predictions)):
        scores.append(float(predictions[index][0]))

    y_true=np.array(label)
    pre=np.array(scores)
    fpr, tpr, thre = metrics.roc_curve(y_true, pre, pos_label=1)
    auc=roc_auc_score(y_true, pre)
    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, pre)
    area = metrics.auc(recall, precision)
    return auc,fpr, tpr, thre,area



def getHitRation():
    hit=0
    for idx in _testRatings:
        train_diease=[]#在训练集中用过的药
        users, items= [], []
        users_c,items_c=[],[]
        scores={}
        u=int(idx[0])
        i=int(idx[1])#需要验证的药
        la=int(idx[2])
        if la==0:
            continue
        else:
            
            for idxx in _train:
                uu=int(idxx[0])
                ii=int(idxx[1])
                if uu==u:
                    train_diease.append(ii)
            for index in range(313):   
                users.append(_uSim[u])
                items.append(_iSim[index])
                items_c.append(_DiDrAMat[index])
                users_c.append(_DiDrAMat[:,u])
        
        predictions = _model.predict([np.array(users), np.array(users_c),np.array(items),np.array(items_c)],
                                 batch_size=100, verbose=0)
        for index in range(len(predictions)):
            scores[index]=float(predictions[index][0])
        for diease in train_diease:
            scores.pop(diease)
    
        scores=sorted(scores.items(),key=operator.itemgetter(1),reverse=True)
        
        for k in range(1):
            if int(scores[k][0])==i:
                if hit>100:
                    print("药物ID: ")
                    print(u)
                    for kk in range(10):
                        print(scores[kk][0])
                        print(scores[kk][1])
                hit=hit+1
    print(hit)
    return hit
            
        
            
                

    


