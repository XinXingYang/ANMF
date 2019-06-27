'''
Created on Aug 8, 2016
Processing datasets. 

@author: XinXing Yang 
'''
import scipy.sparse as sp
import numpy as np
import operator 
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
     
        
        self.trainMatrix = self.load_rating_file_as_matrix('Data\\train.rating')
        self.testRatings = self.load_rating_file_as_list('Data\\test.rating')
        self.uSimMat = self.load_user_sim_file('Data\\DrugSim.txt')
        self.iSimMat,self.Sim_order = self.load_item_sim_file('Data\\DiseaseSim.txt')
        self.DiDrAMat=self.load_c_matrix('Data\\DiDrA.txt')

    def load_rating_file_as_matrix(self, filename):

        train_list=[]
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                train_list.append([user,item,rating])
                line = f.readline()    
        return train_list

       
    def load_c_matrix(self, filename):
        DiDrMat=np.loadtxt(filename)
        train_list=[]
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        row,col=DiDrMat.shape       
        for i in range(row):
            for j in range(col):
                DiDrMat[i][j]=DiDrMat[i][j]+np.random.normal(0,0.2)
        return DiDrMat
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                user, item ,rating= int(arr[0]), int(arr[1]),int(arr[2])
                ratingList.append([user, item,rating])
                line = f.readline()
        return ratingList

    def load_user_sim_file(self, filename):
        uSimMat=[]
        with open(filename, "r") as f:
            for line in f.readlines():
                temp=[]
                line=line.strip('\n')
                arr=line.split()
                for item in arr:
                    temp.append(float(item))
                uSimMat.append(temp)
        return uSimMat

    def load_item_sim_file(self, filename):
        iSimMat=[]
        with open(filename, "r") as f:
            for line in f.readlines():
                temp=[]
                line=line.strip('\n')
                arr=line.split()
                for item in arr:
                    temp.append(float(item))
                iSimMat.append(temp)
        negativeList = []
        with open('Data\\negitive.rating') as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                user,item=int(arr[0]),int(arr[1])
                negativeList.append([user,item])
                line = f.readline()
        with open('Data\\test.rating') as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                user,item=int(arr[0]),int(arr[1])
                negativeList.append([user,item])
                line = f.readline()
        oreder_iSim=[]
        for drug in range(593):
            temp=[]
            for item in negativeList:
                if item[0]==drug:
                    temp.append(item[1])
            oreder_iSim.append(temp)
        return iSimMat, oreder_iSim
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line=line.strip('\n')
                arr = line.split("\t")
                user,item=int(arr[0]),int(arr[1])
                negativeList.append([user,item])
                line = f.readline()
        return negativeList
    

