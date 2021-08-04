#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:56:43 2020

@author: dungtran
"""
import pandas as pd 
import math
import numpy as np

########################## Main Tree Algorithm ###############################
class TreeNode(object):
    
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf
        self.defautl_label = None # a defualt label for unexpected an missing attribute

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label
    
    def set_default_label(self, label):
        self.defautl_label = label;

#caculate entropy 
def entropy(freq):
   
    if int(freq[0]) == 0 or int(freq[1]) == 0: return 0
    prob_0 = int(freq[0])/ (int(freq[0]) + int(freq[1]))
    temp = - prob_0*math.log(prob_0) - (1-prob_0)*math.log(1-prob_0)
    return temp

class DecisionTreeID3(object):
    
    #initilize Data
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-3):
        self.root = None
        self.max_depth = max_depth     # limitation depth of the tree 
        self.Ntrain = 0
        self.min_gain = min_gain      # minimum threshold for the information gain
        self.min_samples_split = min_samples_split 
    #Fit data to train
    def fit(self, data, target):
        
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
    
        ids = range(self.Ntrain)  
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0) # create a root of tree
        
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children: #check if the node is leaf node
                    self._set_label(node)
                else: 
                    node.set_default_label(self.default_label(node.ids)) 
                queue += node.children
            else:
                self._set_label(node)
       
    # take the default_label that is the most common class    
    def default_label(self,ids): 
        
        if len(ids) == 0: return 0
       
        count_y = 0
        count_n = 0
        
        for element in ids: 
          
             if self.target[element] == "yes" or self.target[element] == True: 
                 count_y += 1
             else: 
                 count_n += 1
           
        if count_y > count_n: 
            check = True
        else:
            check = False
            
        return  check 
      
    #caculate entropy for each atribute    
    def _entropy(self, ids):
        
        if len(ids) == 0: return 0
       
        count_y = 0
        count_n = 0
        freq = []
        for element in ids: 
          
             if self.target[element] == "yes" or self.target[element] == True or self.target[element] == 1: 
                 count_y += 1
             else: 
                 count_n += 1
           
        freq.append(count_y)
        freq.append(count_n)
        return entropy(freq)


    def _set_label(self, node):
       
        node.set_label(self.target[node.ids].mode()[0]) # most frequent label
 
      
    #building a ID3 Tree
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
      
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
          
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append(sub_ids)
            
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
        
            # information gain
            H_S= 0
            for split in splits:
                H_S += len(split)*self._entropy(split)/len(ids)
            
            gain = node.entropy - H_S 
            if gain < self.min_gain: continue # stop if small gain 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
      
        node.set_properties(best_attribute, order)
        
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
       
        
        return child_nodes


    #Classify new entry
    def classify(self, new_data,n):
        
        
        labels = []
        for i in range(n):
            x = new_data.iloc[i, :]
           
            # start from root and recursively travel if not meet a leaf 
            node = self.root  
            while node.children: 
             
                attribute = x[node.split_attribute]
                if attribute not in node.order: 
                     
                       labels.append(node.defautl_label)
                       break
                else:
                       node = node.children[node.order.index(x[node.split_attribute])]
            if node.label == None:
                continue
            else: 
                labels.append(node.label)
            
        return labels


################################Part2#########################################

filename_1 = "train.csv"

def raw_data():
  data = pd.read_csv(filename_1)
  
  train = data.iloc[:650, ]
  test = data.iloc[650:,  :]
  y = train[train.columns[1]]
  drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Survived']
  x = train.drop(drop_elements, axis = 1)
  tree = DecisionTreeID3(max_depth = 5, min_samples_split = 2, min_gain = 1e-4)
  tree.fit(x,y)
      
  y_test = test[test.columns[1]].to_list()
  x_test = test.drop(drop_elements, axis = 1)
  n = len(y_test)
  result = tree.classify(x_test,n)
  acc = 0
  
  for i in range(n):
      if int(result[i]) == int(y_test[i]): 
              acc += 1
  print(" The total accuracy in raw test set: ",acc/n)

#Process data to get better performence
def process_data(filename):
    
   data = pd.read_csv(filename)
   
   #create a new column named: Has_cabin based on Cabin
   data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
   
   # Create new feature FamilySize as a combination of SibSp and Parch
   data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
   
   # Create new feature IsAlone from FamilySize
   data['IsAlone'] = 0
   data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
   
   # Mapping Sex
   data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
   
   # Mapping Fare
   data['Fare'] = data['Fare'].fillna(data['Fare'].median())
   data.loc[ data['Fare'] <= 7.91, 'Fare'] 						        = 0
   data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
   data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
   data.loc[ data['Fare'] > 31, 'Fare'] 							        = 3
   data['Fare'] = data['Fare'].astype(int)
   
   #Mapping Embark
   data['Embarked'] = data['Embarked'].fillna('S')
   data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
   
   # Mapping Age
   age_avg = data['Age'].mean()
   data.loc[np.isnan(data['Age']), 'Age'] = age_avg
   data['Age'] = data['Age'].astype(int)
   data.loc[ data['Age'] <= 16, 'Age'] 					       = 0
   data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
   data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
   data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
   data.loc[ data['Age'] > 64, 'Age'] = 4
   
   drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
   data = data.drop(drop_elements, axis = 1)
   
   return data

def new_data():
 
  data = process_data(filename_1)
      
  train = data.iloc[:650, ]
  test = data.iloc[650:,  :]
  
  y = train[train.columns[0]]
  x = train.drop(["Survived"], axis = 1)
  tree = DecisionTreeID3(max_depth = 5, min_samples_split = 2, min_gain = 1e-4)
  tree.fit(x,y)
      
  y_test = test[test.columns[0]].to_list()
  x_test = test.drop(["Survived"], axis = 1)
  n = len(y_test)
  result = tree.classify(x_test,n)
  acc = 0
  
  for i in range(n):
     if int(result[i]) == int(y_test[i]): 
              acc += 1
  print(" The total accuracy in cleaned test set: ",acc/n)
       
##############################Main Function###################################
    
    
def main():
   
    print("The result of part2: \n")
    raw_data()
    new_data()
      
       
main()