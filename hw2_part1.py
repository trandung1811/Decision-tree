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
        self.order = None       # orderof values of split_attribute in children
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
    
    #showing the tree#
    def show(self):
       
       node = self.root
       print("root: ", node.split_attribute, "\n")
       print("['None']: ", node.defautl_label)
       self.find(node)
       
    def find(self, node):
        
        
        if not node.children:
      
            return 
        else: 
    
            if node.split_attribute != "level":
               print([node.split_attribute], end = "-->")
               print("['None']:", node.defautl_label, end = ", ")
        for i in range(len(node.order)-1, -1, -1):
            if node.children[i].children: 
                print([node.order[i]], end = "-->")
            else: 
                print([node.order[i]], ": ", node.children[i].label, end = ", ")
            if node.order[i] == "Mid":
                print()
            self.find(node.children[i])
            
        print()
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



#########################  part 1#############################
training_data = [
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
] 

test_ = ({"level": "Junior","lang": "Java","tweets": "yes","phd": "no"}, 
{"level": "Junior","lang": "Java","tweets": "yes","phd": "yes"}, {"level": "Intern"}, {"level": "Senior"}
 )

def part_1():
    
    target = []
    data_ = []
    for item in training_data: 
        target.append(item[1])
        data_.append(item[0])
    data = pd.DataFrame(target)
    y = data[data.columns[-1]]
    x = pd.DataFrame(data_)
    tree = DecisionTreeID3(max_depth = 3)
    tree.fit(x,y)
    
    print("The tree shape are: \n")
    tree.show()
    print()
    print("The predicted class: \n")
    x_test = pd.DataFrame(test_)
    n = len(test_)
    result = tree.classify(x_test,n)
    for i in range(0,n):   
        if result[i] == True: 
            Str = "Hire"
        else: 
            Str = "Do not Hire"
        print(test_[i],"   ", Str)
    print("\n")
       
##############################Main Function###################################
    
    
def main():
     
    part_1()     
       
main()