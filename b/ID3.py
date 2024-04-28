from __future__ import print_function 
import numpy as np 
import pandas as pd
import time 

class Infomation(object):
    def __init__(self, age, gender, debility, fever, digestive_issues, chronic_pain):
        self.age = self._age(age)           
        self.gender = self._gender(gender)
        self.debility = self._debility(debility)
        self.fever = self._fever(fever)
        self.digestive_issues = self._digestive_issues(digestive_issues)
        self.chronic_pain = self._chronic_pain(chronic_pain)

    
    def _age(self, age):
        res = ">60"
        if 0 <= age < 18:
            res = "1-18"
        elif 18 <= age < 35:
            res = "18-35"
        elif 35 <= age < 60:
            res = "35-60"
        return res
    
    def _gender(self, gender):
        gender = gender.lower()
        if gender == "female" or gender == "male":
            return gender
        if gender == "nu":
            return "female"
        return "male"
    
    def _debility(self, debility):
        debility = debility.lower()
        if debility == "yes" or debility == "no":
            return debility
        else:
            if debility == "co": return "yes"
            return "no"

    def _fever(self, fever):
        if fever < 36: return "<36"
        elif 36 <= fever < 38: return "36-38"
        elif 38 <= fever <= 40: return "38-40"
        return ">40"
    
    def _digestive_issues(self, digestive_issues):
        digestive_issues = digestive_issues.lower()
        if digestive_issues == "yes" or digestive_issues == "no":
            return digestive_issues
        else:
            if digestive_issues == "co": return "yes"
            return "no"
        
    def _chronic_pain(self, chronic_pain):
        chronic_pain = chronic_pain.lower()
        rate = ["luon luon", "lien tuc", "moi luc"]
        if ( chronic_pain == "thinh thoang" ): return "Occasionally"
        elif chronic_pain in rate : return "always"
        elif chronic_pain == "thuong xuyen": return "often"
        return "sometimes"
    
    def get_information(self):
        return [self.age, self.gender, self.debility, self.fever, self.digestive_issues, self.chronic_pain]



class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           
        self.entropy = entropy   
        self.depth = depth       
        self.split_attribute = None 
        self.children = children 
        self.order = None       
        self.label = None       

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        
        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, ids):
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids]
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        target_ids = [i + 1 for i in node.ids]  
        node.set_label(self.target[target_ids].mode()[0]) 
    
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue 
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
            if min(map(len, splits)) < self.min_samples_split: continue
            HxS= 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS 
            if gain < self.min_gain: continue 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes
    
    def predict(self, new_data):
        labels = None
        x = new_data.iloc[0, :]
        node = self.root
        while node.children: 
            value = x[node.split_attribute]
            node = node.children[node.order.index(value)]
        labels = node.label
        return labels

if __name__ == "__main__":
    df = pd.read_csv('diseases.csv', index_col = 0, parse_dates = True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTreeID3(max_depth = 6, min_samples_split = 2)
    tree.fit(X, y)

    data = []
    age = int(input("Vui lòng nhập tuổi của bạn: \n"))
    gender = input("Vui lòng nhập giới tính của bạn: \n")
    debility = input("Bạn có đang bị suy nhược cơ thể? (Yes or No) \n")
    fever = float(input("Nhiệt độ cơ thể của bạn là: \n"))
    digestive_issues = input("Bạn có bị vấn đề về đường tiêu hóa không (Yes or No): \n")
    chronic_pain = input("Bạn có hay bị các cơn đau kéo dài không: \n")
    
    info = Infomation(age,gender,debility,fever,digestive_issues,chronic_pain)

    data.append(info.get_information())
    res = pd.DataFrame(data, columns=["Age", "Gender", "Debility", "Fever", "Digestive_issues", "Chronic_pain"])
    time.sleep(1)
    print("--------LOADING-------- \n")
    time.sleep(2)
    print("Theo dự đoán của tôi \nKết quả ung thư của bạn là: \n")
    time.sleep(1)
    print(tree.predict(res))
