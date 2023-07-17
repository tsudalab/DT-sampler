import traceback
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os

@dataclass
class Node:
    id: int
    x: int = None  # input feature assigned to the node
    type_: str = None
    gini: float = None
    positive: int = 0
    negative: int = 0
    importance: float = None

class DecisionTree:
    def __init__(self):
        self.nodes = {}
        self.tree_structure = {}
        self.l_leaf = {}
        self.r_leaf = {}
        self.trained = False  # True if the tree has been fit
        self.l_leaf_stat = {}
        self.r_leaf_stat = {}
        
    def fit_solution(self, solution, X, y, is_sample_leaf):
        self.nodes = {}
        self.tree_structure = {}
        self.l_leaf = {}
        self.r_leaf = {}
        self.trained = True
        # build the decision tree
        try:
            v_var = solution['v']

            for k, v in v_var.items():
                self.nodes[k] = Node(k)
                self.tree_structure[k] = []
                self.nodes[k].type_ = v

            i_allocated = 1
            for k, v in v_var.items():
                if v == "A":
                    self.tree_structure[k].append(i_allocated+1)
                    self.tree_structure[k].append(i_allocated+2)
                    i_allocated += 2
                elif v == "B" or v == "C":
                    self.tree_structure[k].append(i_allocated+1)
                    i_allocated += 1

            a_var = solution['a']
            for k, v in a_var.items():
                self.nodes[k].x = v

            cl_var = solution['cl']
            for k, v in cl_var.items():
                if self.nodes[k].type_ == 'C' or self.nodes[k].type_ == 'D':
                    self.l_leaf[k] = v
                    self.l_leaf_stat[k] = [0,0]

            cr_var = solution['cr']
            for k, v in cr_var.items():
                if self.nodes[k].type_ == 'B' or self.nodes[k].type_ == 'D':
                    self.r_leaf[k] = v
                    self.r_leaf_stat[k] = [0,0]
                    #self.nodes[k].leaf = True

            self.trained = True

        except Exception as e:
            print("\n exception!!!!\n")
            traceback.print_exc()
            raise e

        data = np.concatenate([X,y.reshape(-1,1)], axis=1)
        for i, example in enumerate(data):
            self.update_one_item(example)
        
        if is_sample_leaf == False:
            for k,v in self.l_leaf.items():
                self.l_leaf[k] = 0 if self.l_leaf_stat[k][0]>=self.l_leaf_stat[k][1] else 1
            for k,v in self.r_leaf.items():
                self.r_leaf[k] = 0 if self.r_leaf_stat[k][0]>=self.r_leaf_stat[k][1] else 1

        self.compute_gini()

    def print_tree(self):
        print(self.nodes)
        print(self.tree_structure)
        print("left leaves: ", self.l_leaf)
        print("left leaf stat: ",self.l_leaf_stat)
        print("right leaves:", self.r_leaf)
        print("right leaf stat: ",self.r_leaf_stat)
        
    def predict(self, item):
        """ Predicts the class of the item passed as argument."""

        if not self.trained:
            raise ValueError('Classifier has not been trained or no solution have been found!')

        # create a dictionary of pairs (feature_number, feature_value)
        item_data = {i: item[i - 1] for i in range(1, len(item) + 1)}
        current_node = self.nodes[1]  # get the tree root
        
        while True:
            if current_node.x in item_data:
                if item_data[current_node.x] == 0:
                    # next node is left child
                    if current_node.type_ == 'C' or current_node.type_ == 'D':
                        y = self.l_leaf[current_node.id]
                        if sum(self.l_leaf_stat[current_node.id])==0:
                            prob = 0
                        else:
                            prob = max(self.l_leaf_stat[current_node.id]) / sum(self.l_leaf_stat[current_node.id])
                        break;
                    else:
                        next_node = self.nodes[self.tree_structure[current_node.id][0]]
                else:
                    if current_node.type_ == 'B' or current_node.type_ == 'D':
                        y = self.r_leaf[current_node.id]
                        if sum(self.r_leaf_stat[current_node.id]) == 0:
                            prob = 0
                        else:
                            prob = max(self.r_leaf_stat[current_node.id]) / sum(self.r_leaf_stat[current_node.id])
                        break;

                    # next node is right child
                    elif current_node.type_ == 'C':
                        next_node = self.nodes[self.tree_structure[current_node.id][0]]
                    else:
                        next_node = self.nodes[self.tree_structure[current_node.id][1]]

                current_node = next_node
        return y, prob
    
    def update_one_item(self, item):
        # create a dictionary of pairs (feature_number, feature_value)
        item_data = {i: item[i - 1] for i in range(1, len(item) + 1)}
        current_node = self.nodes[1]  # get the tree root
        
        while True:
            if current_node.x in item_data:
                if item[-1] == 0:
                    self.nodes[current_node.id].negative += 1
                else:
                    self.nodes[current_node.id].positive += 1
                if item_data[current_node.x] == 0:
                    # next node is left child
                    if current_node.type_ == 'C' or current_node.type_ == 'D':
                        y = self.l_leaf[current_node.id]
                        if current_node.id not in self.l_leaf_stat:
                            self.l_leaf_stat[current_node.id] = [0,0]
                        if item[-1]==0:
                            self.l_leaf_stat[current_node.id][0] += 1
                        else:
                            self.l_leaf_stat[current_node.id][1] += 1
                        break;
                    else:
                        next_node = self.nodes[self.tree_structure[current_node.id][0]]
                else:
                    if current_node.type_ == 'B' or current_node.type_ == 'D':
                        y = self.r_leaf[current_node.id]
                        if current_node.id not in self.r_leaf_stat:
                            self.r_leaf_stat[current_node.id] = [0,0]
                        if item[-1]==0:
                            self.r_leaf_stat[current_node.id][0] += 1
                        else:
                            self.r_leaf_stat[current_node.id][1] += 1
                        break;

                    # next node is right child
                    elif current_node.type_ == 'C':
                        next_node = self.nodes[self.tree_structure[current_node.id][0]]
                    else:
                        next_node = self.nodes[self.tree_structure[current_node.id][1]]
                current_node = next_node

    def compute_gini(self):
        for i in range(1, len(self.nodes)+1):
            if self.nodes[i].positive == 0 or self.nodes[i].negative == 0:
                self.nodes[i].gini = 0
            else:
                self.nodes[i].gini = 1 - (self.nodes[i].positive**2+self.nodes[i].negative**2)/(self.nodes[i].positive+self.nodes[i].negative)**2
        
        importance_sum = 0
        for i in range(1, len(self.nodes)+1):
            N_parent = self.nodes[i].positive+self.nodes[i].negative
            if i in self.r_leaf_stat:
                tree_right_n = self.r_leaf_stat[i][0]
                tree_right_p = self.r_leaf_stat[i][1]
                if tree_right_n == 0 or tree_right_p==0:
                    tree_right_gini = 0
                else:
                    tree_right_gini = 1-(tree_right_n**2+tree_right_p**2)/((tree_right_n+tree_right_p)**2)
            else:
                tree_right_n, tree_right_p = 0,0
                tree_right_gini = 0
                
            if i in self.l_leaf_stat:
                tree_left_n = self.l_leaf_stat[i][0]
                tree_left_p = self.l_leaf_stat[i][1]
                if tree_left_n == 0 or tree_left_p==0:
                    tree_left_gini = 0
                else:
                    tree_left_gini = 1-(tree_left_n**2+tree_left_p**2)/((tree_left_n+tree_left_p)**2)
            else:
                tree_left_n, tree_left_p = 0,0
                tree_left_gini = 0
            
            if self.nodes[i].type_ == 'A':
                tree_left = self.nodes[self.tree_structure[i][0]]
                tree_right = self.nodes[self.tree_structure[i][1]]
                self.nodes[i].importance = N_parent*self.nodes[i].gini-(tree_left.positive+tree_left.negative)*tree_left.gini-(tree_right.positive+tree_right.negative)*tree_right.gini
            elif self.nodes[i].type_ == 'B': 
                tree_left = self.nodes[self.tree_structure[i][0]]
                self.nodes[i].importance = N_parent*self.nodes[i].gini-(tree_left.positive+tree_left.negative)*tree_left.gini-(tree_right_n+tree_right_p)*tree_right_gini
            elif self.nodes[i].type_ == 'C': 
                tree_right = self.nodes[self.tree_structure[i][0]]
                self.nodes[i].importance = N_parent*self.nodes[i].gini-(tree_right.positive+tree_right.negative)*tree_right.gini-(tree_left_n+tree_left_p)*tree_left_gini
            else:  
                self.nodes[i].importance = N_parent*self.nodes[i].gini-(tree_right_n+tree_right_p)*tree_right_gini-(tree_left_n+tree_left_p)*tree_left_gini
            importance_sum += self.nodes[i].importance

        for i in range(1, len(self.nodes)+1):
            self.nodes[i].importance /= importance_sum
                