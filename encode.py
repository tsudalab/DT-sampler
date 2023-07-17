import matplotlib.pyplot as plt
import re
from z3 import *
from math import floor, ceil
import traceback
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.special import gamma, factorial
import time
import pyunigen
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from pyunigen import Sampler
import os

set_option(max_args=100000000, max_lines=100000000, max_depth=100000000, max_visited=100000000)
set_param('parallel.enable', True)

# Global variables to be used later on.
#var = {}  # to store the variables of the csp
#s = None

def set_csp(pos_x, neg_x, n, k, true_n, export_path):
    n_b = int((n - 1) / 2)   # the number of branch nodes

    global var
    var = {}

    def get_child(i):
        """
        Given a node i, returns all its possible branch children.
        get_child(i) = [i + 1: min(2i+1, n_b)]), i = 1,...,n_b-1
        """
        return tuple([_ for _ in range(i + 1, min(2 * i + 1, n_b) + 1)])

    
    # variables "vl" and "vr"
    # "vl_i" is true iff branch node i has a left branch child.
    # "vr_i" is true iff branch node i has a right branch child.
    for i in range(1, n_b + 1):
        var['VL%i' % i] = Bool( 'VL%i' % i)
        var['VR%i' % i] = Bool( 'VR%i' % i)

    # variables "cl" and "cr"
    # "cl_i" is true iff class of the left leaf child of branch node i is 1.
    # "cr_i" is true iff class of the right leaf child of branch node i is 1.
    for i in range(1, n_b + 1):
        var['cl%i' % i] = Bool('cl%i' % i)
        var['cr%i' % i] = Bool('cr%i' % i)

    if n_b!=1:
        # variable "l" 
        # "l_ij" is true iff node i has node j as left child
        for i in range(1, n_b + 1):
            for j in get_child(i):
                var['l%i,%i' % (i, j)] = Bool('l%i,%i' % (i, j))

        # variable "r" 
        # "r_ij" is true iff node i has node j as right child
        for i in range(1, n_b + 1):
            for j in get_child(i):
                var['r%i,%i' % (i, j)] = Bool('r%i,%i' % (i, j))

    # variable "a"
    # "a_rj" is true iff feature r is assigned to node j
    for j in range(1, n_b + 1):
        for i in range(1, k + 1):
            var['a%i,%i' % (i, j)] = Bool('a%i,%i' % (i, j))

    # variable "u"
    # "u_rj" is true iff feature r is being discriminated against by node j
    for i in range(1, k + 1):
        for j in range(1, n_b + 1):
            var['u%i,%i' % (i, j)] = Bool('u%i,%i' % (i, j))

    if true_n != 0:
        # variable "d0"
        # "d0_rj" is true iff feature r is discriminated for value 0 by node j or by one of its ancestors
        for i in range(1, k + 1):
            for j in range(1, n_b + 1):
                var['d0%i,%i' % (i, j)] = Bool('d0%i,%i' % (i, j))

        # variable "d1"
        # "d1_rj" is true iff feature r is discriminated for value 1 by node j or by one of its ancestors
        for i in range(1, k + 1):
            for j in range(1, n_b + 1):
                var['d1%i,%i' % (i, j)] = Bool('d1%i,%i' % (i, j))

        if true_n != len(pos_x)+len(neg_x):
            # variable "Aux"
            # "Aux_i" is true iff example i is classfied correctly.
            for j in range(len(neg_x)):
                var['Aux0%i'%j] = Bool('Aux0%i'%j)
            for j in range(len(pos_x)):
                var['Aux1%i'%j] = Bool('Aux1%i'%j)

    global var_dic
    var_dic = {}
    var_list = list(var.keys())
    
    index = 1
    for v in var_list:
        var_dic[v] = str(index)
        index += 1

    global var_num
    var_num = index

    global s
        
    if export_path != None:
        s = Goal()
    else:
        s = Solver()
    

    # Constraints of tree structure
    # Constraint 1: We use "vl_i" (resp. "vr_i") to denote whether the i_th node has a left (resp. right) branch child or not.
    for j in range(1, n_b + 1):
        s.add(Implies(var['VL%i' % j], Not(var['cl%i' % j])))
        s.add(Implies(var['VR%i' % j], Not(var['cr%i' % j])))
    
    for i in range(1, n_b + 1):
        #left
        sum_list1 = []
        for j in get_child(i):
            sum_list1.append(var['l%i,%i' % (i, j)])

        if len(sum_list1) == 0:
            s.add(Not(var['VL%i' % i]))
            continue;
  
        f = z3.PbEq([(x,1) for x in sum_list1], 1)
        s.add(Implies(var['VL%i' % i], f))
        s.add(Implies(Not(var['VL%i' % i]), Not(Or(sum_list1))))
    
    for i in range(1, n_b + 1):
        #right
        sum_list2 = []
        for j in get_child(i):
            sum_list2.append(var['r%i,%i' % (i, j)])

        if len(sum_list2) == 0:
            s.add(Not(var['VR%i' % i]))
            continue;

        f = z3.PbEq([(x,1) for x in sum_list2], 1)
        s.add(Implies(var['VR%i' % i], f))
        s.add(Implies(Not(var['VR%i' % i]), Not(Or(sum_list2))))


    # Constraint 2: any branch nodes can only have one parent
    for j in range(2, n_b+1):
        l = []
        for i in range(floor(j/2), j):
            if "l%i,%i" % (i, j) in var:
                l.append(var["l%i,%i"%(i, j)])
                l.append(var["r%i,%i"%(i, j)])
        if len(l) <= 1:
            continue;
        s.add(z3.PbEq([(x,1) for x in l], 1))

    # Constraint 3 : The IDs of branch nodes are assigned according to level order of the tree.
    for i in range(1, n_b):
        for j in range(2, n_b+1):
            l1 = []
            l2 = []
            if "l%i,%i" % (i, j) in var:
                for ii in range(1, i):
                    for jj in range(j, n_b+1):
                        if "l%i,%i"%(ii,jj) in var:
                            l1.append(var["l%i,%i"%(ii,jj)])
                        if "r%i,%i"%(ii,jj) in var:
                            l1.append(var["r%i,%i"%(ii,jj)])
                if len(l1) > 0:
                    s.add(Implies(var["l%i,%i" % (i, j)], Not(Or(l1))))

            if "r%i,%i" % (i, j) in var:
                for ii in range(1, i+1):
                    for jj in range(j, n_b+1):
                        if i == ii and j == jj:
                            if "l%i,%i"%(ii,jj) in var:
                                l2.append(var["l%i,%i"%(ii,jj)])
                            continue;
                        if "l%i,%i"%(ii,jj) in var:
                            l2.append(var["l%i,%i"%(ii,jj)])
                        if "r%i,%i"%(ii,jj) in var:
                            l2.append(var["r%i,%i"%(ii,jj)])
                if len(l2) > 0:
                    s.add(Implies(var["r%i,%i" % (i, j)], Not(Or(l2))))


    # Constraints of learning
    # Constraint 4: a branch node is assigned exactly one feature
    for j in range(1, n_b + 1):
        sum_list = []
        for r in range(1, k + 1):
            sum_list.append(var['a%i,%i' % (r, j)])
        
        s.add(z3.PbEq([(x,1) for x in sum_list], 1))


    # Constraint 5: Variable "u_rj" has the information of whether the r_th feature is discriminated at any node on the path from the root to this node.
    for r in range(1, k + 1):
        s.add(var['u%i,%i' % (r, 1)] == var['a%i,%i' % (r, 1)])
        for j in range(2, n_b + 1):
            and_list = []
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1:  
                    if j in get_child(i):
                        pji = Or([var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j)]])
                    and_list.append(Implies(And([var['u%i,%i' % (r, i)], pji]), Not(var['a%i,%i' % (r, j)])))
                    or_list.append(And([var['u%i,%i' % (r, i)], pji]))
            s.add(And(and_list))
            s.add(var['u%i,%i' % (r, j)] == Or([var['a%i,%i' % (r, j)], *or_list]))
    
    if true_n == 0:
        return
    # Constraint 6: to track if the r_th feature was discriminated negatively along the path from the root to j_th node
    for r in range(1, k + 1):
        s.add(Not(var['d0%i,%i' % (r, 1)]))  # d0r,1 = 0

        for j in range(2, n_b + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1 and 'r%i,%i' % (i, j) in var:
                    if j in get_child(i):
                        pji = Or([var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j)]])
                    or_list.append(Or([And([pji, var['d0%i,%i' % (r, i)]]), And([var['a%i,%i' % (r, i)], var['r%i,%i' % (i, j)]])]))

            s.add(var['d0%i,%i' % (r, j)] == Or(or_list))

    # Constraint 7: to track if the r_th feature was discriminated positively along the path from the root to j_th node
    for r in range(1, k + 1):
        s.add(Not(var['d1%i,%i' % (r, 1)]))  # d1r,1 = 0
        for j in range(2, n_b + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1 and 'l%i,%i' % (i, j) in var:
                    if j in get_child(i):
                        pji = Or([var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j)]])
                    or_list.append(Or([And([pji, var['d1%i,%i' % (r, i)]]), And([var['a%i,%i' % (r, i)], var['l%i,%i' % (i, j)]])]))

            s.add(var['d1%i,%i' % (r, j)] == Or(or_list))
  

    # Constraint 8: constrain the accuracy
    judge_list = []
    for i, example in enumerate(neg_x):
        f = []
        for j in range(1, n_b + 1):
            or_list = []
            or_list1 = []
            or_list2 = []
            for r in range(1, k + 1):
                if example[r - 1] == 0:  
                    if n_b > 1:
                        or_list.append(var['d0%i,%i' % (r, j)]) 
                    or_list1.append(var['a%i,%i' % (r, j)]) 
                else:
                    if n_b > 1:
                        or_list.append(var['d1%i,%i' % (r, j)]) 
                    or_list2.append(var['a%i,%i' % (r, j)])

            f1 = Or([*or_list, *or_list1])
            f2 = Or([*or_list, *or_list2])
            f.append(Implies(And([var['VL%i' % j], Not(var['VR%i' % j]), var['cr%i' % j]]), f1))
            f.append(Implies(And([Not(var['VL%i' % j]), Not(var['VR%i' % j]), var['cr%i' % j]]), f1))
            f.append(Implies(And([Not(var['VL%i' % j]), var['VR%i' % j],var['cl%i' % j]]), f2))
            f.append(Implies(And([Not(var['VL%i' % j]), Not(var['VR%i' % j]), var['cl%i' % j]]), f2))

        if true_n == len(pos_x)+len(neg_x):
            s.add(And(f))
        else:
            s.add(And(f)==var['Aux0%i' % i])
            judge_list.append(var['Aux0%i' % i])

    for i, example in enumerate(pos_x):
        f=[]
        for j in range(1, n_b + 1):
            or_list = []
            or_list1 = []
            or_list2 = []
            for r in range(1, k + 1):
                if example[r - 1] == 0:
                    if n_b > 1:
                        or_list.append(var['d0%i,%i' % (r, j)])  
                    or_list1.append(var['a%i,%i' % (r, j)])  
                else:
                    if n_b > 1:
                        or_list.append(var['d1%i,%i' % (r, j)]) 
                    or_list2.append(var['a%i,%i' % (r, j)])  

            f1 = Or([*or_list, *or_list1])
            f2 = Or([*or_list, *or_list2])
            f.append(Implies(And([var['VL%i' % j],Not(var['VR%i' % j]), Not(var['cr%i' % j])]), f1))
            f.append(Implies(And([Not(var['VL%i' % j]), Not(var['VR%i' % j]), Not(var['cr%i' % j])]), f1))
            f.append(Implies(And([Not(var['VL%i' % j]), var['VR%i' % j], Not(var['cl%i' % j])]), f2))
            f.append(Implies(And([Not(var['VL%i' % j]), Not(var['VR%i' % j]), Not(var['cl%i' % j])]), f2))
        if true_n == len(pos_x)+len(neg_x):
            s.add(And(f))
        else:
            s.add(And(f)==var['Aux1%i' % i])
            judge_list.append(var['Aux1%i' % i])

    if true_n != len(pos_x)+len(neg_x): 
        s.add(z3.PbGe([(x,1) for x in judge_list], true_n))
        
def to_CNF(cnf):
    if type(cnf) != list:
        l = eval(cnf.strip('\n'))
    else:
        l = cnf
    index = 0
    c = ""
    while index < len(l):
        if l[index] == "or":
            c += to_CNF(l[index + 1]) + " "
            index += 2
        elif l[index] == "not":
            c += "-" + l[index + 1][0] + " "
            index += 2
        else:
            c += l[index] + " "
            index += 1
    return c

def export_CNF(cnf, filename, is_leaf_sampling):
    def match_func(matchobj):
        global var_dic
        return "\"" + str(var_dic[matchobj.group(0)]) + "\""

    def match_func2(matchobj):
        global var_num
        match = matchobj.group(0)
        for i in range(len(match)):
            if match[i] == "!":
                s = match[i+1:]
        return "\"" + str(int(s)+var_num) + "\""
    
    max_n = 0
    count = 0
    cnf_file = ""

    ind = []
    if is_leaf_sampling:
        for k,v in var_dic.items():
            if k.startswith("VL") or k.startswith("VR") or k.startswith("a") or k.startswith("cr") or k.startswith("cl") :
                ind.append(v)
    else:
        for k,v in var_dic.items():
            if k.startswith("VL") or k.startswith("VR") or k.startswith("a"):
                ind.append(v)
            
    with open(filename, 'w') as f:
        f.write('c ' + str(var_dic) + '\n')
        for line in cnf:
            count+=1
            global cnf_str
            cnf_str = ''
            cnf_str = re.sub('(VL|VR|cl|cr|Aux0|Aux1)[0-9]+|((l|r|p|a|u|d0|d1)[0-9]+,[0-9]+)', match_func, str(line).replace('\n', ' '))
            cnf_str = re.sub('[^()\s]+![0-9]+', match_func2, cnf_str)
            cnf_str = cnf_str.replace(" ", "")
            cnf_str = cnf_str.replace("(", "[")
            cnf_str = cnf_str.replace(")", "]")
            cnf_str = cnf_str.replace("Or", '"or", ')
            cnf_str = cnf_str.replace("Not", '"not", ')
            cnf_str = '[' + cnf_str + ']'
            cnf_str = to_CNF(cnf_str).strip(' ') + ' 0\n'
            if cnf_str == '':
                print("error" + str(line))
            max_n = max(max_n, max([abs(int(x)) for x in cnf_str.split(" ")]))
            
            cnf_file += cnf_str
        f.write("c ind "+ " ".join([str(x) for x in ind])+" 0\n")
        f.write("p cnf " + str(max_n) + " " + str(count)+"\n")
        f.write(cnf_file)


def get_solution(x_values, y_values, target_nodes, true_n, export_path, is_leaf_sampling, seed=0, pre_sol=None):
    set_option('smt.random_seed', seed )

    n = target_nodes  

    # select only the rows where the target feature equals 1
    pos_x = x_values[y_values.astype(np.bool), :]

    # select only the rows where the target feature equals 0
    neg_x = x_values[~y_values.astype(np.bool), :]

    k = len(x_values[0])
    set_csp(pos_x, neg_x, n, k, true_n, export_path)

    global s
    global var

    if pre_sol != None:
        for sol_ in pre_sol:
            exp = []
            for k, v in var.items():
                if is_true(sol_.eval(v)) == 1:
                    exp.append(v)
                else:
                    exp.append(Not(v))
            s.add(Not(And(exp)))

    if export_path != None:
        #tac = Then('simplify', 'bit-blast', 'tseitin-cnf')
        tac = Then( 'simplify', 'card2bv', 'simplify', 'bit-blast', 'tseitin-cnf')
        cnf = tac(s)[0]
        export_CNF(cnf, export_path, is_leaf_sampling)
        return True

    status = s.check()

    if status == z3.sat:

        m = s.model()
        a_var = {}
        cl_var = {}
        cr_var = {}
        vl_var = []
        vr_var = []


        for k, v in var.items():
            try:
                if k.startswith('VL') and is_true(m.eval(v)) == 1:
                    vl_var.append(int(k[2:]))
                elif k.startswith('VR') and is_true(m.eval(v)) == 1:
                    vr_var.append(int(k[2:]))
                elif k.startswith('a') and is_true(m.eval(v)) == 1:
                    feature = k[1:].partition(',')[0]
                    node = k[1:].partition(',')[2]
                    a_var[int(node)] = int(feature)
                elif k.startswith('cl'):
                    cl_var[int(k[2:])] = 1 if is_true(m.eval(v)) else 0
                elif k.startswith('cr'):
                    cr_var[int(k[2:])] = 1 if is_true(m.eval(v)) else 0

            except Exception as e:
                traceback.print_exc()
                raise e
        
        v_var = {}
        for i in range(1,len(cl_var)+1):
            if i in vl_var and i in vr_var:
                v_var[i] = 'A'
            if i in vl_var and i not in vr_var:
                v_var[i] = 'B'
            if i not in vl_var and i in vr_var:
                v_var[i] = 'C'
            if i not in vl_var and i not in vr_var:
                v_var[i] = 'D'
        
        solution = {'v': v_var, 'a':a_var, 'cl': cl_var, 'cr':cr_var}
    else:
        return None
    
    if pre_sol != None:
        return solution, m
    return solution

if __name__ == '__main__':
    print("cpu: ", os.uname()[1])

    data = pd.read_csv('./mouse.csv',delimiter=',', header=None).to_numpy()
    
    X = data[:,:-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 50, random_state=2)
    
    sk = SelectKBest(chi2, k=10).fit(X_train, y_train)
    X_train = sk.transform(X_train)
    X_test = sk.transform(X_test)
    
    print("X_train:", X_train.shape)

    start = time.time()
    sol = get_solution(X_train, y_train, 9, 45 , None)#export_path="test.cnf")
    print(sol)
    print("time:", time.time()-start)