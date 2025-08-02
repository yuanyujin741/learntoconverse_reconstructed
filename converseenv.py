# -*- coding: utf-8 -*-
"""
代码的主要修改都记录在这里，方便后面修改增进。
2025/5/17修改：
    改了一下午怎么也找不到bug，结果发现是python3转python2的时候出问题了；
    证据：使用aifc(python3环境)以及pentropy_v3(python3代码)，得到的结果是正确的: rewardlist: [0.0, 0.161408828237398, 1.7763568394002505e-15, -1.7763568394002505e-15, 0.0, 0.0, 0.0].
    方案：寻找新的3to2的工具，实现python3转python2的功能。
    注意：发现使用其他工具也解决不聊这个问题。可能是python环境与gurobi的建通问题导致的。
2025/5/27修改：
    现在新添加一个关于自动由NK求解得到其他超参数的函数。
2025/6/15修改：
    现在修改Regions2Matrix的计算方法，也就是说，使用entropydict_all实现计算，而不是直接使用entropydict。
2025/6/17修改：
    使用expr2vec进行嵌入，方便进一步处理。这里只测试嵌入的实现，再训练之前测试expr2vec关于模型加载的能力，避免多次加载模型。
    （按照我的理解应该是不会重复加载的，应该是只会加载3次，也就是每个模型加载一次。）
"""
# 库导入
import numpy as np
from helpful_files.pentropy_v3 import *
import itertools
import json
import copy
import random
import time
from collections import defaultdict, Counter
from gurobipy import Model, GRB, LinExpr, GurobiError
import gurobipy as gp
import os
import pdb
from datetime import datetime
# from hyperparams_new.RF_regression import predict_wrapper # 链接不上但是可以正常使用这个。
from helpful_files.expr2vec import *
#random.seed(42)
#np.random.seed(42)

# 全局变量导入
RENDER = "non-selftest" # 是否在每次step和reset的时候渲染。
episode = 0 # 全局变量继续保留，但是添加一个新的innerepisode变量，用于记录每个环境的episode的次数。
point_num = 12 # sample per 1, at draw.
GENERATE_SIZE = 30 # original: 30
generate_size = GENERATE_SIZE # z's num
subset_size = 30 # z's num from original vars.
comb_size = 150
NEWINEQ_NUM = 1000 # get NEWINEQ_num ineqs from candidate regions.默认值。
entropydict = EntropyEqDict()
entropydict_all = EntropyEqDict()
effective_vars = []
globalN = 5
globalK = 5
total_time = 0.0
total_exprs = 0

# 需要用到的函数
def gurobi_solver(ent_num, ine_constraints, regions,ori_obj_coef,effective_idx_gurobi):
    try:
        ine_list = []
        # global result_slope
        # 创建 Gurobi 模型
        model = Model("entropy_minimization")

        # 禁用输出日志（可选）
        model.setParam("OutputFlag", 0)

        # 创建变量
        variables = []
        for i in range(ent_num):
            var = model.addVar(name="V{}".format(i), lb=0)
            variables.append(var)
        obj_expr = gp.quicksum(ori_obj_coef * variables)
        model.setObjective(obj_expr, GRB.MINIMIZE)
        # model.setObjective(variables[-1], GRB.MINIMIZE)
        # var_names = ["V" + str(i) for i in range(ent_num)]
        # variables = model.addVars(var_names, lb=0, obj=[0.0] * (ent_num), name="V")
        # variables[var_names[-1]].obj = 1.0  # 设置目标函数的系数
        # model.setObjective(variables[var_names[-1]], GRB.MINIMIZE)

        # 添加不等式约束
        for ine in ine_constraints[:-1]:
            model.addConstr(LinExpr(ine[:-1], variables) >= ine[-1])

        # 添加等式约束 M = M_value
        M_cons = ine_constraints[-1]
        model.addConstr(LinExpr(M_cons[:-1], variables) == M_cons[-1])
        # 添加等式约束 M = M_value
        # model.addConstr(variables[var_names[-2]] == M_value)

        model.optimize()

        # 检查求解状态
        if model.status == GRB.OPTIMAL:
            dual_values = model.getAttr('Pi', model.getConstrs())

            indices = [i for i, val in enumerate(dual_values) if abs(val) > 1e-5 and i < len(regions.exprs)]
            # slope = dual_values[-1]
            # result_slope.append(slope)

            for index in indices:
                if index not in effective_idx_gurobi:
                    effective_idx_gurobi.append(index)
                    effective_idx_gurobi.sort()
            
            # 获取目标函数值
            optimal_value = model.objVal
            # print(f"Optimal value: {optimal_value}")
        
            return optimal_value
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Trying to find the IIS...")
            model.computeIIS()
            model.write("model.ilp")
            print("IIS written to model.ilp. Check this file for conflicting constraints.")
            for c in model.getConstrs():
                if c.IISConstr:
                    print("约束 {} 导致不可行".format(c.constrName))
                    ine_list.append(c.constrName)
            for v in model.getVars():
                if v.IISLB or v.IISUB:
                    print("变量 {} 的边界条件导致不可行".format(v.varName))
            return ine_list
        else:
            # print(f"Model status: {model.status}")
            return None
    except GurobiError as e:
        # print(f"Gurobi Error: {e}")
        return None
def dual_solver(prob_cons_num,expr_num,dual_obj_coef,trans_ine_cons):
    """
    作用: 二次规划求解器, 输入为主问题的线性规划模型, 输出为主问题的最优解和可行解的索引

    :return: 主问题的最优解和可行解的索引
    :rtype: tuple
    
    """
    try:
        effective_idx_dual = []
        # 创建 Gurobi 模型
        model = Model("secondary LP")

        # 启用输出日志
        model.setParam("OutputFlag", 0)

        # 创建变量
        variables = []
        for i in range(expr_num):
            var = model.addVar(name="Y{}".format(i), lb=0)
            variables.append(var)
        var_z = model.addVar(name="Z",lb=-GRB.INFINITY, ub=GRB.INFINITY)
        variables.append(var_z)
        # print("expr_num",expr_num)
        # print("len_var",len(variables))
        # objective_expr = gp.quicksum(coef_dual[i] * variables[i] for i in range(expr_num)) + M_value * variables[-1]
        objective_expr = gp.quicksum(dual_obj_coef * variables)
        # objective_expr = gp.quicksum(variables)
        model.setObjective(objective_expr, GRB.MAXIMIZE)

        # 添加不等式约束
        for ine in trans_ine_cons:
            model.addConstr(LinExpr(ine[:-1], variables) == ine[-1])

        # 添加等式约束 M = M_value
        # model.addConstr(variables[-1] == M_value)

        model.optimize()

        # 检查求解状态
        if model.status == GRB.OPTIMAL:
            # 获取目标函数值
            optimal_value = model.objVal
            # print(f"Optimal value: {optimal_value}")

            # 获取变量的最优解
            solution_values = {var.varName: var.x for var in model.getVars()}
            # print(f"Solution: {solution_values}")

            for var_name, var_value in solution_values.items():
                if var_value > 0:
                    index = int(var_name[1:])
                    if index < expr_num - prob_cons_num and index is not None:
                        effective_idx_dual.append(index)
            return solution_values, effective_idx_dual
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Trying to find the IIS...")
            model.computeIIS()
            model.write("model.ilp")
            print("IIS written to model.ilp. Check this file for conflicting constraints.")
            return None,None
        else:
            print("Model status: {}".format(model.status))
            return None,None
    except GurobiError as e:
        print("Gurobi Error: {}".format(e))
        return None,None
def find_min_effective_indices(dual_value,regions):
    """
    用来找到全部的直线的有效不等式索引, 已经去重和排序.

    :param dual_value: 双变量的系数矩阵
    :return: 有效不等式索引的列表

    """
    non_zero_counts = np.count_nonzero(dual_value, axis=1)
    old_slope = 0
    single_cut_idx = []
    all_cut_idx = set()
    old_slope = dual_value[0][-1]
    min_row = 0
    min_cnt = non_zero_counts[0]
    for idx,row in enumerate(dual_value):
        if row[-1] != old_slope:
            for i,value in enumerate(dual_value[min_row]):
                if value > 0 and i < len(regions.exprs):
                    single_cut_idx.append(i)
            # print(f"slope:{old_slope},index{single_cut_idx}")
            all_cut_idx = all_cut_idx.union(set(single_cut_idx))
            old_slope = row[-1]
            min_cnt = non_zero_counts[idx]
            min_row = idx
            single_cut_idx = []
            continue
        if non_zero_counts[idx] < min_cnt:
            min_cnt = non_zero_counts[idx]
            min_row = idx
    for i,value in enumerate(dual_value[min_row]):
        if value > 0 and i < len(regions.exprs):
            single_cut_idx.append(i)
    all_cut_idx = all_cut_idx.union(set(single_cut_idx))
    # print(f"slope:{old_slope},index{single_cut_idx}")
    # print("all",all_cut_idx)
    return sorted(list(all_cut_idx))
def calculate_square(plot_data):
    """
    输入要绘制的点， 就可以得到对应的面积了（也就是折线下面的面积啊）。
    """
    x_data = [item[0] for item in plot_data]
    y_data = [item[1] for item in plot_data]
    square = 0
    for i in range(1,len(x_data)):
        square += (x_data[i] - x_data[i-1]) * (y_data[i] + y_data[i-1])
    return square / 2
def Regions2Matrix(entropydict, regions, forRL = False):
    """
    Convert regions to a matrix in the form of list.
    主要修改在这里，也就是直接使用entropydict_all。
    : param: forRL 判断是否是用于强化学习的筛选过程的，如果是，那么使用的是entropydict，所以说
    """
    ine_constraints = []
    ent_num = len(entropydict.redict) + 3
    if forRL:
        ent_num = entropydict.max_index() + 1 + 3 # +1, for start from 0; +3 for M、R、Constant
    for expr in regions.exprs:
        row = [0] * ent_num
        for term in expr.terms:
            row[entropydict[term.to_ent_str()]] = term.coef
        row[-1] = expr.value
        non_zero_values = [i for i in row if i != 0]
        non_zero_count = len(non_zero_values)
        if non_zero_count == 0:
            print("0 row")
            print(expr)
        elif sum(row[:-1]) < 0:
            print("negative row")
            print(expr)
        else:
            ine_constraints.append(row)
    return ine_constraints
def AddProblemConstraints2Matrix(Xrvs_cons, Wrvs_cons, entropydict, ine_constraints, ent_num):
    # R >= H(X)
    prob_cons_num = 0
    for key in Xrvs_cons:
        if entropydict.get(key) != None:
            row3 = [0] * ent_num
            row3[entropydict[key]] = -1
            row3[-2] = 1
            ine_constraints.append(row3)
            prob_cons_num += 1

    # M >= H(Z)
    row5 = [0] * ent_num
    row5[entropydict["Z1"]] = -1
    row5[-3] = 1
    ine_constraints.append(row5)
    prob_cons_num += 1

    # H(W1,..,Wn) >= n
    for key in Wrvs_cons:
        if entropydict.get(key) != None:
            rvs = key.split(",")
            row3 = [0] * ent_num
            row3[entropydict[key]] = 1
            row3[-1] = len(rvs)
            # print(row3)
            ine_constraints.append(row3)
            prob_cons_num += 1

    # M = M_value
    row5 = [0] * ent_num
    ine_constraints.append(row5)

    return ine_constraints, prob_cons_num
# 首先引入一些cutsetbound需要的函数
def preprocessing_single(x_vars,y_vars,z_vars,expand_vars):
    for index,x in enumerate(x_vars):
        y = y_vars[index]
        new_var = x + y
        new_var = Iutils.sort_elements(new_var)
        if new_var not in expand_vars:
            expand_vars.append(new_var)
        
        if len(z_vars) != 0:
            z = z_vars[index]
            if len(z) != 0:
                new_var = x + z
                new_var = Iutils.sort_elements(new_var)
                if new_var not in expand_vars:
                    expand_vars.append(new_var)
                
                new_var = x + y + z
                new_var = Iutils.sort_elements(new_var)
                if new_var not in expand_vars:
                    expand_vars.append(new_var)
def generate_single_inequalities(ix_vars,iy_vars,iz_vars,hx_vars,hy_vars,entropydict,regions):
    """
        更新不等式集regions，由变量生成不等式
    
        :param vars: I(x;y|z)中的z集合
        :param single_vars: 单变量集合，用于生成排列组合(x,y)
        :param  entropydict: 熵字典，存储所有互信息扩展后的联合熵变量的值，
        保证优化变量与不等式矩阵一一对应
        :param regions: 不等式集，调用函数前为上一轮迭代得到的有效不等式，
        调用函数后增加新变量生成的不等式

    """
    # generate I(x;y) & I(x;y|z)
    for index,x in enumerate(ix_vars):
        y = iy_vars[index]
        x = Ivar.rv(x)
        y = Ivar.rv(y)
        
        # print("XY",x,y,z)
        term_x = Comp(set(x))
        term_y = Comp(set(y))
        # I(x;y)
        if len(iz_vars) == 0:
            term = Term.I(term_x, term_y)
        
        # I(x;y|z) or I(x;y)
        else:
            z = iz_vars[index]
            if len(z) == 0:
                term = Term.I(term_x, term_y)
            else:
                z = Ivar.rv(z)
                term_z = Comp(set(z))
                term = Term.Ic(term_x, term_y, term_z)
            
        terms = [term]
        print(term)
        expr = Expr.empty()
        regions.append_expr(expr.inequality(terms=terms, edict=entropydict))

    # generate H(x;y)
    for index,x in enumerate(hx_vars):
        y = hy_vars[index]
        x = Ivar.rv(x)
        y = Ivar.rv(y)
        
        term_x = Comp(set(x))
        term_y = Comp(set(y))
        term = Term.Hc(term_x, term_y)

        terms = [term]
        print(term)
        expr = Expr.empty()
        regions.append_expr(expr.inequality(terms=terms, edict=entropydict))
def create_cutset_bound(N,K,user_perm,file_perm,Wkey,vars):
    regions = Region.empty()
    entropydict = EntropyEqDict() # 因为一开始就是这样初始化的所以就不输入一个region了。

    ix_vars = []
    iy_vars = []
    iz_vars = []
    hx_vars = []
    hy_vars = []

    for s in range(1,min(N,K)+1):
        print(s)
        X_list = []
        coef = N // s
        i = 0
        for num in range(coef):
            X_comb = []
            for j in range(K):
                X_comb.append(((i + j) % N) + 1)
            X_comb_str = "".join(map(str,X_comb))
            X_list.append("X" + X_comb_str)
            i += s
        Z_list = ["Z" + str(i) for i in range(1,s+1)]
        cutset_list = X_list + Z_list

        print(cutset_list)

        for index,var in enumerate(cutset_list):
            if index != 0:
                ix_vars.append([var])
                if [var] not in vars:
                    vars.append([var])
                y_var = cutset_list[:index]
                iy_vars.append(y_var)
                if y_var not in vars:
                    vars.append(y_var)
        
        hx_vars.append(cutset_list)
        keyZ_list = [elem for elem in cutset_list if elem.startswith("Z")]
        keyX_list = [elem for elem in cutset_list if elem.startswith("X")]
        keyW_new_set = set()
        for keyZ in keyZ_list:
            userindex = int(keyZ[1])
            fileindex_set = set(['W'+ s[userindex] for s in keyX_list])
            keyW_new_set = keyW_new_set.union(fileindex_set)
        hy_vars.append(list(keyW_new_set))
        vars.append(list(keyW_new_set))

        print("cutset_list",cutset_list)
        print("ix_vars",ix_vars)
        print("iy_vars",iy_vars)
        print("hx_vars",hx_vars)
        print("hy_vars",hy_vars)

    print(len(vars))
    print(vars)

    expand_vars = vars[:]
    preprocessing_single(ix_vars+hx_vars,iy_vars+hy_vars,iz_vars,expand_vars)
    print("len of vars",len(vars))
    print("len of expand vars",len(expand_vars))
    # print("expand_vars",expand_vars)
    Xrvs_cons = []
    Wrvs_cons = []
    Iutils.symmetrize(user_perm,file_perm,expand_vars,entropydict,Xrvs_cons,Wrvs_cons)
    Iutils.problem_constraints_process(Wkey,entropydict)
    entropydict.regenerate_keys()
    # print(len(entropydict.redict))
    # print(entropydict.redict)
    # print(entropydict)
    print("Xrvs_cons",Xrvs_cons)
    print("Wrvs_cons",Wrvs_cons)
    generate_single_inequalities(ix_vars,iy_vars,iz_vars,hx_vars,hy_vars,entropydict,regions)

    regions.sort_exprs()
    cutset_regions = regions.copy()

    for expr in cutset_regions.exprs:
        print(expr)
    print("num of exprs",len(cutset_regions.exprs))
    regions.reduce_redundant_expr()

    return regions, entropydict, vars
# 保存json文件需要的函数
# 构建数据结构
def create_experiment_data(globalN,globalK,GENERATE_SIZE,subset_size,comb_size,env,evaluate_rewards):
    """
    实际的结构需要结合write_json函数一起看，这里只是单次测试的效果啊。
    """
    experiment_data = {
        "metadata": {
            "experiment_id":None, # waiting to be assigned
            "experiment_time": datetime.now().isoformat()
        },
        "hyperparameters": {
            "N": globalN,
            "K": globalK,
            "GENERATE_SIZE": GENERATE_SIZE,
            "subset_size": subset_size,
            "comb_size": comb_size,
            "NEWINEQ_NUM": env.NEWINEQ_NUM
        },
        "results": {
            "evaluate_rewards": evaluate_rewards
        }
    }
    return experiment_data
def write_json(experiment_data,filename=f"/home/yxd/yyj/learntoconverse-master/envs/hyperparams/hyperparams_analysis.json"):
    """
    保存为json文件
    """
    new_data = copy.deepcopy(experiment_data)
    try:
        with open(filename, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = {
            "format_version":"1.0",
            "creation_time":datetime.now().isoformat(),
            "experiments_data":[]
        }
    new_data["metadata"]["experiment_id"] = len(all_data["experiments_data"]) # 酸了，id也从0开始得了。
    all_data["experiments_data"].append(new_data)
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=4,ensure_ascii=False,default=str)
# 新建一个用来表示entropydict变化的函数
class EntropyEqDictChange():
    """
    主要是用来看entropydict_all的变化情况，也就是说，应该是不怎么发生变化才是比较合理的。
    """
    def __init__(self, entropydict_old):
        self.entropydict_old = entropydict_old.copy()
    def update(self, entropydict_new):
        self.entropydict_new = entropydict_new.copy()
        self.UnUsedIndex = []
        for i in range(self.entropydict_new.max_index()+1):
            INNEW = i in self.entropydict_new.redict.keys()
            INOLD = i in self.entropydict_old.redict.keys()
            if INOLD and not INNEW:
                #print(f"* bug found! INOLD and !INNEW, i={i}")
                pass
            elif not INOLD and INNEW:
                #print(f"** new added index found! INNEW and !INOLD, i={i}")
                pass
            elif INOLD and INNEW:
                continue
            else:
                self.UnUsedIndex.append(i)
                continue
        #print(f"*** {len(self.UnUsedIndex)} unused index: {self.UnUsedIndex}")
        self.entropydict_old = self.entropydict_new.copy()
def Regions2VecMatrix(regions):
    """
    直接使用Expr2Vec函数，实现从regions到VecMatrix的转换。
    """
    global total_time
    global total_exprs
    t_start = time.time()
    RegionStrList = [str(expr) for expr in regions.exprs]
    print(len(RegionStrList))
    VecMatrixList = Expr2Vec(ExprStrList = RegionStrList)
    t_end = time.time()
    total_time += (t_end - t_start)
    total_exprs += len(RegionStrList)
    return np.array(VecMatrixList)

class ConverseEnv():
    """
    默认NK=33的环境。
    """
    def __init__(self,N=3,K=3,MAXEPISODE = 10,REPEATDONE=5,RENDER = RENDER,usepredict_hyperparams = False):
        self.square = 0  # 实际上就是在更新之前的square的值。
        self.N = N
        self.K = K
        # 新添加的关于直接由NK计算其他超参数的函数。因为初始化之后就不要重新修改NK了所以也就不用重新修改超参数。
        global hyperparams
        # if usepredict_hyperparams:
        #     hyperparams = predict_wrapper(self.N,self.K)

        #     global NEWINEQ_NUM
        #     global GENERATE_SIZE
        #     global generate_size
        #     global subset_size
        #     global comb_size
        #     # hyperparams = dict()
            
        #     NEWINEQ_NUM = hyperparams["NEWINEQ_NUM"]
        #     GENERATE_SIZE = hyperparams["GENERATE_SIZE"]
        #     generate_size = hyperparams["GENERATE_SIZE"]
        #     subset_size = hyperparams["subset_size"]
        #     comb_size = hyperparams["comb_size"]
        #     print(f"hyperparams in {self.N}, {self.K} are:",hyperparams)
        self.user_perm = list(itertools.permutations(range(1, K + 1)))
        self.file_perm = list(itertools.permutations(range(1, N + 1)))
        self.MAXEPISODE = MAXEPISODE
        self.REPEATDONE = REPEATDONE
        self.NEWINEQ_NUM = NEWINEQ_NUM
        self.RENDER = RENDER
        self.rewardlist = []
        # self.plot_data = []
        self.plot_data_list = []
        # self.X_combinations = []
        # self.ine_constraints = []
        # self.vars = []
        # self.state = []
        # self.ob = []
        # self.done = False
        # self.regions = Region.empty()
        # self.regions_candidate = Region.empty()
        # self.entropydict = EntropyEqDict()
        # self.entropydict_all = EntropyEqDict()
        # self.Xrvs_cons = []
        # self.Wrvs_cons = []
        # self.Wkey = []
        # self.oriall_vars = []
        self.pic_storage_path = "~/yyj/save/"
    def render(self,namestring = ""):

        #pdb.set_trace()
        plt.rcParams.update({'font.size': 8})  # 减小字体大小

        N = self.N
        K = self.K
        plot_data = self.plot_data
        X_combinations = self.X_combinations
        ine_constraints = self.ine_constraints
        vars = self.vars
        # 绘制图像

        # plot the cut-set bound
        Iutils.plot_cutset_bound(N,K,point_num)
        Iutils.plot_inner_bound(N,K)


        x = [item[0] for item in plot_data]
        y = [item[1] for item in plot_data]
        result_slope = Iutils.compute_slopes(x,y)
        point_x = []
        point_y = []
        for i in range(1,len(result_slope)):
            if result_slope[i-1] != result_slope[i]:
                point_x.append(x[i])
                point_y.append(y[i])
        plt.scatter(point_x,point_y,color="red")
        
        for xi, yi in zip(point_x, point_y):
            label = "({:.3f}, {:.3f})".format(xi, yi)
            plt.annotate(label,  
                        (xi, yi),  
                        textcoords="offset points",  
                        xytext=(30, 5),  
                        ha='center')  
        plt.plot(x, y, color='red', linewidth=2, label='computed outer bound')

        
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("M")
        plt.ylabel("R")
        plt.legend()
        plt.title("Case(N,K)=({},{}),episode{}\nX_type={}\nvars:{},cons:{}".format(N,K,episode,X_combinations,len(vars),ine_constraints.shape))
        save_dir = os.path.join(os.path.expanduser(self.pic_storage_path),"N,K={}{}".format(N,K))#
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            plt.savefig('{}/{},N={},K={}.png'.format(save_dir,namestring,N,K))
        except:
            print("savefig error")
            print(f"result: {point_x},{point_y}")
        plt.close()
    def reset(self):
        """
        实现cutsetbound的求解，得到目标ob和state的描述。
        """
        N = self.N
        K = self.K
        user_perm = self.user_perm
        file_perm = self.file_perm
        self.rewardlist = []
        self.plot_data_list = []
        global episode
        episode = 0
        single_vars = []
        vars = [] # [["W1","W2"]]
        Wrvs = [] # ["W1","W2"]
        necessary_vars = [] # single + vars
        W_combinations = [] # [["W1"],["W2"],["W1","W2"]]
        for i in range(1, N+1):
            single_vars.append("W" + str(i))
            Wrvs.append("W" + str(i))
        for i in range(1, K+1):
            single_vars.append("Z" + str(i))
        X_combinations = ["X" + item for item in Iutils.generate_combinations(N, K)]
        if N == 2 and K == 2:
            X_combinations = ["X12"]
        elif N == 2 and K == 3:
            X_combinations = ["X112"]
        elif N == 3 and K == 2:
            X_combinations = ["X12","X13"]
        elif N == 3 and K == 3:
            X_combinations = ["X112","X113","X123"]
        elif N == 2 and K == 4:
            X_combinations = ["X1112","X1122"]
        elif N == 4 and K == 2:
            X_combinations = ["X12","X13","X14"]
        elif N == 4 and K == 3:
            X_combinations = ["X112","X113","X114","X123"]
        elif N == 4 and K == 4:
            X_combinations = ["X1112","X1123","X1234"]
        elif N == 6 and K == 4:
            X_combinations = ["X1112","X1123","X1234"]
        elif N == 5 and K == 5:
            X_combinations = ["X11112","X11123","X11234","X12345"]
        
        Xrvs_cons = Iutils.symmetry_vars(user_perm,file_perm,X_combinations) # 约束之后的结果. 还不知道是做什么用的, 现在.

        for item in X_combinations:
            single_vars.append(item)
        for var in single_vars:
            # vars.append([var])
            necessary_vars.append([var])
        vars.append(Wrvs)
        necessary_vars.append(Wrvs)
        for r in range(N+1):
            # 生成指定长度的所有组合
            combos = itertools.combinations(Wrvs, r+1)
            for combo in combos:
                W_combinations.append(list(combo))
        Wrvs_cons = Iutils.symmetry_vars(user_perm,file_perm,W_combinations) # 不同熵, 只保留一个.
        Wkey = ','.join(sorted(Wrvs, key=Iutils.sort_key))
        self.Wkey = Wkey

        regions,entropydict,vars = create_cutset_bound(N,K,user_perm,file_perm,Wkey,vars)
        for expr in regions.exprs:
            expr.sort_terms()
        ine_constraints = Regions2Matrix(entropydict, regions) # prob solve
        ent_num = len(entropydict.redict) + 3
        ine_constraints, prob_cons_num = AddProblemConstraints2Matrix(Xrvs_cons, Wrvs_cons, entropydict, ine_constraints, ent_num)
        ine_constraints = np.array(ine_constraints)
        #print(ine_constraints.shape)
        ine_constraints = ine_constraints.astype(np.float64)
        expr_num = ine_constraints.shape[0] - 1 # 不等式数量
        ent_num = ine_constraints.shape[1] - 1 # 减之前用于生成矩阵，减之后就是实际的熵变量数量
        ori_obj_coef = np.zeros(ent_num)
        ori_obj_coef[-1] = 1
        #print(ent_num)
        # solve problem
        plot_data = [] # 存放M值和对应的结果
        effective_idx_gurobi = [] # 有效不等式的索引
        all_eff_indx = set() # 所有有效不等式的索引
        result_slope = [] # 斜率
        ori_slope = [] # 原目标函数的斜率
        M_space = np.linspace(0,N,point_num*N+1)
        dual_value = [] # 存放对偶问题的解
        effective_idx_cut = [] # 存放切割点的索引
        # print(M_space)
        for M_value in M_space:
            # 根据M_value更新约束矩阵，添加等式约束
            ine_constraints = list(ine_constraints[:-1])
            row = [0] * (ent_num + 1)
            row[-3] = 1
            row[-1] = M_value
            ine_constraints.append(row)
            ine_constraints = np.array(ine_constraints)
            
            # print("ine_constraints")
            # print(ine_constraints)
            # print("ori_obj_coef",ori_obj_coef)

            # # 更新对偶问题约束矩阵
            expr_num = ine_constraints.shape[0] - 1
            trans_ine_cons = ine_constraints.T[:-1] # 对偶问题的约束矩阵 是原约束矩阵的转置
            dual_obj_coef = ine_constraints[:,-1] # 原约束的常量 是对偶问题目标函数的系数
            trans_ine_cons = np.hstack((trans_ine_cons, ori_obj_coef.T.reshape(-1, 1))) # 原目标函数的系数，是对偶问题约束的常量
            # print("shape",trans_ine_cons.shape)
            # print("trans_ine_cons")
            # print(trans_ine_cons)
            # print("dual_obj_coef",dual_obj_coef)

            # 求解原LP问题
            result = gurobi_solver(ent_num, ine_constraints, regions,ori_obj_coef,effective_idx_gurobi)
            if type(result) == list:
                bad = Region.empty()
                for ine in result:
                    idx = int(ine[1:])
                    print("type:{},value:{}".format(type(idx),idx))
                    terms = []
                    row = ine_constraints[idx]
                    for i in range(len(row) - 1):
                        coef = row[i]
                        if coef != 0:
                            if entropydict.get_keys_by_value(i) != None:
                                term_x = entropydict.get_keys_by_value(i)[0]
                                var_str = term_x.split(",")
                                if var_str not in vars:
                                    vars.append(var_str) # 添加有效不等式中的变量
                                term_x = Comp.jes(term_x)
                                term = Term(x=[term_x.copy()],coef=int(coef),termtype=TermType.H)
                                terms.append(term) 
                    expr = Expr(terms, eqtype="ge", value=row[-1])
                    expr.sort_terms()
                    # print("expr",expr)
                    bad.append_expr(expr)
                for expr in bad.exprs:
                    print(expr)
        
            # 求解对偶LP问题
            if result != 0:
                solution_values, effective_idx_dual = dual_solver(prob_cons_num,expr_num,dual_obj_coef,trans_ine_cons)
                if solution_values is not None:
                    dual_value.append(list(solution_values.values()))

                # print("effective_indices",effective_idx_dual)
                if effective_idx_dual is not None:
                    all_eff_indx = all_eff_indx.union(set(effective_idx_dual))
                effective_idx_dual = sorted(list(all_eff_indx))
            
            plot_data.append((M_value, result))
        if len(dual_value) > 0:
            effective_idx_cut = find_min_effective_indices(dual_value,regions)

        self.square = calculate_square(plot_data)
        oriall_vars = []
        # in episode <= 10, start here.
        print("oriall vars",len(oriall_vars))
        # print("same times",same_times)
        #episode += 1
        # 更新regions、vars
        print("1.更新regions、vars")
        print("number of effective exprs:",len(effective_idx_cut))
        # print(effective_indices)
        effective_constraints = [ine_constraints[i] for i in effective_idx_cut]
        regions = Region.empty() # 更新regions
        ori_vars = vars[:]
        oriall_vars += vars[:]
        entropydict_all = entropydict.copy()
        vars = []
        effective_vars = []
        effective_idx_cut = []
        count_dict = {}
        # regions = cutset_regions.copy()
        # print("dict",regions.exprdict)
        # 下面这里很奇怪, 但是其实就是把矩阵形式的不等式转化为region形式, 保存regions.
        for row in effective_constraints:
            terms = []
            for i in range(len(row) - 1):
                coef = row[i]
                if coef != 0:
                    if entropydict.get_keys_by_value(i) != None:
                        term_x = entropydict.get_keys_by_value(i)[0]
                        var_str = term_x.split(",")
                        if term_x in count_dict:
                            count_dict[term_x] += 1
                        else:
                            count_dict[term_x] = 1
                        if var_str not in vars:
                            vars.append(var_str) # 添加有效不等式中的变量
                        term_x = Comp.jes(term_x)
                        term = Term(x=[term_x.copy()],coef=int(coef),termtype=TermType.H)
                        terms.append(term) 
            expr = Expr(terms, eqtype="ge", value=row[-1])
            expr.sort_terms()
            # print("expr",expr)
            regions.append_expr(expr)
        # for expr in regions.exprs:
        #     print(expr)
        effective_vars = vars[:]
        # print("effective vars",vars)
        print("len of effecitve vars",len(vars))
        count_dict = sorted(count_dict.items(), key=lambda item: item[1],reverse=True)
        count_dict = dict(count_dict)
        print(count_dict)

        # 生成新vars
        print("2.生成新vars")
        # 方式2：ori vars的交并集
        generate_vars = []
        for i in range(len(ori_vars)):
            for j in range(i + 1, len(ori_vars)):
                union = list(set(ori_vars[i]) | set(ori_vars[j]))
                if len(union) <= episode+1:
                    # print("lenunion",len(union))
                    union.sort()
                    generate_vars.append(union)
                ins = list(set(ori_vars[i]) & set(ori_vars[j]))
                if ins:
                    ins.sort()
                    generate_vars.append(ins)
        
        global generate_size
        random_indices = []
        if generate_size > len(generate_vars): # 补齐, 防止越界
            generate_size = len(generate_vars)
        elif len(generate_vars) >= GENERATE_SIZE: # 在返回回去
            generate_size = GENERATE_SIZE
        random_indices = np.random.choice(len(generate_vars), size=generate_size, replace=False)
        selected_vars = [generate_vars[i] for i in random_indices]

        for var in selected_vars:
            if var not in vars:
                vars.append(var)
        # 上面是来自于生成的熵直接转化来的。下面是随机生成的
        result_subsets = Iutils.generate_random_subsets(single_vars, subset_size, 2, episode+4)
        for subset in result_subsets:
            if subset not in vars and subset not in oriall_vars:
                vars.append(subset)

        # 对vars进行封闭集和对称性处理，生成entropydict
        print("3.对vars进行封闭集和对称性处理，生成entropydict")
        expand_vars = vars[:]
        combs,combinations = Iutils.generate_combs(single_vars,comb_size)
        Iutils.preprocessing_combs(vars,single_vars,expand_vars,combs)
        for var in necessary_vars:
            if var not in expand_vars:
                expand_vars.append(var)
        print("before symmetrize num of expanded vars:",len(expand_vars))
        entropydict = EntropyEqDict()
        # Xrvs_cons = []
        # Wrvs_cons = []
        print("all_before",len(entropydict_all.redict))
        before_sym_time = time.time()
        Iutils.symmetrize_by_dict(user_perm,file_perm,expand_vars,entropydict,entropydict_all)
        print("-----------symmetrize time",time.time()-before_sym_time,"-----------")
        print("after symmetreize num of expanded vars:",len(entropydict.redict))
        print("all_after",len(entropydict_all.redict))
        # Iutils.symmetrize(user_perm,file_perm,expand_vars,entropydict,Xrvs_cons,Wrvs_cons)
        # print(entropydict)

        #pdb.set_trace()

        print("4.问题约束")
        Iutils.problem_constraints_process(Wkey,entropydict)
        entropydict.regenerate_keys()
        Iutils.problem_constraints_process(Wkey,entropydict_all)
        #entropydict_all.regenerate_keys()

        # 生成不等式集，并合并相同不等式
        print("5.生成不等式集，并合并相同不等式")
        regions_candidate = Region.empty()
        Iutils.generate_inequalities_combs(vars,entropydict,regions_candidate,combinations)
        #print("before reducing",len(regions.exprs))
        regions_candidate.reduce_redundant_expr()
        #print("number of exprs",len(regions.exprs))

        #pdb.set_trace()
        ent_num = len(entropydict.redict) + 3 # 因为更新过dict。
        regions_matrix = Regions2VecMatrix(regions)
        regions_candidate_matrix = Regions2VecMatrix(regions_candidate)
        ob = tuple(
            [
                regions_matrix[:,:-1],# A
                regions_matrix[:,-1].reshape(-1),# b
                np.array([N,K,self.square]),# info
                regions_candidate_matrix[:,:-1],
                regions_candidate_matrix[:,-1].reshape(-1)
            ] # 分开为A和b没有实际的意义
        )
        #pdb.set_trace()
        print(f"**** regions_matrix: {regions_matrix.shape}, regions_candidate_matrix: {regions_candidate_matrix.shape}\nentropydict_all.max_index()+1+3: {entropydict_all.max_index()+1+3}")
    
        # 传参
        self.X_combinations = X_combinations
        self.ine_constraints = ine_constraints
        self.vars = vars
        self.ob = ob
        self.done = False
        self.plot_data = plot_data
        self.plot_data_list.append(plot_data)
        self.entropydict = entropydict
        self.entropydict_all = entropydict_all
        self.regions_candidate = regions_candidate
        self.regions = regions
        self.Xrvs_cons = Xrvs_cons
        self.Wrvs_cons = Wrvs_cons
        self.necessary_vars = necessary_vars
        self.single_vars = single_vars
        self.oriall_vars = oriall_vars
        self.effective_idx_cut = effective_idx_cut
        if self.RENDER=="selftest":
            self.render(f"time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')},episode_{episode}")
        return ob,{}
    def step(self,action):
        """
        按照action的索引，选择对应的不等式。更新regions。
        主要的逻辑是：
            基于原始的regions（这里是effective regions），和生成的新candidate_regions。
            得到用来计算的regions, 计算求解之后得到有效边界，重新生成candidate_regions.
        
        """
        global episode
        N = self.N
        K = self.K
        user_perm = self.user_perm
        file_perm = self.file_perm        
        X_combinations = self.X_combinations
        ine_constraints = self.ine_constraints
        vars = self.vars
        ob = self.ob
        entropydict = self.entropydict
        entropydict_all = self.entropydict_all
        regions_candidate = copy.deepcopy(self.regions_candidate)
        regions = copy.deepcopy(self.regions)
        Xrvs_cons = self.Xrvs_cons
        Wrvs_cons = self.Wrvs_cons
        necessary_vars = self.necessary_vars
        single_vars = self.single_vars
        oriall_vars = self.oriall_vars
        effective_idx_cut = self.effective_idx_cut
        Wkey = self.Wkey
        #pdb.set_trace()
        # new added codes.
        #pdb.set_trace() # regions: [H({Z1}) + H({X12,X21}) - H({W1,W2}) >= 0.0, H({Z1}) + H({X12,Z1}) - H({W1,W2}) >= 0.0, H({X12}) + H({Z1}) - H({X12,Z1}) >= 0.0]
        for i in range(len(action)):
            regions.append_expr(regions_candidate.exprs[action[i]])
        #pdb.set_trace() # 这里是新添加了action之后的regions。
        regions.reduce_redundant_expr()

        # 生成不等式矩阵
        print("6.生成不等式矩阵")
        ine_constraints = Regions2Matrix(entropydict, regions)
        ent_num = len(entropydict.redict) + 3
        ine_constraints, prob_cons_num = AddProblemConstraints2Matrix(Xrvs_cons, Wrvs_cons, entropydict, ine_constraints, ent_num)
        ent_num -= 1 # 减1是因为最后一列是常数项

        #pdb.set_trace()
        try:
            self.entropydictchange.update(entropydict_all)
        except:
            self.entropydictchange = EntropyEqDictChange(entropydict_all)

        # 问题求解
        plot_data = []
        effective_idx_gurobi = []
        all_eff_indx = set()
        result_slope = []
        ori_slope = []
        M_space = np.linspace(0,N,point_num*N+1)
        dual_value = []
        # print(M_space)
        t_solve = 0
        # print("shape of ine_constraints",ine_constraints.shape)
        for M_value in M_space:
            s = time.time()
            # 根据M_value更新约束矩阵，添加等式约束
            ine_constraints = list(ine_constraints[:-1])
            row = [0] * (ent_num + 1)
            row[-3] = 1
            row[-1] = M_value
            ine_constraints.append(row)

            ine_constraints = np.array(ine_constraints)
            if M_value == 0:
                print("shape of ine_constraints",ine_constraints.shape)
            ine_constraints = ine_constraints.astype(np.float64)

            ori_obj_coef = np.zeros(ent_num)
            ori_obj_coef[-1] = 1
            # print("ine_constraints")
            # print(ine_constraints)
            # print("ori_obj_coef",ori_obj_coef)

            # # 更新对偶问题约束矩阵
            expr_num = ine_constraints.shape[0] - 1
            trans_ine_cons = ine_constraints.T[:-1] # 对偶问题的约束矩阵 是原约束矩阵的转置
            dual_obj_coef = ine_constraints[:,-1] # 原约束的常量 是对偶问题目标函数的系数
            trans_ine_cons = np.hstack((trans_ine_cons, ori_obj_coef.T.reshape(-1, 1))) # 原目标函数的系数，是对偶问题约束的常量
            # print("shape",trans_ine_cons.shape)
            # print("trans_ine_cons")
            # print(trans_ine_cons)
            # print("dual_obj_coef",dual_obj_coef)

            count_dict = {}
            # 求解原LP问题
            before_lp_time = time.time()
            result = gurobi_solver(ent_num, ine_constraints, regions,ori_obj_coef,effective_idx_gurobi)
            if type(result) == list:
                bad = Region.empty()
                for ine in result:
                    idx = int(ine[1:])
                    #print(f"type:{type(idx)},value:{idx}")
                    terms = []
                    row = ine_constraints[idx]
                    for i in range(len(row) - 1):
                        coef = row[i]
                        if coef != 0:
                            if entropydict.get_keys_by_value(i) != None:
                                term_x = entropydict.get_keys_by_value(i)[0]
                                var_str = term_x.split(",")
                                if term_x in count_dict:
                                    count_dict[term_x] += 1
                                else:
                                    count_dict[term_x] = 1
                                if var_str not in vars:
                                    vars.append(var_str) # 添加有效不等式中的变量
                                term_x = Comp.jes(term_x)
                                term = Term(x=[term_x.copy()],coef=int(coef),termtype=TermType.H)
                                terms.append(term) 
                    expr = Expr(terms, eqtype="ge", value=row[-1])
                    expr.sort_terms()
                    # print("expr",expr)
                    bad.append_expr(expr)
                for expr in bad.exprs:
                    print(expr)

            if result > 0:
                # 求解对偶LP问题
                solution_values, effective_idx_dual = dual_solver(prob_cons_num,expr_num,dual_obj_coef,trans_ine_cons)
                if solution_values is not None:
                    dual_value.append(list(solution_values.values()))

                # print("effective_indices",effective_idx_dual)
                if effective_idx_dual is not None:
                    all_eff_indx = all_eff_indx.union(set(effective_idx_dual))
                effective_idx_dual = sorted(list(all_eff_indx))
            plot_data.append((M_value, result))
            e = time.time()
            t = e - s
            t_solve += t

        print("-------------Lp time: ",time.time()-before_lp_time,"-------------")
        print("solve time",t_solve)
        if len(dual_value) > 0:
            effective_idx_cut = find_min_effective_indices(dual_value,regions)

        # 下面按照前面的方法更新旧的东西啊。
        # 都是从reset中复制过来的啊。
        reward = np.exp(calculate_square(plot_data)) - np.exp(self.square)
        self.square = calculate_square(plot_data)
        self.rewardlist.append(reward)
        episode += 1
        print("oriall vars",len(oriall_vars))
        # print("same times",same_times)
        # 更新regions、vars
        print("1.更新regions、vars")
        print("number of effective exprs:",len(effective_idx_cut))
        # print(effective_indices)
        effective_constraints = [ine_constraints[i] for i in effective_idx_cut]
        regions = Region.empty() # 更新regions
        ori_vars = vars[:]
        oriall_vars += vars[:]
        #entropydict_all = entropydict.copy()
        vars = []
        effective_vars = []
        effective_idx_cut = []
        index = 0
        count_dict = {}
        # regions = cutset_regions.copy()
        # print("dict",regions.exprdict)
        # 下面这里很奇怪, 但是其实就是把矩阵形式的不等式转化为region形式, 保存regions.
        for row in effective_constraints:
            terms = []
            for i in range(len(row) - 1):
                coef = row[i]
                if coef != 0:
                    if entropydict.get_keys_by_value(i) != None:
                        term_x = entropydict.get_keys_by_value(i)[0]
                        var_str = term_x.split(",")
                        if term_x in count_dict:
                            count_dict[term_x] += 1
                        else:
                            count_dict[term_x] = 1
                        if var_str not in vars:
                            vars.append(var_str) # 添加有效不等式中的变量
                        term_x = Comp.jes(term_x)
                        term = Term(x=[term_x.copy()],coef=int(coef),termtype=TermType.H)
                        terms.append(term) 
            expr = Expr(terms, eqtype="ge", value=row[-1])
            expr.sort_terms()
            # print("expr",expr)
            regions.append_expr(expr)
        #pdb.set_trace()# 新的有效边界啊。
        # for expr in regions.exprs:
        #     print(expr)
        effective_vars = vars[:]
        # print("effective vars",vars)
        print("len of effecitve vars",len(vars))
        count_dict = sorted(count_dict.items(), key=lambda item: item[1],reverse=True)
        count_dict = dict(count_dict)
        print(count_dict)

        # 生成新vars
        print("2.生成新vars")
        # 方式2：ori vars的交并集
        generate_vars = []
        for i in range(len(ori_vars)):
            for j in range(i + 1, len(ori_vars)):
                union = list(set(ori_vars[i]) | set(ori_vars[j]))
                if len(union) <= episode+1:
                    # print("lenunion",len(union))
                    union.sort()
                    generate_vars.append(union)
                ins = list(set(ori_vars[i]) & set(ori_vars[j]))
                if ins:
                    ins.sort()
                    generate_vars.append(ins)
        
        global generate_size
        random_indices = []
        if generate_size > len(generate_vars): # 补齐, 防止越界
            generate_size = len(generate_vars)
        elif len(generate_vars) >= GENERATE_SIZE: # 在返回回去
            generate_size = GENERATE_SIZE
            random_indices = np.random.choice(len(generate_vars), size=generate_size, replace=False)
        selected_vars = [generate_vars[i] for i in random_indices]

        for var in selected_vars:
            if var not in vars:
                vars.append(var)
        # 上面是来自于生成的熵直接转化来的。下面是随机生成的
        result_subsets = Iutils.generate_random_subsets(single_vars, subset_size, 2, episode+4)
        for subset in result_subsets:
            if subset not in vars and subset not in oriall_vars:
                vars.append(subset)

        # 对vars进行封闭集和对称性处理，生成entropydict
        print("3.对vars进行封闭集和对称性处理，生成entropydict")
        expand_vars = vars[:]
        combs,combinations = Iutils.generate_combs(single_vars,comb_size)
        Iutils.preprocessing_combs(vars,single_vars,expand_vars,combs)
        for var in necessary_vars:
            if var not in expand_vars:
                expand_vars.append(var)
        print("before symmetrize num of expanded vars:",len(expand_vars))
        entropydict = EntropyEqDict()
        # Xrvs_cons = []
        # Wrvs_cons = []
        print("all_before",len(entropydict_all.redict))
        before_sym_time = time.time()
        Iutils.symmetrize_by_dict(user_perm,file_perm,expand_vars,entropydict,entropydict_all)
        print("-----------symmetrize time",time.time()-before_sym_time,"-----------")
        print("after symmetreize num of expanded vars:",len(entropydict.redict))
        print("all_after",len(entropydict_all.redict))
        # Iutils.symmetrize(user_perm,file_perm,expand_vars,entropydict,Xrvs_cons,Wrvs_cons)
        # print(entropydict)

        print("4.问题约束")
        Iutils.problem_constraints_process(Wkey,entropydict)
        entropydict.regenerate_keys()
        Iutils.problem_constraints_process(Wkey,entropydict_all)
        #entropydict_all.regenerate_keys()

        # 生成不等式集，并合并相同不等式
        print("5.生成不等式集，并合并相同不等式")
        regions_candidate = Region.empty()
        Iutils.generate_inequalities_combs(vars,entropydict,regions_candidate,combinations)
        #print("before reducing",len(regions.exprs))
        regions.reduce_redundant_expr()
        #print("number of exprs",len(regions.exprs))

        #pdb.set_trace()
        ent_num = len(entropydict.redict) + 3 # 因为更新过dict。
        regions_matrix = Regions2VecMatrix(regions)
        regions_candidate_matrix = Regions2VecMatrix(regions_candidate)
        ob = tuple(
            [
                regions_matrix[:,:-1],# A
                regions_matrix[:,-1].reshape(-1),# b
                np.array([N,K,self.square]),# info
                regions_candidate_matrix[:,:-1],
                regions_candidate_matrix[:,-1].reshape(-1)
            ]
        )

        #pdb.set_trace()
        print(f"**** regions_matrix: {regions_matrix.shape}, regions_candidate_matrix: {regions_candidate_matrix.shape}\nentropydict_all.max_index()+1+3: {entropydict_all.max_index()+1+3}")

        #pdb.set_trace()
        # 终止条件分析。
        done = False
        doneflag = ''
        if episode >= self.MAXEPISODE:
            done = True
            doneflag = "因为episode数目达到最大值"
        rescent_reward = self.rewardlist[-5:]
        print("rewardlist: {}".format(self.rewardlist))
        print("recent reward: {}".format(rescent_reward))
        rescent_reward = [round(i,4) for i in rescent_reward]
        print("recent reward: {}".format(rescent_reward))
        if rescent_reward.count(0) >= self.REPEATDONE: # recent reward: [0.0, -0.0, 0.0, 0.0, 0.0] 也是可以的。
            done = True
            doneflag = "因为连续5次reward为0"

        # 传参
        self.X_combinations = X_combinations
        self.ine_constraints = ine_constraints
        self.vars = vars
        self.ob = ob
        self.done = done
        self.plot_data = plot_data
        self.plot_data_list.append(plot_data)
        self.entropydict = entropydict
        self.entropydict_all = entropydict_all
        self.regions_candidate = regions_candidate
        self.regions = regions
        self.Xrvs_cons = Xrvs_cons
        self.Wrvs_cons = Wrvs_cons
        self.necessary_vars = necessary_vars
        self.single_vars = single_vars
        self.oriall_vars = oriall_vars
        self.effective_idx_cut = effective_idx_cut

        if self.RENDER=="selftest":
            self.render(f"time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')},episode_{episode},{doneflag}")
        return ob, reward, done

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envmode', '-em',type=str, default="selftest") # choose from "selftest","hyperparam"
    parser.add_argument('--generate_size', '-gs', type=int, default=30)
    parser.add_argument('--subset_size', '-ss', type=int, default=30)
    parser.add_argument('--comb_size', '-cs', type=int, default=150)
    parser.add_argument('--NEWINEQ_NUM', '-n', type=int, default=100)
    parser.add_argument('--N', '-N', type=int, default=3)
    parser.add_argument('--K', '-K', type=int, default=3)
    parser.add_argument('--evaluate_time', '-et', type=int, default=1)
    args = parser.parse_args() # namespace type variables
    params = vars(args) # dict type variables

    if params['envmode'] == "selftest":
        # 环境自测试;render in env.
        # env: aifc; N=2,K=2.done.
        globalN = params['N']
        globalK = params['K']
        env = ConverseEnv(N=globalN,K=globalK,usepredict_hyperparams=False)
        print("hyperparams:","NEWINEQ_NUM:{},generate_size:{},GENERATE_SIZE:{},subset_size:{},comb_size:{}".format(NEWINEQ_NUM,generate_size,GENERATE_SIZE,subset_size,comb_size))
        ob,info = env.reset()
        print(ob[0].shape,ob[-2].shape)
        #env.render("reset_for_cutsetbound")
        action = np.random.randint(0, ob[3].shape[0], size=NEWINEQ_NUM)
        action = sorted(action)
        done = False
        print("---------------------------",len(env.regions.exprs),"-----------")
        for i in range(20):
            if done:
                break
            ob, r, done = env.step(action)
            print("---------------------------",len(env.regions.exprs),"-----------")
            #env.render("at episode {}with hyperparams".format(i))
            action = np.random.randint(0, ob[3].shape[0], size=NEWINEQ_NUM)
            action = sorted(action)
        print("hyperparams:","NEWINEQ_NUM:{},generate_size:{},GENERATE_SIZE:{},subset_size:{},comb_size:{}".format(NEWINEQ_NUM,generate_size,GENERATE_SIZE,subset_size,comb_size))
        print(f"total_exprs:{total_exprs},total_time:{total_time},average_time:{total_time/total_exprs}")
    elif params['envmode'] == "hyperparam":
        # render here.
        GENERATE_SIZE = params['generate_size']
        generate_size = GENERATE_SIZE
        subset_size = params['subset_size']
        comb_size = params['comb_size']
        NEWINEQ_NUM = params['NEWINEQ_NUM']
        globalN = params['N']
        globalK = params['K']
        RENDER = params['envmode']
        evaluate_time = params['evaluate_time']
        hyperparams = {"NEWINEQ_NUM":NEWINEQ_NUM,"generate_size":generate_size,"GENERATE_SIZE":GENERATE_SIZE,"subset_size":subset_size,"comb_size":comb_size}
        #evaluate_rewards = []
        for i in range(evaluate_time):
            print(f"N={globalN},K={globalK},evaluate_time={i}")
            env = ConverseEnv(N=globalN,K=globalK,RENDER=RENDER,usepredict_hyperparams=False) # 这里先使用旧的结果看看不改变的话，效果如何呢？
            ob,info = env.reset()
            #print(ob[0].shape,ob[-2].shape)
            #env.render("reset_for_cutsetbound")
            action = np.random.choice(ob[3].shape[0], size=min(NEWINEQ_NUM,ob[3].shape[0]), replace=False)
            action = sorted(action)
            done = False
            #print("----------",len(env.regions.exprs),"-----------")
            for j in range(20):
                if done:
                    break
                ob, r, done = env.step(action)
                action = np.random.choice(ob[3].shape[0], size=min(NEWINEQ_NUM,ob[3].shape[0]), replace=False)
                action = sorted(action)
                #print("----------",len(env.regions.exprs),"-----------")
            #evaluate_rewards.append(env.rewardlist)
            #env.render(f"gs_{GENERATE_SIZE},ss_{subset_size},cs_{comb_size},n_{env.NEWINEQ_NUM},et_{i},episode_{episode}")
            #pdb.set_trace()
            experiment_data = create_experiment_data(globalN,globalK,GENERATE_SIZE,subset_size,comb_size,env,evaluate_rewards=env.rewardlist)
            write_json(experiment_data)
    elif params["envmode"] == "newfunctest":
        env = ConverseEnv(N=2,K=2,RENDER="newfunctest",usepredict_hyperparams=False)
        ob,info = env.reset()
        VecMatrix = Regions2VecMatrix(env.regions)
        pdb.set_trace()
        print(VecMatrix)

        

"""
测试用例：
python converseenv.py -em selftest
"""
"""
提示词：完成对hyperparams_analysis.json文件的分析。写一个代码完成这个工作。具体而言0. 从parser中得到N和K的值，也就是给定超参数N和K；然后分析json文件 1. 给定一次实验，将revaluate_rewards求和得到这次实验的总的reward 2. 对相同超参数下的rewards求平均，得到给定超参数下总的rewards的均值 3. 不同超参数的均值进行比较；注意这里最好找到一种方法可以同时分析四个超参数与结果（结果就是指定超参数下的<单次实验的总的reward>的均值）的关系。 将结果保存在output_dir = "/home/yxd/yyj/learntoconverse-master/envs/hyperparams/analysis_result"，因为我使用的是linux系统（所以图像必须显示保存，而不是直接显示出来）
"""
"""测试：
python converseenv.py -em newfunctest
"""
"""
python converseenv.py  > envlog.log 2>&1
"""