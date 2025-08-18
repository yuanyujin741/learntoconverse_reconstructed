"""
主要目标
- 把需要设计的变量设置为类的内部变量
- 需要传递的参数也转换为内部变量
"""
# region import
from pentropy_main import *
import itertools
import json
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
import time
from collections import defaultdict, Counter
import datetime
print = print_

# region EnvClass
class ConverseEnv():
    def __init__(self,N:int,K:int,point_num:int=12,generate_size:int=30,subset_size:int=30,comb_size:int=30):
        """
        主要用于各项参数的设置。
        """
        self.N = N
        self.K = K
        self.point_num = point_num # 采样点密度
        self.generate_size = generate_size
        self.subset_size = subset_size
        self.comb_size = comb_size
        self.user_perm = list(itertools.permutations(range(1, K + 1)))
        self.file_perm = list(itertools.permutations(range(1, N + 1)))
        self.start_time = time.time()
        self.episode = 0
        self.square = 0
        self.plot_data_list = []
        self.reward_list = []
        self.DoneThreashold = 5 # 连续五次reward为0就认为是done了
        self.DoneEpsilon = 1e-6
        self.max_apisode_num = 10
    def reset(self,seed:int = None):
        """
        reset实现的是，重置环境；一直是到生成待选不等式为止。
        """
        # self restore
        N = self.N
        K = self.K
        user_perm = self.user_perm
        file_perm = self.file_perm
        subset_size = self.subset_size
        comb_size = self.comb_size
        self.episode = 0

        # ORIGINAL CODES
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # generate all random variables, stores in single_vars
        single_vars = [] # 单变量=W+Z+reduced_X
        Wrvs = [] # 原始W变量，也可以理解为一个全部W组成的单变量
        W_combinations = [] # W变量组合之后的全部变量
        Wrvs_cons = [] # 对前面的全部组合进行对称化去除之后的W_combinations
        vars = [] # 用来在cutsetbound中使用的对象
        necessary_vars = [] # 后面迭代的时候使用的。

        for i in range(1, N+1):
            single_vars.append("W" + str(i))
            Wrvs.append("W" + str(i))
        for i in range(1, K+1):
            single_vars.append("Z" + str(i))
        vars.append(Wrvs)

        X_combinations = ["X" + item for item in Iutils.generate_combinations(N, K)]
        X_combinations = get_reduced_X_combination(N,K,False)

        Xrvs_cons = Iutils.symmetry_vars(user_perm,file_perm,X_combinations)

        for item in X_combinations:
            single_vars.append(item)
        for var in single_vars:
            # vars.append([var])
            necessary_vars.append([var])
        necessary_vars.append(Wrvs)
        for r in range(N+1): # 这里似乎有点冗余，但是问题不大。
            # 生成指定长度的所有组合
            combos = itertools.combinations(Wrvs, r+1)
            for combo in combos:
                W_combinations.append(list(combo))
        Wrvs_cons = Iutils.symmetry_vars(user_perm,file_perm,W_combinations)
        print(Wrvs_cons)
        print(single_vars)
        Wkey = ','.join(sorted(Wrvs, key=Iutils.sort_key))
        print(Wkey)
        print(Xrvs_cons)
        necessary_vars = [] # 单变量+W_all变量
        vars = [] # 上面的necessary_vars+generated_sets，作为Z用于生成不等式；变量集合，元素为列表，一个列表对应一个联合熵，列表元素为字符串
        entropydict = EntropyEqDict()
        index = 0
        episode = 0

        for var in single_vars:
            vars.append([var])
            necessary_vars.append([var])
        # necessary_vars = Iutils.symmetry_vars(N,K,vars)
        necessary_vars.append(Wrvs)
        vars.append(Wrvs)

        sets = Iutils.generate_random_subsets(single_vars,subset_size,2,episode+4)
        # print(sets)
        vars += sets
        print(necessary_vars)
        print(vars)

        expand_vars = vars[:]

        combs,combinations = Iutils.generate_combs(single_vars,comb_size)
        Iutils.preprocessing_combs(vars,single_vars,expand_vars,combs)    

        entropydict = EntropyEqDict()
        entropydict_all = EntropyEqDict()

        print(len(expand_vars))
        print(expand_vars)
        Iutils.symmetrize_by_dict_simple(N,K,expand_vars,entropydict_all)
        # Iutils.symmetrize_by_dict(user_perm,file_perm,expand_vars,entropydict,entropydict_all)
        print(len(entropydict.redict))

        Iutils.problem_constraints_process(N,K,Wkey,entropydict_all)
        entropydict_all.regenerate_keys()

        for var in expand_vars:
            var_str = ",".join(sorted(var,key=Iutils.sort_key))
            entropydict[var_str] = entropydict_all.get(var_str)
        entropydict.regenerate_keys()

        print(len(entropydict.redict))
        print(entropydict.redict)

        # generate inequalities
        regions = Region.empty()
        regions_candidate = Region.empty() # 这里是初始化为empty的region，后续更新为cutsetbound
        Iutils.generate_inequalities_combs(vars,entropydict,regions_candidate,combinations) # 打印的是返回值Ixyz_list
        regions_candidate.reduce_redundant_expr()

        # encoding
        VecMatrix = Regions2VecMatrix(regions,N,K,necessary_vars)
        CandidateVecMatrix = Regions2VecMatrix(regions_candidate,N,K,necessary_vars)
        ob = tuple(
            [
                VecMatrix[:,:-1],# A
                VecMatrix[:,-1].reshape(-1),# b
                np.array([N,K,self.square]),# info
                CandidateVecMatrix[:,:-1],
                CandidateVecMatrix[:,-1].reshape(-1)
            ] # 分开为A和b没有实际的意义
        )
        #print(ob)

        self.regions_candidate = regions_candidate
        self.regions = regions
        self.entropydict = entropydict
        self.Xrvs_cons = Xrvs_cons
        self.Wrvs_cons = Wrvs_cons
        self.single_vars = single_vars
        self.necessary_vars = necessary_vars
        self.Wkey = Wkey
        self.entropydict_all = entropydict_all
        self.X_combinations = X_combinations
        self.vars = vars
        self.oriall_vars = []
        self.effctive_vars_last = []
        return ob, {}
    def step(self,action:list[int]):
        """
        执行。
        :regions: 上次的有效边界；
        :regions_candidate：上次等待使用的边界。
        """
        # self thing
        entropydict = copy.deepcopy(self.entropydict)
        Xrvs_cons = self.Xrvs_cons
        Wrvs_cons = self.Wrvs_cons
        N = self.N
        K = self.K
        point_num = self.point_num
        single_vars = self.single_vars
        subset_size = self.subset_size
        comb_size = self.comb_size
        generate_size = self.generate_size
        necessary_vars = self.necessary_vars
        Wkey = self.Wkey
        entropydict_all = copy.deepcopy(self.entropydict_all)
        vars = self.vars
        oriall_vars = self.oriall_vars
        effctive_vars_last = self.effctive_vars_last

        # recover from the candidate regions
        regions_candidate = copy.deepcopy(self.regions_candidate)
        regions = copy.deepcopy(self.regions)
        for i in range(len(action)):
            regions.append_expr(regions_candidate.exprs[action[i]])
        regions.reduce_redundant_expr()

        # ORIGINAL_CODES
        # 生成不等式矩阵
        # print("6.生成不等式矩阵")
        ent_num = len(entropydict.redict) + 3
        ine_constraints = Regions2Matrix(entropydict,regions)
        
        # additional constraints
        ine_constraints,prob_cons_num = AddProblemConstrains2Matrix(Xrvs_cons,Wrvs_cons,entropydict,ent_num,ine_constraints)
        ent_num -= 1 # 实际变量数量
        

        # 问题求解
        plot_data = []
        effective_idx_gurobi = []
        all_eff_indx = set()
        M_space = np.linspace(0,N,point_num*N+1)
        dual_value = []
        # print(M_space)
        t_solve = 0
        # print("shape of ine_constraints",ine_constraints.shape)
        for M_value in M_space:
            s = time.perf_counter()
            # 根据M_value更新约束矩阵，添加等式约束
            ine_constraints = list(ine_constraints[:-1])
            row = [0] * (ent_num + 1)
            row[-3] = 1
            row[-1] = M_value
            ine_constraints.append(row)

            ine_constraints = np.array(ine_constraints)
            if M_value == 0:
                pass
                # print("shape of ine_constraints",ine_constraints.shape)
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
            # print(trans_ine_cons)
            # print("dual_obj_coef",dual_obj_coef)

            # 求解原LP问题
            result,effective_idx_gurobi = gurobi_solver(effective_idx_gurobi,ori_obj_coef,ent_num,ine_constraints,regions)
            if type(result) == list:
                bad = Region.empty()
                for ine in result:
                    idx = int(ine[1:])
                    # print(f"type:{type(idx)},value:{idx}")
                    terms = []
                    row = ine_constraints[idx]
                    for i in range(len(row) - 1):
                        coef = row[i]
                        if coef != 0:
                            if entropydict.get_keys_by_value(i) != None:
                                term_x = entropydict.get_keys_by_value(i)[0]
                                var_str = term_x.split(",")
                                # if term_x in count_dict:
                                #     count_dict[term_x] += 1
                                # else:
                                #     count_dict[term_x] = 1
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
                    pass
                    # print(expr)

            elif result > 0:
                # 求解对偶LP问题
                solution_values, effective_idx_dual = dual_solver(expr_num,dual_obj_coef,trans_ine_cons,prob_cons_num)
                if solution_values is not None:
                    dual_value.append(list(solution_values.values()))

                # print("effective_indices",effective_idx_dual)
                if effective_idx_dual is not None:
                    all_eff_indx = all_eff_indx.union(set(effective_idx_dual))
                effective_idx_dual = sorted(list(all_eff_indx))
            plot_data.append((M_value, result))
            e = time.perf_counter()
            t = e - s
            t_solve += t
        # print("solve time",t_solve)
        if len(dual_value) > 0:
            effective_idx_cut = find_min_effective_indices(dual_value,regions)
        else:
            effective_idx_cut = []

        # ---- RENDER RECORDER ----
        self.plot_data_list.append(plot_data)
        self.ine_constraints = copy.deepcopy(ine_constraints)
        self.render_vars = vars

        # new episode here
        self.episode += 1
        episode = self.episode
        # print("oriall vars",len(oriall_vars))
        # print("same times",same_times)
        # print(entropydict_all.eqdict)
        # 更新regions、vars
        # print("1.更新regions、vars")
        # print("number of effective exprs:",len(effective_idx_cut))
        # print(effective_indices)
        effective_constraints = [ine_constraints[i] for i in effective_idx_cut]
        regions = Region.empty() # 更新regions
        ori_vars = vars[:]
        oriall_vars += vars[:]
        # entropydict_all = entropydict.copy()
        vars = []
        effective_vars = []
        effective_idx_cut = []
        add_vars = set()
        del_vars = set()
        index = 0
        count_dict = {}
        # regions = cutset_regions.copy()
        # print("dict",regions.exprdict)
        for row in effective_constraints:
            terms = []
            term_list = []
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
                        term_list.append(coef)
                        term_list.append(term_x)
                        term_x = Comp.jes(term_x)
                        term = Term(x=[term_x.copy()],coef=int(coef),termtype=TermType.H)
                        terms.append(term)
            # print(term_list)
            # x_vars, y_vars, z_vars = extract_single_var_conditional(term_list)
            # print(f"x_vars:{x_vars}, y_vars:{y_vars}, z_vars:{z_vars}")
            expr = Expr(terms, eqtype="ge", value=row[-1])
            expr.sort_terms()
            # print("expr",expr)
            regions.append_expr(expr)
        for expr in regions.exprs:
            pass
            # print(expr)
            
        
        effective_vars = vars[:]
        add_vars = set(map(tuple, effective_vars)) - set(map(tuple, effctive_vars_last))
        del_vars = set(map(tuple, effctive_vars_last)) - set(map(tuple, effective_vars))
        effctive_vars_last = vars[:]
        
        # print(f"add vars:{add_vars}")
        # print(f"del vars:{del_vars}")

        # print("effective vars",vars)
        # print("len of effecitve vars",len(vars))
        count_dict = sorted(count_dict.items(), key=lambda item: item[1],reverse=True)
        count_dict = dict(count_dict)
        # print(count_dict)

        # 生成新vars
        # print("2.生成新vars")
        # 方式1：effective vars的子集
        # generate_vars = Iutils.generate_random_subsets(effective_vars,generate_size,2,episode+3)
        
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

        # 方式3：effctive vars的交并集
        # generate_vars = []
        # for i in range(len(effective_vars)):
        #     for j in range(i + 1, len(effective_vars)):
        #         union = list(set(effective_vars[i]) | set(effective_vars[j]))
        #         if len(union) <= episode+1:
        #             # print("lenunion",len(union))
        #             union.sort()
        #             generate_vars.append(union)
        #         ins = list(set(effective_vars[i]) & set(effective_vars[j]))
        #         if ins:
        #             ins.sort()
        #             generate_vars.append(ins)
        
        if generate_size > len(generate_vars):
            generate_size = len(generate_vars)
        random_indices = np.random.choice(len(generate_vars), size=generate_size, replace=False)
        selected_vars = [generate_vars[i] for i in random_indices]
        # selected_vars = generate_vars[:]

        for var in selected_vars:
            if var not in vars:
                vars.append(var)
        # print(len(vars))
        # for var in single_vars:
        #     if [var] not in vars:
        #         vars.append([var])
        # print(len(vars))


        # 随机引入子集
        result_subsets = Iutils.generate_random_subsets(single_vars, subset_size, 2, episode+4)
        for subset in result_subsets:
            if subset not in vars and subset not in oriall_vars:
                vars.append(subset)

        # 显示vars构成
        # print("ori vars",len(ori_vars))
        # print(ori_vars)
        # print("generate vars",len(generate_vars))
        # print(generate_vars)
        # print("selected vars",len(selected_vars))
        # print(selected_vars)
        # print("random vars",len(result_subsets))
        # print(result_subsets)
        # print("number of varibles:",len(vars))

        # 对vars进行封闭集和对称性处理，生成entropydict
        # print("3.对vars进行封闭集和对称性处理，生成entropydict")
        expand_vars = vars[:]
        combs,combinations = Iutils.generate_combs(single_vars,comb_size)
        Iutils.preprocessing_combs(vars,single_vars,expand_vars,combs)
        for var in necessary_vars:
            if var not in expand_vars:
                expand_vars.append(var)
        # print("before symmetrize num of expanded vars:",len(expand_vars))
        entropydict = EntropyEqDict()
        # Xrvs_cons = []
        # Wrvs_cons = []
        # print("all_before_eqdict",len(entropydict_all.eqdict))
        # print("all_before_redict",len(entropydict_all.redict))
        # Iutils.symmetrize_by_dict(user_perm,file_perm,expand_vars,entropydict,entropydict_all)
        Iutils.symmetrize_by_dict_simple(N,K,expand_vars,entropydict_all)

        
        
        # Iutils.symmetrize_simple(N,K,expand_vars,entropydict)
        # print("after symmetreize num of expanded vars:",len(entropydict.redict))
        # print("all_after_eq",len(entropydict_all.eqdict))
        # print("all_after_re",len(entropydict_all.redict))
        # Iutils.symmetrize(user_perm,file_perm,expand_vars,entropydict,Xrvs_cons,Wrvs_cons)
        # print(entropydict)

        # 问题约束
        # print("4.问题约束")
        Iutils.problem_constraints_process(N,K,Wkey,entropydict_all)
        entropydict_all.regenerate_keys()
        for var in expand_vars:
            var_str = ",".join(sorted(var,key=Iutils.sort_key))
            entropydict[var_str] = entropydict_all.get(var_str)
        entropydict.regenerate_keys()
        # print(entropydict.redict)
        # print("entropydictall",len(entropydict_all.redict))
        # print("number of problem variebles",len(entropydict.redict))
        # print("number of all the variebles",len(entropydict.eqdict))

        # 生成不等式集，并合并相同不等式
        # print("5.生成不等式集，并合并相同不等式")
        regions_candidate = copy.deepcopy(regions)
        Iutils.generate_inequalities_combs(vars,entropydict,regions_candidate,combinations)
        # print("before reducing",len(regions.exprs))
        regions_candidate.reduce_redundant_expr()
        # print("number of exprs",len(regions.exprs))

        # self thing
        self.entropydict = entropydict
        self.entropydict_all = entropydict_all
        self.regions = regions
        self.regions_candidate = regions_candidate
        self.vars = vars
        self.oriall_vars = oriall_vars
        self.effctive_vars_last = effctive_vars_last

        # encoding
        VecMatrix = Regions2VecMatrix(regions,N,K,necessary_vars)
        CandidateVecMatrix = Regions2VecMatrix(regions_candidate,N,K,necessary_vars)
        ob = tuple(
            [
                VecMatrix[:,:-1],# A
                VecMatrix[:,-1].reshape(-1),# b
                np.array([N,K,self.square]),# info
                CandidateVecMatrix[:,:-1],
                CandidateVecMatrix[:,-1].reshape(-1)
            ] # 分开为A和b没有实际的意义
        )
        #print(ob)

        # reward calculator
        if len(self.plot_data_list) == 1:
            reward = np.exp(calculate_square(self.plot_data_list[-1])) - 0
        else:
            reward = np.exp(calculate_square(self.plot_data_list[-1])) - np.exp(calculate_square(self.plot_data_list[-2]))
        self.reward_list.append(reward)
        # done logic
        done = (len(self.reward_list) >= self.DoneThreashold and np.average(self.reward_list[-self.DoneThreashold]) <= self.DoneEpsilon) or self.episode >= self.max_apisode_num
        return ob, reward, done

    def render(self,namestring:str|None=None,save_dir:str=None,render_intime:bool=False):
        """
        :param: render_intime: 如果为是，那就直接是在windows系统中直接展示；否则的话是保存为图片。
        """
        plt.rcParams.update({'font.size': 8})  # 减小字体大小

        N = self.N
        K = self.K
        plot_data = self.plot_data_list[-1]
        X_combinations = self.X_combinations
        ine_constraints = self.ine_constraints
        vars = self.render_vars
        point_num = self.point_num
        episode = self.episode

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
        if render_intime:
            plt.show()
        else: # 否则就要保存为图片。没有进行测试过。
            if save_dir == None:
                save_dir = os.path.join(os.path.expanduser(self.pic_storage_path),"N,K={}{}".format(N,K))#
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            try:
                plt.savefig('{}/{}.png'.format(save_dir,namestring))
            except:
                print("savefig error")
                print(f"result: {point_x},{point_y}")
        plt.close()

    def close(self):
        pass

if __name__ == "__main__":
    start_env_time = datetime.datetime.now()
    env = ConverseEnv(6,4)
    ob,info = env.reset(seed = 42)
    for i in range(15):
        action = list(range(ob[-1].shape[0]))
        ob, reward, done = env.step(action)
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        # 修复时间差显示 - 使用str方法并替换字符
        time_diff = datetime.datetime.now() - start_env_time
        total_time = str(time_diff).replace(':', '-').replace('.', '-')
        env.render(save_dir=f"E:/pycharm/python_doc/learntoconverse_reconstructed/01example/best_from_xxy_20250808/test_result/N={env.N},K={env.K}",namestring=f"episode={env.episode},time={time_now},total_time={total_time},total_reward = {round(sum(env.reward_list),5)}") # step 之后才可以进行render哎。
        if done:
            break
    print(env.reward_list)