from ast import Raise
import itertools
import re
import numpy as np
import random
import time
import matplotlib.pyplot as plt


class Iutils:
    """
    A class containing utility functions for the AI for Converse project.
    """

    @staticmethod
    def find_combinations(N, K):
        def backtrack(start, current_combination, current_sum):
            # If the current combination has N elements and the sum is K, add to results
            if len(current_combination) == N and current_sum == K:
                results.append(list(current_combination))
                return
            # If the current combination has more than N elements or the sum exceeds K, return
            if len(current_combination) > N or current_sum > K:
                return
            # Try adding each number from start to 0 to the current combination
            for i in range(start, -1, -1):
                current_combination.append(i)
                backtrack(i, current_combination, current_sum + i)
                current_combination.pop()

        results = []
        backtrack(K, [], 0)
        return results
    
    @staticmethod
    def generate_combinations(N, K):
        # 生成所有可能的组合
        combinations = [''.join(p) for p in itertools.product(map(str, range(1, N + 1)), repeat=K)]
        return combinations

    @staticmethod
    def replace_by_userindex(s, user_perms):
        # Step 1: Remove all non-alphanumeric characters except commas
        s = re.sub(r"[^a-zA-Z0-9,]", "", s)

        # Step 2: Split the input string by commas
        elements = s.split(",")

        # Step 3: Generate all possible permutations of the numbers
        permutations = user_perms
        # print("permutations", permutations)

        # Step 4: Apply each permutation to the original string
        results = []
        for perm in permutations:
            perm_dict = {str(i + 1): str(perm[i]) for i in range(len(perm))}
            new_elements = []
            for e in elements:
                if e.startswith("Z"):
                    prefix = e[0]
                    number = "".join(filter(str.isdigit, e))
                    new_number = "".join(perm_dict[digit] for digit in number)
                    new_elements.append(prefix + new_number)
                elif e.startswith("X"):
                    prefix = e[0]
                    number = "".join(filter(str.isdigit, e))
                    # new_number = "".join(number[perm[i] - 1] for i in range(len(perm)))
                    new_number = [''] * len(number)
                    for i in range(len(perm)):
                        new_index = int(perm_dict[str(i + 1)]) - 1
                        new_number[new_index] = number[i]
                    new_number = ''.join(new_number)
                    # if perm == (3,1,2) and s == "W1,X112,Z1":
                    #     print("new_number",new_number)
                    new_elements.append(prefix + new_number)
                else:
                    new_elements.append(e)
            results.append(",".join(new_elements))

        return results

    @staticmethod
    def replace_by_fileindex(s, file_perms):
        # Step 1: Split the input string by commas
        elements = s.split(",")

        # Step 2: Generate all possible permutations of the numbers
        # permutations = list(itertools.permutations(range(1, N + 1)))
        permutations = file_perms

        results = []

        # Step 3: Apply each permutation to the original string
        for perm in permutations:
            perm_dict = {str(i + 1): str(perm[i]) for i in range(len(perm))}
            new_elements = []
            for e in elements:
                if e.startswith("W") or e.startswith("X"):
                    prefix = e[0]
                    number = "".join(filter(str.isdigit, e))
                    new_number = "".join(perm_dict[digit] for digit in number)
                    new_elements.append(prefix + new_number)
                else:
                    new_elements.append(e)
            results.append(",".join(new_elements))
        return results

    @staticmethod
    def sort_key(element):
        # 定义排序键
        prefix_order = {"W": 0, "X": 2, "Z": 1}
        # 拆分元素
        items = element.split(',')
        # 获取元素个数
        num_elements = len(items)
        # 获取每个元素的排序键
        item_keys = [(prefix_order.get(item[0], 3), int("".join(filter(str.isdigit, item)))) for item in items]
        return (num_elements, item_keys)
    
    @staticmethod
    def sort_elements(elements):
        # 对元素进行排序
        sorted_elements = sorted(elements, key=Iutils.sort_key)
        return sorted_elements

    @staticmethod
    def get_all_subsets(val_list):
        # 生成所有子集，包括空集和自身
        subsets = []
        for r in range(len(val_list) + 1):
            subsets.extend(itertools.combinations(val_list, r))
        return subsets

    @staticmethod
    def replace_by_combined_rules(s, user_perms, file_perms):
        # Step 1: Get all results from replace_by_userindex
        userindex_results = Iutils.replace_by_userindex(s, user_perms)
        userindex_results = list(set(userindex_results))
        # print("userindex_results", userindex_results)

        # Step 2: Apply replace_by_fileindex to each result from replace_by_userindex
        final_results = set()  # Use a set to store unique results
        for result in userindex_results:
            fileindex_results = Iutils.replace_by_fileindex(result, file_perms)
            # print("fileindex_results", fileindex_results)
            for fileindex_result in fileindex_results:
                # Normalize the result by sorting the elements
                elements = fileindex_result.split(",")
                sorted_elements = ",".join(sorted(elements, key=Iutils.sort_key))

                final_results.add(sorted_elements)
        final_results.add(s)
        
        # Step 3: Sort the final results
        sorted_results = Iutils.sort_elements(list(final_results))

        return sorted_results

    @staticmethod
    def generate_random_subsets(lst, num_subsets, min_size, max_size):
        """
        从列表 lst 中随机生成多个个数不同的子集
        :param lst: 输入的列表
        :param num_subsets: 要生成的子集数量
        :param min_size: 子集中元素的最小个数
        :param max_size: 子集中元素的最大个数
        :return: 包含多个子集的列表
        """
        if type(lst[0]) == list:
            str_lst = []
            for item in lst:
                item_str = ",".join(sorted(item,key=Iutils.sort_key))
                str_lst.append(item_str)
        else:
            str_lst = lst[:]
        # print("str_list",str_lst)
        if max_size > len(lst):
            max_size = len(lst)
        total_subsets = (2 ** len(lst)) - 1 - len(lst)
        if num_subsets > total_subsets:
            num_subsets = total_subsets
        subsets = []
        for _ in range(num_subsets):
            # 随机确定当前子集的元素个数
            # random.seed(47)
            subset_size = random.randint(min_size, max_size)
            # 从原列表中随机选取指定数量的不重复元素构成子集
            subset = random.sample(str_lst, subset_size)
            temp_set = set()
            for elm in subset:
                if ',' in elm:
                    sub_elms = elm.split(',')
                    for sub_elm in sub_elms:
                        temp_set.add(sub_elm)
                else:
                    temp_set.add(elm)
            subset = Iutils.sort_elements(list(temp_set))
            # print("sub",subset)
            subsets.append(subset)
        return subsets
    
    @staticmethod
    def symmetry_vars(user_perms,file_perms,vars):
        """
        对目标变量进行对称化处理，只保留对称变量中的第一个
        :param: vars:待处理的变量集
        :return: del_vars: 删除对称变量后的变量集
        
        """
        all_symmetricentropy = []
        # cnt = 0
        del_vars = vars[:]
        sym_vars = []
        flag = False
        for item in vars:
            if type(item) == list:
                flag = True
                item_key = ','.join(sorted(item, key=Iutils.sort_key))
            else:
                item_key = item
            symmetricentropy = Iutils.replace_by_combined_rules(item_key, user_perms, file_perms)
            symmetricentropy = Iutils.sort_elements(symmetricentropy)
            if item_key in all_symmetricentropy:
                del_vars.remove(item)
            else:
                for ent in symmetricentropy:
                    all_symmetricentropy.append(ent)
        if flag:
            for var in del_vars:
                var_key = ','.join(sorted(var, key=Iutils.sort_key))
                sym_vars.append(var_key)
        else:
            sym_vars = del_vars[:]
        return sym_vars

    @staticmethod
    def generate_combs(single_vars, size):
        """
            随机生成size数量的两变量排列组合，分别返回preprocessing函数
            和generate_inequality函数对应的排列组合
            :param: single_vars:单变量
            :param: size:生成变量集大小
            :return: combs:preprocessing函数对应的排列组合，元素类型为字符串
            :return: combinations:generate_inequality函数对应的排列组合，元素类型为随机变量类(Ivar)
        """
        num = random.randint(1, 100)
        var_len = len(single_vars)
        total_size = int(var_len * (var_len - 1) / 2)
        if size > total_size:
            size = total_size
        # num = 20
        print("comb size",size)

        ori_combs = list(itertools.combinations(single_vars, 2))
        # 使用 random.sample 从 ori_combs 中随机抽取 size 个元素
        random.seed(num)
        combs = random.sample(ori_combs, size)

        comb_vars = Ivar.rv(single_vars)
        ori_combinations = list(itertools.combinations(comb_vars, 2))
        # 使用 random.sample 从 ori_combinations 中随机抽取 size 个元素
        random.seed(num)
        combinations = random.sample(ori_combinations, size)

        # print("1", combs)
        # print("1", combinations)
        return combs, combinations
    
    def preprocessing_combs(vars,single_vars,expand_vars,combs):
        """
        对联合熵变量集的预处理，确保集合对后续处理中的运算封闭

        :param vars: I(x;y|z)对应的{z}
        :param single_vars: 单变量集合，用于生成排列组合(x,y)
        :param expand_vars: 经过预处理扩充的变量集，用于更新熵字典，元素为字符串列表
        :param combs:筛选的排列组合，元素类型为字符串
        """

        varlist_W = [ivar for ivar in single_vars if ivar[0] == 'W']
        for var in vars:
            # original version
            if all('W' not in item for item in var):    
                new_var = varlist_W + var
                new_var = Iutils.sort_elements(new_var)
                # if var_str == "X11,Z1,Z2":
                #     print(new_var)
                if new_var not in expand_vars:
                    expand_vars.append(new_var)
            # when var contains W,Z,X, {Z,X} in dict(constraint 3)
            type_list = [s[0] for s in var]
            if set(["W", "Z", "X"]).issubset(set(type_list)) and len(var) == 1:
                new_var = [elem for elem in var if elem not in varlist_W]
                new_var = Iutils.sort_elements(new_var)
                if new_var not in expand_vars:
                    expand_vars.append(new_var)
            # I(x;y|z) terms in dict
            # comb_cnt = 0
            for comb in combs:
                # print("combcnt:",comb_cnt)
                # comb_cnt += 1
                x, y = comb
                diff_var = set(var) - set(comb)

                # I(x;y) {x,y} in dict
                if not diff_var:
                    new_var = list(comb)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                else:
                    # {z} in dict
                    new_var = list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars and len(new_var) != 0:
                        expand_vars.append(new_var)
                    
                    # {x,z} in dict
                    new_var = [x] + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                    
                    # {y,z} in dict
                    new_var = [y] + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                    
                    # {x,y,z} in dict
                    new_var = list(comb) + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
        
    
    @staticmethod
    def preprocessing(vars,single_vars,expand_vars):
        """
        对联合熵变量集的预处理，确保集合对后续处理中的运算封闭

        :param vars: I(x;y|z)对应的{z}
        :param single_vars: 单变量集合，用于生成排列组合(x,y)
        :param expand_vars: 经过预处理扩充的变量集，用于更新熵字典，元素为字符串列表

        """
        # expand_vars = vars[:] # 经过预处理扩充的变量集
        varlist_W = [ivar for ivar in single_vars if ivar[0] == 'W']
        N = len(varlist_W)
        combs = list(itertools.combinations(single_vars, 2))
        # cnt = 0
        for var in vars:

            # original version
            if all('W' not in item for item in var):    
                new_var = varlist_W + var
                new_var = Iutils.sort_elements(new_var)
                # if var_str == "X11,Z1,Z2":
                #     print(new_var)
                if new_var not in expand_vars:
                    expand_vars.append(new_var)
            # when var contains Z,X, {Zk,Xd1-dK,Wdk} in dict(constraint 3)
            if set(["Z", "X"]).issubset(set(s[0] for s in var)):
                keyW_list = [elem for elem in var if elem.startswith("W")]
                keyW_set = set(keyW_list)
                keyX_list = [elem for elem in var if elem.startswith("X")]
                keyZ_list = [elem for elem in var if elem.startswith("Z")]
                keyW_new_set = set()
                for keyZ in keyZ_list:
                    userindex = int(keyZ[1])
                    fileindex_set = set(['W'+ s[userindex] for s in keyX_list])
                    keyW_new_set = keyW_new_set.union(fileindex_set)
                
                if len(keyW_set.union(keyW_new_set)) != N and len(keyW_set.intersection(keyW_new_set)) != 0:
                    new_var = [elem for elem in var if elem not in keyW_new_set]
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)

            # I(x;y|z) terms in dict
            # comb_cnt = 0
            for comb in combs:
                # print("combcnt:",comb_cnt)
                # comb_cnt += 1
                x, y = comb
                diff_var = set(var) - set(comb)

                # I(x;y) {x,y} in dict
                if not diff_var:
                    new_var = list(comb)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                else:
                    # {z} in dict
                    new_var = list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars and len(new_var) != 0:
                        expand_vars.append(new_var)
                    
                    # {x,z} in dict
                    new_var = [x] + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                    
                    # {y,z} in dict
                    new_var = [y] + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
                    
                    # {x,y,z} in dict
                    new_var = list(comb) + list(diff_var)
                    new_var = Iutils.sort_elements(new_var)
                    if new_var not in expand_vars:
                        expand_vars.append(new_var)
    
    @staticmethod
    def symmetrize(user_perms,file_perms,expand_vars,entropydict):
        print("symmetrize")
        s = time.perf_counter()
        symentropy = []
        same_ent = 0
        index = 0
        for item in expand_vars:
            # print(index)
            item_key = ','.join(sorted(item, key=Iutils.sort_key))
            # 对没有进行处理的联合变量进行处理
            if entropydict.get(item_key) is None:
                symmetricentropy = Iutils.replace_by_combined_rules(item_key, user_perms, file_perms) 
                symentropy.append(symmetricentropy[0])
                same_ent -= 1
                for symitem in symmetricentropy:
                    # print(symitem)
                    entropydict[symitem] = index
                    same_ent += 1
                index += 1
        e = time.perf_counter()
        t = e - s
        print(f"symmetrize time:{t}")
        # for ent in symentropy:
        #     ent_list = ent.split(",")
        #     if not set(["Z"]).issubset(set(s[0] for s in ent_list)) and not set(["X"]).issubset(set(s[0] for s in ent_list)):
        #         Wrvs_cons.append(ent)
            
        #     if len(ent_list) == 1 and ent[0] == "X":
        #         Xrvs_cons.append(ent)
        # Wrvs_cons = set(Wrvs_cons)
        # Xrvs_cons = set(Xrvs_cons)     
        print("number of varibles",len(symentropy))
        return
    


    @staticmethod
    def symmetrize_simple(N, K, expand_vars, entropydict):
        print("symmetrize")
        s = time.perf_counter()
        symentropy = []
        same_ent = 0
        index = len(entropydict.redict)
        # print("index",index)
        # entropydict, unclassified_vars = classify_vars(expand_vars, N, K)
        for item in expand_vars:
            # print(index)
            # print("item",item)
            W_indices = [int(elem[1]) for elem in item if elem.startswith("W")]
            Z_indices = [int(elem[1]) for elem in item if elem.startswith("Z")]
            file_perms = Iutils.get_permutations(W_indices, N)
            # print("file_perms",file_perms)
            user_perms = Iutils.get_permutations(Z_indices, K)
            # print("user_perms",user_perms)
            item_key = ','.join(sorted(item, key=Iutils.sort_key))
            # 对没有进行处理的联合变量进行处理
            if entropydict.get(item_key) is None:
                # print("item_key",item_key)
                symmetricentropy = Iutils.replace_by_combined_rules(item_key, user_perms, file_perms) 
                # print("symmetricentropy",symmetricentropy)
                # symentropy.append(symmetricentropy[0])
                same_ent -= 1
                if entropydict.get(symmetricentropy[0]) is None:
                    # print("1")
                    for symitem in symmetricentropy:
                        # print(symitem)
                        entropydict[symitem] = index
                        same_ent += 1
                    index += 1
                else:
                    # print("0")
                    entropydict[item_key] = entropydict[symmetricentropy[0]]
                    same_ent += 1
        e = time.perf_counter()
        t = e - s
        print(f"symmetrize time:{t}")
        print("number of varibles",len(entropydict.redict))
        return
    
    @staticmethod
    def symmetrize_by_dict_simple(N,K,expand_vars,entropydict_all):
        index = len(entropydict_all.redict)
        cnt_get = 0
        cnt_notget = 0
        t_get = 0
        t_notget = 0
        for var in expand_vars:
            var_str = ",".join(sorted(var,key=Iutils.sort_key))
            if entropydict_all.get(var_str) is None:
                s2 = time.perf_counter()
                W_indices = [int(elem[1]) for elem in var if elem.startswith("W")]
                Z_indices = [int(elem[1]) for elem in var if elem.startswith("Z")]
                file_perms = Iutils.get_permutations(W_indices, N)
                # print("file_perms",file_perms)
                user_perms = Iutils.get_permutations(Z_indices, K)
                # print("user_perms",user_perms)
                symmetricentropy = Iutils.replace_by_combined_rules(var_str, user_perms, file_perms) 
                if entropydict_all.get(symmetricentropy[0]) is not None:
                    # print("get base")
                    # print(entropydict_all.get(symmetricentropy[0]))
                    for symitem in symmetricentropy:
                        # print(symitem)
                        if entropydict_all.get(symitem) is None:
                            entropydict_all[symitem] = entropydict_all[symmetricentropy[0]]
                else:
                    # print("not get base")
                    # print(index)
                    for symitem in symmetricentropy:
                        # print(symitem)
                        entropydict_all[symitem] = index
                    index += 1
                e2 = time.perf_counter()
                t2 = e2 - s2
                # print("t2",t2)
                t_notget += t2
                cnt_notget += 1
            else:
                # print("not get")
                cnt_get += 1
                continue
        print(f"get:{cnt_get},time:{t_get}")
        print(f"not get:{cnt_notget},time:{t_notget}")
        return
    
    @staticmethod
    def symmetrize_by_dict_simple_ori(N,K,expand_vars,entropydict,entropydict_all):
        index = len(entropydict_all.redict)
        cnt_get = 0
        cnt_notget = 0
        t_get = 0
        t_notget = 0
        for var in expand_vars:
            var_str = ",".join(sorted(var,key=Iutils.sort_key))
            if entropydict_all.get(var_str) is not None:
                s1 = time.perf_counter()
                # print("get")
                entropydict[var_str] = entropydict_all.get(var_str)
                e1 = time.perf_counter()
                cnt_get += 1
                t1 = e1 - s1
                # print("t1",t1)
                t_get += t1
            else:
                # print("not get")
                s2 = time.perf_counter()
                W_indices = [int(elem[1]) for elem in var if elem.startswith("W")]
                Z_indices = [int(elem[1]) for elem in var if elem.startswith("Z")]
                file_perms = Iutils.get_permutations(W_indices, N)
                # print("file_perms",file_perms)
                user_perms = Iutils.get_permutations(Z_indices, K)
                # print("user_perms",user_perms)
                
                symmetricentropy = Iutils.replace_by_combined_rules(var_str, user_perms, file_perms) 
                if entropydict_all.get(symmetricentropy[0]) is not None:
                    # print("get base")
                    # print(entropydict_all.get(symmetricentropy[0]))
                    for symitem in symmetricentropy:
                        # print(symitem)
                        if entropydict_all.get(symitem) is None:
                            entropydict_all[symitem] = entropydict_all[symmetricentropy[0]]
                else:
                    # print("not get base")
                    # print(index)
                    for symitem in symmetricentropy:
                        # print(symitem)
                        entropydict_all[symitem] = index
                    index += 1
                entropydict[var_str] = entropydict_all.get(var_str)
                e2 = time.perf_counter()
                t2 = e2 - s2
                # print("t2",t2)
                t_notget += t2
                cnt_notget += 1
        print(f"get:{cnt_get},time:{t_get}")
        print(f"not get:{cnt_notget},time:{t_notget}")
        return

    @staticmethod
    def symmetrize_by_dict(user_perm,file_perm,expand_vars,entropydict,entropydict_all):
        index = len(entropydict_all.redict)
        cnt_get = 0
        cnt_notget = 0
        t_get = 0
        t_notget = 0
        for var in expand_vars:
            var_str = ",".join(sorted(var,key=Iutils.sort_key))
            if entropydict_all.get(var_str) is not None:
                s1 = time.perf_counter()
                # print("get")
                # entropydict[var_str] = entropydict_all.get(var_str)
                e1 = time.perf_counter()
                cnt_get += 1
                t1 = e1 - s1
                # print("t1",t1)
                t_get += t1
            else:
                # print("not get")
                s2 = time.perf_counter()
                symmetricentropy = Iutils.replace_by_combined_rules(var_str, user_perm, file_perm) 
                for symitem in symmetricentropy:
                    # print(symitem)
                    entropydict_all[symitem] = index
                index += 1
                # entropydict[var_str] = entropydict_all.get(var_str)
                e2 = time.perf_counter()
                t2 = e2 - s2
                # print("t2",t2)
                t_notget += t2
                cnt_notget += 1
        print(f"get:{cnt_get},time:{t_get}")
        print(f"not get:{cnt_notget},time:{t_notget}")
        return
    
    @staticmethod
    def get_permutations(indices, N):
    
        perms = []
        base_list = [0] * N 
        
        # 将 indices 位置的元素依次设为 1, 2, 3, ...
        ori_perms = itertools.permutations(range(1, len(indices) + 1), len(indices))
        for ori_perm in ori_perms:
            for idx,value in zip(indices,ori_perm):
                base_list[idx-1] = value
            # 获取其余位置的索引
            remaining_pos = [i for i in range(N) if i+1 not in indices]
            # print("remain position",remaining_pos)
            
            # 对剩余位置进行 (N - len(W_indices)) 范围内的任意排列组合
            # 对填充内容进行排列组合
            remaining_permutations = itertools.permutations(range(len(indices)+1, N+1), len(remaining_pos))  
            
            # 将每种排列组合填入剩余位置
            for perm in remaining_permutations:
                # print("perm",perm)
                temp_list = base_list[:]
                for pos, value in zip(remaining_pos, perm):
                    temp_list[pos] = value
                perms.append(temp_list)
        
        return perms

    @staticmethod
    def problem_constraints_process(N, K, Wkey, entropydict):
        """
        通过问题特定约束对联合熵变量集进行合并

        :param Wkey: 包含所有文件的字符串，eg "W1,W2", 可直接作为entropydict的索引
        :param entropydict: 熵字典，用于存储和更新熵相关信息
        :return: 更新后的熵字典
        
        """
        basevalue_W = entropydict[Wkey]
        Wkey_list = Wkey.split(",")
        N = len(Wkey_list)
        entlist = entropydict.redict.items()
        baseentropy = [val[0] for _, val in entlist]
        baseentropy = set(baseentropy)
        index = len(entropydict.redict)
        # print(baseentropy)
        # 1. H(Z_k|W_1, W_2, ..., W_N) = 0
        # 2. H(X_d1,d2,...,d_K|W_1, W_2, ..., W_N) = 0
        for item in baseentropy:
            # print("item",item)
            item_list = item.split(",")
            if set(Wkey_list).issubset(set(item_list)):
                # print("item_list",item_list)
                basevalue = entropydict[Wkey]
                itemvalue = entropydict[item]
                if basevalue != itemvalue:
                    updatekeys = entropydict.get_keys_by_value(itemvalue)
                    entropydict.remove_keys_by_value(itemvalue)
                    entropydict.batch_update(updatekeys, basevalue)
                continue
            # 3. H(W_d_k|Z_k,X_d1,d2,...,d_K) = 0      
            # check if the item contains W, Z, X
            # type_list = [s[0] for s in item]
            if set(["Z", "X"]).issubset(set(s[0] for s in item_list)):
                keyW_list = [elem for elem in item_list if elem.startswith("W")]
                keyW_set = set(keyW_list)
                keyX_list = [elem for elem in item_list if elem.startswith("X")]
                keyZ_list = [elem for elem in item_list if elem.startswith("Z")]
                keyW_new_set = set()
                for keyZ in keyZ_list:
                    userindex = int(keyZ[1])
                    fileindex_set = set(['W'+ s[userindex] for s in keyX_list])
                    keyW_new_set = keyW_new_set.union(fileindex_set)
                
                if len(keyW_set.union(keyW_new_set)) == N:
                    # H(item)=H(W1,W2,...,WN)
                    itemvalue = entropydict[item]
                    if basevalue_W != itemvalue:
                        try:
                            updatekeys = entropydict.get_keys_by_value(itemvalue)
                            # if "W1,X112,Z1" in updatekeys:
                            #     print("first: ", epoch, item)
                            entropydict.batch_update(updatekeys, basevalue_W)
                        except:
                            print("First Part Error keys: ", item, Wkey)
                            print("First Part Error values: ", itemvalue, basevalue_W)
                elif len(keyW_set.intersection(keyW_new_set)) == 0:
                    # simplest form
                    continue
                else:
                    item_list = [elem for elem in item_list if elem not in keyW_new_set]
                    basekey = ','.join(sorted(item_list, key=Iutils.sort_key))
                    if item == "W1,W2,Z1,X112,X113,X132":
                        print(basekey)
                    # if entropydict.get(basekey) == None:
                    #     continue
                    if entropydict.get(basekey) == None:
                        W_indices = [int(elem[1]) for elem in item_list if elem.startswith("W")]
                        Z_indices = [int(elem[1]) for elem in item_list if elem.startswith("Z")]
                        file_perms = Iutils.get_permutations(W_indices, N)
                        # print("file_perms",file_perms)
                        user_perms = Iutils.get_permutations(Z_indices, K)
                        # print("user_perms",user_perms)
                        symmetricentropy = Iutils.replace_by_combined_rules(basekey, user_perms, file_perms)
                        # if item == "W1,W2,Z1,X112,X113,X132":
                        #     print(symmetricentropy[0])
                        if entropydict.get(symmetricentropy[0]) is None:
                            # print("1")
                            for symitem in symmetricentropy:
                                # print(symitem)
                                entropydict[symitem] = index
                            index += 1
                        else:
                            entropydict[basekey] = entropydict[symmetricentropy[0]]
                    basevalue = entropydict[basekey]
                    itemvalue = entropydict[item]
                    if basevalue != itemvalue:
                        try:
                            updatekeys = entropydict.get_keys_by_value(itemvalue)
                            # if "W1,X112,Z1" in updatekeys:
                            #     print("second: ", epoch, item)
                            entropydict.batch_update(updatekeys, basevalue)
                        except:
                            print("Second Part Error keys: ", item, basekey)
                            print("Second Part Error values: ", itemvalue, basevalue)
            
        # entropydict.regenerate_keys()
    


    # 等式约束 有点复杂且用处应该不大，暂时搁置
    # def eq_constraints(N, K, single_vars, Wkey, entropydict, eq_regions):
    #     """
    #     为所有非Z、X类型的单变量两两添加I(X;Y)=0等式约束
    #     :param single_vars: 单变量名列表（如["W1", "W2", "Z1", ...]）
    #     :return: 约束表达式列表，每个元素为("I", X, Y)
    #     """
    #     constraints = []
    #     # 只保留Z、X类型的单变量
    #     zx_vars = [v for v in single_vars if (v.startswith("Z") or v.startswith("X"))]
    #     zx_subsets = Iutils.get_all_subsets(zx_vars)
    #     for i in range(len(filtered_vars)):
    #         for j in range(i + 1, len(filtered_vars)):
    #             x = Ivar.rv(filtered_vars[i])
    #             y = Ivar.rv(filtered_vars[j])
    #             # x = Ivar.rv(x)
    #             # y = Ivar.rv(y)
    #             #print(x,type(x),y,type(y))
    #             term_x = Comp(set(x))
    #             term_y = Comp(set(y))
    #             #print(term_x,type(term_x),term_y,type(term_y))
    #             term = Term.I(term_x, term_y)
    #             terms = [term]
    #             expr = Expr.empty()
    #             eq_regions.append_expr(expr.inequality(terms=terms, edict=entropydict))
    #             # 这里用元组或自定义结构表示I(X;Y)=0
    #     return

    @staticmethod
    def generate_inequalities_combs(vars,entropydict,regions,combinations):
        """
            更新不等式集regions，由变量生成不等式
        
            :param vars: I(x;y|z)中的z集合
            :param single_vars: 单变量集合，用于生成排列组合(x,y)
            :param  entropydict: 熵字典，存储所有互信息扩展后的联合熵变量的值，
            保证优化变量与不等式矩阵一一对应
            :param regions: 不等式集，调用函数前为上一轮迭代得到的有效不等式，
            调用函数后增加新变量生成的不等式
            :param combinations:排列组合，元素类型为随机变量类(Ivar)

        """
        xyz_list = []
        for combination in combinations:
            x, y = combination
            
            # print("XY",x,y,type(x))
            term_x = Comp(set([x]))
            term_y = Comp(set([y]))
            # print(term_x)
            for ori_var in vars:
                
                # print("xyz",xyz)
                var = Ivar.rv(ori_var)
                # 原逻辑
                diffset = set(var) - set(combination)
                # print("diff_set",diffset)
                term_z = Comp(diffset)
                # print("z",term_z)

                # 现逻辑
                # term_z = Comp(var)
                # xyz = [x,y,ori_var]
                # print("xyz",xyz)
                # xyz_list.append(xyz)

                if term_z.is_empty():
                    term = Term.I(term_x, term_y)
                    xyz = [x,y]
                    xyz_list.append(xyz)
                else:
                    term = Term.Ic(term_x, term_y, term_z)
                    xyz = [x,y,ori_var]
                    xyz_list.append(xyz)
                terms = [term]
                # print("term",term)
                expr = Expr.empty()
                # expr.inequality(terms=terms, edict=entropydict)
                expand_expr = expr.inequality(terms=terms, edict=entropydict)
                if expand_expr.is_empty():
                    xyz_list.pop()
                    continue
                key = expand_expr.terms_to_str()
                # 区域中存在该不等式
                if regions.exprdict.get(key) is None:
                    regions.exprdict[key] = expand_expr
                    regions.exprs.append(expand_expr)
                    # print("1",expand_expr)
                    # print("0",xyz_list[-1])
                else:
                    xyz_list.pop()

                # # 新区域比原区域紧,如何删除原表达式？
                # elif expand_expr.value > expr.exprdict[key].value:
                #     expr.exprdict[key] = expr
                #     expr.exprs.append(expr)
                # regions.append_expr(expr.inequality(terms=terms, edict=entropydict))
        return xyz_list
    
    @staticmethod
    def generate_inequalities(vars,single_vars,entropydict,regions):
        """
            更新不等式集regions，由变量生成不等式
        
            :param vars: I(x;y|z)中的z集合
            :param single_vars: 单变量集合，用于生成排列组合(x,y)
            :param  entropydict: 熵字典，存储所有互信息扩展后的联合熵变量的值，
            保证优化变量与不等式矩阵一一对应
            :param regions: 不等式集，调用函数前为上一轮迭代得到的有效不等式，
            调用函数后增加新变量生成的不等式

        """
        comb_vars = Ivar.rv(single_vars)
        combinations = list(itertools.combinations(comb_vars, 2))
        # comb_cnt = 0
        cnt = 0
        term_all = []
        for combination in combinations:
            x, y = combination
            x_str = [x.name]
            y_str = [y.name]
            term_x = Comp(set([x]))
            term_y = Comp(set([y]))
            # cnt = 0
            for var_list in vars:
                # print("var",var)
                # print(f"comb:{comb_cnt},var:{cnt}")
                # last_len = len(regions.exprs)
                
                all_vars = set(x_str).union(set(y_str)).union(set(var_list))
                # print("all",all_vars)
                x_count = sum(1 for v in all_vars if v.startswith('X'))
                # print("Xcount",x_count)
                # if x_count > 1:
                #     continue

                var = Ivar.rv(var_list)
                diffset = set(var) - set(combination)
                term_z = Comp(diffset)
                if term_z.is_empty():
                    term = Term.I(term_x, term_y)
                else:
                    term = Term.Ic(term_x, term_y, term_z)
                terms = [term]
                # if terms not in term_all:
                #     term_all.append(terms)
                #     # print(cnt,term)
                #     cnt += 1
                # print("all",term_all)
                expr = Expr.empty()
                # expr.inequality(terms=terms, edict=entropydict)
                regions.append_expr(expr.inequality(terms=terms, edict=entropydict))


    def Regions2Matrix(entropydict, regions):
        """
        Convert regions to a matrix in the form of list.
        """
        ine_constraints = []
        ent_num = len(entropydict.redict) + 3
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

        ent_num -= 1 # 实际变量数量

        return ine_constraints, prob_cons_num, ent_num
    
    @staticmethod
    def compute_slopes(x, y):
        slopes = []
        for i in range(1, len(x)):
            slope = (y[i] - y[i-1]) / (x[i] - x[i-1]) if x[i] != x[i-1] else np.inf  # 防止除0错误
            slope = round(slope,3)
            slopes.append(slope)
        return slopes   
    
    def plot_cutset_bound(N,K,point_num):
        M = np.linspace(0,N,point_num*N+1)
        R = []
        for s in range(1,min(N,K)+1):
            R_s = s - s / (N // s) * M
            R_s = R_s.tolist()
            R.append(R_s)
        R = np.array(R)
        R_cutset = R.max(axis=0)

        result_slope = Iutils.compute_slopes(M,R_cutset)
        point_x = []
        point_y = []
        for i in range(1,len(result_slope)):
            if result_slope[i-1] != result_slope[i]:
                point_x.append(M[i])
                point_y.append(R_cutset[i])
        plt.scatter(point_x,point_y,color='blue')

        # for xi, yi in zip(point_x, point_y):
        #     label = f"({round(xi,3)}, {round(yi,3)})"
        #     plt.annotate(label,  
        #                 (xi, yi),  
        #                 textcoords="offset points",  
        #                 xytext=(-40, -5),  
        #                 ha='center',
        #                 color="blue") 
        plt.plot(M,R_cutset,color='blue', linestyle=':', linewidth=2, label='cut-set bound')


    @staticmethod
    def plot_inner_bound(N,K):
        # 第一组点
        inner_point_x1 = []
        inner_point_y1 = []
        for t in range(K + 1):
            x1 = t * ((N - 1) * t + K - N) / (K * (K - 1))
            y1 = N * (K - t) / K
            inner_point_x1.append(x1)
            inner_point_y1.append(y1)

        # 第二组点
        inner_point_x2 = np.linspace(0, N, K+1)
        inner_point_y2 = []
        for M in inner_point_x2:
            coef1 = 1 / (1 + K * M / N)
            coef2 = N / K
            coef = min(coef1, coef2)
            # coef = coef1
            y2 = K * (1 - M / N) * coef
            inner_point_y2.append(y2)

        # 合并两组点
        all_x = inner_point_x1 + inner_point_x2.tolist()
        all_y = inner_point_y1 + inner_point_y2
        # 按 x 值排序
        sorted_points = sorted(zip(all_x, all_y), key=lambda point: point[0])

        # 处理相同 x 值，取较小的 y 值
        inner_x = []
        inner_y = []
        i = 0
        while i < len(sorted_points):
            current_x = sorted_points[i][0]
            min_y = sorted_points[i][1]
            j = i + 1
            if j < len(sorted_points) and sorted_points[j][0] == current_x:
                min_y = min(min_y, sorted_points[j][1])
            new_point = (current_x, min_y)
            # 检查添加新点后是否保持凸性
            if new_point[0] not in inner_x:
                inner_x.append(current_x)
                inner_y.append(min_y)
                while len(inner_x) >= 3:
                    x0, y0 = inner_x[-3], inner_y[-3]
                    x1, y1 = inner_x[-2], inner_y[-2]
                    x2, y2 = inner_x[-1], inner_y[-1]
                    # 判断是否满足凸性条件（斜率单调递减）
                    k1 = (y1 - y0) / (x1 - x0)
                    k2 = (y2 - y1) / (x2 - x1)
                    if k1 < k2:
                        break
                    else:
                        inner_x.pop(-2)
                        inner_y.pop(-2)
            i = j
        inner_x.sort()
        inner_y.sort(reverse=True)
        plt.plot(inner_x, inner_y, linestyle='--', linewidth=3, color="black",label="inner bound")
        plt.scatter(inner_x[1:-1], inner_y[1:-1], color="black")
        # for xi, yi in zip(inner_x[1:-1], inner_y[1:-1]):
        #     label = f"({round(xi,3)}, {round(yi,3)})"
        #     plt.annotate(label,  
        #                 (xi, yi),  
        #                 textcoords="offset points",  
        #                 xytext=(-30, -5),  
        #                 ha='center',
        #                 color="black")
    
    @staticmethod
    def plot_bound(x,y,color,label,line=1,annotate=0):
        result_slope1 = Iutils.compute_slopes(x,y)
        point_x = []
        point_y = []
        for i in range(1,len(result_slope1)):
            if result_slope1[i-1] != result_slope1[i]:
                point_x.append(x[i])
                point_y.append(y[i])
        plt.scatter(point_x,point_y,color=color)
        
        if annotate == True:
            for xi, yi in zip(point_x, point_y):
                label = f"({round(xi,3)}, {round(yi,3)})"
                plt.annotate(label,  
                            (xi, yi),  
                            textcoords="offset points",  
                            xytext=(30, 5),  
                            ha='center')  
        plt.plot(x, y, color=color, linewidth=line, label=label)
        return
    
    def find_line_equation(point1, point2):
        # 提取两点的坐标
        x1, y1 = point1
        x2, y2 = point2

        # 计算斜率
        if x2 - x1 == 0:
            # 处理垂直直线的情况
            return f"x = {x1}"
        slope = (y2 - y1) / (x2 - x1)

        # 计算截距
        intercept = y1 - slope * x1

        return slope,intercept
    
    def find_intersection(m1, b1, m2, b2):
        """
        该函数用于计算两条直线的交点
        :param m1: 第一条直线的斜率
        :param b1: 第一条直线的截距
        :param m2: 第二条直线的斜率
        :param b2: 第二条直线的截距
        :return: 交点的坐标 (x, y)，若两直线平行则返回 None
        """
        # 检查两条直线是否平行
        if m1 == m2:
            return None

        # 计算交点的 x 坐标
        x = (b2 - b1) / (m1 - m2)
        # 计算交点的 y 坐标
        y = m1 * x + b1

        return (x, y)


class Ivar:
    """Random variable class"""

    def __init__(self, name, region=None):
        self.name = name
        self.region = None

    def rv(names):
        if type(names) == list:
            return [Ivar(name) for name in names]
        elif type(names) == str:
            return [
                Ivar(name) for name in re.sub(r"[^a-zA-Z0-9,]", "", names).split(",")
            ]

    def __hash__(self):
        # print(f"Calculating hash for Ivar: {self.name}")
        return hash(self.name)
        
    def __eq__(self, other):
        # print(f"Comparing Ivar: {self.name} with {other.name if isinstance(other, Ivar) else other}")
        if isinstance(other, Ivar):
            return self.name == other.name
        return False
        
    def __repr__(self) -> str:
        return self.name


class Comp:
    """Compound random variable"""

    def __init__(self, varset):
        self.varset = varset

    def empty():
        return Comp(set())

    def is_empty(self):
        return len(self.varset) == 0

    def jes(varlist):
        if type(varlist) == list:
            return Comp(set(varlist))
        elif type(varlist) == str:
            namelist = re.sub(r"[^a-zA-Z0-9,]", "", varlist).split(",")
            return Comp(set(Ivar.rv(namelist)))
        elif type(varlist) == tuple:
            return Comp(set([*varlist]))
        elif type(varlist) == set:
            return Comp(varlist)

    def copy(self):
        return Comp(self.varset)

    def length(self):
        return len(self.varset)

    def to_list(self):
        return sorted(list(self.varset), key=lambda v: (v.name[0], int(v.name[1:])))

    def to_str_list(self):
        # return sorted(
        #     [v.name for v in self.varset], key=lambda name: (name[0], int(name[1:]))
        # )
        return sorted(
            [v.name for v in self.varset], key=Iutils.sort_key)

    def to_str(self):
        return ",".join(self.to_str_list())

    def __repr__(self) -> str:
        return "{" + self.to_str() + "}"

    def __add__(self, other):
        return Comp(self.varset.union(other.varset))
    
    # def __eq__(self, other):
    #     if isinstance(other, Comp):
    #         return self.varset == other.varset
    #     return False


class TermType:
    H = 0
    Hc = 1
    I = 2
    Ic = 3
    NoneType = -1


class Term:
    """Term class: H(X), H(X|Y), etc."""

    def __init__(self, x, z=None, coef=1, value=None, termtype=TermType.NoneType):
        self.x = x
        if z is None:
            self.z = Comp.empty()
        else:
            self.z = z
        self.coef = coef
        self.value = value
        self.termtype = termtype

    def copy(self):
        return Term([a.copy() for a in self.x], self.z.copy(), self.coef)

    # H(X)
    def H(x):
        x = Comp.jes(x)
        return Term([x.copy()], Comp.empty(), termtype=TermType.H)

    # H(X|Z)
    def Hc(x, z):
        return Term([x.copy()], z.copy(), termtype=TermType.Hc)

    # I(X;Y)
    def I(x, y):
        return Term([x.copy(), y.copy()], Comp.empty(), termtype=TermType.I)

    # I(X;Y|Z)
    def Ic(x, y, z):
        return Term([x.copy(), y.copy()], z.copy(), termtype=TermType.Ic)

    def Hc_eq_zero(self, edict):
        # H(x|z) = 0
        if len(self.x) == 1 and not self.z.is_empty():
            # H(x,z) = H(z)
            key_z = self.z.to_str()
            key_xz = (self.x[0] + self.z).to_str() # 把长度为1的列表中的项提取出来
            # print("key_xz", key_xz)
            value_z = edict[key_z] 
            value_xz = edict[key_xz]
            if value_xz != value_z:
                xz_eq_keys = edict.get_keys_by_value(value_xz)
                # edict.remove_keys_by_value(value_xz)
                edict.batch_update(xz_eq_keys, value_z)
        # I() = 0
        else:
            pass
        return edict

    def simplify(self, edict):
        terms = []
        coef = self.coef
        if self.termtype == TermType.H:
            key_x = self.x[0].to_str()
            value_x = edict[key_x]
            return coef*Term.H(edict.get_keys_by_value(value_x)[0])
        elif self.termtype == TermType.Hc:
            # H(x|z) = H(x,z) - H(z)
            key_xz = (self.x[0] + self.z).to_str()
            key_z = self.z.to_str()
            value_xz = edict[key_xz]
            value_z = edict[key_z]
            term_xz = coef*Term.H(edict.get_keys_by_value(value_xz)[0])
            terms.append(term_xz)
            term_z = -coef*Term.H(edict.get_keys_by_value(value_z)[0])
            terms.append(term_z)
        elif self.termtype == TermType.I:
            # I(x;y) = H(x) + H(y) - H(x,y)
            key_x = self.x[0].to_str()
            key_y = self.x[1].to_str()
            key_xy = (self.x[0] + self.x[1]).to_str()
            value_x = edict[key_x]
            value_y = edict[key_y]
            value_xy = edict[key_xy]
            term_x = coef*Term.H(edict.get_keys_by_value(value_x)[0])
            terms.append(term_x)
            term_y = coef*Term.H(edict.get_keys_by_value(value_y)[0])
            terms.append(term_y)
            term_xy = -coef*Term.H(edict.get_keys_by_value(value_xy)[0])
            terms.append(term_xy)
        elif self.termtype == TermType.Ic:
            # I(x;y|z) = H(x,z) + H(y,z) - H(z) - H(x,y,z)
            # print(self.x[0],self.x[1],self.z)
            # print("self.x",self.x)

            # key_xz = (Iutils.sort_key(self.x[0] + self.z)).to_str()
            # key_yz = (Iutils.sort_key(self.x[1] + self.z)).to_str()
            # key_z = self.z.to_str()
            # key_xyz = (Iutils.sort_key(self.x[0] + self.x[1] + self.z)).to_str()
            key_xz = (self.x[0] + self.z).to_str()
            key_yz = (self.x[1] + self.z).to_str()
            key_z = self.z.to_str()
            key_xyz = (self.x[0] + self.x[1] + self.z).to_str()
            # print("key_xz",key_xz)
            # print("key_yz",key_yz)
            # print("key_z",key_z)
            # print("key_xyz",key_xyz)
            value_xz = edict[key_xz]
            value_yz = edict[key_yz]
            value_z = edict[key_z]
            value_xyz = edict[key_xyz]
            term_xz = coef*Term.H(edict.get_keys_by_value(value_xz)[0])
            terms.append(term_xz)
            term_yz = coef*Term.H(edict.get_keys_by_value(value_yz)[0])
            terms.append(term_yz)
            term_z = -coef*Term.H(edict.get_keys_by_value(value_z)[0])
            terms.append(term_z)
            term_xyz = -coef*Term.H(edict.get_keys_by_value(value_xyz)[0])
            terms.append(term_xyz)
        return terms

    def equal(self, other):
        if isinstance(other, list):
            return self.x[0].to_str()==other[0] and self.x[1].to_str()==other[1] and self.z.to_str()==other[2]
        elif isinstance(other, Term):
            return self.x[0]==other.x[0] and self.x[1]==other.x[1] and self.z==other.z
        else:
            return False
    
    def to_ent_str(self):
        if len(self.x) == 1:
            if self.z.is_empty():
                return f"{self.x[0].to_str()}"
            else:
                Raise(TypeError("Entropy term type error"))
        else:
            Raise(TypeError("Entropy term type error"))
    
    def to_str(self):
        return self.__repr__()

    def __add__(self, other):
        if self.termtype == other.termtype:
            return Term(self.x, self.z, self.coef + other.coef)
        else:
            print(self.termtype, other.termtype)
            print(self.x, other.x)
            print(self.z, other.z)
            print(self.coef, other.coef)
            raise TypeError("Different term type")
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Term(self.x, self.z, self.coef * other)
        else:
            raise TypeError("Multiplication only supports scalar")
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self) -> str:
        if np.abs(self.coef - 1) < 1e-5:
            coef = ""
        elif np.abs(self.coef + 1) < 1e-5:
            coef = "-"
        else:
            coef = (np.round(self.coef, 5)).astype(str)
        if len(self.x) == 1:
            if self.z.is_empty():
                return f"{coef}H({self.x[0]})"
            else:
                return f"{coef}H({self.x[0].to_str()}|{self.z.to_str()})"
        else:
            if self.z.is_empty():
                return f"{coef}I({self.x[0].to_str()};{self.x[1].to_str()})"
            else:
                return f"{coef}I({self.x[0].to_str()};{self.x[1].to_str()}|{self.z.to_str()})"


class Expr:
    """Expression class: entropy inequality expression"""

    def __init__(self, terms, eqtype="ge", value=0, indexdict=None):
        # default: >= 0
        self.terms = terms
        self.eqtype = eqtype
        self.value = value
        self.indexdict = indexdict

    def copy(self):
        return Expr([term.copy() for term in self.terms], self.eqtype, self.value, self.indexdict.copy())
    
    def empty():
        return Expr([])
    
    def is_empty(self):
        return len(self.terms) == 0
    
    def add_terms(self, terms=None):
        if terms is None:
            terms = self.terms
        count = 0
        indexdict = {}
        baseterms = []
        for term in terms:
            key = term.x[0].to_str()
            if indexdict.get(key) is None:
                indexdict[key] = count
                baseterms.append(term)
                count += 1
            else:
                index = indexdict[key]
                baseterms[index] += term
        baseterms = [term for term in baseterms if np.abs(term.coef) > 1e-5]
        indexdict = {v.x[0].to_str(): k for k, v in enumerate(baseterms)}
        return baseterms, indexdict

    def inequality(self, terms, edict, eqtype="ge", value=0):
        entropy_terms = []
        for term in terms:
            # print(term)
            baseterms = term.simplify(edict)
    
            # print("baseterms:",baseterms)
            if isinstance(baseterms, list):
                entropy_terms.extend(baseterms)
            else:
                entropy_terms.append(baseterms)
        terms, indexdict = self.add_terms(entropy_terms)
        # print("terms",terms)
        return Expr(terms, eqtype, value, indexdict)
    
    def sort_terms(self):
        # Sort terms by coefficient magnitude first, then by term string representation using Iutils.sort_key
        self.terms.sort(key=lambda term: (-term.coef, Iutils.sort_key(term.x[0].to_str())))

    def __add__(self, other):
        if self.eqtype == other.eqtype:
            terms = self.terms + other.terms
            values = self.value + other.value
            terms, indexdict = self.add_terms(terms)
            return Expr(terms, self.eqtype, values, indexdict)
        else:
            raise TypeError("Different inequality type")
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expr([term*other for term in self.terms], self.eqtype, self.value*other, self.indexdict)
        else:
            raise TypeError("Multiplication only supports scalar")
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def terms_to_str(self):
        result = " + ".join([term.to_str() for term in self.terms])
        result = re.sub(r"\+\s*-\s*", "- ", result)
        if result.startswith("+ "):
            result = result[2:]
        return result

    def eqtype_to_str(self):
        if self.eqtype == "ge":
            return ">="
        elif self.eqtype == "le":
            return "<="
        else:
            return "="

    def __repr__(self) -> str:
        result = (
            " + ".join([term.to_str() for term in self.terms])
            + f" {self.eqtype_to_str()} {self.value}"
        )
        result = re.sub(r"\+\s*-\s*", "- ", result)
        if result.startswith("+ "):
            result = result[2:]
        return result


class Region:
    """Region class: contains a list of expressions"""

    def __init__(self, exprs):
        self.exprs = exprs
        self.exprdict = {}

    # def copy(self):
    #     return Region([expr.copy() for expr in self.exprs])
    
    def copy(self):
        new_exprs = [expr.copy() for expr in self.exprs]
        new_exprdict = self.exprdict.copy()
        new_region = Region(new_exprs)
        new_region.exprdict = new_exprdict
        return new_region
    
    def empty():
        return Region([])

    def append_expr(self, expr):
        if expr.is_empty():
            return
        key = expr.terms_to_str()
        # 区域中不存在该不等式
        if self.exprdict.get(key) is None:
            self.exprdict[key] = expr
            self.exprs.append(expr)
        # 新区域比原区域紧,如何删除原表达式？
        elif expr.value > self.exprdict[key].value:
            self.exprdict[key] = expr
            self.exprs.append(expr)

    def sort_exprs(self):
        for expr in self.exprs:
            expr.sort_terms()
    
    def reduce_redundant_expr(self):
        exprdict = {}
        for expr in self.exprs:
            key = expr.terms_to_str()
            if exprdict.get(key) is None:
                exprdict[key] = expr
            else:
                exprdict[key] = expr if expr.value > exprdict[key].value else exprdict[key]
        self.exprs = list(exprdict.values())
        
    def save_region_to_txt(self, filename="Region.txt"):
        with open(filename, 'w') as f:
            for expr in self.exprs:
                f.write("%s\n" % expr)
                
    @staticmethod
    def read_from_txt(filename):
        """Create a Region object by reading expressions from a file"""
        exprs = []
        with open(filename, 'r') as f:
            for line in f:
                expr = Region.parse_expr(line.strip())
                exprs.append(expr)
        return Region(exprs)

    @staticmethod
    def parse_expr(line):
        """Parse an individual line into an Expr object"""
        # Match terms and coefficients
        term_pattern = r'([+-]?\d*)(H\(\{([A-Za-z0-9,]+)\}\))'
        terms = []
        total_value = 0
        eqtype = 'ge'  # Default to '>='
        
        # Find all terms in the line
        matches = re.findall(term_pattern, line)
        for match in matches:
            coefficient = match[0]
            term = match[1]
            variables = match[2].split(',')
            
            # Handle coefficient (could be empty, which means 1 or -1)
            if coefficient == '':
                coefficient = 1
            elif coefficient == '-':
                coefficient = -1
            else:
                coefficient = int(coefficient)
            
            terms.append((coefficient, variables))
        
        # Find inequality type (>=, <=, or other)
        if '>=' in line:
            eqtype = 'ge'
            total_value = int(line.split('>=')[-1].strip())
        elif '<=' in line:
            eqtype = 'le'
            total_value = int(line.split('<=')[-1].strip())
        elif '=' in line:
            eqtype = 'eq'
            total_value = int(line.split('=')[-1].strip())

        # Create and return an Expr object
        return Expr(terms, eqtype, total_value)
    
    # unfinished
    def FME(self, edict):
        entropylist = edict.redict.items()
        entropys = [val[0] for _, val in entropylist]
        print(entropys)
        # sort the exprs
        epoch = 0
        for entropy in entropys:
            if entropy == "W1,W2":
                continue
            plus_exprs = []
            minus_exprs = []
            index_count = 0
            index_list = []
            for expr in self.exprs:
                if expr.indexdict.get(entropy) is not None:
                    index_list.append(index_count)
                    index = expr.indexdict[entropy]
                    # normallize the coefficient
                    coef = expr.terms[index].coef
                    expr = np.abs(1/coef)*expr
                    if coef > 0:
                        plus_exprs.append(expr)
                    else:
                        minus_exprs.append(expr)
                index_count += 1
            # remove the exprs
            index_list.reverse()
            for index in index_list:
                self.exprs.pop(index)
            # get all exprs after removing an entropy
            for plus_expr in plus_exprs:
                for minus_expr in minus_exprs:
                    self.append_expr(plus_expr + minus_expr)
            # sort and simplify
            self.sort_exprs(edict)
            self.reduce_redundant_expr()
            print(f"reduced {entropy}")
            print(f"last exprs {len(self.exprs)}")
            epoch += 1
            if epoch == 6:
                break

class EntropyEqDict:
    """Dictionary of entropy equations"""

    def __init__(self):
        # key to value
        self.eqdict = {}
        # value to keys
        self.redict = {}

    def copy(self):
        newdict = EntropyEqDict()
        newdict.eqdict = self.eqdict.copy()
        newdict.redict = self.redict.copy()
        return newdict

    def get(self, key):
        return self.eqdict.get(key)

    def get_keys_by_value(self, value):
        return self.redict.get(value)

    def remove_keys_by_value(self, value):
        del self.redict[value]
    
    def remove_key(self, key):
        del self.eqdict[key]

    def batch_update(self, keys, value):
        for key in keys:
            self.eqdict[key] = value
        # self.regenerate_keys()

    def update_dict(self, other):
        # 更新 eqdict
        for key, value in other.eqdict.items():
            if key in self.eqdict:
                # 如果键已存在，则跳过
                continue
            else:
                self[key] = value

            # # 更新 redict
            # if value in self.redict:
            #     if key not in self.redict[value]:
            #         self.redict[value].append(key)
            # else:
            #     self.redict[value] = [key]


    def regenerate_keys(self):
        # 获取所有键值对并排序
        sorted_items = sorted(self.eqdict.items(), key=lambda item: (item[1], Iutils.sort_key(item[0])))
        # 重新赋值，使其连续不中断
        last_value = 0
        current_value = 0
        self.redict = {}
        for key, value in sorted_items:
            if last_value != value:
                last_value = value
                current_value += 1
            self[key] = current_value
        #     if self.redict.get(current_value) is None:
        #         self.redict[current_value] = [key]
        #     else:
        #         self.redict[current_value].append(key)
        # for key, value in self.redict.items():
        #     self.redict[key] = sorted(value, key=lambda v: (len(v.split(","))))

    def save_dict_to_txt(self):
        with open('EntropyEqDict.txt', 'w') as f:
            for key, value in self.eqdict.items():
                f.write("%s:%s\n" % (key, value))
        with open('EntropyReDict.txt', 'w') as f:
            for key, value in self.redict.items():
                f.write("%s:%s\n" % (key, value))

    def __setitem__(self, key, value):
        self.eqdict[key] = value
        if self.redict.get(value) is None:
            self.redict[value] = [key]
        else:
            self.redict[value].append(key)

    def __getitem__(self, key):
        return self.eqdict[key]

    def __contains__(self, key):
        return key in self.eqdict

    def __repr__(self) -> str:
        return str(self.eqdict) + "\n" + str(self.redict)
