# utils.py

# region general
import numpy as np
import torch
if not torch.cuda.is_available():
    print("warning! cuda is not available!")
from torch import nn
import torch.nn.functional as F
import random
import os
import time
import datetime
import copy
import pdb
from collections import defaultdict
import json
import multiprocessing
from multiprocessing import Pool
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 听说这个库可以助力实现并行运算；但是实际上一直用的都是matplotlib的agg哎。
from helpful_files.pentropy_v3 import Iutils # 主要是v3有些代码修改了一下，懒得改main了。直接用v3的代码。
from helpful_files.pentropy_main import calculate_square # ajusted for env_v2
from typing import Literal
import cProfile

class Config():
    def __init__(self, DBM = True, continue_training = False, envs_NK=[[2,2],[2,3]], use_pretrained_model=False, pretrained_model_id = None,pretrained_model_checkpoint = "latest", new_ineq_num_factor = 0.5, num_worker = 1, rewardtype:Literal["innerbound_ratio", "original"] = "original", test_mode = False, test_task_id = "000", test_task_models = [], test_num = 3, test_subject = "Nothing"):

        self.DBM = DBM
        self.continue_training = continue_training
        self.set_task_id() # 基于是否继续训练设置一个合理的id
        self.envs_NK = envs_NK # ,[3,2],[3,3]
        self.use_pretrained_model = use_pretrained_model
        self.pretrained_model_id = pretrained_model_id
        assert self.use_pretrained_model == False or self.pretrained_model_id != None
        assert self.use_pretrained_model == False or self.continue_training == False # 总之就是，从原来的最大的id继续训练（保存在该id处）；要么指定一个id，从这个id，训练出来一个新的id（自动寻找的最大的id）
        self.pretrained_model_checkpoint = pretrained_model_checkpoint
        assert (type(self.pretrained_model_checkpoint)==str and self.pretrained_model_checkpoint=="latest") or type(self.pretrained_model_checkpoint) == int
        self.new_ineq_num_factor = new_ineq_num_factor
        self.log_dir = f"02all_data/{self.task_id}"
        self.num_worker = num_worker
        self.rewardtype = rewardtype
        self.set_innerbound_reward() # 直接根据envs_NK设置内边界对应的奖励。记得在reward处进行奖励的更新。
        self.test_mode = test_mode
        assert (not (self.test_mode and self.use_pretrained_model)) and (not (self.test_mode and self.continue_training))
        self.test_task_id = test_task_id
        self.test_task_models = test_task_models
        self.test_num = test_num
        self.env_version = "v2" # in ["v1", "v2"]
        assert self.env_version in ["v1", "v2"], "env_version must be v1 or v2"
        assert self.env_version=="v1" or self.rewardtype!="innerbound_ratio", "converseenv_v1才可以使用innerbound_ratio"
        self.test_subject = test_subject
        self.cProfile_switch_on = False
        self.about_training = """ Testing the performance of the FiLM layer. Without FiLM here. """

    def set_task_id(self):
        # 设置数据目录路径（根据实际情况调整）
        data_dir = "02all_data"
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 获取目录中所有条目并筛选出三位数字格式的ID
        task_ids = []
        for entry in os.listdir(data_dir):
            if entry.isdigit() and len(entry) == 3:  # 假设ID是三位数字格式
                task_ids.append(int(entry))

        # 生成新ID（如果没有现有ID则从000开始，否则取最大ID+1）
        new_id = max(task_ids) + 1 if task_ids else 0
        if self.continue_training:
            self.task_id = f"{new_id - 1:03d}"  # 格式化为三位数字字符串
        else:
            self.task_id = f"{new_id:03d}"  # 格式化为三位数字字符串
    
    def set_policy_config(self, max_rollout_num = 500, n_directions = 4, evaluate_time = 3, learning_rate = 0.01, gamma = 0.97, std = 0.02, using_FiLM:bool=True):
        """
        :param: max_rollout_num: 最大rollout数，也就是最大进行多少次全部envs的rollout；取代了max_step
        """
        self.max_rollout_num = max_rollout_num
        self.n_directions = n_directions
        assert n_directions%2 == 0, "n_directions must be even"
        self.evaluate_time = evaluate_time
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.std = std
        self.rollout_length = 10 # 默认超过十次之后就直接结束了，不确定是否有用。
        self.using_FiLM = using_FiLM

    def print_config(self):
        print('-'*3,"config",'-'*3)
        for key, value in self.__dict__.items():
            print(f"* {key}: {value}")
        print('-'*3,"------",'-'*3)

    def set_attention_network_config(self,device = "cpu"):
        """
        directly from the original value.
        """
        self.network_param = {}
        self.network_param['hsize'] = 64
        self.network_param['numlayers'] = 2
        self.network_param['embed'] = 100
        self.network_param["numvars"] = 384-1 # obdim - 1
        #self.network_param['rowembed'] = 200
        self.device = device

    def set_innerbound_reward(self):
        """
        直接根据envs_NK设置内边界对应的奖励。
        """
        self.innerbound_reward = {}
        self.total_reward = {}
        for env_NK in self.envs_NK:
            innerbound = Iutils.plot_inner_bound(env_NK[0],env_NK[1],not_plot=True)
            cutsetbound = Iutils.plot_cutset_bound(env_NK[0],env_NK[1],point_num=12,not_plot=True)
            self.innerbound_reward[f"[{env_NK[0]},{env_NK[1]}]"] = np.exp(calculate_square(list(zip(innerbound[0],innerbound[1])))) - np.exp(calculate_square(list(zip(cutsetbound[0],cutsetbound[1]))))
            self.total_reward[f"[{env_NK[0]},{env_NK[1]}]"] = np.exp(calculate_square(list(zip(innerbound[0],innerbound[1])))) - np.exp(0)

def set_seed(args):
    """
    注意给env也要设置seed啊：
    state, _ = env.reset(seed=args.seed)
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def save_log(config:Config, log:dict):
    """
    保存log到指定的目录
    """
    log_path = f"02all_data/{config.task_id}/log.json"
    with open(log_path,"w+",encoding="utf-8") as f:
        json.dump(log, f,indent=4)
    print(f"log saved to {log_path}")

def load_log(config:Config):
    with open(f"02all_data/{config.task_id}/log.json","r") as f:
        log = json.load(f)
    return log

def create_new_log(config:Config):
    """
    主要修改在于将metadata改到了前面。
    log 结构都显示在这里了。
    注意我们这里似乎没有loss，因为我们没有target，所以没有差距这一说，相应的就没有loss的说法了。
    """
    log = {
        "meta_data":[
            {
                "config": {
                    k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                    for k, v in vars(config).items()
                },
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_time_in_seconds": time.time(),
                "end_time": None,
                "end_time_in_seconds": None,
            }
        ], # different training for different meta data
        "evaluate_average_rewards": [],
        "clocktime": [], # 指的是rollout的总时间（含参数更新）
        "results": [], # 每个rollout的并行计算原始输出，防止信息丢失；下面都是预处理之后的结果。# 结构上，每个worker有一个single_worker_rewards作为一次rollout的返回结果；各个worker的保存在workers_rewards中，然后一起放入rewards。最后几个是evaluate的结果啊。
        "evaluate_each_rewards":[],
        "evaluate_env_rewards": [], # 这个是每个env的平均奖励，shape为(rollout_num, evaluate_time)
        "rollout_time": [], # 值得是单纯的rollout环境的时间，不包含进行参数更新的时间。
    } # 不方便直接观察了，但是也可以在写一个函数实现动态可视化。
    return log

def add_new_meta_data(log:dict, config:Config):
    log["meta_data"].append({
        "config": {
            k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
            for k, v in vars(config).items()
        },
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time_in_seconds": time.time(),
        "end_time": None,
        "end_time_in_seconds": None,
    })

def save_results_2_log(results:list[tuple], log:dict, config:Config):
    """
    将results的结果存储在log之中。
    :param: results: [(0, [0.161408828237398, 0.688021175654887], [6, 6]), (1, [0.161408828237398, 0.6880211756548862], [6, 6]), (2, [0.161408828237398, 0.6880211756548862], [6, 6]), (3, [0.161408828237398, 0.688021175654887], [6, 6]), (4, [0.161408828237398, 0.688021175654887], [6, 6]), (5, [0.161408828237398, 0.6880211756548871], [6, 6]), (6, [0.161408828237398, 0.6880211756548862], [6, 6])]
    :param: log: 要存储的log
    """
    log["results"].append(results)
    log["evaluate_each_rewards"].append([results[config.n_directions + i][1] for i in range(config.evaluate_time)])
    log["evaluate_env_rewards"].append(np.mean(log["evaluate_each_rewards"],axis = 1)[-1].tolist())
    log["evaluate_average_rewards"].append(np.mean(log["evaluate_each_rewards"][-1]))
    assert len(log["evaluate_each_rewards"]) == len(log["evaluate_average_rewards"]) and len(log["results"]) == len(log["evaluate_average_rewards"]), "log长度错误！！"

# region envs
from converseenv_v2 import ConverseEnv
class ConverseEnvWrapper(ConverseEnv):
    def __init__(self, N=3,K=3,RENDER = "dont_render",epsilon_id=0, FORTEST_model:int|str = "latest", test_number:int=0):
        super().__init__(N=N,K=K,RENDER=RENDER)
        self.epsilon_id = epsilon_id
        self.FORTEST_model = FORTEST_model
        self.test_number = test_number
        
def make_same_epsilon_env(config,epsilon_id):
    envs = []
    for NK in config.envs_NK:
        env = ConverseEnvWrapper(N = NK[0], K = NK[1],epsilon_id=epsilon_id)
        envs.append(env)
    return envs

def make_converse_env(config:Config):
    envs = []
    total_epsilon = config.n_directions + config.evaluate_time
    for epsilon_id in range(total_epsilon):
        envs.append(make_same_epsilon_env(config,epsilon_id))
    if config.DBM:
        print(f"total envs（N,K,epsilon_id）: {[[[env.N,env.K,env.epsilon_id] for env in env_s] for env_s in envs]}")
    return envs

def make_same_model_env(config:Config, model_id:int|str, test_number):
    envs = []
    for NK in config.envs_NK:
        env = ConverseEnvWrapper(N = NK[0], K = NK[1], FORTEST_model=model_id,test_number=test_number)
        envs.append(env)
    return envs

def make_test_env(config:Config):
    """
    思路大致是，对全部的结果进行测试。
    全部的NK，生成一个list，具有相同的model；然后多个list，重复config.test_num次，同时使用不同的model，得到最后的结果唉。
    """
    envs = []
    for model in config.test_task_models:
        for test_number in range(config.test_num):
            envs.append(make_same_model_env(config=config,model_id=model,test_number=test_number))
    if config.DBM:
        print("test envs are: ", [[(env.N, env.K, env.FORTEST_model) for env in env_s] for env_s in envs])
    return envs

# region policy
class AttentionPolicy(nn.Module):
    """
    这一部分代码由ai直接使用pytorch重构。嗯，deepseek-R1。
    因为不是自己写的，所以这里对config的使用不足，就这样吧，懒得改了。
    人类对代码进行检查，感觉没问题，后续做debug再说吧。
    """
    def __init__(self, policy_params,device = "cpu",using_FiLM:bool=True):
        super().__init__()
        self.device = device
        self.numvars = policy_params['numvars']
        hsize = policy_params['hsize']
        numlayers = policy_params['numlayers']
        embeddeddim = policy_params['embed']
        self.using_FiLM = using_FiLM

        # 参数（NK）编码器
        if self.using_FiLM:
            self.param_embed_dim = 4
            self.param_encoder = nn.Sequential(
                nn.Linear(2, self.param_embed_dim),
                nn.Tanh()
            ).to(self.device)

        # 构建MLP和FiLM生成器
        self.layers = nn.ModuleList() # 这个是映射层哎。
        if self.using_FiLM:
            self.film_generators = nn.ModuleList()# 这个是FiLM生成器，用于生成每个层的scale和bias。
        
        input_dim = self.numvars + 1 # 输入维度处理
        for i in range(numlayers):
            if i == 0:
                layer = nn.Linear(input_dim, hsize)
            else:
                layer = nn.Linear(hsize, hsize)
            self.layers.append(layer.to(self.device))
            
            if self.using_FiLM:
                # FiML生成器
                film_gen = nn.Sequential(
                    nn.Linear(self.param_embed_dim, 2*hsize),
                    nn.Tanh()
                ).to(self.device)
                self.film_generators.append(film_gen)
        
        # 输出层
        self.final_layer = nn.Linear(hsize, embeddeddim)
        
        # 初始化权重
        self._init_weights()
        
        # 保持filter兼容，不使用filter。
        #self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.numvars+1,))
        
        self.t = 0

    def _init_weights(self):
        """
        对全部的非重复的nn.Linear进行初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def apply_film(self, x, param_embed, layer_idx):
        film_params = self.film_generators[layer_idx](param_embed)
        scale, bias = torch.chunk(film_params, 2, dim=-1)
        return scale * x + bias

    def forward(self, ob, update=True, num=1, n_value=None, k_value=None) -> list:
        t1 = time.time()
        A, b, c0, cutsa, cutsb = ob
        
        # 数据预处理
        baseob_original = torch.from_numpy(np.column_stack((A, b))).float().to(self.device)
        ob_original = torch.from_numpy(np.column_stack((cutsa, cutsb))).float().to(self.device)
        
        try:
            totalob_original = torch.cat([baseob_original, ob_original], dim=0)
        except Exception as e:
            print(f"Error concatenating: {e}")
            return []
        
        # 数据归一化
        totalob_original = (totalob_original - totalob_original.min()) / (totalob_original.max() - totalob_original.min() + 1e-8)
        
        # 应用filter
        # totalob_np = self.observation_filter(totalob_original.numpy(), update=update)
        # totalob = torch.from_numpy(totalob_np).float().to(self.device)
        totalob = totalob_original # 因为没有使用filter，在这里啊。

        # 分割数据
        baseob = totalob[:A.shape[0]]
        ob = totalob[A.shape[0]:]
        
        if self.using_FiLM:
            # 参数编码
            NK = torch.tensor([[n_value, k_value]], dtype=torch.float).to(self.device)
            NK_embed = self.param_encoder(NK)
        
        # 前向传播
        x_base = baseob
        for i, layer in enumerate(self.layers):
            x_base = layer(x_base)
            x_base = F.tanh(x_base)
            if self.using_FiLM:
                x_base = self.apply_film(x_base, NK_embed, i)
        base_embed = self.final_layer(x_base)
        
        x_ob = ob
        for i, layer in enumerate(self.layers):
            x_ob = layer(x_ob)
            x_ob = F.tanh(x_ob)
            if self.using_FiLM:
                x_ob = self.apply_film(x_ob, NK_embed, i)
        cut_embed = self.final_layer(x_ob)
        
        # 计算attention
        attention_map = torch.mm(cut_embed, base_embed.t())
        score = attention_map.mean(dim=1)
        
        # 概率计算
        score = score - score.max()
        prob = F.softmax(score, dim=0)
        
        # 动作选择
        if num == 1:
            action = torch.multinomial(prob, 1).item()
        else:
            _, indices = torch.topk(prob, num)
            action = indices.tolist()
        
        self.t += time.time() - t1
        return action

    def get_weights(self):
        """
        The get_weights method is already correct as it moves weights to CPU before converting to numpy arrays, 
        which is the standard practice since numpy only works with CPU data.
        """
        return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

    def update_weights(self, weights_dict):
        """
        Update the policy's weights using a dictionary of numpy arrays.
        """
        self.load_state_dict({k: torch.from_numpy(v).to(self.device) for k, v in weights_dict.items()})

    # def get_weights_plus_stats(self):
    #     mu, std = self.observation_filter.get_stats()
    #     return self.get_weights(), mu, std

# region rollout
def create_epsilon_table(config:Config,policy:AttentionPolicy):
    """
    创建epsilon表。是的，对称建立的哎。
    基本没有问题哎。
    """
    epsilon_table = []
    # 原始参数
    original_state_dict = policy.state_dict()

    for _ in range(config.n_directions // 2):
        epsilon = {}
        neg_epsilon = {}
        for key, param in original_state_dict.items():
            epsilon[key] = torch.randn_like(param) * config.std # 原来是正太分布，现在是config.std的正态分布
            neg_epsilon[key] = -epsilon[key]
        epsilon_table.append(epsilon)
        epsilon_table.append(neg_epsilon)
    
    assert len(epsilon_table) == config.n_directions, "len(epsilon_table) != config.n_directions"
    return epsilon_table

def rollout_test_workers(envs_s:list[list[ConverseEnvWrapper]], original_policy:AttentionPolicy,config:Config):
    """
    测试用的rollout函数。
    尝试使用GPU在这里，因为，似乎以其他的方式解决了之前在训练时候的设备的问题。
    """
    num_task = config.test_num * len(config.test_task_models)
    num_workers = config.num_worker
    results = []
    # try not put it to cpu, error occurred.
    original_policy = original_policy.to("cpu")
    original_policy.device = "cpu"
    try:
        with Pool(processes=num_workers) as pool:
            try:
                async_results = []
                for i in range(num_task):
                    policy = copy.deepcopy(original_policy)
                    model_id = envs_s[i][0].FORTEST_model
                    if type(model_id) == int:
                        policy_path = f"02all_data/{config.test_task_id}/checkpoint/model_{model_id}.pth"
                    elif type(model_id) == str:
                        policy_path = f"02all_data/{config.test_task_id}/model.pth"
                    policy.load_state_dict(torch.load(policy_path))
                    async_results.append(
                        pool.apply_async(rollout_envs, args=(
                            envs_s[i],
                            policy,
                            config,
                            model_id
                        ))
                    )
                results = [list(async_result.get()) for async_result in async_results]
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                pool.terminate()
                pool.join()
                raise KeyboardInterrupt
    except:
        print("Error in multiprocessing")
    return results

def rollout_workers(envs_s:list,epsilon_table:list,original_policy:AttentionPolicy,config:Config):
    """
    并行计算rollout
    参考文件01example/多进程测试文件/07
    添加了keyboardInterrupt的处理。
    :param: envs_s: 环境列表，而且是双层的哎。
    """
    num_task = config.evaluate_time + config.n_directions
    num_worker = config.num_worker
    results = []
    original_policy_cpu = copy.deepcopy(original_policy).to("cpu")
    original_policy_cpu.device = "cpu"  # 这里和前面不太一样，这里的device是我们自己定义的，主要再forward的时候决定新建的tensor的位置；而前面的to cpu是把tensor转移到cpu上哎。
    try:
        with Pool(processes=num_worker) as pool:
            try:
                epsilon_ids = range(num_task) # 值得包含evaluate的全部的ids啊
                async_results = []
                for epsilon_id in epsilon_ids:
                    policy = copy.deepcopy(original_policy_cpu)  
                    if epsilon_id < config.n_directions:
                        added_state_dict = add_state_dict(policy.state_dict(),epsilon_table[epsilon_id],"cpu")
                        policy.load_state_dict(added_state_dict)
                        hasdiff = any(not torch.equal(original_policy_cpu.state_dict()[key],policy.state_dict()[key]) for key in policy.state_dict())
                        assert hasdiff, "policy is equal to origional policy."
                    async_results.append(
                        pool.apply_async(rollout_envs,args=(
                            envs_s[epsilon_id],
                            policy,
                            config,
                            epsilon_id
                        ))
                    )
                # 下面是错误代码哎，因为这里没有
                # async_results = [pool.apply_async(rollout_envs,args=(
                #     envs_s[i],
                #     copy.deepcopy(original_policy_copy).load_state_dict(add_state_dict(original_policy_copy.state_dict(),epsilon_table[i],"cpu")) if i<config.n_directions else copy.deepcopy(original_policy_copy),
                #     config,
                #     i
                # )) for i in epsilon_ids]
                results = [list(async_result.get()) for async_result in async_results]
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                pool.terminate()
                pool.join()
                raise KeyboardInterrupt
    except:
        print("Error in multiprocessing")
    return results


def rollout_envs(envs:list[ConverseEnvWrapper], policy:AttentionPolicy, config:Config, epsilon_id):
    """
    直接来自alg_utils移植，只是修改了参数传递的方式。
    adding NEWINQ_NUM, modified by Yuan at 2025/5/18
    adding NEWINQ_NUMlist, modified by Yuan at 2025/5/27
    已经对输出进行了重定向。重定向的位置是：02all_data/{config.task_id}/rollout_output/{epsilon_id}.txt下面
    :param:epsilon_id: epsilon_id when training; model_id when testing.
    """
    if config.test_mode:
        assert envs[0].FORTEST_model == epsilon_id, "envs[0].FORTEST_model != epsilon_id"
        os.makedirs(f"02all_data/{config.test_task_id}/test_results", exist_ok=True)
        output_path = f"02all_data/{config.test_task_id}/test_results/model={epsilon_id}_pid={os.getpid()}.txt"
    else:
        assert envs[0].epsilon_id == epsilon_id, "envs[0].epsilon_id != epsilon_id"
        os.makedirs(f"02all_data/{config.task_id}/rollout_output",exist_ok=True)
        output_path = f"02all_data/{config.task_id}/rollout_output/{epsilon_id}.txt"
    with open(output_path,"w+",encoding="utf-8") as f:
        sys.stdout = f
        sys.stderr = f # tqdm默认输出

        rewards = []
        times = []
        for i, env in enumerate(envs):
            if config.cProfile_switch_on:
                pr = cProfile.Profile()
                pr.enable()
            r, t = rollout(env= env, policy=policy, rollout_length=config.rollout_length, gamma=config.gamma, NumFactor=config.new_ineq_num_factor,config = config)
            rewards.append(r)
            times.append(t)
            if config.cProfile_switch_on:
                pr.disable()
                pr.print_stats(sort='cumtime')

    return epsilon_id, rewards, times

def rollout(env:ConverseEnvWrapper, policy, rollout_length, gamma, NumFactor,config:Config):
    """
    删掉num_rollouts，因为都是只做一次rollout就好了。针对的是一个环境进行rollout哎。
    修改删除act，直接使用policy处理ob等的。
    without rewriting this function.by Yuan at 2025/5/18.
    change NUM to NumFactor to use Factor here.
    """
    this_rollout_start_time = datetime.datetime.now()
    ob, _ = env.reset()
    factor = 1.0 # factor is gamma**n
    #ob = env.reset()
    done = False
    t = 0
    rsum = 0
    while not done and t <= rollout_length:
        if config.test_mode == True and config.env_version == "v1":
            time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            total_reward = 0.0 if len(env.rewardlist) == 0 else sum(env.rewardlist)
            env.render(namestring=f"env_N={env.N}_K={env.K}_model={env.FORTEST_model}_test={env.test_number}_step={t}_time={time_now}_totalreward={total_reward}", save_dir = f"02all_data/{config.test_task_id}/test_pics/env_v2_test")
        action = policy(ob,num=int(NumFactor*len(ob[-1])),n_value=env.N, k_value=env.K)
        #print(action)
        #time.sleep(60)
        ob, r, done = env.step(action)
        if config.test_mode == True and config.env_version == "v2":
            # 因为必须要在step之后进行render，所以需要分别处理两个不同的环境。
            time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            total_reward = sum(env.reward_list)
            total_time = str(datetime.datetime.now() - this_rollout_start_time).replace(':', '-').replace('.', '-')
            env.render(namestring=f"model={env.FORTEST_model}_test={env.test_number}_step={t}_time={time_now}_totaltime={total_time}_totalreward={round(total_reward,5)}", save_dir = f"02all_data/{config.test_task_id}/test_pics/{config.test_subject}/{env.N}_{env.K}")
        rsum += r * factor
        factor *= gamma
        t += 1
        print("select action:",action, " from len of ",len(ob[-1])) 
    #rewards.append(rsum)
    #times.append(t)
    if config.rewardtype == "innerbound_ratio":
        rsum = rsum / config.innerbound_reward[f"[{env.N},{env.K}]"]
    return rsum,t # 每个环境的返回值从[]重塑为float

def add_state_dict(state_dict1, state_dict2,device):
    """
    将两个state_dict逐元素相加，done by deepseek V3
    :param state_dict1: 第一个模型状态字典
    :param state_dict2: 第二个模型状态字典
    :return: 相加后的新状态字典
    """
    result = {}
    for key in state_dict1:
        if key in state_dict2:
            # 确保张量在相同设备上
            tensor1 = state_dict1[key].to(device)
            tensor2 = state_dict2[key].to(device)
            result[key] = tensor1 + tensor2
        else:
            raise KeyError(f"Key {key} not found in both state_dicts")
    return result

# region update
def get_grad(epsilon_table:list,config:Config,log:dict):
    """
    按照计算grad的公式：grad = np.mean(epsilon_table * train_rewards_table[:,np.newaxis], axis=0) / delta_std
    其中train_rewards_table已经进行了归一化处理，也就是减去了均值，除以了标准差。
    """
    epsilon_weight = [result[1] for result in log["results"][-1]]
    epsilon_weight = epsilon_weight[:config.n_directions]
    epsilon_weight = np.mean(epsilon_weight, axis = 1)
    norm_epsilon_weight = ( epsilon_weight - np.mean(epsilon_weight) )/ (np.std(epsilon_weight) + 1e-8)
    
    grad = {}
    for i, epsilon in enumerate(epsilon_table):
        for key in epsilon:
            value = epsilon[key]
            if i == 0:
                grad[key] = value*norm_epsilon_weight[i]/(config.n_directions * config.std)
            else:
                grad[key] += value*norm_epsilon_weight[i]/(config.n_directions * config.std)
    return grad

# region render
def draw_rewards(log:dict, config:Config):
    """
    绘制奖励曲线，同时进行保存操作。
    """
    save_path = f"02all_data/{config.task_id}/reward.png"
    plt.figure(figsize=(12,6))

    # plot here
    plt.plot(log["evaluate_average_rewards"], label='Average Rewards', alpha = 1, color = "red")
    envs_reward_dict = {}
    for env_id in range(len(config.envs_NK)):
        envs_reward_dict[env_id] = [ log["evaluate_env_rewards"][i][env_id] for i in range(len(log["evaluate_env_rewards"])) ]
    assert len(envs_reward_dict[0]) == len(log["evaluate_env_rewards"])
    for env_id in range(len(config.envs_NK)):
        plt.plot(envs_reward_dict[env_id],label=str(config.envs_NK[env_id]),alpha = 0.3)

    plt.xlabel('Evaluation rollout number')
    plt.ylabel('Rewards')
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.savefig(save_path)
    plt.close()

def draw_rewards_20points(log:dict, config:Config):
    """
    绘制奖励曲线，同时进行保存操作。
    """
    save_path = f"02all_data/{config.task_id}/reward.png"
    plt.figure(figsize=(12,6))

    # 提取奖励数据
    rewards_data = log["evaluate_average_rewards"]
    
    # 第一步：绘制平均奖励曲线（所有点都显示）
    plt.plot(rewards_data, label='Average Rewards', alpha=1, color="red")
    # 添加所有数据点（使用散点图）
    

    # 标注点选择（与原始逻辑相同）
    total_points = len(rewards_data)
    max_annotations = 20
    
    if total_points <= max_annotations:
        annotation_indices = list(range(total_points))
    else:
        step = total_points / max_annotations
        annotation_indices = [int(i * step) for i in range(max_annotations)]
        if annotation_indices[-1] != total_points - 1:
            annotation_indices[-1] = total_points - 1
    
    # 为平均奖励曲线添加数值标注
    for pos in annotation_indices:
        if 0 <= pos < total_points:
            plt.scatter(pos, rewards_data[pos], color='red', s=10, zorder=5, alpha=0.7)
            reward_val = rewards_data[pos]
            plt.annotate(f'{reward_val:.3f}', 
                        xy=(pos, reward_val), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        zorder=6)  # 提高zorder确保标注在顶层

    # 处理每个环境的奖励数据
    envs_reward_dict = {}
    for env_id in range(len(config.envs_NK)):
        envs_reward_dict[env_id] = [log["evaluate_env_rewards"][i][env_id] 
                                   for i in range(len(log["evaluate_env_rewards"]))]
    
    # 颜色列表用于区分不同环境
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 绘制每个环境的奖励曲线（所有点都显示）
    for env_id in range(len(config.envs_NK)):
        color = colors[env_id % len(colors)]
        env_rewards = envs_reward_dict[env_id]
        
        # 绘制曲线和所有数据点
        plt.plot(env_rewards, alpha=0.5, color=color, label=str(config.envs_NK[env_id]))
        
        
        # 为每个环境曲线添加数值标注（仅在选定点上）
        for pos in annotation_indices:
            if 0 <= pos < len(env_rewards):
                plt.scatter(pos, env_rewards[pos], color=color, s=5, alpha=0.7, zorder=4)
                reward_val = env_rewards[pos]
                plt.annotate(f'{reward_val:.3f}', 
                            xy=(pos, reward_val), 
                            xytext=(5, -10), 
                            textcoords='offset points',
                            fontsize=6,
                            alpha=0.8,
                            color=color,
                            zorder=6)

    plt.xlabel('Evaluation rollout number')
    plt.ylabel('Rewards')
    plt.title('Training Rewards (All Points Displayed)')
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def draw_times(log:dict, config:Config):
    """
    绘制时间曲线，同时进行保存操作。
    """
    save_path = f"02all_data/{config.task_id}/time.png"
    plt.figure()
    plt.plot(log["clocktime"], label='Clock Time', alpha = 1, color = "red")
    plt.scatter(range(len(log["clocktime"])), log["clocktime"], color='red', s=30, zorder=5,)
    plt.xlabel(f'Evaluation with {config.num_worker} workers and {config.new_ineq_num_factor}% added')
    plt.ylabel('Time')
    plt.legend()
    plt.grid(alpha = 0.2)
    plt.savefig(save_path)
    plt.close()

def explain_test_results(results:list[list[int|list[list]]], config:Config):
    """
    test done.
    :return: reward_results: row for different model; column for different envs
    """
    reward_results = []
    for model_num, model_id in enumerate(config.test_task_models):
        this_model_result = [results[i + model_num * config.test_num][1] for i in range(config.test_num)]
        this_model_result = np.mean(np.array(this_model_result),axis = 0)
        reward_results.append(this_model_result.tolist())
    
    format_str = " ".join(["{: <15}" for _ in range(len(config.envs_NK) + 1)])
    print("tested task id: ", config.test_task_id)
    print(format_str.format(*["model\\env_NK",*[str(NK) for NK in config.envs_NK]]))
    for row,model_id in zip(reward_results,config.test_task_models):
        print(format_str.format(*[model_id,*row]))
    return reward_results

# region MAIN
if __name__ == "__main__":
    """
    测试结果表明，基本上是一致的，除了原始的result使用的是tuple，而reload之后使用的是list.
    """
    log = {'meta_data': [{'config': {'DBM': True, 'continue_training': False, 'task_id': '005', 'envs_NK': [[2, 2], [2, 3]], 'use_pretrained_model': False, 'pretrained_model_id': 'None', 'new_ineq_num_factor': 0.5, 'log_dir': '02all_data/005', 'num_worker': 3, 'about_training': '改成3个看看并行计算的效果哎。\n **   ** \n**** ****\n*********\n *******\n  *****\n   ***\n    *', 'max_rollout_num': 1000, 'n_directions': 4, 'evaluate_time': 3, 'learning_rate': 0.01, 'gamma': 0.97, 'std': 0.02, 'rollout_length': 10, 'network_param': {'hsize': 64, 'numlayers': 2, 'embed': 100, 'numvars': 383}, 'device': 'cuda'}, 'start_time': '2025-08-03 14:59:40', 'start_time_in_seconds': 1754204380.1347613, 'end_time': None, 'end_time_in_seconds': None}, {'config': {'DBM': True, 'continue_training': True, 'task_id': '005', 'envs_NK': [[2, 2], [2, 3]], 'use_pretrained_model': False, 'pretrained_model_id': 'None', 'new_ineq_num_factor': 0.5, 'log_dir': '02all_data/005', 'num_worker': 2, 'about_training': 'continue training test here.as expected should be added to 005.\n **   ** \n**** ****\n*********\n *******\n  *****\n   ***\n    *', 'max_rollout_num': 1000, 'n_directions': 4, 'evaluate_time': 3, 'learning_rate': 0.01, 'gamma': 0.97, 'std': 0.02, 'rollout_length': 10, 'network_param': {'hsize': 64, 'numlayers': 2, 'embed': 100, 'numvars': 383}, 'device': 'cuda'}, 'start_time': '2025-08-03 15:15:01', 'start_time_in_seconds': 1754205301.849048, 'end_time': None, 'end_time_in_seconds': None}, {'config': {'DBM': True, 'continue_training': True, 'task_id': '005', 'envs_NK': [[2, 2], [2, 3]], 'use_pretrained_model': False, 'pretrained_model_id': 'None', 'new_ineq_num_factor': 0.5, 'log_dir': '02all_data/005', 'num_worker': 1, 'about_training': 'continue training test here.as expected should be added to 005.\n **   ** \n**** ****\n*********\n *******\n  *****\n   ***\n    *', 'max_rollout_num': 1000, 'n_directions': 4, 'evaluate_time': 3, 'learning_rate': 0.01, 'gamma': 0.97, 'std': 0.02, 'rollout_length': 10, 'network_param': {'hsize': 64, 'numlayers': 2, 'embed': 100, 'numvars': 383}, 'device': 'cuda'}, 'start_time': '2025-08-03 15:19:58', 'start_time_in_seconds': 1754205598.3363388, 'end_time': None, 'end_time_in_seconds': None}], 'evaluate_average_rewards': [0.10452009330354793, 0.161408828237398, 0.161408828237398, 0.161408828237398, 0.161408828237398, 0.161408828237398, 0.24554696541202006, 0.24554696541202023, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037, 0.24554696541202023, 0.24554696541202023, 0.24554696541202037, 0.24554696541202037, 0.24554696541202023, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037, 0.24554696541202037], 'clocktime': [19.99359703063965, 20.401638507843018, 19.43332004547119, 18.861104249954224, 19.172513246536255, 20.27242684364319, 19.61225724220276, 19.733516454696655, 22.218499183654785, 21.85263204574585, 22.10595464706421, 22.088513374328613, 21.997992515563965, 21.69546866416931, 21.194518089294434, 21.402024030685425, 22.09985637664795, 28.315980434417725, 28.218089818954468, 27.303302764892578, 27.463107347488403, 28.074164867401123], 'results': [[[0, [0.161408828237398, 0.3296851025866419], [6, 6]], [1, [0.161408828237398, 0.0], [6, 5]], [2, [0.0, 0.161408828237398], [5, 6]], [3, [0.161408828237398, 0.161408828237398], [6, 6]], [4, [0.1428940751090934, 0.161408828237398], [10, 6]], [5, [0.0, 0.16140882823739802], [5, 6]], [6, [0.0, 0.16140882823739808], [5, 6]]], [[0, [0.161408828237398, 0.0933369136118829], [6, 6]], [1, [0.0, 0.161408828237398], [5, 6]], [2, [0.161408828237398, 0.17172192989276924], [6, 6]], [3, [0.161408828237398, 0.161408828237398], [6, 6]], [4, [0.161408828237398, 0.16140882823739802], [6, 6]], [5, [0.161408828237398, 0.16140882823739802], [6, 6]], [6, [0.161408828237398, 0.161408828237398], [6, 6]]], [[0, [-1.6212329256859447e-15, 0.161408828237398], [5, 6]], [1, [0.161408828237398, 0.0], [6, 5]], [2, [0.161408828237398, 0.32968510258664185], [6, 6]], [3, [0.161408828237398, 0.0], [6, 5]], [4, [0.161408828237398, 0.161408828237398], [6, 6]], [5, [0.161408828237398, 0.161408828237398], [6, 6]], [6, [0.161408828237398, 0.161408828237398], [6, 6]]], [[0, [0.0, 0.0], [5, 5]], [1, [0.161408828237398, 0.161408828237398], [6, 6]], [2, [0.161408828237398, 0.0], [6, 5]], [3, [0.161408828237398, 0.0], [6, 5]], [4, [0.161408828237398, 0.161408828237398], [6, 6]], [5, [0.161408828237398, 0.161408828237398], [6, 6]], [6, [0.161408828237398, 0.161408828237398], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.161408828237398], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.0, 0.161408828237398], [5, 6]], [4, [0.161408828237398, 0.161408828237398], [6, 6]], [5, [0.161408828237398, 0.161408828237398], [6, 6]], [6, [0.161408828237398, 0.161408828237398], [6, 6]]], [[0, [0.161408828237398, 0.0], [6, 5]], [1, [0.161408828237398, 0.16140882823739883], [6, 6]], [2, [0.0, 0.14731347949391074], [5, 9]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.16140882823739805], [6, 6]], [5, [0.161408828237398, 0.161408828237398], [6, 6]], [6, [0.161408828237398, 0.16140882823739802], [6, 6]]], [[0, [0.15656656339027605, 0.16140882823739802], [7, 6]], [1, [0.0, 0.0], [5, 5]], [2, [0.0, 0.161408828237398], [5, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664185], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664185], [6, 6]]], [[0, [0.161408828237398, 0.32463681435616454], [6, 7]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.161408828237398, 0.161408828237398], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664185], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.15186956648856775, 0.32968510258664274], [8, 6]], [1, [0.161408828237398, 0.32968510258664196], [6, 6]], [2, [0.161408828237398, 0.3296851025866419], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.0, 0.3246368143561654], [5, 7]], [1, [0.161408828237398, 0.32968510258664196], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.3296851025866427], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.161408828237398, 0.3296851025866419], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.3296851025866419], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.161408828237398], [6, 6]], [1, [0.1428940751090934, 0.32968510258664274], [10, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.3296851025866419], [6, 6]], [5, [0.161408828237398, 0.3296851025866427], [6, 6]], [6, [0.161408828237398, 0.3296851025866427], [6, 6]]], [[0, [0.161408828237398, 0.161408828237398], [6, 6]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.15186956648856775, 0.32968510258664274], [8, 6]], [3, [0.0, 0.32968510258664274], [5, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.0, 0.0], [5, 5]], [1, [0.161408828237398, 0.0], [6, 5]], [2, [0.161408828237398, 0.32968510258664196], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664196], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.14731347949391074, 0.32968510258664274], [9, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.3296851025866427], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.0, 0.32968510258664274], [5, 6]], [1, [0.161408828237398, 0.0], [6, 5]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.15656656339027605], [6, 7]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.0, 0.32968510258664274], [5, 6]], [1, [0.161408828237398, 0.161408828237398], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.0, 0.32968510258664274], [5, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.1565665633902768], [6, 7]], [1, [0.0, 0.32968510258664274], [5, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.3149900403765462], [6, 9]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]], [[0, [0.161408828237398, 0.32968510258664274], [6, 6]], [1, [0.161408828237398, 0.32968510258664274], [6, 6]], [2, [0.161408828237398, 0.32968510258664274], [6, 6]], [3, [0.161408828237398, 0.32968510258664274], [6, 6]], [4, [0.161408828237398, 0.32968510258664274], [6, 6]], [5, [0.161408828237398, 0.32968510258664274], [6, 6]], [6, [0.161408828237398, 0.32968510258664274], [6, 6]]]], 'evaluate_each_rewards': [[[0.1428940751090934, 0.161408828237398], [0.0, 0.16140882823739802], [0.0, 0.16140882823739808]], [[0.161408828237398, 0.16140882823739802], [0.161408828237398, 0.16140882823739802], [0.161408828237398, 0.161408828237398]], [[0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398]], [[0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398]], [[0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398]], [[0.161408828237398, 0.16140882823739805], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.16140882823739802]], [[0.161408828237398, 0.32968510258664185], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664185]], [[0.161408828237398, 0.32968510258664185], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.3296851025866419], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.3296851025866419], [0.161408828237398, 0.3296851025866427], [0.161408828237398, 0.3296851025866427]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664196], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], [[0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]]], 'evaluate_env_rewards': [[0.047631358369697806, 0.16140882823739802], [0.161408828237398, 0.16140882823739802], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.161408828237398], [0.161408828237398, 0.16140882823739802], [0.161408828237398, 0.32968510258664213], [0.161408828237398, 0.32968510258664246], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664246], [0.161408828237398, 0.32968510258664246], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664246], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274], [0.161408828237398, 0.32968510258664274]], 'rollout_time': [19.94765853881836, 20.396638870239258, 19.428133726119995, 18.85479187965393, 19.166483163833618, 20.266430139541626, 19.60674214363098, 19.727531909942627, 22.175246953964233, 21.841633319854736, 22.09855818748474, 22.08252215385437, 21.9869863986969, 21.689460515975952, 21.18757390975952, 21.396015882492065, 22.089462995529175, 28.26794934272766, 28.20909070968628, 27.29230046272278, 27.45611023902893, 28.066166877746582, 27.25330376625061]}
    config = Config()
    os.makedirs(f"02all_data/{config.task_id}",exist_ok=True)
    draw_rewards_20points(log,config)
    draw_times(log,config)